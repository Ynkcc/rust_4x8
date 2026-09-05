"""banqi/nnue/samples.py — NNUE 样本流式累积器。

把 NNUE 样本的「解析/过滤/标签混合」逻辑从 NnueSampleDataset 中抽出，
使其既能从 JSONL 文件批量装载（离线 CLI），也能在主训练闭环中逐个
episode dict 流式摄入（NnueDistillWorker 蒸馏），二者的数据契约一致
（PyO3 episode_to_dict / serialize.rs JSON）：

- nnue_meta:     特征布局元信息（feature_dim 等，随变体推导）
- nnue_features: {"mover": [[索引...]每步], "opponent": [[索引...]每步]}
- completed_qs / mcts_values: 搜索价值（行棋方视角）
- game_results:  终局回报（行棋方视角）
- is_full_search: 算力随机化标记

标签: y = value_weight * 搜索价值 + (1 - value_weight) * 终局回报。
"""

from __future__ import annotations

import json
import logging
import threading
from typing import Dict, List, Optional

import torch

logger = logging.getLogger(__name__)


class NnueSampleBuffer:
    """NNUE 训练样本的线程安全流式累积器。

    add_episode 直接吃主闭环自对弈产出的 episode dict；
    ingest_jsonl 复用同一解析路径装载离线 JSONL。
    feature_dim 由首条 nnue_meta 推导，维度不一致的样本静默丢弃并计数。
    """

    def __init__(
        self,
        value_source: str = "completed_q",
        value_weight: float = 0.7,
        full_only: bool = False,
        dual_perspective: bool = False,
        max_samples: int = 2_000_000,
    ) -> None:
        if value_source not in ("completed_q", "mcts_value"):
            raise ValueError(f"未知 value_source: {value_source}")
        self.value_key = value_source + "s"  # episode dict 字段为复数形式
        self.value_weight = float(value_weight)
        self.full_only = bool(full_only)
        self.dual_perspective = bool(dual_perspective)
        self.max_samples = int(max_samples)

        self.feature_dim: Optional[int] = None
        self.features: List[List[int]] = []
        self.targets: List[float] = []
        # 统计：摄入 episode 数 / 缺 NNUE 字段跳过数 / 维度不匹配丢弃数
        self.episodes = 0
        self.skipped_episodes = 0
        self.dropped_episodes = 0
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ #
    # 摄入
    # ------------------------------------------------------------------ #
    def add_episode(self, ep: Dict) -> int:
        """摄入一个 episode dict，返回本次新增样本数（0 = 被跳过/丢弃）。

        缺 nnue_features / nnue_meta 字段（自对弈未开 collect_nnue_features）
        时静默跳过并计数，不抛异常——蒸馏 worker 依赖该行为安全旁路。
        """
        nnue = ep.get("nnue_features")
        meta = ep.get("nnue_meta")
        if nnue is None or meta is None:
            with self._lock:
                self.skipped_episodes += 1
            return 0
        dim = int(meta["feature_dim"])
        with self._lock:
            if self.feature_dim is None:
                self.feature_dim = dim
            elif dim != self.feature_dim:
                self.dropped_episodes += 1
                return 0
            n = self._append_locked(
                nnue["mover"],
                nnue.get("opponent") or [],
                ep[self.value_key],
                ep["game_results"],
                ep.get("is_full_search") or [True] * len(nnue["mover"]),
            )
            self.episodes += 1
            self._trim_locked()
        return n

    def ingest_jsonl(self, path: str) -> int:
        """从 episode JSONL 文件批量装载（与离线 CLI 契约一致）。"""
        n_eps = 0
        with open(path, encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    ep = json.loads(line)
                except json.JSONDecodeError as e:
                    logger.warning("跳过非法 JSON 行 %s:%d: %s", path, line_no, e)
                    continue
                if self.add_episode(ep):
                    n_eps += 1
        logger.info("加载 %s: %d 个 episode, 累计 %d 个 NNUE 样本", path, n_eps, len(self))
        return n_eps

    def _append_locked(
        self,
        movers: List[List[int]],
        opponents: List[List[int]],
        values: List[float],
        results: List[float],
        full_flags: List[bool],
    ) -> int:
        w = self.value_weight
        n = 0
        for i, feats in enumerate(movers):
            if self.full_only and not full_flags[i]:
                continue
            self.features.append(feats)
            self.targets.append(w * values[i] + (1.0 - w) * results[i])
            n += 1
            if self.dual_perspective and i < len(opponents):
                self.features.append(opponents[i])
                self.targets.append(
                    w * (-values[i]) + (1.0 - w) * (-results[i])
                )
                n += 1
        return n

    def _trim_locked(self) -> None:
        """超过容量上限时丢弃最旧样本（FIFO），防止常驻闭环内存无界增长。"""
        if self.max_samples > 0 and len(self.features) > self.max_samples:
            excess = len(self.features) - self.max_samples
            del self.features[:excess]
            del self.targets[:excess]

    # ------------------------------------------------------------------ #
    # 导出
    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return len(self.features)

    def to_tensors(self):
        """导出 (features, targets, feature_dim)。空 buffer 返回 (None, None, None)。"""
        with self._lock:
            if not self.features or self.feature_dim is None:
                return None, None, None
            feats = [list(f) for f in self.features]
            tgts = torch.tensor(self.targets, dtype=torch.float32).unsqueeze(1).clone()
            dim = self.feature_dim
        return (feats, tgts, dim)

    def to_dataset(self):
        """物化为 NnueSampleDataset（惰性导入避免 train.py 循环依赖）。"""
        from banqi.nnue.train import NnueSampleDataset

        feats, tgts, dim = self.to_tensors()
        if feats is None:
            raise ValueError("NnueSampleBuffer 为空，无法构建数据集")
        return NnueSampleDataset(
            jsonl_paths=None, features=feats, targets=tgts, feature_dim=dim
        )

    def stats(self) -> Dict[str, int]:
        with self._lock:
            return {
                "samples": len(self.features),
                "episodes": self.episodes,
                "skipped_episodes": self.skipped_episodes,
                "dropped_episodes": self.dropped_episodes,
            }
