"""banqi/training/buffer.py — 向量化 replay buffer 与 episode 转换。

DataBuffer 把自对弈 episode 转换出的 sample dict 列表压缩进内存向量
（board / scalar / policy / value / mask / root_visit），并按 config
的 VALUE_TARGET_MODE 计算价值目标、过滤异常样本。
"""

from __future__ import annotations

import numpy as np
import torch
from typing import Dict, List

from banqi.constants import build_constants
from banqi.variant import Variant


def episode_to_samples(episode_dict: Dict) -> List[Dict]:
    """
    把一个 episode dict（来自 self_play 队列）转换为 DataBuffer 可消费的
    sample dict 列表，字段与 Mongo GameDocument.samples 一致
    （含 health_diff，与归档数据同步）。
    """
    samples = []
    n = len(episode_dict["boards"])
    health_diffs = episode_dict.get("health_diffs") or [0.0] * n
    # 策略头验证 ground truth：
    #   - rule_selfplay 数据带 teacher_actions（温度采样前的启发式/规则最优动作）
    #   - 自对弈数据带 actions（MCTS 实际选择的最优动作）作为 fallback
    teacher_actions = episode_dict.get("teacher_actions")
    actions = episode_dict.get("actions")
    for step_idx, (board, scalar, policy, mcts_val, completed_q,
                    root_visit, game_result, mask) in enumerate(zip(
        episode_dict["boards"], episode_dict["scalars"], episode_dict["policies"],
        episode_dict["mcts_values"], episode_dict["completed_qs"],
        episode_dict["root_visits"], episode_dict["game_results"],
        episode_dict["action_masks"],
    )):
        teacher_action = None
        if teacher_actions is not None and step_idx < len(teacher_actions):
            teacher_action = int(teacher_actions[step_idx])
        elif actions is not None and step_idx < len(actions):
            teacher_action = int(actions[step_idx])
        samples.append({
            "board_state": board,
            "scalar_state": scalar,
            "policy_probs": policy,
            "mcts_value": float(mcts_val),
            "completed_q": float(completed_q),
            "root_visit_count": int(root_visit),
            "game_result_value": float(game_result),
            "action_mask": mask,
            "teacher_action": teacher_action,
            "health_diff": float(health_diffs[step_idx]),
        })
    return samples


class DataBuffer:
    """向量化缓冲区（维度来自变体），value 目标按 config.VALUE_TARGET_MODE 计算。"""

    def __init__(self, capacity: int, variant: Variant, cfg) -> None:
        self.capacity = capacity
        self.variant = variant
        self.cfg = cfg
        self.C = build_constants(variant)
        self.boards: List[np.ndarray] = []
        self.scalars: List[np.ndarray] = []
        self.probs: List[np.ndarray] = []
        self.values: List[float] = []
        self.masks: List[np.ndarray] = []
        self.root_visits: List[int] = []
        # anneal 模式下 game_result 的权重（0~1），由 TrainWorker 按轮更新
        self.value_result_weight = 0.0
        # 累计丢弃的异常样本数（NaN/Inf/非法策略），供 TB 数据质量监控
        self.total_dropped = 0

    def _target_value(self, s: Dict) -> float:
        """按 value 目标模式计算训练 target：
          mcts  -> mcts_value（搜索/教师平滑评估，噪声小）
          game  -> game_result_value（AlphaZero 标准，终局真值 ±1）
          mixed -> 固定 0.5/0.5 混合
          anneal-> (1-w)*mcts_value + w*game_result，w 按轮退火
        """
        mode = self.cfg.VALUE_TARGET_MODE
        mv = s.get('mcts_value', 0.0)
        gr = s.get('game_result_value', 0.0)
        if mode == "game":
            return float(gr)
        if mode == "mixed":
            return 0.5 * float(mv) + 0.5 * float(gr)
        if mode == "anneal":
            w = self.value_result_weight
            return (1.0 - w) * float(mv) + w * float(gr)
        return float(mv)  # mcts（默认）

    def add_samples(self, samples: List[Dict]) -> None:
        C = self.C
        dropped = 0
        for s in samples:
            board = np.array(s['board_state'], dtype=np.float32).reshape(
                C.TOTAL_INPUT_CHANNELS, C.BOARD_ROWS, C.BOARD_COLS
            )
            scalar_arr = np.array(s['scalar_state'], dtype=np.float32)
            if scalar_arr.shape[0] > C.SCALAR_FEATURE_COUNT:
                scalar_arr = scalar_arr[:C.SCALAR_FEATURE_COUNT]
            probs = np.array(s['policy_probs'], dtype=np.float32)
            mask = np.array(s['action_mask'], dtype=np.float32)
            target_val = self._target_value(s)

            # ---- NaN/Inf 与非法策略/价值目标过滤（来源校验，防污染训练）----
            # 丢弃含非有限值的 board/scalar/policy/mask/value，以及 policy 含
            # 负值或行和≈0 的样本（此类样本会让 log_softmax/交叉熵产生 NaN 或
            # 梯度消失）。value target 来自 mcts_value/game_result 的组合，若
            # 上游 mcts_value 为 NaN（权重被污染的后遗症）会得到 NaN target，
            # 应在此处拦截而非累积进 buffer。
            if (
                not np.isfinite(board).all()
                or not np.isfinite(scalar_arr).all()
                or not np.isfinite(probs).all()
                or not np.isfinite(mask).all()
                or not np.isfinite(target_val)
                or (probs < 0.0).any()
                or probs.sum() <= 0.0
            ):
                dropped += 1
                continue

            self.boards.append(board)
            self.scalars.append(scalar_arr)
            self.probs.append(probs)
            self.values.append(target_val)
            self.masks.append(mask)
            self.root_visits.append(int(s.get('root_visit_count', 0)))

        if dropped:
            self.total_dropped += dropped
            print(
                f"[TR-{self.variant.id}] ⚠️ DataBuffer 丢弃 {dropped} 个异常样本"
                f"（累计 {self.total_dropped}，NaN/Inf/非法策略），Blocked 来自自对弈或冷存储"
            )

        if len(self.boards) > self.capacity:
            excess = len(self.boards) - self.capacity
            for attr in ("boards", "scalars", "probs", "values", "masks", "root_visits"):
                setattr(self, attr, getattr(self, attr)[excess:])

    def __len__(self) -> int:
        return len(self.boards)

    def get_batch(self, indices):
        b = torch.from_numpy(np.stack([self.boards[i] for i in indices]))
        s = torch.from_numpy(np.stack([self.scalars[i] for i in indices]))
        p = torch.from_numpy(np.stack([self.probs[i] for i in indices]))
        v = torch.tensor([self.values[i] for i in indices], dtype=torch.float32)
        m = torch.from_numpy(np.stack([self.masks[i] for i in indices]))
        return b, s, p, v, m
