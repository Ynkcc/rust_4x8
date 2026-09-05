"""banqi/nnue/distill.py — 主训练闭环内的 NNUE 蒸馏 worker。

NnueDistillWorker 以 daemon 线程常驻主训练闭环（trainer_cli/runners/selfplay.py）：
- 从数据旁路队列（TeeQueue 分流）流式消费主闭环 MCTS 自对弈 episode dict，
  累积到 NnueSampleBuffer（稀疏特征 + 混合价值标签）；
- 同步把含 NNUE 特征的 episode 追加落盘 JSONL（留档供离线复训）；
- TrainWorker 每 save_checkpoint 后 set ckpt_event，每 N 次 checkpoint
  用当前累积样本调用 train_nnue 蒸馏训练，导出 .nnue 到 output_dir。

闭环收益：.nnue 落地后即可被 ExpectimaxEngine（"expectimax:<path>.nnue"
选手 / ExpectimaxSidecar / Tauri GUI）直接热加载，无需人工干预。
"""

from __future__ import annotations

import json
import os
import threading
import time
from typing import Dict, Optional

from banqi.config import Config
from banqi.nnue.samples import NnueSampleBuffer
from banqi.variant import Variant


class NnueDistillWorker(threading.Thread):
    """NNUE 蒸馏 worker：数据旁路消费 + 周期蒸馏训练 + .nnue 导出。"""

    def __init__(
        self,
        variant: Variant,
        cfg: Config,
        side_queue,
        stop_event: threading.Event,
        ckpt_event: threading.Event,
        tag: str = "[NNUE]",
    ) -> None:
        super().__init__(name=f"NnueDistill-{variant.id}", daemon=True)
        self.variant = variant
        self.cfg = cfg
        self.side_queue = side_queue
        self.stop_event = stop_event
        self.ckpt_event = ckpt_event
        self.tag = tag

        self.buffer = NnueSampleBuffer(
            value_source=getattr(cfg, "NNUE_VALUE_SOURCE", "completed_q"),
            value_weight=float(getattr(cfg, "NNUE_VALUE_WEIGHT", 0.7)),
            full_only=bool(getattr(cfg, "NNUE_FULL_ONLY", False)),
            dual_perspective=bool(getattr(cfg, "NNUE_DUAL_PERSPECTIVE", True)),
            max_samples=int(getattr(cfg, "NNUE_MAX_SAMPLES", 2_000_000)),
        )

        self.data_dir = getattr(cfg, "NNUE_DISTILL_DATA_DIR", "") or "data/nnue"
        self.output_dir = getattr(cfg, "NNUE_DISTILL_OUTPUT_DIR", "") or "models/nnue"
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        self._jsonl_path = os.path.join(
            self.data_dir, f"distill_{variant.id}_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"
        )
        self._jsonl_lock = threading.Lock()

        self.checkpoints_seen = 0
        self.distill_rounds = 0
        self.last_export_path: Optional[str] = None

    # ------------------------------------------------------------------ #
    # 主循环
    # ------------------------------------------------------------------ #
    def run(self) -> None:
        print(f"{self.tag} 🧠 NNUE 蒸馏 worker 已启动 "
              f"(数据: {self._jsonl_path}, 输出: {self.output_dir})")
        while not self.stop_event.is_set():
            try:
                ep = self.side_queue.get(timeout=1.0)
            except Exception:
                ep = None
            if ep is not None:
                self._ingest(ep)
            # checkpoint 事件（TrainWorker save_checkpoint 后置位）
            if self.ckpt_event.is_set():
                self.ckpt_event.clear()
                self._on_checkpoint()
        # 退出前若积累足够样本则做一次最终蒸馏
        if len(self.buffer) > 0 and self.last_export_path is None:
            self._distill()
        self._log_stats("退出")

    def _ingest(self, ep: Dict) -> None:
        n = self.buffer.add_episode(ep)
        if n > 0:
            self._append_jsonl(ep)

    def _append_jsonl(self, ep: Dict) -> None:
        try:
            line = json.dumps(ep, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            return
        with self._jsonl_lock:
            try:
                with open(self._jsonl_path, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
            except OSError as exc:
                print(f"{self.tag} ⚠️ NNUE JSONL 落盘失败: {exc}")

    def _on_checkpoint(self) -> None:
        self.checkpoints_seen += 1
        every = max(int(getattr(self.cfg, "NNUE_DISTILL_EVERY_N_CHECKPOINTS", 5)), 1)
        if self.checkpoints_seen % every != 0:
            return
        self._distill()

    # ------------------------------------------------------------------ #
    # 蒸馏训练
    # ------------------------------------------------------------------ #
    def _distill(self) -> None:
        min_samples = max(int(getattr(self.cfg, "NNUE_DISTILL_MIN_SAMPLES", 50_000)), 1)
        if len(self.buffer) < min_samples:
            print(f"{self.tag} 样本不足，跳过蒸馏 "
                  f"({len(self.buffer)}/{min_samples}, checkpoint #{self.checkpoints_seen})")
            return
        try:
            import torch

            from banqi.nnue.train import train_nnue

            dataset = self.buffer.to_dataset()
            self.distill_rounds += 1
            out = os.path.join(self.output_dir, f"{self.variant.id}_v{self.distill_rounds}.nnue")
            latest = os.path.join(self.output_dir, f"{self.variant.id}_latest.nnue")
            t0 = time.time()
            train_nnue(
                dataset,
                epochs=int(getattr(self.cfg, "NNUE_EPOCHS", 20)),
                batch_size=int(getattr(self.cfg, "NNUE_BATCH_SIZE", 256)),
                lr=float(getattr(self.cfg, "NNUE_LR", 1e-3)),
                output_nnue=out,
                checkpoint=os.path.splitext(out)[0] + ".pth",
            )
            # latest 原子替换（os.replace），供 expectimax 选手/sidecar 热加载
            tmp = latest + ".tmp"
            with open(out, "rb") as src, open(tmp, "wb") as dst:
                dst.write(src.read())
            os.replace(tmp, latest)
            self.last_export_path = latest
            print(f"{self.tag} ✅ 第 {self.distill_rounds} 次蒸馏完成: {latest} "
                  f"(样本={len(dataset)}, 耗时={time.time() - t0:.1f}s)")
        except Exception as exc:  # noqa: BLE001 — 蒸馏失败不阻塞主闭环
            print(f"{self.tag} ⚠️ NNUE 蒸馏失败: {exc}")

    # ------------------------------------------------------------------ #
    def stats(self) -> Dict[str, int]:
        s = self.buffer.stats()
        s.update({
            "checkpoints_seen": self.checkpoints_seen,
            "distill_rounds": self.distill_rounds,
        })
        return s

    def _log_stats(self, when: str) -> None:
        s = self.stats()
        print(f"{self.tag} {when}: 样本={s['samples']} episode={s['episodes']} "
              f"(缺字段跳过={s['skipped_episodes']}, 维度不符丢弃={s['dropped_episodes']}), "
              f"蒸馏={s['distill_rounds']} 次")
