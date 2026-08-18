"""banqi/training/worker.py — 训练 worker 线程。

TrainWorker 在独立线程消费 self_play 队列，把 episode 转换的 sample 写入
DataBuffer，按 (new_samples/batch)×epochs 限制训练量（避免旧数据反复训练），
并在 checkpoint 时保存 model/optimizer/scheduler/global_step + 训练监控。
"""

from __future__ import annotations

import os
import time
import copy
import pickle
import threading
from collections import deque
from typing import Dict, Optional

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from banqi.constants import build_constants
from banqi.variant import Variant

from .buffer import DataBuffer, episode_to_samples
from .losses import run_training_epochs, _resolve_device


def _compute_lr_scale(global_step: int, cfg) -> float:
    if global_step <= 0:
        return 1.0
    if cfg.LR_FINDER_MODE == "linear":
        frac = min(global_step, cfg.LR_FINDER_STEPS) / max(cfg.LR_FINDER_STEPS, 1)
        return max(cfg.LR_MIN_SCALE, frac)
    half = max(cfg.LR_FINDER_STEPS // 2, 1)
    if global_step < half:
        return max(cfg.LR_MIN_SCALE, global_step / half)
    else:
        return max(cfg.LR_MIN_SCALE, 2.0 - global_step / half)


class TrainWorker:
    def __init__(self, variant: Variant, cfg, data_queue, stop_event,
                 ckpt_dir: Optional[str] = None, device=None, run_dir: Optional[str] = None):
        self.variant = variant
        self.cfg = cfg
        self.C = build_constants(variant)
        self.data_queue = data_queue
        self.stop_event = stop_event
        self.ckpt_dir = ckpt_dir or os.path.join("checkpoints", variant.id)
        self.run_dir = run_dir
        self.device = device or _resolve_device(getattr(cfg, "TRAIN_DEVICE", "auto"))
        os.makedirs(self.ckpt_dir, exist_ok=True)

        # 监控：每轮训练时长、最近 ckpt 路径、最近一次 epoch loss 分解
        self.metrics = {
            "train_duration": 0.0,
            "last_ckpt_path": None,
            "last_epoch_losses": None,
            "last_lr": 0.0,
            "global_step": 0,
        }

        # S3 默认 CPU（避免与主训练 GPU 争抢）；GPU 推理时显式 device
        self.desired_sp_device = (
            "cuda" if getattr(cfg, "SELF_PLAY_DEVICE", "cpu").startswith("cuda") else "cpu"
        )

        self._last_ckpt_lock = threading.Lock()
        self._warmup_done = False
        self._init_model_and_checkpoint()

    def _init_model_and_checkpoint(self):
        cfg = self.cfg
        from banqi.net import BanqiNet

        if os.path.exists(self.last_ckpt_path()):  # resume
            print(f"[TR-{self.variant.id}] 从 checkpoint 恢复: {self.last_ckpt_path()}")
            ckpt = torch.load(self.last_ckpt_path(), map_location=self.device, weights_only=False)
            model = BanqiNet(self.C, cfg)
            model.load_state_dict(ckpt["model_state"])
            self.model = model.to(self.device)
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=cfg.LEARNING_RATE * _compute_lr_scale(ckpt.get("global_step", 0), cfg),
                weight_decay=cfg.WEIGHT_DECAY,
            )
            self.optimizer.load_state_dict(ckpt["optimizer_state"])
            self.scheduler = lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=cfg.LR_SCHEDULER_TMAX, eta_min=cfg.LR_SCHEDULER_ETAMIN
            )
            if "scheduler_state" in ckpt:
                self.scheduler.load_state_dict(ckpt["scheduler_state"])
            self.global_step = ckpt.get("global_step", 0)
            self.metrics["global_step"] = self.global_step
            self.start_global_step = self.global_step
            self.start_total_samples = ckpt.get("total_samples", 0)
            self.version = ckpt.get("version", 0) + 1
            print(f"[TR-{self.variant.id}] 恢复 global_step={self.global_step}, "
                  f"version={self.version}")
        else:
            self.model = BanqiNet(self.C, cfg).to(self.device)
            self.optimizer = optim.AdamW(
                self.model.parameters(), lr=cfg.LEARNING_RATE,
                weight_decay=cfg.WEIGHT_DECAY
            )
            self.scheduler = lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=cfg.LR_SCHEDULER_TMAX, eta_min=cfg.LR_SCHEDULER_ETAMIN
            )
            self.global_step = 0
            self.start_global_step = 0
            self.start_total_samples = 0
            self.version = 0

        self.model.eval()
        buffer_capacity = int(
            cfg.REPLAY_CAPACITY_FRAC * (cfg.REPLAY_CAPACITY_BASE
                                       + cfg.REPLAY_CAPACITY_PER_EP * cfg.TRAIN_STEPS)
        )
        self.buffer = DataBuffer(buffer_capacity, self.variant, cfg)

        # 显式记录 value 目标模式（终端日志，便于复现/调试）
        print(f"[TR-{self.variant.id}] 价值目标模式={cfg.VALUE_TARGET_MODE}，"
              f"buffer 容量={buffer_capacity}，TRAIN_DEVICE={self.device}")

        # ---- value 目标退火（VALUE_TARGET_MODE='anneal' 时）----
        # 退火权重 w：前 N 轮用 mcts 平滑评估，后段切到 game_result 真值。
        # 每轮训练前按 (round_idx / anneal_rounds) 更新 buffer.value_result_weight。
        self.anneal_rounds = getattr(cfg, "VALUE_TARGET_ANNEAL_ROUNDS", 0)

        if cfg.INIT_FROM_CHECKPOINT:
            self._load_pretrained(cfg.INIT_FROM_CHECKPOINT)

    def _load_pretrained(self, ckpt_path: str):
        """从指定 checkpoint 导入权重（仅 model + optimizer），重置 global_step。"""
        print(f"[TR-{self.variant.id}] 加载预训练权重: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        try:
            self.model.load_state_dict(ckpt["model_state"])
        except RuntimeError as e:
            print(f"[TR-{self.variant.id}] ⚠️ state_dict 不完全匹配（跨变体/结构），"
                  f"忽略不匹配键: {e}")
            sd = ckpt["model_state"]
            own = self.model.state_dict()
            filtered = {k: v for k, v in sd.items()
                        if k in own and v.shape == own[k].shape}
            own.update(filtered)
            self.model.load_state_dict(own)
        if "optimizer_state" in ckpt:
            try:
                self.optimizer.load_state_dict(ckpt["optimizer_state"])
            except Exception:
                pass
        self.global_step = 0
        self.metrics["global_step"] = 0

    def last_ckpt_path(self):
        return os.path.join(self.ckpt_dir, "last.ckpt")

    def get_inference_model(self):
        return self.model

    def get_global_step(self):
        return self.global_step

    def get_model_version(self):
        return self.version

    def get_checkpoint_path(self):
        return self.last_ckpt_path()

    def save_checkpoint(self, new_samples: int = 0, total_samples: Optional[int] = None,
                        round_idx: int = 0):
        path = self.last_ckpt_path()
        self.model.eval()
        snapshot = {
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "total_samples": total_samples if total_samples is not None else self.start_total_samples,
            "version": self.version,
            "pytorch_version": torch.__version__,
        }
        torch.save(snapshot, path)
        with self._last_ckpt_lock:
            self.metrics["last_ckpt_path"] = path
            self.metrics["last_global_step"] = self.global_step
        print(f"[TR-{self.variant.id}] 💾 checkpoint 已保存: {path} "
              f"(global_step={self.global_step}, v{self.version})")

    def _anneal_value_weight(self, round_idx: int):
        if self.anneal_rounds and self.buffer.cfg.VALUE_TARGET_MODE == "anneal":
            w = min(1.0, (round_idx + 1) / max(self.anneal_rounds, 1))
            self.buffer.value_result_weight = w
            print(f"[TR-{self.variant.id}] value 目标退火 w={w:.3f} (round {round_idx})")

    def run(self, rounds: int = 100000):
        cfg = self.cfg
        last_processed_round = 0
        version = self.version
        total_samples = self.start_total_samples
        while not self.stop_event.is_set():
            try:
                episode_dict = self.data_queue.get(timeout=2.0)
            except Exception:
                continue
            if episode_dict is None:
                break

            t0 = time.time()
            samples = episode_to_samples(episode_dict)
            self.buffer.add_samples(samples)
            new_samples = len(samples)
            total_samples += new_samples
            round_idx = episode_dict.get("round_idx", last_processed_round)

            if len(self.buffer) < cfg.TRAIN_MIN_SAMPLES:
                print(f"[TR-{self.variant.id}] 等待足够样本进行训练: "
                      f"{len(self.buffer)}/{cfg.TRAIN_MIN_SAMPLES}")
                self._maybe_save_early(episode_dict)
                last_processed_round = round_idx
                continue

            self._anneal_value_weight(round_idx)

            # ---- 训练量限制：与新增样本量匹配，避免旧数据反复训练 ----
            # 每轮批次 ≈ new_samples/batch × epochs，超过则封顶。数据量足够大
            # （如冷存储预热）时 max_batches=None 自动恢复对完整 buffer 的训练。
            if new_samples >= cfg.REPLAY_CAPACITY_BASE // 4:
                max_batches = None
            else:
                max_batches = int(
                    (new_samples / cfg.TRAIN_BATCH) * cfg.TRAIN_EPOCHS_PER_ROUND + 0.5
                )
                max_batches = max(max_batches, 1)

            self.model.train()
            epoch_results, total_batches = run_training_epochs(
                self.model, self.optimizer, self.scheduler, self.buffer,
                cfg.TRAIN_EPOCHS_PER_ROUND, self.device, max_batches=max_batches,
            )
            self.model.eval()

            self.global_step += total_batches
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.metrics["train_duration"] = time.time() - t0
            self.metrics["global_step"] = self.global_step
            self.metrics["last_lr"] = current_lr
            last_losses = epoch_results[-1] if epoch_results else None
            self.metrics["last_epoch_losses"] = last_losses
            if last_losses is not None:
                print(f"[TR-{self.variant.id}] round {round_idx}: "
                      f"epoch_avg_loss={last_losses[0]:.4f} "
                      f"(policy={last_losses[1]:.4f}, value={last_losses[2]:.4f}) "
                      f"grad_norm={last_losses[3]:.3f} entropy={last_losses[4]:.3f} "
                      f"value_mean={last_losses[5]:.3f} value_std={last_losses[6]:.3f} "
                      f"lr={current_lr:.2e} duration={self.metrics['train_duration']:.1f}s "
                      f"buffer={len(self.buffer)} global_step={self.global_step}")

            self.save_checkpoint(new_samples=new_samples, total_samples=total_samples,
                                 round_idx=round_idx)
            last_processed_round = round_idx
            version += 1
            self.version = version

            if last_processed_round >= rounds - 1:
                print(f"[TR-{self.variant.id}] 达到训练轮数上限 {rounds}，退出训练 worker")
                break

    def _maybe_save_early(self, episode_dict):
        # 预热阶段（样本不足）也定期保存，避免长期无 checkpoint
        if (episode_dict.get("round_idx", 0) % 10 == 0) and not os.path.exists(
                self.last_ckpt_path()):
            self.save_checkpoint(round_idx=episode_dict.get("round_idx", 0))
