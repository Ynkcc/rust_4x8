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
from banqi.tb_logger import add_scalar
from banqi.variant import Variant

from .buffer import DataBuffer, episode_to_samples
from .losses import run_training_epochs, _resolve_device
from .eval import (
    build_fixed_eval,
    eval_match,
    eval_policy_accuracy,
    eval_value_drift,
    prefill_from_archive,
    select_balanced_fixed_samples,
)


def _compute_lr_scale(global_step: int, cfg) -> float:
    """线性 warmup 系数：前 LR_DECAY_STEPS 步从 MIN_LR 线性升到 1.0。

    仅用于 checkpoint 恢复时对初始 lr 做缩放，避免从 0 直接起训导致的震荡；
    之后的余弦下降交由 CosineAnnealingLR 处理。
    """
    if global_step <= 0:
        return 1.0
    decay_steps = max(int(getattr(cfg, "LR_DECAY_STEPS", 1000) or 1000), 1)
    min_lr = float(getattr(cfg, "MIN_LR", 1e-6) or 1e-6)
    frac = min(global_step, decay_steps) / decay_steps
    # 目标 = MIN_LR + (1 - MIN_LR)*frac，最小不低于 MIN_LR
    return max(min_lr, 1.0 - (1.0 - min_lr) * (1.0 - frac))


def _is_stopped(stop_event) -> bool:
    if stop_event is None:
        return False
    if isinstance(stop_event, list):
        return bool(stop_event[0])
    if hasattr(stop_event, "is_set"):
        return stop_event.is_set()
    return bool(stop_event)


class TrainWorker(threading.Thread):
    def __init__(self, arg1, arg2, arg3=None, arg4=None,
                 ckpt_dir: Optional[str] = None, device=None, run_dir: Optional[str] = None):
        if isinstance(arg1, Variant):
            variant = arg1
            cfg = arg2
            data_queue = arg3
            stop_event = arg4
        elif isinstance(arg3, Variant):
            # 兼容 (data_q, stop_flag, variant) 参数顺序
            data_queue = arg1
            stop_event = arg2
            variant = arg3
            from banqi.config import make_config
            cfg = arg4 if arg4 is not None else make_config(variant.id)
        else:
            # 兼容 (data_q, stop_flag, variant_id_str)
            data_queue = arg1
            stop_event = arg2
            from banqi.variant import get_variant
            from banqi.config import make_config
            variant = get_variant(str(arg3))
            cfg = arg4 if arg4 is not None else make_config(variant.id)

        super().__init__(name=f"TrainWorker-{variant.id}", daemon=True)
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
        self._stats_lock = threading.Lock()
        self.round_num = 0
        self.total_loss_sum = 0.0
        self.total_policy_loss_sum = 0.0
        self.total_value_loss_sum = 0.0
        self.round_history: deque = deque(maxlen=1000)

        self._warmup_done = False
        self._raw_sample_pool: List[Dict] = []
        self._fixed_eval: Optional[Dict] = None
        self._prev_weights: Optional[Dict[str, torch.Tensor]] = None
        self._init_model_and_checkpoint()

    def _init_model_and_checkpoint(self):
        cfg = self.cfg
        from banqi.nn_model import BanqiNet

        if os.path.exists(self.last_ckpt_path()):  # resume
            print(f"[TR-{self.variant.id}] 从 checkpoint 恢复: {self.last_ckpt_path()}")
            ckpt = torch.load(self.last_ckpt_path(), map_location=self.device, weights_only=False)
            model = BanqiNet(self.variant)
            model.load_state_dict(ckpt["model_state"])
            self.model = model.to(self.device)
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=cfg.LEARNING_RATE * _compute_lr_scale(ckpt.get("global_step", 0), cfg),
                weight_decay=cfg.WEIGHT_DECAY,
            )
            self.optimizer.load_state_dict(ckpt["optimizer_state"])
            self.scheduler = lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=cfg.LR_DECAY_STEPS, eta_min=cfg.MIN_LR
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
            self.model = BanqiNet(self.variant).to(self.device)
            self.optimizer = optim.AdamW(
                self.model.parameters(), lr=cfg.LEARNING_RATE,
                weight_decay=cfg.WEIGHT_DECAY
            )
            self.scheduler = lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=cfg.LR_DECAY_STEPS, eta_min=cfg.MIN_LR
            )
            self.global_step = 0
            self.start_global_step = 0
            self.start_total_samples = 0
            self.version = 0

        buffer_capacity = cfg.MAX_SAMPLE_BUFFER_SIZE
        self.buffer = DataBuffer(buffer_capacity, self.variant, cfg)

        # 冷存储预填充 & 固定验证集生成
        fixed_archive = prefill_from_archive(self.buffer, self.variant, cfg)
        if fixed_archive is not None:
            self._fixed_eval = fixed_archive

        # 显式记录 value 目标模式（终端日志，便于复现/调试）
        print(f"[TR-{self.variant.id}] 价值目标模式={cfg.VALUE_TARGET_MODE}，"
              f"buffer 容量={buffer_capacity}，TRAIN_DEVICE={self.device}")

        # ---- value 目标退火（VALUE_TARGET_MODE='anneal' 时）----
        # 退火权重 w：前 N 轮用 mcts 平滑评估，后段切到 game_result 真值。
        # 每轮训练前按 (round_idx / anneal_rounds) 更新 buffer.value_result_weight。
        self.anneal_rounds = getattr(cfg, "VALUE_TARGET_ANNEAL_ROUNDS", 0)

        init_ckpt = getattr(cfg, "INIT_FROM_CHECKPOINT", None)
        if init_ckpt:
            self._load_pretrained(init_ckpt)

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

    def _ensure_fixed_eval_from_selfplay(self, samples: List[Dict]) -> None:
        """无归档时，从本会话自对弈原始样本池中按终局结果分层构建固定验证集。"""
        if self._fixed_eval is not None:
            return
        n_fixed = getattr(self.cfg, "VALUE_DRIFT_NUM_POSITIONS", 128)
        if n_fixed <= 0:
            return
        self._raw_sample_pool.extend(samples)
        if len(self._raw_sample_pool) < n_fixed:
            return
        pool = select_balanced_fixed_samples(self._raw_sample_pool, n_fixed)
        fixed = build_fixed_eval(pool, self.variant) if pool else None
        if fixed is not None:
            self._fixed_eval = fixed
            self._raw_sample_pool = []

    def _safe_qsize(self) -> int:
        """线程安全地读取数据队列积压。"""
        try:
            if hasattr(self.data_queue, "qsize"):
                qsize = self.data_queue.qsize()
                return int(qsize) if qsize is not None else -1
            return -1
        except Exception:
            return -1

    def run(self, rounds: int = 100000):
        cfg = self.cfg
        last_processed_round = 0
        version = self.version
        total_samples = self.start_total_samples
        while not _is_stopped(self.stop_event):
            try:
                episode_dict = self.data_queue.get(timeout=2.0)
            except Exception:
                continue
            if episode_dict is None:
                break

            t0 = time.time()
            samples = episode_to_samples(episode_dict)
            self._ensure_fixed_eval_from_selfplay(samples)
            self.buffer.add_samples(samples)
            new_samples = len(samples)
            total_samples += new_samples
            round_idx = episode_dict.get("round_idx", last_processed_round)

            min_samples = getattr(cfg, "MIN_SAMPLES_TO_START", getattr(cfg, "TRAIN_MIN_SAMPLES", 100))
            if len(self.buffer) < min_samples:
                print(f"[TR-{self.variant.id}] 等待足够样本进行训练: "
                      f"{len(self.buffer)}/{min_samples}")
                self._maybe_save_early(episode_dict)
                last_processed_round = round_idx
                continue

            self._anneal_value_weight(round_idx)

            # ---- 训练量限制：与新增样本量匹配，避免旧数据反复训练 ----
            capacity_base = getattr(cfg, "MAX_SAMPLE_BUFFER_SIZE", 50000)
            if new_samples >= capacity_base // 4:
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
                with self._stats_lock:
                    self.round_num = round_idx
                    self.total_loss_sum += last_losses[0] * total_batches
                    self.total_policy_loss_sum += last_losses[1] * total_batches
                    self.total_value_loss_sum += last_losses[2] * total_batches
                    self.round_history.append({
                        "round": round_idx,
                        "train_loss": last_losses[0],
                        "policy_loss": last_losses[1],
                        "value_loss": last_losses[2],
                        "grad_norm": last_losses[3],
                        "entropy": last_losses[4],
                        "lr": current_lr,
                        "global_step": self.global_step,
                    })

                print(f"[TR-{self.variant.id}] round {round_idx}: "
                      f"epoch_avg_loss={last_losses[0]:.4f} "
                      f"(policy={last_losses[1]:.4f}, value={last_losses[2]:.4f}) "
                      f"grad_norm={last_losses[3]:.3f} entropy={last_losses[4]:.3f} "
                      f"value_mean={last_losses[5]:.3f} value_std={last_losses[6]:.3f} "
                      f"lr={current_lr:.2e} duration={self.metrics['train_duration']:.1f}s "
                      f"buffer={len(self.buffer)} global_step={self.global_step}")

                # 恢复 TensorBoard 训练过程标量记录
                step = self.global_step
                tag = f"[TR-{self.variant.id}]"
                add_scalar("train/loss", last_losses[0], step)
                add_scalar("train/policy_loss", last_losses[1], step)
                add_scalar("train/value_loss", last_losses[2], step)
                add_scalar("train/grad_norm", last_losses[3], step)
                add_scalar("train/policy_entropy", last_losses[4], step)
                add_scalar("train/value_mean", last_losses[5], step)
                add_scalar("train/value_std", last_losses[6], step)
                add_scalar("train/lr", current_lr, step)
                add_scalar("train/buffer_size", len(self.buffer), step)
                add_scalar("queue/backlog", self._safe_qsize(), step)
                if cfg.VALUE_TARGET_MODE == "anneal":
                    add_scalar("train/value_anneal_w", self.buffer.value_result_weight, step)

                # 固定验证集评估（价值漂移与策略命中率）
                eval_value_drift(self.model, self.device, self._fixed_eval, step, tag, round_idx)
                eval_policy_accuracy(self.model, self.device, self._fixed_eval, step, tag, round_idx)

                # 周期性对战评估
                eval_match_rounds = getattr(cfg, "EVAL_MATCH_ROUNDS", 10)
                if eval_match_rounds > 0 and (round_idx > 0) and (round_idx % eval_match_rounds == 0):
                    eval_match(
                        self.model, self.device, self.variant, cfg,
                        self._prev_weights, round_idx, step, tag
                    )

                # 缓存本轮权重快照（供下一轮 vs prev 对战评估）
                self._prev_weights = {
                    k: v.detach().to("cpu").clone()
                    for k, v in self.model.state_dict().items()
                }

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

    def stats(self) -> Dict[str, float]:
        with self._stats_lock:
            total = max(1, self.global_step)
            return {
                "round_num": self.round_num,
                "total_batches": self.global_step,
                "avg_loss": self.total_loss_sum / total,
                "avg_policy_loss": self.total_policy_loss_sum / total,
                "avg_value_loss": self.total_value_loss_sum / total,
            }

    def round_history_snapshot(self) -> List[Dict]:
        """返回逐轮指标历史的浅拷贝。"""
        with self._stats_lock:
            return list(self.round_history)

    def finalize(self) -> None:
        """最终落盘 checkpoint。"""
        self.save_checkpoint()
        print(f"[TR-{self.variant.id}] 🎉 最终 Checkpoint 已保存")
