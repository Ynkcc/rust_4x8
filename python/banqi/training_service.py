"""banqi/training_service.py — 训练消费者模块（共享实现，4x2 / 4x4 / 4x8 通用）

从数据队列消费自对弈 episode，填充向量化 replay buffer，持续迭代训练，
周期性导出 checkpoint（.pt 供 Rust 推理 / .pth 供训练恢复）。
以线程形式运行（TrainWorker），由 banqi.train 编排。

统一行为要点：
  - value 目标模式（mcts / game / mixed / anneal）由 config.VALUE_TARGET_MODE 控制
  - 每轮训练批次数与新数据量匹配（max_batches，防过拟合旧分布）
  - 冷存储预填充 / 固定验证集价值漂移监控按 config 开关启用
  - checkpoint 复用 banqi.checkpoint（含 variant 维度校验）
"""

from __future__ import annotations

import os
import queue
import random
import threading
import time
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from banqi.checkpoint import load_checkpoint, save_checkpoint
from banqi.config import make_config
from banqi.constants import build_constants
from banqi.data_augmentation import make_augmentor
from banqi.nn_model import BanqiNet
from banqi.tb_logger import add_scalar  # TensorBoard 训练日志（未启用时为 no-op）
from banqi.variant import Variant


def _resolve_device(spec: str) -> "torch.device":
    """按 config.TRAIN_DEVICE 解析训练设备；auto = CUDA 可用则用 CUDA。"""
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


# ============================================================================
# 向量化数据缓冲
# ============================================================================

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
        for s in samples:
            board = np.array(s['board_state'], dtype=np.float32).reshape(
                C.TOTAL_INPUT_CHANNELS, C.BOARD_ROWS, C.BOARD_COLS
            )
            self.boards.append(board)
            scalar_arr = np.array(s['scalar_state'], dtype=np.float32)
            if scalar_arr.shape[0] > C.SCALAR_FEATURE_COUNT:
                scalar_arr = scalar_arr[:C.SCALAR_FEATURE_COUNT]
            self.scalars.append(scalar_arr)
            self.probs.append(np.array(s['policy_probs'], dtype=np.float32))
            self.values.append(self._target_value(s))
            self.masks.append(np.array(s['action_mask'], dtype=np.float32))
            self.root_visits.append(int(s.get('root_visit_count', 0)))

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


def episode_to_samples(episode_dict: Dict) -> List[Dict]:
    """
    把一个 episode dict（来自 self_play 队列）转换为 DataBuffer 可消费的
    sample dict 列表，字段与 Mongo GameDocument.samples 一致
    （含 step_in_game / health_diff，与归档数据同步）。
    """
    samples = []
    n = len(episode_dict["boards"])
    step_ids = episode_dict.get("step_in_game") or list(range(n))
    health_diffs = episode_dict.get("health_diffs") or [0.0] * n
    for step_idx, (board, scalar, policy, mcts_val, completed_q,
                    root_visit, game_result, mask) in enumerate(zip(
        episode_dict["boards"], episode_dict["scalars"], episode_dict["policies"],
        episode_dict["mcts_values"], episode_dict["completed_qs"],
        episode_dict["root_visits"], episode_dict["game_results"],
        episode_dict["action_masks"],
    )):
        samples.append({
            "board_state": board,
            "scalar_state": scalar,
            "policy_probs": policy,
            "mcts_value": float(mcts_val),
            "completed_q": float(completed_q),
            "root_visit_count": int(root_visit),
            "game_result_value": float(game_result),
            "action_mask": mask,
            "step_in_game": int(step_ids[step_idx]),
            "health_diff": float(health_diffs[step_idx]),
        })
    return samples


# ============================================================================
# 训练步骤
# ============================================================================

def train_step(model, optimizer, batch_data, device):
    model.train()
    boards_t, scalars_t, target_probs_t, target_values_t, masks_t = batch_data

    boards_t = boards_t.to(device, non_blocking=True)
    scalars_t = scalars_t.to(device, non_blocking=True)
    target_probs_t = target_probs_t.to(device, non_blocking=True)
    target_values_t = target_values_t.to(device, non_blocking=True).view(-1, 1)
    masks_t = masks_t.to(device, non_blocking=True)

    optimizer.zero_grad()
    logits, values = model(boards_t, scalars_t)

    masked_logits = logits + (masks_t - 1.0) * 1e9
    log_probs = F.log_softmax(masked_logits, dim=1)
    policy_loss = -torch.sum(target_probs_t * log_probs, dim=1).mean()

    value_loss = F.mse_loss(values, target_values_t)
    total_loss = policy_loss + value_loss

    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    return total_loss.item(), policy_loss.item(), value_loss.item()


def run_training_epochs(model, optimizer, scheduler, buffer, num_epochs,
                        device, max_batches: Optional[int] = None):
    """
    在完整 replay buffer 上训练指定个 epoch。
    scheduler.step() 按 batch 步进以匹配 CosineAnnealingLR 的 T_max (batch 数)。

    max_batches: 限制本轮总训练批次数。当每轮新增数据量远小于 buffer（如 RL
    自对弈慢、每轮仅几百样本而 buffer 上万）时，若每轮对整个 buffer 训练多
    epoch，旧数据会被反复训练导致过拟合/棋力退化。限制训练量与新数据量匹配：
      每轮批次 ≈ 新样本数/batch × epochs，数据量大时自动恢复全覆盖训练。

    返回 (epoch 平均 loss 列表, 累计训练 batch 数)。
    """
    total_batches = 0
    epoch_results = []
    for epoch in range(num_epochs):
        indices = list(range(len(buffer)))
        random.shuffle(indices)
        num_batches = len(indices) // buffer.cfg.TRAIN_BATCH
        if num_batches == 0:
            break
        if max_batches is not None:
            remaining = max_batches - total_batches
            if remaining <= 0:
                break
            num_batches = min(num_batches, remaining)
        batch_total_l, batch_pol_l, batch_val_l = 0.0, 0.0, 0.0
        for step in range(num_batches):
            batch_indices = indices[step * buffer.cfg.TRAIN_BATCH: (step + 1) * buffer.cfg.TRAIN_BATCH]
            batch_data = buffer.get_batch(batch_indices)
            tl, pl, vl = train_step(model, optimizer, batch_data, device)
            scheduler.step()
            batch_total_l += tl
            batch_pol_l += pl
            batch_val_l += vl
            total_batches += 1

        epoch_results.append((
            batch_total_l / num_batches,
            batch_pol_l / num_batches,
            batch_val_l / num_batches,
        ))
    return epoch_results, total_batches


# ============================================================================
# 训练消费者线程
# ============================================================================

class TrainWorker(threading.Thread):
    """
    消费者线程：从数据队列消费 episode，填充 replay buffer 并训练。
    训练期间缓存 0 元素（确保 sklearn/keras 风格的优雅等待）。
    """

    def __init__(
        self,
        data_q: "queue.Queue",
        stop_flag: "List[bool]",
        variant: Variant,
        model: Optional[BanqiNet] = None,
    ) -> None:
        super().__init__(name=f"TrainWorker-{variant.id}", daemon=True)
        self.variant = variant
        self.cfg = make_config(variant.id)
        self.tag = f"[TR-{variant.id}]"
        self.data_q = data_q
        self.stop_flag = stop_flag

        self.device = _resolve_device(self.cfg.TRAIN_DEVICE)
        # 吞吐优化：TF32 + cudnn auto-tune（训练端，GPU 时启用）
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True
            torch.set_float32_matmul_precision("high")
        print(f"{self.tag} 训练设备: {self.device}（config.TRAIN_DEVICE={self.cfg.TRAIN_DEVICE!r}）")

        self.model = model if model is not None else BanqiNet(variant).to(self.device)
        # weight_decay=1e-4：轻正则化，抑制小数据量下的过拟合/价值头漂移
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=self.cfg.LEARNING_RATE, weight_decay=self.cfg.WEIGHT_DECAY
        )
        self.scheduler = lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.cfg.LR_DECAY_STEPS, eta_min=self.cfg.MIN_LR,
        )
        # 恢复 checkpoint（含维度校验；失败回退权重/全新模型）
        load_checkpoint(self.model, self.optimizer, self.scheduler,
                        self.cfg.MODEL_PATH, self.cfg.STATE_DICT_PATH,
                        self.device, variant)
        # 立即导出一次，确保 Rust 侧有可用的 .pt（全新模型也导出初始）
        save_checkpoint(self.model, self.optimizer, self.scheduler,
                        self.cfg.MODEL_PATH, self.cfg.STATE_DICT_PATH,
                        self.device, variant)

        self.buffer = DataBuffer(self.cfg.MAX_SAMPLE_BUFFER_SIZE, variant, self.cfg)
        # 冷存储预填充：启动时从归档加载历史局复用，避免训练从"空 buffer + 少量
        # 新局"开始就过度拟合当轮数据（config.ARCHIVE_PREFILL_GAMES>0 时启用）。
        self._prefill_from_archive()
        self._last_round_new_samples = 0
        self.round_num = 0
        self.total_batches_trained = 0
        self.total_loss_sum = 0.0
        self.total_policy_loss_sum = 0.0
        self.total_value_loss_sum = 0.0
        self.round_history: List[Dict] = []
        self._stats_lock = threading.Lock()
        # 固定价值验证集（价值漂移监控）：优先由归档预填充构建；
        # 无归档时降级为"用本会话前 N 条自对弈样本构建"。
        self._fixed_eval = None
        self._raw_sample_pool: List[Dict] = []

    # ---- 冷存储预填充 + 固定验证集 ----

    def _prefill_from_archive(self) -> None:
        """从冷存储归档加载历史 episode 预填充训练 buffer（config.ARCHIVE_PREFILL_GAMES>0）。"""
        n_games = getattr(self.cfg, "ARCHIVE_PREFILL_GAMES", 0)
        if not n_games:
            return
        from banqi.storage import episode_dict_to_samples, load_jsonl_episodes
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        dirs = [
            self.cfg.ARCHIVE_PREFILL_DIR,
            self.variant.archive_dir or "",
            os.path.join(here, "training_data", f"archive_{self.variant.id}"),
            os.path.join(here, "training_data", f"archive_{self.variant.id}_imitate"),
        ]
        archive_dir = next((d for d in dirs if d and os.path.isdir(d)), None)
        if not archive_dir:
            print(f"{self.tag} ⚠️ 冷存储预填充：未找到归档目录，跳过")
            return
        try:
            t0 = time.time()
            episodes = load_jsonl_episodes(archive_dir, limit_games=n_games)
            samples: List[Dict] = []
            for ep in episodes:
                samples.extend(episode_dict_to_samples(ep))
            if samples:
                self.buffer.add_samples(samples)
                print(f"{self.tag} 🗃️ 冷存储预填充: 从 {archive_dir} 加载 "
                      f"{len(episodes)} 局 → {len(samples)} 样本 "
                      f"(Buffer={len(self.buffer)}, 耗时 {time.time() - t0:.1f}s)")
            # 固定验证集（价值漂移监控）：取前 N 条局面及其终局结果
            n_fixed = self.cfg.VALUE_DRIFT_NUM_POSITIONS
            if n_fixed > 0:
                self._fixed_eval = self._build_fixed_eval(samples[:n_fixed])
                if self._fixed_eval:
                    print(f"{self.tag} 🎯 固定价值验证集（归档）"
                          f"{len(self._fixed_eval['boards'])} 局面已就绪")
        except Exception as e:  # pragma: no cover
            print(f"{self.tag} ⚠️ 冷存储预填充失败 ({e})，继续正常训练")

    def _build_fixed_eval(self, samples: List[Dict]) -> Optional[Dict]:
        if not samples:
            return None
        C = build_constants(self.variant)
        try:
            return {
                "boards": np.stack([np.array(s['board_state'], dtype=np.float32).reshape(
                    C.TOTAL_INPUT_CHANNELS, C.BOARD_ROWS, C.BOARD_COLS) for s in samples]),
                "scalars": np.stack([np.array(s['scalar_state'], dtype=np.float32)
                                     for s in samples]),
                "results": np.array([s.get('game_result_value', 0.0)
                                     for s in samples], dtype=np.float32),
            }
        except Exception:  # pragma: no cover
            return None

    def _ensure_fixed_eval_from_selfplay(self, samples: List[Dict]) -> None:
        """无归档时，用本会话自对弈原始样本构建固定价值验证集。

        必须在数据增强之前调用（保留原始局面与终局结果）。
        """
        if self._fixed_eval is not None:
            return
        n_fixed = self.cfg.VALUE_DRIFT_NUM_POSITIONS
        if n_fixed <= 0:
            return
        self._raw_sample_pool.extend(samples)
        if len(self._raw_sample_pool) < n_fixed:
            return
        pool = self._raw_sample_pool[:n_fixed]
        fixed = self._build_fixed_eval(pool)
        if fixed is not None:
            self._fixed_eval = fixed
            self._raw_sample_pool = []
            print(f"{self.tag} 🎯 固定价值验证集（自对弈样本）"
                  f"{len(fixed['boards'])} 局面已就绪")
        else:
            self._raw_sample_pool = pool

    # ---- 主循环 ----

    def _drain_new_episodes(self, max_items: int) -> List[Dict]:
        episodes: List[Dict] = []
        try:
            first = self.data_q.get(timeout=0.5)
        except queue.Empty:
            return episodes
        episodes.append(first)
        for _ in range(max_items - 1):
            try:
                episodes.append(self.data_q.get_nowait())
            except queue.Empty:
                break
        return episodes

    def run(self) -> None:
        cfg = self.cfg
        print(f"{self.tag} 🚀 训练线程启动（batch={cfg.TRAIN_BATCH}, "
              f"MinSamples={cfg.MIN_SAMPLES_TO_START}, "
              f"Epochs/Round={cfg.TRAIN_EPOCHS_PER_ROUND}, "
              f"ValueTarget={cfg.VALUE_TARGET_MODE}）...")

        while not self.stop_flag[0]:
            episodes = self._drain_new_episodes(cfg.QUEUE_FETCH_BATCH)
            if not episodes:
                if self.stop_flag[0]:
                    break
                continue

            train_samples: List[Dict] = []
            for ep in episodes:
                has_data = ep.get("num_samples", 0) > 0 or (
                    ep.get("samples") or ep.get("boards"))
                if not has_data:
                    continue
                train_samples.extend(episode_to_samples(ep))

            # 无归档时用自对弈原始样本构建固定验证集（价值漂移监控）
            self._ensure_fixed_eval_from_selfplay(train_samples)

            # 对称增强（仅训练侧）：archive_q 保存的仍是原始 episode
            aug_count = 0
            if cfg.DATA_AUGMENT_ENABLED and train_samples:
                AUG = make_augmentor(self.variant)
                transforms = [t.strip() for t in cfg.DATA_AUGMENT_TRANSFORMS.split(",") if t.strip()]
                raw_count = len(train_samples)
                train_samples = AUG.augment_samples(
                    train_samples,
                    transforms=transforms,
                    keep_original=cfg.DATA_AUGMENT_KEEP_ORIGINAL,
                )
                aug_count = len(train_samples) - raw_count

            if train_samples:
                self.buffer.add_samples(train_samples)

            aug_note = f"（增强 +{aug_count}）" if aug_count else ""
            print(f"{self.tag} 📥 消费 {len(episodes)} 局 → "
                  f"train: {len(train_samples)}{aug_note} → Buffer={len(self.buffer)}")

            min_required = max(cfg.TRAIN_BATCH, cfg.MIN_SAMPLES_TO_START)
            if len(self.buffer) < min_required:
                print(f"{self.tag} ⚠️ Buffer={len(self.buffer)} < {min_required}，暂不训练，等待更多")
                continue
            # 记录本轮新增样本量（含增强），用于限制训练量
            self._last_round_new_samples = len(train_samples)
            self._train_round()

    def _train_round(self) -> None:
        cfg = self.cfg
        # 每轮训练批次数与新数据量匹配，防过拟合旧分布（详见 run_training_epochs）
        n_new = max(32, self._last_round_new_samples)
        per_epoch_batches = max(1, n_new // cfg.TRAIN_BATCH)
        max_batches = per_epoch_batches * cfg.TRAIN_EPOCHS_PER_ROUND
        full_cover = (len(self.buffer) // cfg.TRAIN_BATCH) * cfg.TRAIN_EPOCHS_PER_ROUND
        max_batches = min(max_batches, full_cover)

        epoch_results, batches_in_round = run_training_epochs(
            self.model, self.optimizer, self.scheduler,
            self.buffer, cfg.TRAIN_EPOCHS_PER_ROUND, self.device,
            max_batches=max_batches,
        )

        with self._stats_lock:
            self.total_batches_trained += batches_in_round
            if epoch_results:
                self.total_loss_sum += sum(r[0] for r in epoch_results)
                self.total_policy_loss_sum += sum(r[1] for r in epoch_results)
                self.total_value_loss_sum += sum(r[2] for r in epoch_results)
                last_avg_l, last_avg_p, last_avg_v = epoch_results[-1]
            else:
                last_avg_l = last_avg_p = last_avg_v = 0.0
            entry: Dict = {
                "round": self.round_num,
                "batches": batches_in_round,
                "train_loss": last_avg_l,
                "train_policy_loss": last_avg_p,
                "train_value_loss": last_avg_v,
                "lr": self.optimizer.param_groups[0]['lr'],
            }
            self.round_history.append(entry)

        if epoch_results:
            print(f"{self.tag} ✅ Round#{self.round_num} | {batches_in_round} 批次 | "
                  f"Loss: {last_avg_l:.4f} (Pol: {last_avg_p:.4f}, Val: {last_avg_v:.4f}) "
                  f"| lr={entry['lr']:.2e}")

        self.round_num += 1
        if self.round_num % cfg.CHECKPOINT_EVERY_N_ROUNDS == 0:
            self._save()

        # value 目标退火（anneal 模式）：每 VALUE_ANNEAL_STEP_ROUNDS 轮增加 game_result 权重
        if cfg.VALUE_TARGET_MODE == "anneal":
            w = cfg.VALUE_ANNEAL_START + \
                (self.round_num // cfg.VALUE_ANNEAL_STEP_ROUNDS) * cfg.VALUE_ANNEAL_INCREMENT
            w = min(1.0, w)
            self.buffer.value_result_weight = w
            print(f"{self.tag} 🔄 value退火权重(game_result)={w:.2f} (Round#{self.round_num})")

        # 固定验证集价值漂移监控
        if (cfg.VALUE_DRIFT_EVAL_ROUNDS > 0 and self._fixed_eval is not None
                and self.round_num % cfg.VALUE_DRIFT_EVAL_ROUNDS == 0):
            self._eval_value_drift()

        # TensorBoard 训练日志（x 轴为累计训练 batch 数）
        if cfg.TENSORBOARD_ENABLED:
            step = self.total_batches_trained
            add_scalar("train/loss", entry["train_loss"], step)
            add_scalar("train/policy_loss", entry["train_policy_loss"], step)
            add_scalar("train/value_loss", entry["train_value_loss"], step)
            add_scalar("train/lr", entry["lr"], step)

    def _save(self) -> None:
        save_checkpoint(self.model, self.optimizer, self.scheduler,
                        self.cfg.MODEL_PATH, self.cfg.STATE_DICT_PATH,
                        self.device, self.variant)

    def _eval_value_drift(self) -> None:
        """在固定验证集上评估价值头输出，检测价值漂移（pred mean/std、与终局相关）。"""
        fixed = self._fixed_eval
        if fixed is None:
            return
        try:
            self.model.eval()
            with torch.inference_mode():
                b = torch.from_numpy(np.ascontiguousarray(fixed["boards"]))
                s = torch.from_numpy(np.ascontiguousarray(fixed["scalars"]))
                _, values = self.model(b, s)
                pred = values.cpu().numpy().reshape(-1).astype(np.float32)
            self.model.train()
            gr = fixed["results"]
            corr = float(np.corrcoef(pred, gr)[0, 1]) if len(pred) > 2 else 0.0
            sep = float(pred[gr > 0].mean() - pred[gr < 0].mean()) if (np.any(gr > 0) and np.any(gr < 0)) else 0.0
            print(f"{self.tag} 📊 价值漂移 Round#{self.round_num}: pred_mean={pred.mean():+.3f} "
                  f"std={pred.std():.3f} corr(终局)={corr:.3f} 胜负区分度={sep:.3f}")
            add_scalar("value_drift/pred_mean", pred.mean())
            add_scalar("value_drift/pred_std", pred.std())
            add_scalar("value_drift/corr_result", corr)
            add_scalar("value_drift/sep", sep)
        except Exception as e:  # pragma: no cover
            print(f"{self.tag} ⚠️ 价值漂移评估失败 ({e})")

    def stats(self) -> Dict[str, float]:
        with self._stats_lock:
            return {
                "round_num": self.round_num,
                "total_batches": self.total_batches_trained,
                "avg_loss": self.total_loss_sum / max(1, self.total_batches_trained),
                "avg_policy_loss": self.total_policy_loss_sum / max(1, self.total_batches_trained),
                "avg_value_loss": self.total_value_loss_sum / max(1, self.total_batches_trained),
            }

    def round_history_snapshot(self) -> List[Dict]:
        """返回逐轮指标历史的浅拷贝（供基线验证/监控线程安全读取）。"""
        with self._stats_lock:
            return list(self.round_history)

    def finalize(self) -> None:
        """最终落盘 checkpoint。"""
        self._save()
        print(f"{self.tag} 🎉 最终 Checkpoint 已保存")
