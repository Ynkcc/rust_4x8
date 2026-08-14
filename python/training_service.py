"""
training_service.py — 训练消费者模块（无 CLI 参数）

从数据队列消费自对弈 episode，填充向量化 replay buffer，持续迭代训练，
周期性导出 checkpoint（.pt 供 Rust 推理 / .pth 供训练恢复）。
以线程形式运行（TrainWorker），由 run_training.py 编排。
"""

from __future__ import annotations

import os
import queue
import random
import threading
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from config import config
from constant import (
    TOTAL_INPUT_CHANNELS,
    BOARD_ROWS,
    BOARD_COLS,
    SCALAR_FEATURE_COUNT,
    ACTION_SPACE_SIZE,
)
from nn_model import BanqiNet

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================================
# 向量化数据缓冲
# ============================================================================

class DataBuffer:
    """向量化缓冲区，优化内存并加速 Tensor 转换"""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.boards: List[np.ndarray] = []
        self.scalars: List[np.ndarray] = []
        self.probs: List[np.ndarray] = []
        self.values: List[float] = []
        self.masks: List[np.ndarray] = []
        self.root_visits: List[int] = []

    def add_samples(self, samples: List[Dict]) -> None:
        for s in samples:
            board = np.array(s['board_state'], dtype=np.float32).reshape(
                TOTAL_INPUT_CHANNELS, BOARD_ROWS, BOARD_COLS
            )
            self.boards.append(board)
            scalar_arr = np.array(s['scalar_state'], dtype=np.float32)
            if scalar_arr.shape[0] > SCALAR_FEATURE_COUNT:
                scalar_arr = scalar_arr[:SCALAR_FEATURE_COUNT]
            self.scalars.append(scalar_arr)
            self.probs.append(np.array(s['policy_probs'], dtype=np.float32))
            val = s.get('game_result_value', s.get('mcts_value', 0.0))
            self.values.append(val)
            self.masks.append(np.array(s['action_mask'], dtype=np.float32))
            self.root_visits.append(int(s.get('root_visit_count', 0)))

        if len(self.boards) > self.capacity:
            excess = len(self.boards) - self.capacity
            self.boards = self.boards[excess:]
            self.scalars = self.scalars[excess:]
            self.probs = self.probs[excess:]
            self.values = self.values[excess:]
            self.masks = self.masks[excess:]
            self.root_visits = self.root_visits[excess:]

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
    sample dict 列表，字段与 Mongo GameDocument.samples 一致。
    """
    samples = []
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
            "step_in_game": step_idx,
        })
    return samples


# ============================================================================
# Checkpoint 保存 / 恢复
# ============================================================================

def save_checkpoint(model, optimizer, scheduler) -> None:
    """
    保存完整训练状态:
    1. .pth: model + optimizer + scheduler 状态（用于断点恢复训练）
    2. .pt: TorchScript（供 Rust 推理加载）
    """
    pt_temp_path = config.MODEL_PATH + ".tmp"
    pth_temp_path = config.STATE_DICT_PATH + ".tmp"

    try:
        model.eval()

        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'model_config': {
                'input_channels': TOTAL_INPUT_CHANNELS,
                'board_rows': BOARD_ROWS,
                'board_cols': BOARD_COLS,
                'scalar_features': SCALAR_FEATURE_COUNT,
                'action_space': ACTION_SPACE_SIZE
            }
        }, pth_temp_path)
        os.replace(pth_temp_path, config.STATE_DICT_PATH)

        with torch.no_grad():
            example_board = torch.randn(1, TOTAL_INPUT_CHANNELS, BOARD_ROWS, BOARD_COLS, device=DEVICE)
            example_scalars = torch.randn(1, SCALAR_FEATURE_COUNT, device=DEVICE)
            traced_model = torch.jit.trace(model, (example_board, example_scalars))
            traced_model.save(pt_temp_path)

        os.replace(pt_temp_path, config.MODEL_PATH)
        print(f"[Training] ✅ Checkpoint 保存成功: {config.STATE_DICT_PATH} + {config.MODEL_PATH}")
    except Exception as e:
        print(f"[Training] ❌ Checkpoint 保存失败: {e}")
        for tmp in [pt_temp_path, pth_temp_path]:
            if os.path.exists(tmp):
                os.remove(tmp)


def load_checkpoint(model, optimizer, scheduler) -> None:
    """
    从 .pth 恢复完整训练状态（model + optimizer + scheduler）。
    如果 .pth 不完整或缺失，回退到仅加载权重 (.pt / 全新模型)。
    """
    state_loaded = False

    if os.path.exists(config.STATE_DICT_PATH):
        try:
            checkpoint = torch.load(config.STATE_DICT_PATH, map_location=DEVICE)
            model.load_state_dict(checkpoint['model_state_dict'])
            if 'optimizer_state_dict' in checkpoint:
                try:
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                except Exception as e_opt:
                    print(f"[Training] ⚠️ Optimizer 状态加载失败 ({e_opt})，保持新初始化")
            if 'scheduler_state_dict' in checkpoint:
                try:
                    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                except Exception as e_sch:
                    print(f"[Training] ⚠️ Scheduler 状态加载失败 ({e_sch})，保持新初始化")
            print(f"[Training] ✅ 从 {config.STATE_DICT_PATH} 恢复完整训练状态")
            state_loaded = True
        except Exception as e:
            print(f"[Training] ⚠️ 完整 .pth 加载失败 ({e})，尝试仅加载权重...")

    if not state_loaded and os.path.exists(config.MODEL_PATH):
        try:
            jit_model = torch.jit.load(config.MODEL_PATH, map_location=DEVICE)
            model.load_state_dict(jit_model.state_dict())
            print(f"[Training] ✅ 从 {config.MODEL_PATH} 加载模型权重 (TorchScript 回退)")
        except Exception as e2:
            print(f"[Training] ⚠️ 权重加载失败 ({e2})，使用全新模型")

    if not state_loaded and not os.path.exists(config.MODEL_PATH) and not os.path.exists(config.STATE_DICT_PATH):
        print("[Training] 📝 初始化全新模型（无 checkpoint）")


# ============================================================================
# 训练步骤
# ============================================================================

def train_step(model, optimizer, batch_data, device):
    model.train()
    boards_t, scalars_t, target_probs_t, target_values_t, masks_t = batch_data

    boards_t = boards_t.to(device)
    scalars_t = scalars_t.to(device)
    target_probs_t = target_probs_t.to(device)
    target_values_t = target_values_t.to(device).view(-1, 1)
    masks_t = masks_t.to(device)

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


@torch.no_grad()
def evaluate(model, buffer, batch_size, device):
    model.eval()
    indices = list(range(len(buffer)))
    random.shuffle(indices)
    num_batches = len(indices) // batch_size
    if num_batches == 0:
        return None

    total_loss_sum = 0.0
    policy_loss_sum = 0.0
    value_loss_sum = 0.0

    for step in range(num_batches):
        batch_indices = indices[step * batch_size : (step + 1) * batch_size]
        boards, scalars, target_probs, target_values, masks = buffer.get_batch(batch_indices)

        boards = boards.to(device)
        scalars = scalars.to(device)
        target_probs = target_probs.to(device)
        target_values = target_values.to(device).view(-1, 1)
        masks = masks.to(device)

        logits, values = model(boards, scalars)
        masked_logits = logits + (masks - 1.0) * 1e9
        log_probs = F.log_softmax(masked_logits, dim=1)
        policy_loss = -torch.sum(target_probs * log_probs, dim=1).mean()
        value_loss = F.mse_loss(values, target_values)
        total_loss = policy_loss + value_loss

        total_loss_sum += total_loss.item()
        policy_loss_sum += policy_loss.item()
        value_loss_sum += value_loss.item()

    return (
        total_loss_sum / num_batches,
        policy_loss_sum / num_batches,
        value_loss_sum / num_batches,
    )


def run_training_epochs(model, optimizer, scheduler, buffer, num_epochs):
    """
    在完整 replay buffer 上训练指定个 epoch。
    scheduler.step() 按 batch 步进以匹配 CosineAnnealingLR 的 T_max (batch 数)。
    返回 (epoch 平均 loss 列表, 累计训练 batch 数)。
    """
    total_batches = 0
    epoch_results = []

    for epoch in range(num_epochs):
        indices = list(range(len(buffer)))
        random.shuffle(indices)
        num_batches = len(indices) // config.TRAIN_BATCH
        if num_batches == 0:
            break

        batch_total_l, batch_pol_l, batch_val_l = 0.0, 0.0, 0.0
        for step in range(num_batches):
            batch_indices = indices[step * config.TRAIN_BATCH : (step + 1) * config.TRAIN_BATCH]
            batch_data = buffer.get_batch(batch_indices)
            tl, pl, vl = train_step(model, optimizer, batch_data, DEVICE)
            scheduler.step()
            batch_total_l += tl
            batch_pol_l += pl
            batch_val_l += vl
            total_batches += 1

        avg_l = batch_total_l / num_batches
        avg_p = batch_pol_l / num_batches
        avg_v = batch_val_l / num_batches
        epoch_results.append((avg_l, avg_p, avg_v))

        if num_epochs > 1:
            print(f"[Training]   Epoch {epoch+1}/{num_epochs} | {num_batches} 批次 | "
                  f"Loss: {avg_l:.4f} (Pol: {avg_p:.4f}, Val: {avg_v:.4f})")

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
        model: Optional[BanqiNet] = None,
    ) -> None:
        super().__init__(name="TrainWorker", daemon=True)
        self.data_q = data_q
        self.stop_flag = stop_flag
        self.model = model if model is not None else BanqiNet().to(DEVICE)

        # 模型 + 优化器 + 调度器
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.LEARNING_RATE)
        self.scheduler = lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.LR_DECAY_STEPS,
            eta_min=config.MIN_LR,
        )
        # 恢复 checkpoint
        load_checkpoint(self.model, self.optimizer, self.scheduler)
        # 立即导出一次，确保 Rust 侧有可用的 .pt（全新模型也导出初始）
        save_checkpoint(self.model, self.optimizer, self.scheduler)

        self.buffer = DataBuffer(config.MAX_SAMPLE_BUFFER_SIZE)
        self.val_buffer = DataBuffer(config.VAL_BUFFER_CAPACITY)

        self.round_num = 0
        self.total_batches_trained = 0
        self.total_loss_sum = 0.0
        self.total_policy_loss_sum = 0.0
        self.total_value_loss_sum = 0.0
        self._stats_lock = threading.Lock()

    def _drain_new_episodes(self, max_items: int) -> List[Dict]:
        """从队列取最多 max_items 个 episode；不足则阻塞等待首个。"""
        episodes: List[Dict] = []
        try:
            first = self.data_q.get(timeout=0.5)
        except queue.Empty:
            return episodes
        episodes.append(first)
        # 尽量把队列里现有数据一次性取光
        for _ in range(max_items - 1):
            try:
                episodes.append(self.data_q.get_nowait())
            except queue.Empty:
                break
        return episodes

    def run(self) -> None:
        print(f"[Training] 🚀 训练线程启动（batch={config.TRAIN_BATCH}, "
              f"MinSamples={config.MIN_SAMPLES_TO_START}, "
              f"Epochs/Round={config.TRAIN_EPOCHS_PER_ROUND}）...")

        while not self.stop_flag[0]:
            episodes = self._drain_new_episodes(config.QUEUE_FETCH_BATCH)
            if not episodes:
                if self.stop_flag[0]:
                    break
                continue

            # 拆分 train / val（沿用原逻辑，尽量按新数据比例拆分）
            split_point = max(1, int(len(episodes) * (1.0 - config.VAL_SPLIT)))
            train_eps = episodes[:split_point]
            val_eps = episodes[split_point:]

            count_train = 0
            for ep in train_eps:
                if ep.get("num_samples", 0) > 0 or (ep.get("samples") or len(ep["boards"])):
                    self.buffer.add_samples(episode_to_samples(ep))
                    count_train += len(ep["boards"])
            count_val = 0
            for ep in val_eps:
                if ep.get("num_samples", 0) > 0 or (ep.get("samples") or len(ep["boards"])):
                    self.val_buffer.add_samples(episode_to_samples(ep))
                    count_val += len(ep["boards"])

            print(f"[Training] 📥 消费 {len(episodes)} 局 → "
                  f"train: {count_train}, val: {count_val} → Buffer={len(self.buffer)}")

            # 最少样本检查
            min_required = max(config.TRAIN_BATCH, config.MIN_SAMPLES_TO_START)
            if len(self.buffer) < min_required:
                print(f"[Training] ⚠️ Buffer={len(self.buffer)} < {min_required}，暂不训练，等待更多")
                continue

            self._train_round()

    def _train_round(self) -> None:
        """对完整 Buffer 训练多个 epoch，并做验证与 checkpoint。"""
        epoch_results, batches_in_round = run_training_epochs(
            self.model, self.optimizer, self.scheduler,
            self.buffer, config.TRAIN_EPOCHS_PER_ROUND,
        )

        with self._stats_lock:
            self.total_batches_trained += batches_in_round
            round_total = sum(r[0] for r in epoch_results)
            round_pol = sum(r[1] for r in epoch_results)
            round_val = sum(r[2] for r in epoch_results)
            self.total_loss_sum += round_total
            self.total_policy_loss_sum += round_pol
            self.total_value_loss_sum += round_val

        if epoch_results:
            last_avg_l, last_avg_p, last_avg_v = epoch_results[-1]
            cur_lr = self.optimizer.param_groups[0]['lr']
            print(f"[Training] ✅ Round#{self.round_num} 结束 | {batches_in_round} 批次 | "
                  f"Loss: {last_avg_l:.4f} (Pol: {last_avg_p:.4f}, Val: {last_avg_v:.4f}) "
                  f"| lr={cur_lr:.2e}")

        # 验证集评估
        min_val_samples = config.TRAIN_BATCH * config.VAL_EVAL_MIN_BATCHES
        if len(self.val_buffer) >= min_val_samples:
            val_result = evaluate(self.model, self.val_buffer, config.TRAIN_BATCH, DEVICE)
            if val_result is not None:
                vl, vp, vv = val_result
                train_ref = epoch_results[-1][0] if epoch_results else 0.0
                flag = " ⚠️ 过拟合?" if vl > train_ref + 0.1 else ""
                print(f"[Training] 📊 验证集: Loss={vl:.4f} (Pol: {vp:.4f}, Val: {vv:.4f}){flag}")

        self.round_num += 1
        if self.round_num % config.CHECKPOINT_EVERY_N_ROUNDS == 0:
            save_checkpoint(self.model, self.optimizer, self.scheduler)

    def stats(self) -> Dict[str, float]:
        with self._stats_lock:
            return {
                "round_num": self.round_num,
                "total_batches": self.total_batches_trained,
                "avg_loss": self.total_loss_sum / max(1, self.total_batches_trained),
                "avg_policy_loss": self.total_policy_loss_sum / max(1, self.total_batches_trained),
                "avg_value_loss": self.total_value_loss_sum / max(1, self.total_batches_trained),
            }

    def finalize(self) -> None:
        """最终落盘 checkpoint。"""
        save_checkpoint(self.model, self.optimizer, self.scheduler)
        print("[Training] 🎉 最终 Checkpoint 已保存")
