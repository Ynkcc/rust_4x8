"""
training_service_mini.py — 4x2 迷你暗棋的训练消费者

从数据队列消费自对弈 episode，填充向量化 replay buffer，迭代训练 MiniBanqiNet，
周期性导出 checkpoint（.pt 供推理 / .pth 供训练恢复）。CPU 训练。
"""
from __future__ import annotations

import os
import queue
import random
import threading
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from config_mini import config
from constant_mini import (
    TOTAL_INPUT_CHANNELS,
    BOARD_ROWS,
    BOARD_COLS,
    SCALAR_FEATURE_COUNT,
    ACTION_SPACE_SIZE,
)
from nn_model_mini import MiniBanqiNet


def _resolve_device(spec: str) -> "torch.device":
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


DEVICE = _resolve_device(config.TRAIN_DEVICE)
print(f"[TrainingMini] 训练设备: {DEVICE}（config.TRAIN_DEVICE={config.TRAIN_DEVICE!r}）")


class DataBuffer:
    """向量化缓冲区，优化内存并加速 Tensor 转换。"""

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


def save_checkpoint(model, optimizer, scheduler) -> None:
    pt_temp_path = config.MODEL_PATH + ".tmp"
    pth_temp_path = config.STATE_DICT_PATH + ".tmp"
    trace_model = getattr(model, "_orig_mod", model)
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

        with torch.inference_mode():
            example_board = torch.randn(1, TOTAL_INPUT_CHANNELS, BOARD_ROWS, BOARD_COLS, device=DEVICE)
            example_scalars = torch.randn(1, SCALAR_FEATURE_COUNT, device=DEVICE)
            traced_model = torch.jit.trace(trace_model, (example_board, example_scalars))
            traced_model.save(pt_temp_path)
        os.replace(pt_temp_path, config.MODEL_PATH)
        print(f"[TrainingMini] ✅ Checkpoint 保存成功: {config.STATE_DICT_PATH} + {config.MODEL_PATH}")
    except Exception as e:
        print(f"[TrainingMini] ❌ Checkpoint 保存失败: {e}")
        for tmp in [pt_temp_path, pth_temp_path]:
            if os.path.exists(tmp):
                os.remove(tmp)


def load_checkpoint(model, optimizer, scheduler) -> None:
    state_loaded = False
    if os.path.exists(config.STATE_DICT_PATH):
        try:
            checkpoint = torch.load(config.STATE_DICT_PATH, map_location=DEVICE)
            model.load_state_dict(checkpoint['model_state_dict'])
            if 'optimizer_state_dict' in checkpoint:
                try:
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                except Exception as e_opt:
                    print(f"[TrainingMini] ⚠️ Optimizer 状态加载失败 ({e_opt})")
            if 'scheduler_state_dict' in checkpoint:
                try:
                    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                except Exception as e_sch:
                    print(f"[TrainingMini] ⚠️ Scheduler 状态加载失败 ({e_sch})")
            print(f"[TrainingMini] ✅ 从 {config.STATE_DICT_PATH} 恢复完整训练状态")
            state_loaded = True
        except Exception as e:
            print(f"[TrainingMini] ⚠️ 完整 .pth 加载失败 ({e})，尝试仅加载权重...")
    if not state_loaded and os.path.exists(config.MODEL_PATH):
        try:
            jit_model = torch.jit.load(config.MODEL_PATH, map_location=DEVICE)
            model.load_state_dict(jit_model.state_dict())
            print(f"[TrainingMini] ✅ 从 {config.MODEL_PATH} 加载模型权重 (TorchScript 回退)")
        except Exception as e2:
            print(f"[TrainingMini] ⚠️ 权重加载失败 ({e2})，使用全新模型")
    if not state_loaded and not os.path.exists(config.MODEL_PATH) and not os.path.exists(config.STATE_DICT_PATH):
        print("[TrainingMini] 📝 初始化全新模型（无 checkpoint）")


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


def run_training_epochs(model, optimizer, scheduler, buffer, num_epochs):
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
            batch_indices = indices[step * config.TRAIN_BATCH:(step + 1) * config.TRAIN_BATCH]
            batch_data = buffer.get_batch(batch_indices)
            tl, pl, vl = train_step(model, optimizer, batch_data, DEVICE)
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


class TrainWorker(threading.Thread):
    def __init__(self, data_q: "queue.Queue", stop_flag: "List[bool]", model: Optional[MiniBanqiNet] = None):
        super().__init__(name="TrainWorkerMini", daemon=True)
        self.data_q = data_q
        self.stop_flag = stop_flag
        self.model = model if model is not None else MiniBanqiNet().to(DEVICE)
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.LEARNING_RATE)
        self.scheduler = lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.LR_DECAY_STEPS, eta_min=config.MIN_LR,
        )
        load_checkpoint(self.model, self.optimizer, self.scheduler)
        save_checkpoint(self.model, self.optimizer, self.scheduler)

        self.buffer = DataBuffer(config.MAX_SAMPLE_BUFFER_SIZE)
        self.round_num = 0
        self.total_batches_trained = 0
        self.total_loss_sum = 0.0
        self.round_history: List[Dict] = []
        self._stats_lock = threading.Lock()

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
        print(f"[TrainingMini] 🚀 训练线程启动（batch={config.TRAIN_BATCH}, "
              f"MinSamples={config.MIN_SAMPLES_TO_START}, "
              f"Epochs/Round={config.TRAIN_EPOCHS_PER_ROUND}）...")
        while not self.stop_flag[0]:
            episodes = self._drain_new_episodes(config.QUEUE_FETCH_BATCH)
            if not episodes:
                if self.stop_flag[0]:
                    break
                continue
            train_samples: List[Dict] = []
            for ep in episodes:
                if ep.get("num_samples", 0) <= 0:
                    continue
                train_samples.extend(episode_to_samples(ep))
            if train_samples:
                self.buffer.add_samples(train_samples)
            print(f"[TrainingMini] 📥 消费 {len(episodes)} 局 → train: {len(train_samples)} "
                  f"→ Buffer={len(self.buffer)}")

            min_required = max(config.TRAIN_BATCH, config.MIN_SAMPLES_TO_START)
            if len(self.buffer) < min_required:
                continue
            self._train_round()

    def _train_round(self) -> None:
        epoch_results, batches_in_round = run_training_epochs(
            self.model, self.optimizer, self.scheduler,
            self.buffer, config.TRAIN_EPOCHS_PER_ROUND,
        )
        with self._stats_lock:
            self.total_batches_trained += batches_in_round
            if epoch_results:
                self.total_loss_sum += sum(r[0] for r in epoch_results)
            if epoch_results:
                last_avg_l, last_avg_p, last_avg_v = epoch_results[-1]
            else:
                last_avg_l = last_avg_p = last_avg_v = 0.0
            entry = {
                "round": self.round_num,
                "batches": batches_in_round,
                "train_loss": last_avg_l,
                "train_policy_loss": last_avg_p,
                "train_value_loss": last_avg_v,
                "lr": self.optimizer.param_groups[0]['lr'],
            }
            self.round_history.append(entry)

        if epoch_results:
            print(f"[TrainingMini] ✅ Round#{self.round_num} | {batches_in_round} 批次 | "
                  f"Loss: {last_avg_l:.4f} (Pol: {last_avg_p:.4f}, Val: {last_avg_v:.4f}) "
                  f"| lr={entry['lr']:.2e}")

        self.round_num += 1
        if self.round_num % config.CHECKPOINT_EVERY_N_ROUNDS == 0:
            save_checkpoint(self.model, self.optimizer, self.scheduler)

    def stats(self) -> Dict[str, float]:
        with self._stats_lock:
            return {
                "round_num": self.round_num,
                "total_batches": self.total_batches_trained,
                "avg_loss": self.total_loss_sum / max(1, self.total_batches_trained),
            }

    def round_history_snapshot(self) -> List[Dict]:
        with self._stats_lock:
            return list(self.round_history)

    def finalize(self) -> None:
        save_checkpoint(self.model, self.optimizer, self.scheduler)
        print("[TrainingMini] 🎉 最终 Checkpoint 已保存")
