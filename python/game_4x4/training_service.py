"""
training_service.py — 4x4 暗棋的训练消费者

从数据队列消费自对弈 episode，填充向量化 replay buffer，迭代训练 BanqiNet，
周期性导出 checkpoint（.pt 供推理 / .pth 供训练恢复）。CPU 训练。
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

# 与 self_play.py 保持一致：限制 torch intra-op 线程数（进程级全局）。
# 小网络多线程反而因线程池调度开销而变慢（实测 32 线程 batch=32 训练
# 比 2 线程慢一个量级），且会拖累同进程的 MCTS 推理。
torch.set_num_threads(int(os.getenv("G4X4_TORCH_THREADS", "2")))

# 使 python/（banqi 共享包所在目录）可导入；append 避免遮蔽本目录同名模块
import os as _os
import sys as _sys
_sys.path.append(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from config import config
from banqi.variant import get_variant
from banqi.constants import build_constants
from banqi.nn_model import BanqiNet
from banqi.data_augmentation import make_augmentor
from tb_logger import add_scalar  # TensorBoard 训练日志（未启用时为 no-op）

VARIANT = get_variant("4x4")
C = build_constants(VARIANT)
TOTAL_INPUT_CHANNELS = C.TOTAL_INPUT_CHANNELS
BOARD_ROWS = C.BOARD_ROWS
BOARD_COLS = C.BOARD_COLS
SCALAR_FEATURE_COUNT = C.SCALAR_FEATURE_COUNT
ACTION_SPACE_SIZE = C.ACTION_SPACE_SIZE
AUG = make_augmentor(VARIANT)
HAS_AUGMENT = True


def _resolve_device(spec: str) -> "torch.device":
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


DEVICE = _resolve_device(config.TRAIN_DEVICE)
print(f"[Training4x4] 训练设备: {DEVICE}（config.TRAIN_DEVICE={config.TRAIN_DEVICE!r}）")


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
        # anneal 模式下 game_result 的权重（0~1），由 TrainWorker 按轮更新
        self.value_result_weight = 0.0

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
            # value 目标可配置（G4X4_VALUE_TARGET）：
            #   mcts  -> mcts_value（搜索/教师平滑评估，噪声小；模仿阶段已用它训练）
            #   game  -> game_result_value（AlphaZero 标准，终局真值 ±1，无自举漂移）
            #   mixed -> 固定 0.5/0.5 混合
            #   anneal-> (1-w)*mcts_value + w*game_result，w 由 TrainWorker 按轮退火
            #            从 mcts_value 起步逐步过渡到 game_result，避免：
            #            (a) 切换断层（价值头从平滑目标突变到 ±1 噪声）；
            #            (b) RL 全程用模型自搜索 mcts_value 的自举闭环漂移。
            target_mode = config.VALUE_TARGET_MODE
            mv = s.get('mcts_value', 0.0)
            gr = s.get('game_result_value', 0.0)
            if target_mode == "game":
                val = gr
            elif target_mode == "mixed":
                val = 0.5 * mv + 0.5 * gr
            elif target_mode == "anneal":
                w = self.value_result_weight
                val = (1.0 - w) * mv + w * gr
            else:  # mcts（默认）
                val = mv
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
            'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
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
        print(f"[Training4x4] ✅ Checkpoint 保存成功: {config.STATE_DICT_PATH} + {config.MODEL_PATH}")
    except Exception as e:
        print(f"[Training4x4] ❌ Checkpoint 保存失败: {e}")
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
                    # 重要：加载 optimizer 状态会恢复旧的 lr（如 2e-3→1e-3），
                    # 覆盖 config.LEARNING_RATE。精化时必须强制使用配置 LR。
                    for pg in optimizer.param_groups:
                        pg['lr'] = config.LEARNING_RATE
                    if hasattr(optimizer, 'param_groups'):
                        print(f"[Training4x4] ℹ️ 恢复 optimizer 状态后强制 lr={config.LEARNING_RATE}")
                except Exception as e_opt:
                    print(f"[Training4x4] ⚠️ Optimizer 状态加载失败 ({e_opt})")
            if 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict'] is not None:
                try:
                    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                    # 重置衰减进度：精化会话从配置 LR 重新开始 cosine 周期，
                    # 避免恢复旧 epoch 导致 lr 立即跌到 eta_min。
                    try:
                        scheduler.last_epoch = -1
                    except Exception:
                        pass
                    # 跨版本防御：显式同步 base_lrs（torch 2.10 后部分版本
                    # load_state_dict 不恢复 base_lrs，导致 cosine 从旧 base 计算）。
                    try:
                        scheduler.base_lrs = [config.LEARNING_RATE] * len(scheduler.base_lrs)
                    except Exception:
                        pass
                    # 修复：checkpoint 持久化了旧 T_max（如 15000）会跨会话自我延续，
                    # 使 LR 沿错误余弦周期衰减（实测 lr 被压到 3.39e-4 而非 ~4.9e-4）。
                    # LR 计划本应由当前 config 决定，显式重置为配置值。
                    try:
                        scheduler.T_max = config.LR_DECAY_STEPS
                    except Exception:
                        pass
                    try:
                        scheduler.eta_min = config.MIN_LR
                    except Exception:
                        pass
                    # 与 last_epoch=-1 语义一致，防未来 torch 版本把 _step_count
                    # 引入余弦计算时再次踩坑。
                    try:
                        scheduler._step_count = 0
                    except Exception:
                        pass
                except Exception as e_sch:
                    print(f"[Training4x4] ⚠️ Scheduler 状态加载失败 ({e_sch})")
            print(f"[Training4x4] ✅ 从 {config.STATE_DICT_PATH} 恢复完整训练状态")
            state_loaded = True
        except Exception as e:
            print(f"[Training4x4] ⚠️ 完整 .pth 加载失败 ({e})，尝试仅加载权重...")
    if not state_loaded and os.path.exists(config.MODEL_PATH):
        try:
            jit_model = torch.jit.load(config.MODEL_PATH, map_location=DEVICE)
            model.load_state_dict(jit_model.state_dict())
            print(f"[Training4x4] ✅ 从 {config.MODEL_PATH} 加载模型权重 (TorchScript 回退)")
        except Exception as e2:
            print(f"[Training4x4] ⚠️ 权重加载失败 ({e2})，使用全新模型")
    if not state_loaded and not os.path.exists(config.MODEL_PATH) and not os.path.exists(config.STATE_DICT_PATH):
        print("[Training4x4] 📝 初始化全新模型（无 checkpoint）")


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
                        max_batches: Optional[int] = None):
    """训练 buffer，可选限制总批次数。

    max_batches: 限制本轮总训练批次数。关键修复——当每轮新增数据量远小于
    buffer（如 RL 自对弈慢、每轮仅几百样本而 buffer 上万）时，若每轮对整个
    buffer 训练多 epoch，每个样本会被反复训练几十次，导致过拟合旧自对弈
    分布、棋力退化（此前 55%→25% 的元凶）。限制训练量与"新数据量"匹配：
      每轮批次 ≈ 新样本数/32 × epochs，GPU 数据量大时自动恢复全覆盖训练。
    """
    total_batches = 0
    epoch_results = []
    for epoch in range(num_epochs):
        indices = list(range(len(buffer)))
        random.shuffle(indices)
        num_batches = len(indices) // config.TRAIN_BATCH
        if num_batches == 0:
            break
        # 本轮剩余可训练批次
        if max_batches is not None:
            remaining = max_batches - total_batches
            if remaining <= 0:
                break
            num_batches = min(num_batches, remaining)
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
    def __init__(self, data_q: "queue.Queue", stop_flag: "List[bool]", model: Optional[BanqiNet] = None):
        super().__init__(name="TrainWorker4x4", daemon=True)
        self.data_q = data_q
        self.stop_flag = stop_flag
        self.model = model if model is not None else BanqiNet(VARIANT).to(DEVICE)
        # weight_decay=1e-4：轻正则化，抑制小数据量下的过拟合/价值头漂移
        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=config.LEARNING_RATE, weight_decay=1e-4)
        self.scheduler = lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.LR_DECAY_STEPS, eta_min=config.MIN_LR,
        )
        load_checkpoint(self.model, self.optimizer, self.scheduler)
        save_checkpoint(self.model, self.optimizer, self.scheduler)

        self.buffer = DataBuffer(config.MAX_SAMPLE_BUFFER_SIZE)
        # 冷存储预填充：启动时从归档加载历史局复用，避免训练从"空 buffer + 少量
        # 新局"开始就过度拟合当轮数据（这是此前 RL 退化的核心原因之一）。
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
        # 无归档时降级为"用本会话前 N 条自对弈样本构建"（见
        # _ensure_fixed_eval_from_selfplay），保证监控在任何机器上都生效。
        self._fixed_eval = None
        self._raw_sample_pool: List[Dict] = []

    def _prefill_from_archive(self) -> None:
        """从冷存储归档加载历史 episode 预填充训练 buffer（复用训练数据）。

        自动选择归档目录：显式配置 > 模仿学习归档 > 默认 run_training 归档。
        """
        n_games = getattr(config, "ARCHIVE_PREFILL_GAMES", 0)
        if not n_games:
            return
        here = os.path.dirname(os.path.abspath(__file__))
        dirs = [
            config.ARCHIVE_PREFILL_DIR,
            os.path.join(here, "training_data", "archive_4x4_imitate"),
            os.path.join(here, "training_data", "archive_4x4"),
        ]
        archive_dir = next((d for d in dirs if d and os.path.isdir(d)), None)
        if not archive_dir:
            print("[Training4x4] ⚠️ 冷存储预填充：未找到归档目录，跳过")
            return
        try:
            from storage import load_jsonl_episodes, episode_dict_to_samples
            t0 = time.time()
            episodes = load_jsonl_episodes(archive_dir, limit_games=n_games)
            samples: List[Dict] = []
            for ep in episodes:
                samples.extend(episode_dict_to_samples(ep))
            if samples:
                self.buffer.add_samples(samples)
                print(f"[Training4x4] 🗃️ 冷存储预填充: 从 {archive_dir} 加载 "
                      f"{len(episodes)} 局 → {len(samples)} 样本 (Buffer={len(self.buffer)}, "
                      f"耗时 {time.time()-t0:.1f}s)")
            # 固定验证集（价值漂移监控）：取前 N 条局面及其终局结果，
            # 训练中周期性评估价值头输出，检测漂移是否领先于胜率下降。
            n_fixed = config.VALUE_DRIFT_NUM_POSITIONS
            if n_fixed > 0:
                self._fixed_eval = {
                    "boards": np.stack([np.array(s['board_state'], dtype=np.float32).reshape(
                        TOTAL_INPUT_CHANNELS, BOARD_ROWS, BOARD_COLS) for s in samples[:n_fixed]]),
                    "scalars": np.stack([np.array(s['scalar_state'], dtype=np.float32)
                                         for s in samples[:n_fixed]]),
                    "results": np.array([s.get('game_result_value', 0.0)
                                         for s in samples[:n_fixed]], dtype=np.float32),
                }
                print(f"[Training4x4] 🎯 固定价值验证集 {len(self._fixed_eval['boards'])} 局面已就绪")
        except Exception as e:  # pragma: no cover
            print(f"[Training4x4] ⚠️ 冷存储预填充失败 ({e})，继续正常训练")

    def _ensure_fixed_eval_from_selfplay(self, samples: List[Dict]) -> None:
        """无归档时，用本会话自对弈原始样本构建固定价值验证集。

        自对弈样本自带 game_result_value（终局真值），漂移监控不需要教师数据。
        只在 _fixed_eval 尚未构建时收集前 VALUE_DRIFT_NUM_POSITIONS 条；构建后
        释放样本池。必须在数据增强之前调用（保留原始局面）。
        """
        if self._fixed_eval is not None:
            return
        n_fixed = config.VALUE_DRIFT_NUM_POSITIONS
        if n_fixed <= 0:
            return
        self._raw_sample_pool.extend(samples)
        if len(self._raw_sample_pool) < n_fixed:
            return
        pool = self._raw_sample_pool[:n_fixed]
        try:
            self._fixed_eval = {
                "boards": np.stack([np.array(s['board_state'], dtype=np.float32).reshape(
                    TOTAL_INPUT_CHANNELS, BOARD_ROWS, BOARD_COLS) for s in pool]),
                "scalars": np.stack([np.array(s['scalar_state'], dtype=np.float32)
                                     for s in pool]),
                "results": np.array([s.get('game_result_value', 0.0)
                                     for s in pool], dtype=np.float32),
            }
            self._raw_sample_pool = []
            print(f"[Training4x4] 🎯 固定价值验证集（自对弈样本）"
                  f"{len(self._fixed_eval['boards'])} 局面已就绪")
        except Exception as e:  # pragma: no cover
            print(f"[Training4x4] ⚠️ 自对弈样本构建固定价值验证集失败 ({e})，稍后重试")
            self._raw_sample_pool = pool

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
        print(f"[Training4x4] 🚀 训练线程启动（batch={config.TRAIN_BATCH}, "
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
                has_data = ep.get("num_samples", 0) > 0 or (
                    ep.get("samples") or ep.get("boards"))
                if not has_data:
                    continue
                train_samples.extend(episode_to_samples(ep))

            # 无归档时用自对弈原始样本构建固定验证集（价值漂移监控）。
            # 必须在增强之前调用，保留原始局面与终局结果。
            self._ensure_fixed_eval_from_selfplay(train_samples)

            # 对称增强（仅训练侧）：冷存储 archive_q 保存的仍是原始 episode，
            # 此处增强只作用于训练 replay buffer 的数据源。
            aug_count = 0
            if HAS_AUGMENT and config.DATA_AUGMENT_ENABLED and train_samples:
                transforms = [t.strip() for t in config.DATA_AUGMENT_TRANSFORMS.split(",") if t.strip()]
                raw_count = len(train_samples)
                train_samples = AUG.augment_samples(
                    train_samples,
                    transforms=transforms,
                    keep_original=config.DATA_AUGMENT_KEEP_ORIGINAL,
                )
                aug_count = len(train_samples) - raw_count

            if train_samples:
                self.buffer.add_samples(train_samples)

            aug_note = f"（增强 +{aug_count}）" if aug_count else ""
            print(f"[Training4x4] 📥 消费 {len(episodes)} 局 → train: {len(train_samples)}"
                  f"{aug_note} → Buffer={len(self.buffer)}")

            min_required = max(config.TRAIN_BATCH, config.MIN_SAMPLES_TO_START)
            if len(self.buffer) < min_required:
                continue
            # 记录本轮新增样本量（含增强），用于限制训练量
            self._last_round_new_samples = len(train_samples)
            self._train_round()

    def _train_round(self) -> None:
        # 每轮训练批次数与新数据量匹配，防过拟合旧分布：
        #   max_batches = 新样本/32 × epochs（保证新数据至少被完整训练），
        #   下限 32 批次（防退化），上限 = buffer 全覆盖（GPU 大数据量时自然达标）。
        n_new = max(32, self._last_round_new_samples)
        per_epoch_batches = max(1, n_new // config.TRAIN_BATCH)
        max_batches = per_epoch_batches * config.TRAIN_EPOCHS_PER_ROUND
        full_cover = (len(self.buffer) // config.TRAIN_BATCH) * config.TRAIN_EPOCHS_PER_ROUND
        max_batches = min(max_batches, full_cover)
        epoch_results, batches_in_round = run_training_epochs(
            self.model, self.optimizer, self.scheduler,
            self.buffer, config.TRAIN_EPOCHS_PER_ROUND,
            max_batches=max_batches,
        )
        with self._stats_lock:
            self.total_batches_trained += batches_in_round
            if epoch_results:
                round_total = sum(r[0] for r in epoch_results)
                round_pol = sum(r[1] for r in epoch_results)
                round_val = sum(r[2] for r in epoch_results)
                self.total_loss_sum += round_total
                self.total_policy_loss_sum += round_pol
                self.total_value_loss_sum += round_val
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
            print(f"[Training4x4] ✅ Round#{self.round_num} | {batches_in_round} 批次 | "
                  f"Loss: {last_avg_l:.4f} (Pol: {last_avg_p:.4f}, Val: {last_avg_v:.4f}) "
                  f"| lr={entry['lr']:.2e}")

        self.round_num += 1
        if self.round_num % config.CHECKPOINT_EVERY_N_ROUNDS == 0:
            save_checkpoint(self.model, self.optimizer, self.scheduler)

        # value 目标退火（anneal 模式）：每 VALUE_ANNEAL_STEP_ROUNDS 轮增加
        # game_result 权重，从 mcts_value 平滑过渡到终局真值（防自举漂移）。
        if config.VALUE_TARGET_MODE == "anneal":
            w = config.VALUE_ANNEAL_START + \
                (self.round_num // config.VALUE_ANNEAL_STEP_ROUNDS) * config.VALUE_ANNEAL_INCREMENT
            w = min(1.0, w)
            self.buffer.value_result_weight = w
            print(f"[Training4x4] 🔄 value退火权重(game_result)={w:.2f} (Round#{self.round_num})")

        # 固定验证集价值漂移监控：价值头输出均值/方差/区分度是否领先于胜率下降
        if (config.VALUE_DRIFT_EVAL_ROUNDS > 0 and hasattr(self, "_fixed_eval")
                and self.round_num % config.VALUE_DRIFT_EVAL_ROUNDS == 0):
            self._eval_value_drift()

        # ---- TensorBoard 训练日志（x 轴为累计训练 batch 数）----
        if config.TENSORBOARD_ENABLED:
            step = self.total_batches_trained
            add_scalar("train/loss", entry["train_loss"], step)
            add_scalar("train/policy_loss", entry["train_policy_loss"], step)
            add_scalar("train/value_loss", entry["train_value_loss"], step)
            add_scalar("train/lr", entry["lr"], step)

    def _eval_value_drift(self) -> None:
        """在固定验证集上评估价值头输出，检测价值漂移。

        指标：
          - pred mean/std（漂移 = 预测值整体偏移/方差膨胀）
          - 与终局结果的 Pearson 相关 & 胜负区分度（价值"准度"下降是漂移信号）
        """
        fixed = getattr(self, "_fixed_eval", None)
        if fixed is None:
            return
        try:
            self.model.eval()
            with torch.inference_mode():
                b = torch.from_numpy(np.ascontiguousarray(fixed["boards"]))
                s = torch.from_numpy(np.ascontiguousarray(fixed["scalars"]))
                logits, values = self.model(b, s)
                pred = values.cpu().numpy().reshape(-1).astype(np.float32)
            self.model.train()
            gr = fixed["results"]
            corr = float(np.corrcoef(pred, gr)[0, 1]) if len(pred) > 2 else 0.0
            sep = float(pred[gr > 0].mean() - pred[gr < 0].mean()) if (np.any(gr > 0) and np.any(gr < 0)) else 0.0
            print(f"[Training4x4] 📊 价值漂移 Round#{self.round_num}: pred_mean={pred.mean():+.3f} "
                  f"std={pred.std():.3f} corr(终局)={corr:.3f} 胜负区分度={sep:.3f}")
            add_scalar("value_drift/pred_mean", pred.mean())
            add_scalar("value_drift/pred_std", pred.std())
            add_scalar("value_drift/corr_result", corr)
            add_scalar("value_drift/sep", sep)
        except Exception as e:  # pragma: no cover
            print(f"[Training4x4] ⚠️ 价值漂移评估失败 ({e})")

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
        with self._stats_lock:
            return list(self.round_history)

    def finalize(self) -> None:
        save_checkpoint(self.model, self.optimizer, self.scheduler)
        print("[Training4x4] 🎉 最终 Checkpoint 已保存")
