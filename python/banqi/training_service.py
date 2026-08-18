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
from collections import namedtuple
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


# ============================================================================
# 训练步骤
# ============================================================================

# 单 batch 训练统计（供 TensorBoard 记录）：
#   total/policy/value：三类 loss；grad_norm：clip 前梯度范数（发散预警）；
#   entropy：目标策略平均熵（探索健康度）；value_mean/std：价值目标分布。
TrainStepStats = namedtuple(
    "TrainStepStats", "total policy value grad_norm entropy value_mean value_std"
)
_ZERO_STATS = TrainStepStats(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def train_step(model, optimizer, batch_data, device) -> TrainStepStats:
    model.train()
    boards_t, scalars_t, target_probs_t, target_values_t, masks_t = batch_data

    boards_t = boards_t.to(device, non_blocking=True)
    scalars_t = scalars_t.to(device, non_blocking=True)
    target_probs_t = target_probs_t.to(device, non_blocking=True)
    target_values_t = target_values_t.to(device, non_blocking=True).view(-1, 1)
    masks_t = masks_t.to(device, non_blocking=True)

    # ---- 来源校验：输入/目标任何非有限都跳过该 batch（不更新权重）----
    # 防止脏数据（NaN/Inf 的 board/scalar/policy/mask/value）进入前向传播，
    # 进而在 backward 后经 clip_grad_norm_ + optimizer.step() 一次性污染整份权重。
    finite_inputs = (
        torch.isfinite(boards_t).all()
        and torch.isfinite(scalars_t).all()
        and torch.isfinite(target_probs_t).all()
        and torch.isfinite(target_values_t).all()
        and torch.isfinite(masks_t).all()
    )
    # 每行 target 策略和 > 0 且非负（0*-inf 或全 0 target 会导致 NaN/梯度消失）
    valid_target = bool((target_probs_t >= 0.0).all()) and bool(
        target_probs_t.sum(dim=1).min() > 0.0
    )
    if not finite_inputs or not valid_target:
        print(
            f"[TR] ⚠️ 跳过 1 个异常 batch（输入/策略目标非有限或非法）"
        )
        # 返回一个有限的占位 loss，避免上层把 NaN 累进统计/日志
        return _ZERO_STATS

    optimizer.zero_grad()
    logits, values = model(boards_t, scalars_t)

    # ---- 安全 mask：用 -1e9 屏蔽非法动作（替代 (mask-1)*1e9）----
    # 原实现 logits + (mask-1)*1e9 在 logits 含 +inf 时会产生 inf -> log_softmax
    # 得到 NaN（inf-inf）。改用 masked_fill 只把非法位置置为极大负值，
    # 配合下方梯度有限性检查，从源头杜绝 NaN 传播。
    masked_logits = logits.masked_fill(masks_t < 0.5, -1e9)
    log_probs = F.log_softmax(masked_logits, dim=1)
    policy_loss = -torch.sum(target_probs_t * log_probs, dim=1).mean()

    value_loss = F.mse_loss(values, target_values_t)
    total_loss = policy_loss + value_loss

    # ---- 数值安全：loss / 前向输出非有限则跳过，不污染权重 ----
    if not torch.isfinite(total_loss):
        print(
            f"[TR] ⚠️ 跳过 1 个异常 batch（loss 非有限: "
            f"policy={float(policy_loss):.4f} value={float(value_loss):.4f}），"
            f"不更新权重"
        )
        optimizer.zero_grad()
        return _ZERO_STATS

    total_loss.backward()
    # ---- 梯度有限性检查：NaN/Inf 梯度静默放行是权重被污染的主通道 ----
    # 一旦出现非有限梯度，clip_grad_norm_ 返回 NaN 且 optimizer.step() 会把
    # 整份权重写成 NaN。故在 clip 前显式检测并跳过该 batch。
    grad_ok = all(
        p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()
    )
    if not grad_ok:
        print("[TR] ⚠️ 跳过 1 个异常 batch（检测到非有限梯度），不更新权重")
        optimizer.zero_grad()
        return _ZERO_STATS
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    # 记录目标分布统计（无需梯度）：策略熵 + 价值目标 mean/std，供 TB 观测
    with torch.no_grad():
        log_p = torch.log(target_probs_t.clamp_min(1e-12))
        entropy = float(-(target_probs_t * log_p).sum(dim=1).mean())
        value_mean = float(target_values_t.mean())
        value_std = float(target_values_t.std())

    return TrainStepStats(
        total=total_loss.item(),
        policy=policy_loss.item(),
        value=value_loss.item(),
        grad_norm=float(grad_norm),
        entropy=entropy,
        value_mean=value_mean,
        value_std=value_std,
    )


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
        batch_grad_l, batch_ent_l, batch_vm_l, batch_vs_l = 0.0, 0.0, 0.0, 0.0
        for step in range(num_batches):
            batch_indices = indices[step * buffer.cfg.TRAIN_BATCH: (step + 1) * buffer.cfg.TRAIN_BATCH]
            batch_data = buffer.get_batch(batch_indices)
            s = train_step(model, optimizer, batch_data, device)
            scheduler.step()
            batch_total_l += s.total
            batch_pol_l += s.policy
            batch_val_l += s.value
            batch_grad_l += s.grad_norm
            batch_ent_l += s.entropy
            batch_vm_l += s.value_mean
            batch_vs_l += s.value_std
            total_batches += 1

        epoch_results.append((
            batch_total_l / num_batches,
            batch_pol_l / num_batches,
            batch_val_l / num_batches,
            batch_grad_l / num_batches,
            batch_ent_l / num_batches,
            batch_vm_l / num_batches,
            batch_vs_l / num_batches,
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
        self._last_round_aug_count = 0
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
        # 上一轮训练后的模型权重快照（CPU，供 eval/* vs prev 守门评估）
        self._prev_weights: Optional[Dict[str, torch.Tensor]] = None

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
        aspace = C.ACTION_SPACE_SIZE
        try:
            masks = np.array([s['action_mask'] for s in samples], dtype=np.float32)
            if masks.ndim == 1:  # 兼容缺省 mask 的异常样本：默认全合法
                masks = np.ones((len(samples), aspace), dtype=np.float32)
            return {
                "boards": np.stack([np.array(s['board_state'], dtype=np.float32).reshape(
                    C.TOTAL_INPUT_CHANNELS, C.BOARD_ROWS, C.BOARD_COLS) for s in samples]),
                "scalars": np.stack([np.array(s['scalar_state'], dtype=np.float32)
                                     for s in samples]),
                "results": np.array([s.get('game_result_value', 0.0)
                                     for s in samples], dtype=np.float32),
                "masks": masks,
                # 策略头验证 ground truth：启发式/MCTS 最优动作（缺失时置 -1 跳过）
                "teacher_actions": np.array(
                    [int(s['teacher_action']) if s.get('teacher_action') is not None else -1
                     for s in samples], dtype=np.int64),
            }
        except Exception:  # pragma: no cover
            return None

    @staticmethod
    def _select_balanced_fixed_samples(pool: List[Dict], n_fixed: int) -> List[Dict]:
        """从原始样本池中筛选「满足要求」的固定验证局面（方案 B）。

        要求：
          1. 仅取原始样本（不含数据增强副本），保证验证集是对真实自对弈局面的
             采样，不因增强变换而引入冗余/伪样本；
          2. 按终局结果分层（game_result_value 的符号）均衡覆盖：胜方视角(+1)、
             负方视角(-1)、平局(0)，避免验证集被某一类终局结果垄断，从而更客观
             地反映价值头的整体视角一致性；
          3. 数量封顶为 n_fixed，不足时按到达顺序兜底补齐。

        注意：game_result_value 是「当前行动方视角」的终局真值（胜=+1/负=-1/平=0），
        因此正/负两层天然覆盖了胜负双方的样本视角，是视角对称性的有效代理。
        """
        if not pool or n_fixed <= 0:
            return []
        buckets: Dict[int, List[Dict]] = {1: [], -1: [], 0: []}
        for s in pool:
            gr = s.get("game_result_value", 0.0)
            key = 1 if gr > 0 else (-1 if gr < 0 else 0)
            buckets[key].append(s)
        per_bucket = max(1, n_fixed // 3)
        selected: List[Dict] = []
        for key in (1, -1, 0):
            selected.extend(buckets[key][:per_bucket])
        # 若总量不足 n_fixed，从剩余池中按到达顺序补齐（保持多样性）
        if len(selected) < n_fixed:
            seen = {id(s) for s in selected}
            for s in pool:
                if id(s) in seen:
                    continue
                selected.append(s)
                if len(selected) >= n_fixed:
                    break
        return selected[:n_fixed]

    def _ensure_fixed_eval_from_selfplay(self, samples: List[Dict]) -> None:
        """无归档时，用本会话自对弈原始样本构建固定价值验证集（方案 B）。

        在数据增强之前调用（保留原始局面与终局结果），且**不参与训练**——仅用于
        `_eval_value_drift` 的价值漂移/视角一致性监控。按终局结果分层均衡筛选，
        保证覆盖胜/负/平三类真实局面。
        """
        if self._fixed_eval is not None:
            return
        n_fixed = self.cfg.VALUE_DRIFT_NUM_POSITIONS
        if n_fixed <= 0:
            return
        self._raw_sample_pool.extend(samples)
        if len(self._raw_sample_pool) < n_fixed:
            return
        pool = self._select_balanced_fixed_samples(self._raw_sample_pool, n_fixed)
        fixed = self._build_fixed_eval(pool) if pool else None
        if fixed is not None:
            self._fixed_eval = fixed
            self._raw_sample_pool = []
            results = fixed["results"]
            dist = (int((results > 0).sum()), int((results < 0).sum()), int((results == 0).sum()))
            print(f"{self.tag} 🎯 固定价值验证集（自对弈样本，仅验证不训练）"
                  f"{len(fixed['boards'])} 局面已就绪 | 终局分布 胜/负/平={dist}")
        else:
            # 构建失败：保留最近一轮池子，避免无限累积
            self._raw_sample_pool = self._raw_sample_pool[-n_fixed:]

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
            # 记录本轮增强副本数（供 TB train/augment_count）
            self._last_round_aug_count = aug_count

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
                (last_avg_l, last_avg_p, last_avg_v, last_grad_norm,
                 last_entropy, last_v_mean, last_v_std) = epoch_results[-1]
            else:
                last_avg_l = last_avg_p = last_avg_v = 0.0
                last_grad_norm = last_entropy = last_v_mean = last_v_std = 0.0
            entry: Dict = {
                "round": self.round_num,
                "batches": batches_in_round,
                "train_loss": last_avg_l,
                "train_policy_loss": last_avg_p,
                "train_value_loss": last_avg_v,
                "grad_norm": last_grad_norm,
                "policy_entropy": last_entropy,
                "value_mean": last_v_mean,
                "value_std": last_v_std,
                "lr": self.optimizer.param_groups[0]['lr'],
            }
            self.round_history.append(entry)

        if epoch_results:
            print(f"{self.tag} ✅ Round#{self.round_num} | {batches_in_round} 批次 | "
                  f"Loss: {last_avg_l:.4f} (Pol: {last_avg_p:.4f}, Val: {last_avg_v:.4f}) "
                  f"| grad={last_grad_norm:.3f} | lr={entry['lr']:.2e}")

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

        # 固定验证集价值漂移 + 策略头命中率监控（仅验证，不参与训练）
        if (cfg.VALUE_DRIFT_EVAL_ROUNDS > 0 and self._fixed_eval is not None
                and self.round_num % cfg.VALUE_DRIFT_EVAL_ROUNDS == 0):
            self._eval_value_drift()
            self._eval_policy_accuracy()

        # 对战评估（vs 规则对手 + 上一轮模型守门，仅验证，不参与训练）。
        # 注意：_prev_weights 此时仍是「上一轮训练后」的快照（本轮快照在末尾更新），
        # 因此 eval/* vs prev 反映「本轮模型 vs 上一轮模型」的进步/退化。
        if cfg.EVAL_MATCH_ROUNDS > 0 and self.round_num % cfg.EVAL_MATCH_ROUNDS == 0:
            self._eval_match()

        # TensorBoard 训练日志（x 轴为累计训练 batch 数）
        if cfg.TENSORBOARD_ENABLED:
            step = self.total_batches_trained
            add_scalar("train/loss", entry["train_loss"], step)
            add_scalar("train/policy_loss", entry["train_policy_loss"], step)
            add_scalar("train/value_loss", entry["train_value_loss"], step)
            add_scalar("train/lr", entry["lr"], step)
            # ---- 新增：训练过程健康度 ----
            add_scalar("train/grad_norm", entry["grad_norm"], step)
            add_scalar("train/policy_entropy", entry["policy_entropy"], step)
            add_scalar("data/value_target_mean", entry["value_mean"], step)
            add_scalar("data/value_target_std", entry["value_std"], step)
            add_scalar("train/buffer_size", len(self.buffer), step)
            add_scalar("train/samples_per_round", self._last_round_new_samples, step)
            add_scalar("train/augment_count", self._last_round_aug_count, step)
            add_scalar("train/dropped_samples_total", self.buffer.total_dropped, step)
            add_scalar("queue/backlog", self._safe_qsize(), step)
            if cfg.VALUE_TARGET_MODE == "anneal":
                add_scalar("train/value_anneal_w", self.buffer.value_result_weight, step)

        # 保存本轮训练后权重快照，供下一轮 eval/* vs prev 使用（CPU 副本，避免占显存）
        self._prev_weights = {
            k: v.detach().to("cpu").clone()
            for k, v in self.model.state_dict().items()
        }

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
                # 验证集输入需搬到模型所在设备（GPU 训练时 self.device=cuda:*），
                # 否则 FloatTensor(CPU) vs cuda.FloatTensor 类型不匹配。
                b = torch.from_numpy(np.ascontiguousarray(fixed["boards"])).to(self.device)
                s = torch.from_numpy(np.ascontiguousarray(fixed["scalars"])).to(self.device)
                _, values = self.model(b, s)
                pred = values.cpu().numpy().reshape(-1).astype(np.float32)
            self.model.train()
            gr = fixed["results"]
            corr = float(np.corrcoef(pred, gr)[0, 1]) if len(pred) > 2 else 0.0
            sep = float(pred[gr > 0].mean() - pred[gr < 0].mean()) if (np.any(gr > 0) and np.any(gr < 0)) else 0.0
            print(f"{self.tag} 📊 价值漂移 Round#{self.round_num}: pred_mean={pred.mean():+.3f} "
                  f"std={pred.std():.3f} corr(终局)={corr:.3f} 胜负区分度={sep:.3f}")
            # step 必须显式传入：SummaryWriter 在 global_step=None 时所有点落在
            # step=0 重叠，导致 TensorBoard 上曲线"看起来不动"。与 train/* 同 x 轴。
            step = self.total_batches_trained
            add_scalar("value_drift/pred_mean", pred.mean(), step)
            add_scalar("value_drift/pred_std", pred.std(), step)
            add_scalar("value_drift/corr_result", corr, step)
            add_scalar("value_drift/sep", sep, step)
        except Exception as e:  # pragma: no cover
            print(f"{self.tag} ⚠️ 价值漂移评估失败 ({e})")

    def _eval_policy_accuracy(self) -> None:
        """在固定验证集上评估策略头（Policy）单点命中率。

        计算模型输出的 Top-1 / Top-3 候选动作（mask 掉非法动作后）
        与「启发式/MCTS 最优动作」（teacher_actions）的重合率（命中率），
        用于验证策略头是否学到合理走子常识。验证集不参与训练，仅评估。
        """
        fixed = self._fixed_eval
        if fixed is None:
            return
        teacher = fixed["teacher_actions"]
        if teacher.size == 0 or int((teacher >= 0).sum()) == 0:
            print(f"{self.tag} ⚠️ 策略头验证跳过：验证集无有效 teacher_action")
            return
        try:
            self.model.eval()
            with torch.inference_mode():
                b = torch.from_numpy(np.ascontiguousarray(fixed["boards"])).to(self.device)
                s = torch.from_numpy(np.ascontiguousarray(fixed["scalars"])).to(self.device)
                logits, _ = self.model(b, s)
                logits = logits.cpu().numpy().astype(np.float32)
            self.model.train()
            masks = fixed["masks"].astype(np.float32)
            # 防御：非有限 logits 置为极小（不参与 Top-k），非法动作 mask 为 -1e9，
            # 与训练端 masked_fill 语义一致，避免 NaN/Inf 污染命中率统计。
            ml_all = np.where(np.isfinite(logits), logits, -1e9).copy()
            ml_all = np.where(masks >= 0.5, ml_all, -1e9)
            valid = teacher >= 0
            if int(valid.sum()) == 0:
                print(f"{self.tag} ⚠️ 策略头验证跳过：无有效样本")
                return
            ml = ml_all[valid]
            ta = teacher[valid]
            top1_idx = np.argmax(ml, axis=1)
            # Top-3：无需排序所有动作，用 argpartition 取前 3 个候选
            k = min(3, ml.shape[1])
            topk_idx = np.argpartition(-ml, k - 1, axis=1)[:, :k]
            hit1 = float(np.mean(top1_idx == ta))
            hit3 = float(np.mean(np.any(topk_idx == ta[:, None], axis=1)))
            n_eval = int(valid.sum())
            print(f"{self.tag} 🎯 策略头命中 Round#{self.round_num}: Top-1={hit1:.3f} "
                  f"Top-3={hit3:.3f}（{n_eval} 局面 vs 启发式/MCTS 最优动作）")
            step = self.total_batches_trained
            add_scalar("policy_acc/top1_vs_teacher", hit1, step)
            add_scalar("policy_acc/top3_vs_teacher", hit3, step)
            add_scalar("policy_acc/n_positions", n_eval, step)
        except Exception as e:  # pragma: no cover
            print(f"{self.tag} ⚠️ 策略头验证失败 ({e})")

    def _safe_qsize(self) -> int:
        """线程安全地读取数据队列积压；多进程队列 qsize 可能不可用（返回 -1）。"""
        try:
            qsize = self.data_q.qsize()
            return int(qsize) if qsize is not None else -1
        except Exception:  # noqa: BLE001 - 平台/包装不支持 qsize 时降级
            return -1

    def _eval_match(self) -> None:
        """周期性对战评估：vs 启发式/minimax 规则对手 + 上一轮模型（守门）。

        与 value_drift 同 x 轴（累计训练 batch 数）。评估期间模型切 eval 模式，
        结束后恢复 train。规则对手走统一评估协议（banqi.eval，EVAL_SIMS=64），
        三变体协议一致。vs prev 的局数减半以控制评估耗时。
        """
        from banqi import eval as banqi_eval

        cfg = self.cfg
        n = max(1, cfg.EVAL_MATCH_GAMES)
        opps = [o.strip() for o in cfg.EVAL_MATCH_OPPONENTS.split(",") if o.strip()]
        step = self.total_batches_trained
        self.model.eval()
        cur = banqi_eval.ModelPredictor(self.model, self.device)
        try:
            for opp in opps:
                try:
                    wins, draws, losses, avg_moves = banqi_eval.play_match_stats(
                        cur, n=n, model_sims=banqi_eval.EVAL_SIMS,
                        opponent=opp, variant_id=self.variant.id,
                    )
                    tot = max(1, wins + draws + losses)
                    add_scalar(f"eval/win_rate_vs_{opp}", 100.0 * wins / tot, step)
                    add_scalar(f"eval/draw_rate_vs_{opp}", 100.0 * draws / tot, step)
                    add_scalar(f"eval/loss_rate_vs_{opp}", 100.0 * losses / tot, step)
                    add_scalar(f"eval/avg_game_length_vs_{opp}", avg_moves, step)
                    print(f"{self.tag} ⚔️ Round#{self.round_num} vs {opp}: "
                          f"胜{wins} 平{draws} 负{losses} (n={n}, 平均{avg_moves:.0f}步)")
                except Exception as exc:  # noqa: BLE001 - 单对手失败不中断整体评估
                    print(f"{self.tag} ⚠️ 对战评估 vs {opp} 失败: {exc}")
            # 与上一轮模型对头（守门）：_prev_weights 为上一轮训练后快照
            if cfg.EVAL_MATCH_VS_PREV and self._prev_weights is not None:
                try:
                    prev_model = BanqiNet(self.variant).to(self.device)
                    prev_model.load_state_dict({
                        k: v.to(self.device) for k, v in self._prev_weights.items()
                    })
                    prev_model.eval()
                    prev_pred = banqi_eval.ModelPredictor(prev_model, self.device)
                    n_prev = max(4, n // 2)
                    wins, draws, losses, _ = banqi_eval.play_match_vs(
                        cur, prev_pred, n=n_prev,
                        model_sims=banqi_eval.EVAL_SIMS, variant_id=self.variant.id,
                    )
                    tot = max(1, wins + draws + losses)
                    add_scalar("eval/win_rate_vs_prev", 100.0 * wins / tot, step)
                    add_scalar("eval/draw_rate_vs_prev", 100.0 * draws / tot, step)
                    add_scalar("eval/loss_rate_vs_prev", 100.0 * losses / tot, step)
                    print(f"{self.tag} ⚔️ Round#{self.round_num} vs prev: "
                          f"胜{wins} 平{draws} 负{losses} (n={n_prev})")
                except Exception as exc:  # noqa: BLE001 - 守门失败不影响主流程
                    print(f"{self.tag} ⚠️ 对战评估 vs prev 失败: {exc}")
        finally:
            self.model.train()

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
