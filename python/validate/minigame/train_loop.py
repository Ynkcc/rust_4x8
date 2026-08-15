"""
train_loop.py — 自对弈训练环（微型验证引擎）。

支持两种模式：
  1. 完整自对弈：MCTS 自对弈生成样本 → 填充 replay buffer → 网络优化（policy CE + value MSE）。
  2. 固定 Batch 过拟合：从 Buffer 固定抽 1 个 Batch，锁死数据源，循环训练（供单 Batch 过拟合测试）。

价值标签语义对齐 Rust `finalize_episode`：
  每步 game_result = 该步玩家视角的最终结果（value_label_for_player(player, winner)）。
MCTS 每步生成改进策略 improved_policy 作为策略目标（对齐 get_improved_policy 训练目标）。
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

from .tic_tac_toe import (
    TicTacToe, N_CELLS, CHANNELS, BOARD_ROWS, BOARD_COLS,
    CURRENT, OPPONENT, value_label_for_player,
)
from .gumbel_mcts import GumbelMCTS, mcts_choose


# ============================================================================
# 数据缓冲
# ============================================================================

@dataclass
class ReplayBuffer:
    """极简向量化缓冲：boards / policy_targets / value_targets / masks。"""

    boards: list = field(default_factory=list)
    policy_targets: list = field(default_factory=list)
    value_targets: list = field(default_factory=list)
    masks: list = field(default_factory=list)
    capacity: int = 20000

    def add(self, boards, policies, values, masks):
        self.boards.extend(boards)
        self.policy_targets.extend(policies)
        self.value_targets.extend(values)
        self.masks.extend(masks)
        if len(self.boards) > self.capacity:
            excess = len(self.boards) - self.capacity
            del self.boards[:excess]
            del self.policy_targets[:excess]
            del self.value_targets[:excess]
            del self.masks[:excess]

    def __len__(self) -> int:
        return len(self.boards)

    def sample_batch(self, batch_size: int, rng: random.Random):
        idx = rng.sample(range(len(self.boards)), batch_size)
        boards = torch.from_numpy(
            np.stack([self.boards[i] for i in idx])).float()
        policies = torch.from_numpy(
            np.stack([self.policy_targets[i] for i in idx])).float()
        values = torch.tensor([self.value_targets[i] for i in idx], dtype=torch.float32)
        masks = torch.from_numpy(np.stack([self.masks[i] for i in idx])).float()
        return boards, policies, values, masks


# ============================================================================
# 自对弈
# ============================================================================

def sample_action_from_policy(policy: np.ndarray, legal: list, rng: random.Random) -> int:
    """按改进策略概率采样动作（含随机探索，生成多样局面）。"""
    probs = np.zeros(N_CELLS, dtype=np.float64)
    for a in legal:
        probs[a] = policy[a]
    total = probs.sum()
    if total <= 0:
        return rng.choice(legal)
    r = rng.random() * total
    for a in legal:
        r -= probs[a]
        if r <= 0:
            return a
    return legal[-1]


def self_play_one_game(net, num_simulations: int = 64,
                       max_considered_actions: int = 16, c_scale: float = 1.0,
                       rng: random.Random | None = None):
    """
    用 MCTS 自对弈一局，返回 (样本列表, winner)。
    每步按改进策略**随机采样**（带探索），以生成多样的训练局面。
    价值标签在调用方（finalize_episode）按视角回填。
    """
    if rng is None:
        rng = random.Random()
    env = TicTacToe()
    samples = []  # (board, policy_target, mask, player)
    while not env.is_terminal():
        legal = env.legal_actions()
        player = env.to_play
        mcts = GumbelMCTS(env, net, num_simulations, max_considered_actions, c_scale)
        result = mcts.run()
        if result is None:
            break
        _action, improved_policy, _ = result
        # 温度：按改进策略随机采样（低温探索），保证生成多样局面
        action = sample_action_from_policy(improved_policy, legal, rng)
        board = env.encode()
        mask = np.zeros(N_CELLS, dtype=np.float32)
        for a in legal:
            mask[a] = 1.0
        samples.append((board, improved_policy.astype(np.float32), mask, player))
        env, _term, _win = env.step(action)
    winner = env.winner()
    return samples, winner


def self_play_generate(net, num_games: int, num_simulations: int = 64,
                       max_considered_actions: int = 16, c_scale: float = 1.0,
                       buffer: ReplayBuffer | None = None,
                       seed: int = 0) -> ReplayBuffer:
    """
    生成 num_games 局自对弈数据并填充 buffer。
    价值标签按每步玩家的视角用 value_label_for_player 换算（对齐 finalize_episode）。
    """
    if buffer is None:
        buffer = ReplayBuffer()
    rng = random.Random(seed)
    for _ in range(num_games):
        samples, winner = self_play_one_game(
            net, num_simulations, max_considered_actions, c_scale, rng)
        for board, policy, mask, player in samples:
            value = value_label_for_player(player, winner)
            buffer.add([board], [policy], [value], [mask])
    return buffer


# ============================================================================
# 训练
# ============================================================================

def train_step(net, optimizer, batch, device="cpu"):
    """单步训练：policy CE + value MSE。返回 (total, policy, value) loss。"""
    boards, policies, values, masks = batch
    boards = boards.to(device)
    policies = policies.to(device)
    values = values.to(device).view(-1, 1)
    masks = masks.to(device)

    net.train()
    optimizer.zero_grad()
    logits, values_pred = net(boards)
    masked_logits = logits + (masks - 1.0) * 1e9
    log_probs = F.log_softmax(masked_logits, dim=1)
    policy_loss = -torch.sum(policies * log_probs, dim=1).mean()
    value_loss = F.mse_loss(values_pred, values)
    total = policy_loss + value_loss
    total.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
    optimizer.step()
    return total.item(), policy_loss.item(), value_loss.item()


def train_on_fixed_batch(net, optimizer, buffer: ReplayBuffer, batch_indices,
                         num_steps: int, batch_size: int, rng: random.Random,
                         device="cpu"):
    """
    在固定的一批样本上循环训练 num_steps 步（关闭新数据写入）。
    返回 loss 历史 [(total, policy, value), ...]。
    """
    boards = torch.from_numpy(np.stack([buffer.boards[i] for i in batch_indices])).float()
    policies = torch.from_numpy(
        np.stack([buffer.policy_targets[i] for i in batch_indices])).float()
    values = torch.tensor([buffer.value_targets[i] for i in batch_indices],
                          dtype=torch.float32)
    masks = torch.from_numpy(np.stack([buffer.masks[i] for i in batch_indices])).float()

    history = []
    n = len(batch_indices)
    for _ in range(num_steps):
        # 从固定 batch 中随机抽子 batch 训练
        idx = rng.sample(range(n), min(batch_size, n))
        sub = (
            boards[idx], policies[idx], values[idx], masks[idx]
        )
        tl, pl, vl = train_step(net, optimizer, sub, device)
        history.append((tl, pl, vl))
    return history
