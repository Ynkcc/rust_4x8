"""
network.py — 微型 policy-value 网络（纯 CPU）。

面向 Tic-Tac-Toe：输入 (N, 2, 3, 3) 特征，输出 policy logits (N, 9) + value (N, 1)。

设计考量（本机 PyTorch CPU 上 BatchNorm / 多层残差极慢）：
  采用轻量 CNN（单层卷积 + 无 BatchNorm），在保证表达力的同时把单次评估压到毫秒级，
  使微型引擎能在合理时间内完成自对弈训练。逻辑（policy CE + value MSE/tanh）与生产一致。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .tic_tac_toe import CHANNELS, BOARD_ROWS, BOARD_COLS, N_CELLS


class TicTacToeNet(nn.Module):
    """微型策略-价值网络：输入 (N,2,3,3)，输出 (N,9) logits + (N,1) tanh value。"""

    def __init__(self, hidden: int = 32):
        super().__init__()
        self.conv = nn.Conv2d(CHANNELS, hidden, kernel_size=3, padding=1)
        self.policy_fc = nn.Linear(hidden * BOARD_ROWS * BOARD_COLS, N_CELLS)
        self.value_fc1 = nn.Linear(hidden * BOARD_ROWS * BOARD_COLS, 32)
        self.value_fc2 = nn.Linear(32, 1)

    def forward(self, x):
        # x: (N, 2, 3, 3)
        h = F.relu(self.conv(x))
        h = h.view(h.size(0), -1)
        logits = self.policy_fc(h)
        v = F.relu(self.value_fc1(h))
        value = torch.tanh(self.value_fc2(v))
        return logits, value


def policy_logits_to_probs(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """对合法动作做数值稳定 softmax（对齐 Rust compute_probs_from_logits）。"""
    masked = logits + (mask - 1.0) * 1e9
    return F.softmax(masked, dim=1)


def greedy_action(logits: torch.Tensor, legal: list) -> int:
    """从网络原始 logits 中取合法动作里 logit 最大的一个（纯 Raw 贪婪落子）。"""
    logits_np = logits.detach().cpu().numpy().flatten()
    best, best_v = None, float("-inf")
    for a in legal:
        if logits_np[a] > best_v:
            best_v = logits_np[a]
            best = a
    return best
