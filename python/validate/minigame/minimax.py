"""
minimax.py — Minimax 完美裁判 + 纯随机对手策略（微型验证引擎）。

Minimax 在井字棋上毫秒级算出完美策略，用于黄金基准测试断言"全平、无负局"。
返回值为绝对玩家视角的效用（+1=红/当前先手胜，-1=黑/后手胜，0=平）。
"""

from __future__ import annotations

import random

from .tic_tac_toe import TicTacToe


def minimax(env: TicTacToe) -> float:
    """返回从当前玩家视角的最大效用（+1/0/-1）。"""
    winner = env.winner()
    if winner is not None:
        if winner == 0:
            return 0.0
        # winner 是绝对玩家；从当前玩家视角
        return 1.0 if winner == env.to_play else -1.0
    best = -2.0
    for a in env.legal_actions():
        next_env, _term, _win = env.step(a)
        v = -minimax(next_env)   # 对手视角取负
        if v > best:
            best = v
    return best


def minimax_best_action(env: TicTacToe):
    """返回 Minimax 最优动作（当前玩家视角）。"""
    best_v = -2.0
    best_a = None
    for a in env.legal_actions():
        next_env, _term, _win = env.step(a)
        v = -minimax(next_env)
        if v > best_v:
            best_v = v
            best_a = a
    return best_a


def random_action(env: TicTacToe, rng: random.Random):
    """纯随机对手策略。"""
    legal = env.legal_actions()
    return rng.choice(legal) if legal else None
