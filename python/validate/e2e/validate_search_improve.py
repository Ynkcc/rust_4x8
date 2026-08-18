"""
validate_search_improve.py — 搜索提升率测试（真实暗棋模型 + Rust 真实环境）。

检验对象：Gumbel AlphaZero 的数学核心——搜索后的策略 π 必须严格优于
网络输出的原始先验 P（真实游戏环境下的搜索提升）。

与井字棋 minigame 版本的关键区别：
  - 使用**真实暗棋网络 BanqiNet**，直接加载 `../../banqi_model_latest.pt`；
  - 对局在 **Rust 真实环境**（`banqi_4x8.DarkChess`，4x8 暗棋，352 动作空间）中进行；
  - MCTS 搜索与纯网络贪婪都由 Rust 侧实现（`mcts_search_action` /
    `greedy_action`，共用同一个 Python 网络回调），消除实现差异。

核心：让「带 MCTS 搜索的模型」与「纯粹拿网络 Policy Head 贪婪落子的模型（Raw）」
对弈，MCTS 版本胜率应显著高于纯网络版本（通常 > 70%）。

异常诊断：若 MCTS 打不过甚至输给 Raw Policy，说明根节点 Gumbel 噪声注入、
Sequential Halving 预算分配或 Value 反传累加机制写反。

运行：python3 python/validate/validate_search_improve.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

import banqi_4x8 as b  # pyo3 绑定（Rust 暗棋环境 + MCTS）

import os
import sys

_VALIDATE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PYTHON_DIR = os.path.dirname(_VALIDATE_DIR)
_BANQI_DIR = os.path.join(_PYTHON_DIR, "banqi")
for _d in (_PYTHON_DIR, _BANQI_DIR, _VALIDATE_DIR):
    if _d not in sys.path:
        sys.path.insert(0, _d)

import validate_common  # noqa: F401
from validate_common import Reporter, run_part, require

from banqi.variant import get_variant
from banqi.nn_model import BanqiNet, load_model_weights

VARIANT = get_variant("4x8")
DEVICE = "cpu"
MODEL_PATH = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "banqi_model_latest.pt"
))

# 对弈 / 搜索超参数（搜索侧对齐生产暗棋配置：max_considered=16）
NUM_SIMULATIONS = 32
MAX_CONSIDERED = 16
C_SCALE = 1.0
N_GAMES = 25              # 每轮 2 局抵消先后手，共 2*N_GAMES 局
MAX_STEPS_PER_GAME = 200  # 防死循环保护


def load_real_model() -> BanqiNet:
    require(os.path.exists(MODEL_PATH), f"模型文件不存在: {MODEL_PATH}")
    model = BanqiNet(VARIANT)
    load_model_weights(model, MODEL_PATH, torch.device(DEVICE))
    model.eval()
    return model


def make_predict_fn(model):
    """包装 BanqiNet 为 banqi_4x8 约定的 predict_fn(boards_np, scalars_np)。"""
    def predict(boards, scalars):
        xb = torch.from_numpy(boards).float()
        xs = torch.from_numpy(scalars).float()
        with torch.inference_mode():
            logits, values = model(xb, xs)
        return logits.cpu().numpy(), values.cpu().numpy().reshape(-1)
    return predict


def play_game(model, mcts_first: bool, num_simulations: int) -> int:
    """
    一局真实暗棋对局：红方先手。`mcts_first=True` 时红方用 MCTS、黑方用 Raw；
    反之红方用 Raw、黑方用 MCTS。
    返回全局胜者：1=红胜，-1=黑胜，0=平局。
    """
    env = b.DarkChess()
    predict_fn = make_predict_fn(model)
    steps = 0
    while not env.terminated():
        is_red = env.current_player() == 1
        use_mcts = is_red == mcts_first
        if use_mcts:
            action = env.mcts_search_action(
                predict_fn, num_simulations, MAX_CONSIDERED, C_SCALE)
        else:
            action = env.greedy_action(predict_fn)
        if action is None:  # 无合法动作（终局/异常）
            break
        env.step(action)
        steps += 1
        if steps >= MAX_STEPS_PER_GAME:
            break
    w = env.winner()
    return w if w is not None else 0


def test_search_improvement() -> None:
    rep = Reporter(f"search policy improvement: real BanqiNet on DarkChess "
                   f"(MCTS vs Raw, {2 * N_GAMES} games)")
    model = load_real_model()
    print(f"      加载真实模型: {MODEL_PATH}")
    print(f"      MCTS: sims={NUM_SIMULATIONS}, max_considered={MAX_CONSIDERED}")

    mcts_wins = 0
    raw_wins = 0
    draws = 0
    for _ in range(N_GAMES):
        # 局 1：红=MCTS，黑=Raw
        w1 = play_game(model, mcts_first=True, num_simulations=NUM_SIMULATIONS)
        # 局 2：红=Raw，黑=MCTS（交换先后手，抵消先手偏差）
        w2 = play_game(model, mcts_first=False, num_simulations=NUM_SIMULATIONS)
        if w1 == 1:
            mcts_wins += 1
        elif w1 == -1:
            raw_wins += 1
        else:
            draws += 1
        if w2 == -1:
            mcts_wins += 1
        elif w2 == 1:
            raw_wins += 1
        else:
            draws += 1

    total = 2 * N_GAMES
    mcts_rate = mcts_wins / total
    print(f"      MCTS 胜 {mcts_wins}, Raw 胜 {raw_wins}, 平 {draws} "
          f"(共 {total} 局, MCTS 胜率 {mcts_rate:.1%})")

    rep.check(mcts_wins > raw_wins,
              f"MCTS 胜局数 > Raw ({mcts_wins} > {raw_wins})")
    rep.check(mcts_rate > 0.70,
              f"MCTS 胜率 > 70% ({mcts_rate:.1%})")

    ok = rep.summary()
    if ok:
        print("  ✅ 决策：真实暗棋下搜索显著优于原始策略，Gumbel 噪声 / "
              "Sequential Halving / Value 反传逻辑正确")
    else:
        print("  ❌ 决策：检查根节点 Gumbel 噪声注入、Sequential Halving 预算分配、"
              "Value 反传累加机制，或模型训练不足")
    require(ok, "搜索提升率测试未通过")


def main() -> None:
    run_part("search policy improvement (real DarkChess)", test_search_improvement)


if __name__ == "__main__":
    main()
