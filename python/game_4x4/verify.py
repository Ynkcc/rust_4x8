"""
verify.py — 4x4 暗棋模型对战验证

加载训练好的 Banqi4x4Net，分别与：
  - 随机基线对局（验证模型已学会基本策略）
  - Rust minimax(alpha-beta) 对局（搜索强度标尺）
交替先手统计胜率。

用法：
    python python/game_4x4/verify.py
    # 环境变量：
    #   G4X4_GAMES      对局数（默认 40）
    #   G4X4_MM_DEPTH   minimax 深度（默认 3）
    #   G4X4_SIMS       模型 MCTS 模拟数（默认 64）
"""
from __future__ import annotations

import os
import random
import sys
import time
from typing import Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import torch

import banqi_4x8

from config import config
from constant import ACTION_SPACE_SIZE, BOARD_ROWS, BOARD_COLS, SCALAR_FEATURE_COUNT
from nn_model import Banqi4x4Net, load_model_weights

NUM_GAMES = int(os.getenv("G4X4_GAMES", "40"))
MM_DEPTH = int(os.getenv("G4X4_MM_DEPTH", "3"))
MCTS_SIMS = int(os.getenv("G4X4_SIMS", "64"))
MAX_ACTIONS = 16
MODEL_PATH = config.MODEL_PATH
STATE_DICT_PATH = config.STATE_DICT_PATH


class ModelPredictor:
    def __init__(self, model: Banqi4x4Net, device: "torch.device"):
        self.model = model.to(device).eval()
        self.device = device

    def __call__(self, boards: np.ndarray, scalars: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        with torch.inference_mode():
            b = torch.from_numpy(np.ascontiguousarray(boards)).to(self.device)
            s = torch.from_numpy(np.ascontiguousarray(scalars)).to(self.device)
            logits, value = self.model(b, s)
            return logits.cpu().numpy().astype(np.float32), value.cpu().numpy().reshape(-1).astype(np.float32)


def load_model() -> ModelPredictor:
    device = torch.device("cpu")
    model = Banqi4x4Net()
    if os.path.exists(MODEL_PATH):
        load_model_weights(model, MODEL_PATH, device)
        print(f"[Verify4x4] 已加载模型: {MODEL_PATH}")
    elif os.path.exists(STATE_DICT_PATH):
        state = torch.load(STATE_DICT_PATH, map_location=device)
        model.load_state_dict(state["model_state_dict"])
        print(f"[Verify4x4] 已加载模型: {STATE_DICT_PATH}")
    else:
        print(f"[Verify4x4] ⚠️ 未找到模型，使用随机初始化（对照）")
    return ModelPredictor(model, device)


def play_vs_random(predictor: ModelPredictor, model_is_red: bool) -> int:
    """模型 vs 随机一局。返回 +1 模型胜 / 0 平 / -1 负。"""
    env = banqi_4x8.Game4x4()
    moves = 0
    while not env.terminated():
        if env.winner() is not None:
            break
        is_red_turn = env.current_player() == 1
        model_turn = (is_red_turn == model_is_red)
        if model_turn:
            action = env.mcts_search_action(predictor, MCTS_SIMS, MAX_ACTIONS, 1.0, 0.25)
        else:
            legal = env.legal_moves()
            action = random.choice(legal) if legal else None
        if action is None:
            break
        env.step(action)
        moves += 1
        if moves > 400:
            break
    winner = env.winner()
    if winner == 0 or winner is None:
        return 0
    if model_is_red:
        return 1 if winner == 1 else -1
    return 1 if winner == -1 else -1


def play_vs_minimax(predictor: ModelPredictor, model_is_red: bool, depth: int) -> int:
    """模型 vs minimax(depth) 一局。返回 +1 模型胜 / 0 平 / -1 负。"""
    env = banqi_4x8.Game4x4()
    moves = 0
    while not env.terminated():
        if env.winner() is not None:
            break
        is_red_turn = env.current_player() == 1
        model_turn = (is_red_turn == model_is_red)
        if model_turn:
            action = env.mcts_search_action(predictor, MCTS_SIMS, MAX_ACTIONS, 1.0, 0.25)
        else:
            action = env.minimax_action(depth)
        if action is None:
            break
        env.step(action)
        moves += 1
        if moves > 400:
            break
    winner = env.winner()
    if winner == 0 or winner is None:
        return 0
    if model_is_red:
        return 1 if winner == 1 else -1
    return 1 if winner == -1 else -1


def run_series(play_fn, label: str) -> None:
    wins = draws = 0
    t0 = time.time()
    for i in range(NUM_GAMES):
        model_is_red = (i % 2 == 0)
        w = play_fn(model_is_red)
        if w == 1:
            wins += 1
        elif w == 0:
            draws += 1
    losses = NUM_GAMES - wins - draws
    print(f"  {label}: 胜{wins} 平{draws} 负{losses} 胜率={wins / NUM_GAMES:.1%}  "
          f"耗时{time.time() - t0:.0f}s")


def main() -> None:
    predictor = load_model()
    print(f"[Verify4x4] 模型 MCTS sims={MCTS_SIMS} | minimax depth={MM_DEPTH} | {NUM_GAMES} 局")
    print()
    print("=== 模型 vs 随机基线 ===")
    run_series(lambda red: play_vs_random(predictor, red), f"模型(MCTS {MCTS_SIMS}) vs 随机")
    print()
    print(f"=== 模型 vs minimax(depth={MM_DEPTH}) ===")
    run_series(lambda red: play_vs_minimax(predictor, red, MM_DEPTH),
               f"模型(MCTS {MCTS_SIMS}) vs minimax(depth={MM_DEPTH})")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
