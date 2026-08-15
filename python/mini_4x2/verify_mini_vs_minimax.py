"""
verify_mini_vs_minimax.py — 训练好的 mini 模型 vs Rust expectiminimax(alpha-beta) 对局测试。

对局双方：
  - 模型方（MiniBanqiNet）：Gumbel MCTS（可切换为纯网络贪婪）
  - minimax 方（Rust `MiniDarkChess.minimax_action`）：expectiminimax + alpha-beta
    剪枝，不依赖任何网络，纯规则搜索。

统计：交替先手（模型红 / 模型黑各半），报告模型胜 / 平 / 负与胜率。

用法：
    python python/verify_mini_vs_minimax.py
    # 环境变量：
    #   MINI_VM_GAMES   对局数（默认 30）
    #   MINI_VM_DEPTH   minimax 搜索深度（默认 6）
    #   MINI_VM_SIMS    模型 MCTS 模拟数（默认 32）
    #   MINI_VM_GREEDY  模型用纯网络贪婪（无搜索）时设为 1
"""
from __future__ import annotations

import os
import random
import sys
import time
from typing import List, Tuple

import numpy as np
import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import banqi_4x8
from nn_model_mini import MiniBanqiNet, load_model_weights

_HERE = os.path.dirname(os.path.abspath(__file__))
NUM_GAMES = int(os.getenv("MINI_VM_GAMES", "30"))
MM_DEPTH = int(os.getenv("MINI_VM_DEPTH", "6"))
MCTS_SIMS = int(os.getenv("MINI_VM_SIMS", "32"))
GREEDY = os.getenv("MINI_VM_GREEDY", "0") == "1"
MAX_ACTIONS = 12
MODEL_PATH = os.getenv("MINI_MODEL_PATH", os.path.join(_HERE, "banqi_mini_model_latest.pt"))
STATE_DICT_PATH = os.getenv("MINI_STATE_DICT_PATH", os.path.join(_HERE, "banqi_mini_model_latest.pth"))


class ModelPredictor:
    """把训练好的 MiniBanqiNet 封装成 mcts_search_action 需要的回调。"""

    def __init__(self, model: MiniBanqiNet, device: "torch.device"):
        self.model = model.to(device).eval()
        self.device = device

    def __call__(self, boards: np.ndarray, scalars: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        with torch.inference_mode():
            b = torch.from_numpy(np.ascontiguousarray(boards)).to(self.device)
            s = torch.from_numpy(np.ascontiguousarray(scalars)).to(self.device)
            logits, value = self.model(b, s)
            return logits.cpu().numpy().astype(np.float32), value.cpu().numpy().reshape(-1).astype(np.float32)


def load_model() -> ModelPredictor:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MiniBanqiNet()
    if os.path.exists(MODEL_PATH):
        load_model_weights(model, MODEL_PATH, device)
        print(f"[VM] 已加载模型: {MODEL_PATH}")
    elif os.path.exists(STATE_DICT_PATH):
        state = torch.load(STATE_DICT_PATH, map_location=device)
        model.load_state_dict(state["model_state_dict"])
        print(f"[VM] 已加载模型: {STATE_DICT_PATH}")
    else:
        print(f"[VM] ⚠️ 未找到模型，使用随机初始化（对照）")
    return ModelPredictor(model, device)


def model_action(env, predictor) -> int:
    """模型方动作：MCTS 或纯网络贪婪。"""
    if GREEDY:
        a = env.greedy_action(predictor)
    else:
        a = env.mcts_search_action(predictor, MCTS_SIMS, MAX_ACTIONS, c_visit=1.0, c_scale=0.25)
    return a


def play_one_game(predictor: ModelPredictor, model_is_red: bool, mm_depth: int) -> int:
    """模型 vs minimax 一局。返回 +1 模型胜 / 0 平 / -1 minimax 胜。"""
    env = banqi_4x8.MiniDarkChess()
    moves = 0
    mm_time = 0.0
    while not env.terminated():
        if env.winner() is not None:
            break
        is_red_turn = env.current_player() == 1
        model_turn = (is_red_turn == model_is_red)
        if model_turn:
            action = model_action(env, predictor)
        else:
            t0 = time.time()
            action = env.minimax_action(mm_depth)
            mm_time += time.time() - t0
        if action is None:
            break
        env.step(action)
        moves += 1
        if moves > 400:
            break
    winner = env.winner()  # 1=红胜, -1=黑胜, 0=平
    if winner == 0 or winner is None:
        return 0
    if model_is_red:
        return 1 if winner == 1 else -1
    return 1 if winner == -1 else -1


def main() -> None:
    predictor = load_model()
    print(f"[VM] minimax depth={MM_DEPTH} | 模型={'greedy' if GREEDY else f'MCTS({MCTS_SIMS})'} | {NUM_GAMES} 局")

    results: List[int] = []
    model_wins = draws = 0
    t_start = time.time()
    for i in range(NUM_GAMES):
        model_is_red = (i % 2 == 0)
        w = play_one_game(predictor, model_is_red, MM_DEPTH)
        results.append(w)
        if w == 1:
            model_wins += 1
        elif w == 0:
            draws += 1
        if (i + 1) % 5 == 0 or i + 1 == NUM_GAMES:
            print(f"  [{i+1}/{NUM_GAMES}] 模型胜率={model_wins/(i+1):.2f} "
                  f"(胜{model_wins} 平{draws} 负{i+1-model_wins-draws}) "
                  f"累计 {time.time()-t_start:.0f}s")

    model_wins = results.count(1)
    mm_wins = results.count(-1)
    draws_total = results.count(0)
    print("\n" + "=" * 60)
    print(f"  模型 vs Rust minimax(alpha-beta, depth={MM_DEPTH}) 共 {NUM_GAMES} 局（各半先手）")
    print("=" * 60)
    print(f"  模型胜：{model_wins}（{model_wins/NUM_GAMES:.1%}）")
    print(f"  平局：  {draws_total}（{draws_total/NUM_GAMES:.1%}）")
    print(f"  minimax 胜：{mm_wins}（{mm_wins/NUM_GAMES:.1%}）")
    print("=" * 60)
    win_rate = model_wins / NUM_GAMES
    print(f"  模型胜率 = {win_rate:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
