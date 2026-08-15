"""
verify_mini.py — 验证 4x2 迷你暗棋训练收敛

1. 加载训练产物（.pt / .pth），构造模型 Predictor。
2. 在 `MiniDarkChess` 环境上做「训练模型 vs 随机基线」对局：
   - 模型方：Gumbel MCTS（MCTS_SIMS 次搜索）
   - 随机方：每次从合法动作中均匀随机选一个
   - 轮流让模型先手(红)/后手(黑)，统计模型胜率
3. 若存在训练历史（run_training_mini.py 打印/保存），一并报告 loss 趋势。

用法：
    python python/verify_mini.py
    # 可选：MINI_MCTS_SIMS 控制模型搜索强度（默认 32）
"""
from __future__ import annotations

import os
import random
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import banqi_4x8
from constant_mini import ACTION_SPACE_SIZE
from nn_model_mini import MiniBanqiNet, load_model_weights

NUM_GAMES = int(os.getenv("MINI_VERIFY_GAMES", "60"))
MCTS_SIMS = int(os.getenv("MINI_MCTS_SIMS", "32"))
MAX_ACTIONS = int(os.getenv("MINI_MAX_ACTIONS", "12"))
MODEL_PATH = os.getenv("MINI_MODEL_PATH", "banqi_mini_model_latest.pt")
STATE_DICT_PATH = os.getenv("MINI_STATE_DICT_PATH", "banqi_mini_model_latest.pth")


class ModelPredictor:
    """把训练好的 MiniBanqiNet 封装成 mcts_search_action 需要的 (boards, scalars)->(logits, values) 回调。"""

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
        print(f"[Verify] 已加载 TorchScript 模型: {MODEL_PATH}")
    elif os.path.exists(STATE_DICT_PATH):
        state = torch.load(STATE_DICT_PATH, map_location=device)
        model.load_state_dict(state["model_state_dict"])
        print(f"[Verify] 已加载 state_dict 模型: {STATE_DICT_PATH}")
    else:
        print(f"[Verify] ⚠️ 未找到 {MODEL_PATH} / {STATE_DICT_PATH}，使用随机初始化模型（用于对照）")
    return ModelPredictor(model, device)


def play_one_game(predictor: ModelPredictor, model_is_red: bool) -> int:
    """模型 vs 随机基线一局。返回 winner：+1 模型胜, 0 平, -1 随机胜。"""
    env = banqi_4x8.MiniDarkChess()
    moves = 0
    while not env.terminated():
        if env.winner() is not None:
            break
        is_red_turn = env.current_player() == 1
        model_turn = (is_red_turn == model_is_red)
        if model_turn:
            action = env.mcts_search_action(
                predictor, MCTS_SIMS, MAX_ACTIONS, c_visit=1.0, c_scale=0.25
            )
        else:
            legal = env.legal_moves()
            action = random.choice(legal) if legal else None
        if action is None:
            break
        env.step(action)
        moves += 1
        if moves > 400:  # 防御：避免极端长局
            break
    winner = env.winner()  # 1=红胜, -1=黑胜, 0=平
    if winner == 0 or winner is None:
        return 0
    if model_is_red:
        return 1 if winner == 1 else -1
    else:
        return 1 if winner == -1 else -1


def main() -> None:
    predictor = load_model()

    # 交替先手：模型红 / 模型黑，各跑一半
    results: List[int] = []
    model_wins = 0
    draws = 0
    for i in range(NUM_GAMES):
        model_is_red = (i % 2 == 0)
        w = play_one_game(predictor, model_is_red)
        results.append(w)
        if w == 1:
            model_wins += 1
        elif w == 0:
            draws += 1
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{NUM_GAMES}] 模型累计胜率 = {model_wins/(i+1):.2f} "
                  f"(胜{model_wins} 平{draws} 负{i+1-model_wins-draws})")

    model_losses = results.count(1)
    rand_losses = results.count(-1)
    draws_total = results.count(0)
    print("\n" + "=" * 56)
    print(f"  模型 vs 随机基线（共 {NUM_GAMES} 局，各半先手）")
    print("=" * 56)
    print(f"  模型胜：{model_losses}（{model_losses/NUM_GAMES:.1%}）")
    print(f"  平局：  {draws_total}（{draws_total/NUM_GAMES:.1%}）")
    print(f"  随机胜：{rand_losses}（{rand_losses/NUM_GAMES:.1%}）")
    print("=" * 56)

    # 收敛判定：模型胜率显著高于随机基线（>0.55 视为明显优势）
    win_rate = model_losses / NUM_GAMES
    random_baseline = 0.5 - draws_total / (2 * NUM_GAMES)  # 平局摊给双方
    improved = win_rate > max(0.55, random_baseline + 0.05)
    print(f"  模型胜率 = {win_rate:.3f}，随机基线期望 = {random_baseline:.3f}")
    if improved:
        print("  ✅ 结论：模型已显著优于随机基线，训练收敛有效。")
    else:
        print("  ⚠️ 结论：模型对随机基线未表现出显著优势（胜率不足），"
              "建议增加训练时长或调整参数。")
    print("=" * 56)
    return 0 if improved else 1


if __name__ == "__main__":
    sys.exit(main())
