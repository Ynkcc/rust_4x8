"""
verify_vs_heuristic_mcts.py — 4x4 暗棋：训练模型 vs 纯计算启发式 Gumbel MCTS

对局双方：
  - 模型方（Banqi4x4Net）：Gumbel MCTS（网络先验 + 网络价值）
  - 启发式方（Rust `Game4x4.heuristic_mcts_action`）：纯计算启发式 Gumbel MCTS
    （规则先验 logits + 多特征启发式价值，无需 torch，见 src/ai/heuristic_mcts.rs）

交替先手（模型红/黑各半），统计模型胜 / 平 / 负与胜率。

用法：
    python python/game_4x4/verify_vs_heuristic_mcts.py
    # 环境变量：
    #   G4X4_HM_GAMES   对局数（默认 40）
    #   G4X4_HM_SIMS    启发式 MCTS 模拟数（默认 64）
    #   G4X4_SIMS       模型 MCTS 模拟数（默认 64）
"""
from __future__ import annotations

import os
import sys
import time
from typing import Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import torch

# 与训练侧一致：限制 torch intra-op 线程数（小网络多线程反而更慢）
torch.set_num_threads(int(os.getenv("G4X4_TORCH_THREADS", "2")))

import banqi_4x8

from config import config
from eval_common import play_one, play_match, report as eval_report
from nn_model import Banqi4x4Net, load_model_weights

NUM_GAMES = int(os.getenv("G4X4_HM_GAMES", "40"))
HM_SIMS = int(os.getenv("G4X4_HM_SIMS", "64"))
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
        print(f"[HM] 已加载模型: {MODEL_PATH}")
    elif os.path.exists(STATE_DICT_PATH):
        state = torch.load(STATE_DICT_PATH, map_location=device)
        model.load_state_dict(state["model_state_dict"])
        print(f"[HM] 已加载模型: {STATE_DICT_PATH}")
    else:
        print(f"[HM] ⚠️ 未找到模型，使用随机初始化（对照）")
    return ModelPredictor(model, device)


def play_one_game(predictor: ModelPredictor, model_is_red: bool, hm_sims: int) -> int:
    """模型 vs 启发式 MCTS 一局。返回 +1 模型胜 / 0 平 / -1 启发式胜。"""
    env = banqi_4x8.Game4x4()
    moves = 0
    while not env.terminated():
        if env.winner() is not None:
            break
        is_red_turn = env.current_player() == 1
        model_turn = (is_red_turn == model_is_red)
        if model_turn:
            action = env.mcts_search_action(predictor, MCTS_SIMS, MAX_ACTIONS, 0.25)
        else:
            action = env.heuristic_mcts_action(hm_sims)
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
    print(f"[HM] 模型 MCTS sims={MCTS_SIMS} | 启发式 MCTS sims={HM_SIMS} | {NUM_GAMES} 局（各半先手）")

    wins, draws, losses, blk = play_match(predictor, n=NUM_GAMES, model_sims=MCTS_SIMS)
    import numpy as np
    mean = float(np.mean(blk)) if blk else 0.0
    std = float(np.std(blk)) if blk else 0.0
    print("\n" + "=" * 60)
    print(f"  模型 vs 启发式 Gumbel MCTS(sims={HM_SIMS}) 共 {NUM_GAMES} 局")
    print("=" * 60)
    print(f"  模型胜：{wins}（{wins/NUM_GAMES:.1%}）")
    print(f"  平局：  {draws}（{draws/NUM_GAMES:.1%}）")
    print(f"  启发式胜：{losses}（{losses/NUM_GAMES:.1%}）")
    print("=" * 60)
    print(f"  模型胜率 = {wins/NUM_GAMES:.3f}（块均 {mean:.1f}±{std:.1f}%）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
