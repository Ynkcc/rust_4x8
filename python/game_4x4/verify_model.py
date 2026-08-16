"""
verify_model.py — 指定模型文件验证 vs 启发式 MCTS（用于对比不同 checkpoint）

用法：python verify_model.py <模型.pt 或 .pth 路径> [games]
"""
from __future__ import annotations

import os, sys
import numpy as np
import torch

torch.set_num_threads(int(os.getenv("G4X4_TORCH_THREADS", "2")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import banqi_4x8
from config import config
from nn_model import Banqi4x4Net, load_model_weights
from verify_vs_heuristic_mcts import ModelPredictor


def main():
    model_path = sys.argv[1] if len(sys.argv) > 1 else config.MODEL_PATH
    games = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    hm_sims = int(os.getenv("G4X4_HM_SIMS", "64"))
    mcts_sims = int(os.getenv("G4X4_SIMS", "64"))

    device = torch.device("cpu")
    model = Banqi4x4Net().to(device).eval()
    load_model_weights(model, model_path, device)
    pred = ModelPredictor(model, device)
    print(f"[Verify] 模型={model_path} 对局={games} 模型sims={mcts_sims} 启发式sims={hm_sims}",
          flush=True)

    model_is_red = True
    wins = draws = losses = 0
    for i in range(games):
        env = banqi_4x8.Game4x4()
        moves = 0
        while not env.terminated():
            if env.winner() is not None:
                break
            cur = env.current_player()
            if (cur == 1) == model_is_red:
                a = env.mcts_search_action(pred, mcts_sims, 16, 1.0, 0.25)
            else:
                a = env.heuristic_mcts_action(hm_sims)
            if a is None:
                break
            env.step(a)
            moves += 1
            if moves > 400:
                break
        w = env.winner()
        if w == 0:
            draws += 1
        elif (w == 1) == model_is_red:
            wins += 1
        else:
            losses += 1
        print(f"  #{i+1}: {'胜' if ((w==1)==model_is_red and w!=0) else ('平' if w==0 else '负')} "
              f"(len={moves})", flush=True)
        model_is_red = not model_is_red

    total = wins + draws + losses
    print(f"\n[Verify] 结果: 胜{wins} 平{draws} 负{losses} "
          f"胜率={100*wins/total:.1f}% (n={total})", flush=True)


if __name__ == "__main__":
    main()
