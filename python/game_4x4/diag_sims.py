"""诊断：模型 MCTS 在不同 sims 下 vs 启发式64 的胜率，判断是否搜索深度瓶颈。"""
import os, sys
import numpy as np
import torch
torch.set_num_threads(2)
sys.path.insert(0, "/root/rust_4x8/python/game_4x4")

import banqi_4x8
from config import config
from nn_model import Banqi4x4Net, load_model_weights
from verify_vs_heuristic_mcts import ModelPredictor

device = torch.device("cpu")
model = Banqi4x4Net().to(device).eval()
load_model_weights(model, config.MODEL_PATH, device)
pred = ModelPredictor(model, device)

def obs_of(env):
    b, s = env.observation()
    return (np.asarray(b, dtype=np.float32).reshape(1, 16, 4, 4),
            np.asarray(s, dtype=np.float32).reshape(1, -1))

def play(model_sims, hm_sims, n=20):
    wins = draws = losses = 0
    model_is_red = True
    for i in range(n):
        env = banqi_4x8.Game4x4()
        moves = 0
        while not env.terminated():
            if env.winner() is not None:
                break
            cur = env.current_player()
            if (cur == 1) == model_is_red:
                a = env.mcts_search_action(pred, model_sims, 16, 1.0, 0.25)
            else:
                a = env.heuristic_mcts_action(hm_sims)
            if a is None:
                break
            env.step(a); moves += 1
            if moves > 400:
                break
        w = env.winner()
        if w == 0:
            draws += 1
        elif (w == 1) == model_is_red:
            wins += 1
        else:
            losses += 1
        model_is_red = not model_is_red
    print(f"  model_sims={model_sims} vs hm_sims={hm_sims}: 胜{wins} 平{draws} 负{losses} "
          f"({100*wins/n:.0f}%)", flush=True)

# 不同模型 sims，固定对手 64
for ms in [64, 128, 256, 512]:
    play(ms, 64, n=20)
