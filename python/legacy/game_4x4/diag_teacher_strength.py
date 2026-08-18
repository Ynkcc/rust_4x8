"""诊断：找到 vs 启发式64 能稳定 >80% 的教师（高 sims 启发式 / minimax）。"""
import os, sys
import numpy as np
import torch
torch.set_num_threads(2)
sys.path.insert(0, "/root/rust_4x8/python/game_4x4")

import banqi_4x8

def play(red, black, n=20):
    wins = draws = losses = 0
    red_turn = True
    for i in range(n):
        env = banqi_4x8.Game4x4()
        moves = 0
        while not env.terminated():
            if env.winner() is not None:
                break
            cur = env.current_player()
            a = (red if cur == 1 else black)(env)
            if a is None:
                break
            env.step(a); moves += 1
            if moves > 400:
                break
        w = env.winner()
        if w == 0:
            draws += 1
        elif w == 1:
            wins += 1
        else:
            losses += 1
    print(f"  {wins} 平{draws} 负{losses} = {100*wins/n:.0f}%", flush=True)
    return wins / n

def hm_fn(sims):
    def f(env):
        return env.heuristic_mcts_action(sims)
    return f

def mm_fn(depth):
    def f(env):
        return env.minimax_action(depth)
    return f

hm64 = hm_fn(64)

print("=== 高 sims 启发式教师 vs 启发式64 ===", flush=True)
for s in [512, 1024, 2048]:
    print(f"  hm{s}:", end="")
    play(hm_fn(s), hm64, n=24)

print("=== minimax 教师 vs 启发式64 ===", flush=True)
for d in [1, 2]:
    print(f"  mm{d}:", end="")
    play(mm_fn(d), hm64, n=24)
