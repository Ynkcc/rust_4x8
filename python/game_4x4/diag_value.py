"""诊断：模型价值头的准确度（决定 MCTS 搜索质量）。"""
import os, sys, time
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

# 生成一批自对弈对局，统计模型价值 vs 实际结果的相关性
sims = 64
games = 20
cfg = banqi_4x8.SelfPlayConfig(mcts_sims=sims, max_considered_actions=16, temperature_steps=12)
print(f"[DiagValue] 生成 {games} 局模型自对弈...", flush=True)
t0 = time.time()
eps = banqi_4x8.run_game4x4_parallel_self_play_with_predictor(
    predict_fn=pred, config=cfg, num_workers=4, games_per_worker=5, worker_id=0)
print(f"  {time.time()-t0:.1f}s", flush=True)

all_mv = []
all_gr = []
for e in eps:
    (boards, scalars, policies, mcts_values, completed_qs,
     root_visits, game_results, action_masks, actions) = e.get_samples()
    all_mv.extend(mcts_values)
    all_gr.extend(game_results)

mv = np.array(all_mv)
gr = np.array(all_gr)
print(f"[DiagValue] mcts_value mean={mv.mean():.3f} std={mv.std():.3f}")
print(f"[DiagValue] game_result mean={gr.mean():.3f}")
print(f"[DiagValue] 相关性 corr={np.corrcoef(mv, gr)[0,1]:.3f}")
print(f"[DiagValue] 胜局 mcts_value mean={mv[gr>0].mean():.3f} n={np.sum(gr>0)}")
print(f"[DiagValue] 负局 mcts_value mean={mv[gr<0].mean():.3f} n={np.sum(gr<0)}")
# 价值区分度（胜负局价值差）
if np.sum(gr>0) and np.sum(gr<0):
    sep = mv[gr>0].mean() - mv[gr<0].mean()
    print(f"[DiagValue] 胜负区分度={sep:.3f}")

# 对比启发式价值：启发式自对弈的 mcts_value 与结果
print(f"[DiagValue] === 启发式对照 ===", flush=True)
cfg2 = banqi_4x8.SelfPlayConfig(mcts_sims=sims, max_considered_actions=16, temperature_steps=12)
t0 = time.time()
eps2 = banqi_4x8.run_game4x4_heuristic_self_play(config=cfg2, num_games=20, concurrency=4, worker_id=0)
print(f"  {time.time()-t0:.1f}s", flush=True)
mv2, gr2 = [], []
for e in eps2:
    (boards, scalars, policies, mcts_values, completed_qs,
     root_visits, game_results, action_masks, actions) = e.get_samples()
    mv2.extend(mcts_values); gr2.extend(game_results)
mv2 = np.array(mv2); gr2 = np.array(gr2)
print(f"[DiagValue] 启发式 corr={np.corrcoef(mv2,gr2)[0,1]:.3f}")
print(f"[DiagValue] 启发式 胜局 value={mv2[gr2>0].mean():.3f} 负局={mv2[gr2<0].mean():.3f} 区分度={mv2[gr2>0].mean()-mv2[gr2<0].mean():.3f}")
