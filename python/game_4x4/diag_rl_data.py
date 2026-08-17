"""诊断：RL 自对弈数据 vs 教师自对弈数据的质量对比。

对比指标：对局长度/结果分布、policy 目标熵、mcts_value 与结果相关性。
若 RL 自对弈对局混乱（平局多、价值相关低），说明 RL 数据本身信号弱。
"""
import os, sys, time
import numpy as np
import torch
torch.set_num_threads(2)
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
sys.path.append(os.path.dirname(_HERE))

import banqi_4x8
from config import config
from banqi.variant import get_variant
from banqi.nn_model import BanqiNet, load_model_weights
from verify_vs_heuristic_mcts import ModelPredictor

VARIANT = get_variant("4x4")
device = torch.device("cpu")
model = BanqiNet(VARIANT).to(device).eval()
load_model_weights(model, config.MODEL_PATH, device)
pred = ModelPredictor(model, device)

def analyze(eps, tag):
    results = {"红胜": 0, "黑胜": 0, "平局": 0}
    ents, maxp, lens = [], [], []
    mv, gr = [], []
    for e in eps:
        if e.winner == 1: results["红胜"] += 1
        elif e.winner == -1: results["黑胜"] += 1
        else: results["平局"] += 1
        lens.append(e.game_length)
        (boards, scalars, policies, mcts_values, completed_qs,
         root_visits, game_results, action_masks, actions) = e.get_samples()
        for p, mask, m, g in zip(policies, action_masks, mcts_values, game_results):
            mask = np.array(mask); p = np.array(p)
            v = p[mask == 1]; v = v / (v.sum() + 1e-9)
            ents.append(-np.sum(v * np.log(v + 1e-9)))
            maxp.append(v.max())
            mv.append(m); gr.append(g)
    mv = np.array(mv); gr = np.array(gr)
    print(f"  [{tag}] {results} len={np.mean(lens):.1f} 策略熵={np.mean(ents):.2f} "
          f"maxp={np.mean(maxp):.2f} value_corr={np.corrcoef(mv,gr)[0,1]:.3f} "
          f"区分度={mv[gr>0].mean()-mv[gr<0].mean():.3f}", flush=True)

# 1. RL 自对弈（模型）
cfg = banqi_4x8.SelfPlayConfig(mcts_sims=256, max_considered_actions=16, temperature_steps=12)
t0 = time.time()
eps = banqi_4x8.run_game4x4_parallel_self_play_with_predictor(
    predict_fn=pred, config=cfg, num_workers=4, games_per_worker=5, worker_id=0)
print(f"[Diag] 模型自对弈 {len(eps)} 局 {time.time()-t0:.0f}s", flush=True)
analyze(eps, "模型自对弈")

# 2. 教师自对弈（启发式 512）
cfg2 = banqi_4x8.SelfPlayConfig(mcts_sims=512, max_considered_actions=16, temperature_steps=12)
t0 = time.time()
eps2 = banqi_4x8.run_game4x4_heuristic_self_play(config=cfg2, num_games=20, concurrency=4, worker_id=0)
print(f"[Diag] 教师自对弈 {len(eps2)} 局 {time.time()-t0:.0f}s", flush=True)
analyze(eps2, "教师自对弈")
