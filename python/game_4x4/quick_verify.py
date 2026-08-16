"""
quick_verify.py — 5 分钟快速验证：高 sims 启发式教师模仿学习能否快速提升棋力

流程：
  1. 评估当前 checkpoint 棋力（baseline）
  2. 生成 ~90 局 sims=256 高 sims 教师数据（3 线程并行）
  3. 训练 3 epochs
  4. 再次评估，对比提升
"""
from __future__ import annotations

import os, sys, time
import numpy as np
import torch

torch.set_num_threads(int(os.getenv("G4X4_TORCH_THREADS", "2")))

import banqi_4x8

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import config
from nn_model import Banqi4x4Net, load_model_weights
from training_service import DataBuffer, train_step, save_checkpoint, _resolve_device

DEVICE = _resolve_device(config.TRAIN_DEVICE)


def obs_of(env):
    b, s = env.observation()
    return (np.asarray(b, dtype=np.float32).reshape(1, 16, 4, 4),
            np.asarray(s, dtype=np.float32).reshape(1, -1))


def eval_strength(model, tag, n=6):
    from verify_vs_heuristic_mcts import ModelPredictor
    pred = ModelPredictor(model, DEVICE)

    def greedy(env):
        logits, _ = pred(*obs_of(env))
        legal = env.legal_moves()
        return max(legal, key=lambda a: logits[0][a])

    def mcts64(env):
        return env.mcts_search_action(pred, 64, 16, 0.25)

    def rand_fn(env):
        legal = env.legal_moves()
        return legal[np.random.randint(len(legal))] if legal else None

    def mm2(env):
        return env.minimax_action(2)

    def heur64(env):
        return env.heuristic_mcts_action(64)

    def play(red, black):
        env = banqi_4x8.Game4x4()
        moves = 0
        while not env.terminated():
            if env.winner() is not None:
                break
            a = (red if env.current_player() == 1 else black)(env)
            if a is None:
                break
            env.step(a)
            moves += 1
            if moves > 300:
                break
        w = env.winner()
        return 1 if w == 1 else (-1 if w == -1 else 0)

    def match(red, black, name):
        w = [play(red, black) for _ in range(n)]
        wr = sum(x == 1 for x in w); d = sum(x == 0 for x in w)
        print(f"  [{tag}] {name}: 胜{wr} 平{d} 负{len(w)-wr-d}", flush=True)

    match(greedy, rand_fn, "greedy   vs 随机")
    match(mcts64, rand_fn, "MCTS64   vs 随机")
    match(mcts64, mm2, "MCTS64   vs minimax2")
    match(mcts64, heur64, "MCTS64   vs 启发式64")


def gen_teacher(sims, games, conc=12, threads=3):
    from concurrent.futures import ThreadPoolExecutor, as_completed
    per = max(1, games // threads)

    def _one(_):
        cfg = banqi_4x8.SelfPlayConfig(
            mcts_sims=sims, max_considered_actions=16,
            temperature_steps=12)
        return banqi_4x8.run_game4x4_heuristic_self_play(
            config=cfg, num_games=per, concurrency=conc, worker_id=0)

    t0 = time.time()
    eps = []
    with ThreadPoolExecutor(max_workers=threads) as ex:
        futs = [ex.submit(_one, i) for i in range(threads)]
        for f in as_completed(futs):
            eps.extend(f.result())
    print(f"  [Gen] sims={sims} 教师 {len(eps)} 局 耗时 {time.time()-t0:.1f}s", flush=True)
    return eps


def main():
    t_total = time.time()
    model = Banqi4x4Net().to(DEVICE)
    load_model_weights(model, config.MODEL_PATH, DEVICE)
    print(f"[QV] 加载 checkpoint: {config.MODEL_PATH}", flush=True)

    # 1. baseline
    model.eval()
    print("[QV] === Baseline 棋力 ===", flush=True)
    eval_strength(model, "base", n=6)

    # 2. 生成高 sims 教师数据
    eps = gen_teacher(sims=256, games=90)

    # 3. 训练
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    buf = DataBuffer(6000)
    n = 0
    for e in eps:
        (boards, scalars, policies, mcts_values, completed_qs,
         root_visits, game_results, action_masks, actions) = e.get_samples()
        for board, scalar, policy, mv, gr, mask in zip(
                boards, scalars, policies, mcts_values, game_results, action_masks):
            buf.add_samples([{
                "board_state": board, "scalar_state": scalar,
                "policy_probs": policy, "mcts_value": float(mv),
                "game_result_value": float(gr), "root_visit_count": 0,
                "action_mask": mask,
            }])
            n += 1
    print(f"[QV] 训练样本 {n} (Buffer={len(buf)})", flush=True)
    for epc in range(3):
        idx = list(range(len(buf)))
        np.random.shuffle(idx)
        nb = len(idx) // config.TRAIN_BATCH
        tl = pl = vl = 0.0
        for s in range(nb):
            batch = buf.get_batch(idx[s*config.TRAIN_BATCH:(s+1)*config.TRAIN_BATCH])
            t, p, v = train_step(model, opt, batch, DEVICE)
            tl += t; pl += p; vl += v
        print(f"  epoch{epc}: loss={tl/max(1,nb):.4f} pol={pl/max(1,nb):.4f} val={vl/max(1,nb):.4f}", flush=True)

    # 4. 再评估
    model.eval()
    print("[QV] === 高sims教师模仿后 棋力 ===", flush=True)
    eval_strength(model, "after", n=6)

    save_checkpoint(model, opt, None)
    print(f"[QV] 总耗时 {time.time()-t_total:.0f}s", flush=True)


if __name__ == "__main__":
    main()
