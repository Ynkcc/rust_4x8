"""
train_imitate.py — 启发式教师模仿学习预热（改进版）

用高 sims 启发式 Gumbel MCTS 教师生成高质量训练目标，训练网络模仿强教师。
针对上一版"过拟合近期分布导致退化"的修复：
  1. Buffer 严格 FIFO 上限（cap_samples），新样本挤出最旧样本，保持分布相对稳定。
  2. value 目标用教师 mcts_value（平滑评估），避免 ±1 最终结果的噪声。
  3. 每轮只训练 1-2 epoch（不反复碾压同分布数据）。
  4. 逐步提升教师 sims（sims_schedule），让网络跟随越来越强的教师。

用法：
    python train_imitate.py --games 200 --rounds 8 --sims 512 --epochs 2
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import torch

torch.set_num_threads(int(os.getenv("G4X4_TORCH_THREADS", "2")))

import banqi_4x8

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import config
from nn_model import Banqi4x4Net, load_model_weights
from training_service import train_step, save_checkpoint, _resolve_device
from storage import (
    save_episodes_to_archive, load_jsonl_episodes, episode_dict_to_samples,
    list_jsonl_files,
)
from tb_logger import add_scalar

DEVICE = _resolve_device(config.TRAIN_DEVICE)


class FIFOBuffer:
    """带严格上限、先进先出的样本缓冲。value 用 mcts_value（教师平滑评估）。"""

    def __init__(self, cap):
        self.cap = cap
        self.boards, self.scalars, self.probs, self.values, self.masks = (
            deque(), deque(), deque(), deque(), deque())

    def add(self, samples):
        for s in samples:
            self.boards.append(np.array(s['board_state'], dtype=np.float32).reshape(16, 4, 4))
            self.scalars.append(np.array(s['scalar_state'], dtype=np.float32))
            self.probs.append(np.array(s['policy_probs'], dtype=np.float32))
            self.values.append(float(s['mcts_value']))  # 教师搜索平滑评估
            self.masks.append(np.array(s['action_mask'], dtype=np.float32))
        # 超出上限则挤出最旧（保持分布稳定，避免过拟合最近一批）
        while len(self.boards) > self.cap:
            self.boards.popleft(); self.scalars.popleft()
            self.probs.popleft(); self.values.popleft(); self.masks.popleft()

    def __len__(self):
        return len(self.boards)

    def sample_batch(self, batch_size):
        idx = np.random.choice(len(self.boards), batch_size, replace=False)
        b = torch.from_numpy(np.stack([self.boards[i] for i in idx]))
        s = torch.from_numpy(np.stack([self.scalars[i] for i in idx]))
        p = torch.from_numpy(np.stack([self.probs[i] for i in idx]))
        v = torch.tensor([self.values[i] for i in idx], dtype=torch.float32)
        m = torch.from_numpy(np.stack([self.masks[i] for i in idx]))
        return b, s, p, v, m


def gen_teacher(sims, games, conc=12, threads=3):
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


def episodes_to_samples(eps):
    samples = []
    for e in eps:
        (boards, scalars, policies, mcts_values, completed_qs,
         root_visits, game_results, action_masks, actions) = e.get_samples()
        for board, scalar, policy, mv, mask in zip(
                boards, scalars, policies, mcts_values, action_masks):
            samples.append({
                "board_state": board, "scalar_state": scalar,
                "policy_probs": policy, "mcts_value": float(mv),
                "action_mask": mask,
            })
    return samples


def episodes_to_dicts(eps, tag: str):
    """把 episode 转成可归档的 dict 列表（to_dict 序列化）。"""
    dicts = []
    for e in eps:
        d = dict(e.to_dict())
        d["_tag"] = tag
        dicts.append(d)
    return dicts


def eval_strength(model, tag, n=8):
    from verify_vs_heuristic_mcts import ModelPredictor
    pred = ModelPredictor(model, DEVICE)

    def obs_of(env):
        b, s = env.observation()
        return (np.asarray(b, dtype=np.float32).reshape(1, 16, 4, 4),
                np.asarray(s, dtype=np.float32).reshape(1, -1))

    def greedy(env):
        logits, _ = pred(*obs_of(env))
        return max(env.legal_moves(), key=lambda a: logits[0][a])

    def mcts64(env):
        return env.mcts_search_action(pred, 64, 16, 1.0, 0.25)

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
            env.step(a); moves += 1
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=2, help="每轮训练 epoch（<=2 防过拟合）")
    ap.add_argument("--sims", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--cap", type=int, default=6000, help="buffer 上限")
    ap.add_argument("--gen-threads", type=int, default=3)
    ap.add_argument("--conc", type=int, default=12)
    ap.add_argument("--eval-games", type=int, default=8)
    ap.add_argument("--sims-schedule", action="store_true", help="逐步提升教师 sims")
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--archive-dir", type=str, default="./training_data/archive_4x4_imitate",
                    help="冷存储目录（生成数据归档保存，可复用）")
    ap.add_argument("--load-archive", type=int, default=0,
                    help="启动时从冷存储加载最近 N 局复用（0=不加载）")
    ap.add_argument("--save-archive", action="store_true",
                    help="每轮生成后归档到冷存储")
    args = ap.parse_args()

    # 统一初始化：先加载冷存储样本（如有），再初始化模型/优化器/buffer
    load_samples = []
    if args.load_archive > 0:
        t0 = time.time()
        eps = load_jsonl_episodes(args.archive_dir, limit_games=args.load_archive)
        print(f"[Imitation] 从冷存储加载 {len(eps)} 局复用 "
              f"(耗时 {time.time()-t0:.1f}s)", flush=True)
        for ep in eps:
            load_samples.extend(episode_dict_to_samples(ep))
        print(f"  → 冷存储样本 {len(load_samples)}", flush=True)

    model = Banqi4x4Net().to(DEVICE)
    if args.fresh:
        print("[Imitation] 随机初始化", flush=True)
    else:
        load_model_weights(model, config.MODEL_PATH, DEVICE)
        print(f"[Imitation] 加载 {config.MODEL_PATH}", flush=True)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    buf = FIFOBuffer(args.cap)
    if load_samples:
        buf.add(load_samples)
        print(f"  → Buffer 初始 {len(buf)} 样本（含冷存储复用）", flush=True)

    sims_list = ([args.sims] * args.rounds) if not args.sims_schedule else \
        [64, 128, 256, 384, 512, 512, 512, 512][:args.rounds]

    print(f"[Imitation] rounds={args.rounds} games={args.games} epochs={args.epochs} "
          f"sims={args.sims} cap={args.cap}", flush=True)
    total_t0 = time.time()

    for r in range(args.rounds):
        sims = sims_list[r]
        eps = gen_teacher(sims, args.games, args.conc, args.gen_threads)
        samples = episodes_to_samples(eps)
        buf.add(samples)
        print(f"  Round#{r}: 新样本 {len(samples)} → Buffer={len(buf)}", flush=True)

        # 冷存储归档（可复用）：异步批次写入，不阻塞训练
        if args.save_archive:
            dicts = episodes_to_dicts(eps, tag=f"r{r}_sims{sims}")
            t0a = time.time()
            n_arch = save_episodes_to_archive(
                dicts, args.archive_dir, iteration=r, worker_id=sims)
            print(f"  → 归档 {n_arch} 局到 {args.archive_dir} "
                  f"(耗时 {time.time()-t0a:.1f}s)", flush=True)

        # 每轮训练固定 epochs，随机采样（不遍历全量，保持分布混合）
        for epc in range(args.epochs):
            steps = min(600, max(100, len(buf) // config.TRAIN_BATCH))
            tl = pl = vl = 0.0
            for _ in range(steps):
                batch = buf.sample_batch(config.TRAIN_BATCH)
                t, p, v = train_step(model, opt, batch, DEVICE)
                tl += t; pl += p; vl += v
            print(f"    epoch{epc}: loss={tl/steps:.4f} pol={pl/steps:.4f} val={vl/steps:.4f}", flush=True)
        add_scalar("imitation/loss_policy", pl / steps)
        add_scalar("imitation/loss_value", vl / steps)

        model.eval()
        eval_strength(model, f"R{r}", args.eval_games)
        model.train()
        save_checkpoint(model, opt, None)
        print(f"  Round#{r} 完成，累计 {time.time()-total_t0:.0f}s", flush=True)

    print(f"[Imitation] 总耗时 {time.time()-total_t0:.0f}s", flush=True)
    save_checkpoint(model, opt, None)


if __name__ == "__main__":
    main()
