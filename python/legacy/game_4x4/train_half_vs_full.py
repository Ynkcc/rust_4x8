"""
train_half_vs_full.py — 对比实验：启发式 MCTS 数据「后一半」vs「全部」训练

流程：
  1. 从 training_data/archive_4x4_imitate/*.jsonl 加载启发式 Gumbel MCTS 生成的
     训练数据（每行一个 episode dict，含 boards/scalars/policies/mcts_values/
     game_results/action_masks 等）。
  2. 两个 BanqiNet 用「相同随机种子」初始化（起始权重完全一致）：
       - Model A (half)：仅使用每局「后一半」样本训练
       - Model B (full)：使用每局「全部」样本训练
  3. 完全相同的训练超参（Adam / lr / epochs / batch / 顺序）。
  4. 两模型对战（默认模型 Gumbel MCTS，可切 greedy），交替先手，输出胜率。

用法：
    python python/game_4x4/train_half_vs_full.py
    # 常用参数：
    #   --epochs 8 --batch 64 --lr 1e-3
    #   --games 40 --sims 32 --play-mode mcts
    #   --seed 42
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from typing import Dict, List, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# 确保可导入同目录模块（config/nn_model/training_service）
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np
import torch

torch.set_num_threads(int(os.getenv("G4X4_TORCH_THREADS", "2")))

import banqi_4x8

from config import config
from banqi.variant import get_variant
from banqi.nn_model import BanqiNet
from training_service import train_step  # 复用标准训练步骤（pol loss + val loss）

VARIANT = get_variant("4x4")

DEVICE = torch.device("cpu")


# =============================================================================
# 数据加载
# =============================================================================

def load_episodes(data_dir: str) -> List[Dict]:
    """加载归档 JSONL 中全部 episode dict（启发式 MCTS 生成，样本按时间序）。"""
    files = sorted(glob.glob(os.path.join(data_dir, "*.jsonl")))
    if not files:
        raise FileNotFoundError(f"未在 {data_dir} 找到 *.jsonl 训练数据")
    episodes: List[Dict] = []
    for f in files:
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                episodes.append(json.loads(line))
    print(f"[Data] 加载 {len(files)} 个文件 → {len(episodes)} 局")
    return episodes


def episode_to_samples(ep: Dict) -> List[Dict]:
    """episode dict → 样本列表（保留每步的 board/scalar/policy/value/mask）。"""
    n = len(ep["boards"])
    samples = []
    for i in range(n):
        samples.append({
            "board": np.asarray(ep["boards"][i], dtype=np.float32).reshape(16, 4, 4),
            "scalar": np.asarray(ep["scalars"][i], dtype=np.float32),
            "policy": np.asarray(ep["policies"][i], dtype=np.float32),
            "mcts_value": float(ep["mcts_values"][i]),
            "game_result": float(ep["game_results"][i]),
            "mask": np.asarray(ep["action_masks"][i], dtype=np.float32),
        })
    return samples


def build_datasets(episodes: List[Dict], value_key: str) -> Tuple[List[Dict], List[Dict]]:
    """按每局切分：返回 (half_samples, full_samples)。

    half：每局仅取时间上后一半的样本（samples[len//2:]）。
    full：每局全部样本。
    价值目标统一取 mcts_value 或 game_result。
    """
    half_all: List[Dict] = []
    full_all: List[Dict] = []
    for ep in episodes:
        samples = episode_to_samples(ep)
        n = len(samples)
        for s in samples:
            s["value"] = s[value_key]
        full_all.extend(samples)
        if n > 0:
            half_all.extend(samples[n // 2:])
    print(f"[Data] half 样本 = {len(half_all)}（每局后一半）| full 样本 = {len(full_all)}")
    return half_all, full_all


# =============================================================================
# 训练
# =============================================================================

def train_one(model: "BanqiNet", samples: List[Dict], epochs: int, batch: int,
              lr: float, seed: int, tag: str) -> None:
    """在给定样本集上训练模型（Adam + 标准 train_step）。"""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    indices = np.arange(len(samples))
    t0 = time.time()
    for epoch in range(epochs):
        rng = np.random.RandomState(seed * 1000 + epoch)
        rng.shuffle(indices)
        nb = len(indices) // batch
        if nb == 0:
            print(f"  [{tag}] 样本不足一个 batch，跳过")
            break
        tl = pl = vl = 0.0
        for s in range(nb):
            idx = indices[s * batch:(s + 1) * batch]
            b = torch.from_numpy(np.stack([samples[i]["board"] for i in idx]))
            sc = torch.from_numpy(np.stack([samples[i]["scalar"] for i in idx]))
            p = torch.from_numpy(np.stack([samples[i]["policy"] for i in idx]))
            v = torch.tensor([samples[i]["value"] for i in idx], dtype=torch.float32)
            m = torch.from_numpy(np.stack([samples[i]["mask"] for i in idx]))
            t, pl_, vl_ = train_step(model, optimizer, (b, sc, p, v, m), DEVICE)
            tl += t; pl += pl_; vl += vl_
        tl /= nb; pl /= nb; vl /= nb
        print(f"  [{tag}] epoch{epoch}: loss={tl:.4f} pol={pl:.4f} val={vl:.4f}", flush=True)
    print(f"  [{tag}] 训练完成，{epochs} epochs 耗时 {time.time()-t0:.1f}s", flush=True)


# =============================================================================
# 对战
# =============================================================================

class ModelPredictor:
    def __init__(self, model: "BanqiNet"):
        self.model = model.to(DEVICE).eval()

    def __call__(self, boards: np.ndarray, scalars: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        with torch.inference_mode():
            b = torch.from_numpy(np.ascontiguousarray(boards)).to(DEVICE)
            s = torch.from_numpy(np.ascontiguousarray(scalars)).to(DEVICE)
            logits, value = self.model(b, s)
            return logits.cpu().numpy().astype(np.float32), value.cpu().numpy().reshape(-1).astype(np.float32)


def play_one_game(pred_a: ModelPredictor, pred_b: ModelPredictor, a_is_red: bool,
                  sims: int, play_mode: str, max_actions: int = 16) -> int:
    """模型 A vs 模型 B 一局。返回 +1 A胜 / 0 平 / -1 B胜。"""
    env = banqi_4x8.Game4x4()
    moves = 0
    while not env.terminated():
        if env.winner() is not None:
            break
        is_red_turn = env.current_player() == 1
        a_turn = (is_red_turn == a_is_red)
        pred = pred_a if a_turn else pred_b
        if play_mode == "greedy":
            b, s = env.observation()
            board = np.asarray(b, dtype=np.float32).reshape(1, 16, 4, 4)
            scalars = np.asarray(s, dtype=np.float32).reshape(1, -1)
            logits, _ = pred(board, scalars)
            legal = env.legal_moves()
            action = max(legal, key=lambda a: logits[0][a]) if legal else None
        else:
            action = env.mcts_search_action(pred, sims, max_actions, 0.25)
        if action is None:
            break
        env.step(action)
        moves += 1
        if moves > 400:
            break
    winner = env.winner()  # 1=红胜, -1=黑胜, 0/None=平
    if winner == 0 or winner is None:
        return 0
    if a_is_red:
        return 1 if winner == 1 else -1
    return 1 if winner == -1 else -1


def run_match(pred_a: ModelPredictor, pred_b: ModelPredictor, games: int,
              sims: int, play_mode: str) -> None:
    """交替先手对战，输出胜率。"""
    wins = draws = losses = 0
    t0 = time.time()
    for i in range(games):
        a_is_red = (i % 2 == 0)
        w = play_one_game(pred_a, pred_b, a_is_red, sims, play_mode)
        if w == 1:
            wins += 1
        elif w == 0:
            draws += 1
        else:
            losses += 1
        if (i + 1) % 10 == 0 or i + 1 == games:
            print(f"  [{i+1}/{games}] A 胜率={wins/(i+1):.3f} "
                  f"(A胜{wins} 平{draws} B胜{losses}) 累计{time.time()-t0:.0f}s", flush=True)
    print("\n" + "=" * 60)
    print(f"  对战结果（{play_mode}，sims={sims}，{games} 局，交替先手）")
    print("=" * 60)
    print(f"  A(后一半) 胜：{wins}（{wins/games:.1%}）")
    print(f"  平局：       {draws}（{draws/games:.1%}）")
    print(f"  B(全部) 胜： {losses}（{losses/games:.1%}）")
    print("=" * 60)
    print(f"  A(后一半) 胜率 = {wins/games:.3f}")
    print(f"  B(全部)   胜率 = {losses/games:.3f}")


# =============================================================================
# main
# =============================================================================

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=str,
                    default=os.path.join(_HERE, "training_data", "archive_4x4_imitate"))
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--games", type=int, default=40, help="对战局数")
    ap.add_argument("--sims", type=int, default=32, help="对战模型 MCTS 模拟数")
    ap.add_argument("--play-mode", choices=["mcts", "greedy"], default="mcts")
    ap.add_argument("--value-key", choices=["mcts_value", "game_result"], default="mcts_value",
                    help="训练价值目标（mcts_value=启发式搜索平滑值，game_result=最终±1）")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print("=" * 64)
    print("  实验：启发式 MCTS 数据「后一半」vs「全部」训练")
    print("=" * 64)
    print(f"  数据目录   = {args.data_dir}")
    print(f"  训练       = epochs={args.epochs} batch={args.batch} lr={args.lr} "
          f"value={args.value_key}")
    print(f"  对战       = {args.play_mode} sims={args.sims} {args.games} 局")
    print("=" * 64)

    # ---- 1. 加载数据 ----
    t0 = time.time()
    episodes = load_episodes(args.data_dir)
    half_samples, full_samples = build_datasets(episodes, args.value_key)
    print(f"[Data] 加载耗时 {time.time()-t0:.1f}s")

    # ---- 2. 同种子初始化两个模型（起始权重完全一致） ----
    torch.manual_seed(args.seed)
    model_a = BanqiNet(VARIANT).to(DEVICE)   # 后一半
    model_b = BanqiNet(VARIANT).to(DEVICE)   # 全部
    model_b.load_state_dict(model_a.state_dict())
    print(f"[Init] 两个模型使用相同随机种子 seed={args.seed} 初始化（权重一致）")

    # ---- 3. 分别训练 ----
    print("\n=== 训练 Model A（仅每局后一半）===")
    train_one(model_a, half_samples, args.epochs, args.batch, args.lr, args.seed, "A-half")
    print("\n=== 训练 Model B（每局全部）===")
    train_one(model_b, full_samples, args.epochs, args.batch, args.lr, args.seed, "B-full")

    # ---- 4. 对战 ----
    print("\n=== 对战：A(后一半) vs B(全部) ===")
    run_match(ModelPredictor(model_a), ModelPredictor(model_b),
              args.games, args.sims, args.play_mode)
    return 0


if __name__ == "__main__":
    sys.exit(main())
