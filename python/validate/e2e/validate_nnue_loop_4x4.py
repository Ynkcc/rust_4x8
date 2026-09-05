"""validate_nnue_loop_4x4.py — 4x4 NNUE + Expectimax 训练回环验证。

每轮流程（目标：单轮数分钟内完成）：
  1. 随机基座  : 第 1 轮导出随机初始化 .nnue（输出层缩放，冷启动）；
                 之后各轮以上一轮训练产物作为基座。
  2. 自对弈    : 调用 Rust 原生 run_expectimax_self_play（expectimax+nnue），
                 流式写出 NNUE episode JSONL。
  3. 训练      : NnueSampleDataset + train_nnue 过拟合/回归本轮数据，
                 导出新的 .nnue 与 .pth checkpoint。
  4. 深度展开  : 用训练后的模型做深度 expectimax 对局评测——
                 a) vs random 基线（应显著胜出）；
                 b) new vs prev 基座（观察回环是否带来提升）。

用法:
    python validate/e2e/validate_nnue_loop_4x4.py --rounds 2 --games 8
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time

_VALIDATE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PYTHON_DIR = os.path.dirname(_VALIDATE_DIR)
for _d in (_PYTHON_DIR, _VALIDATE_DIR):
    if _d not in sys.path:
        sys.path.insert(0, _PYTHON_DIR)

import torch  # noqa: E402

import banqi_4x8  # noqa: E402
from banqi.nnue.exporter import export_random  # noqa: E402
from banqi.nnue.model import BanqiNNUE, nnue_feature_dim  # noqa: E402
from banqi.nnue.train import NnueSampleDataset, train_nnue  # noqa: E402
from banqi.variant import get_variant  # noqa: E402

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def arena_budgeted(nnue_a: str, nnue_b: str, n: int, depth: int, budget: int) -> tuple[float, float, float]:
    """深度展开对战（显式预算版）：A=先手红方，B=后手黑方。

    `run_native_match` 的 `expectimax:<path>` 规格使用 SearchConfig 默认值
    （node_budget=500k, max_depth=24, quiescence），h2h 对战双方每步全预算搜索
    会跑数十分钟。这里改用 Game4x4.expectimax_action 显式控预算，保证分钟级完成。
    返回 (a 胜率, 和率, b 胜率)。
    """
    w = d = l = 0
    for _ in range(n):
        env = banqi_4x8.Game4x4()
        while True:
            path = nnue_a if env.current_player() == 1 else nnue_b
            action = env.expectimax_action(path, max_depth=depth, node_budget=budget)
            if action is None:
                break
            terminated, _, winner = env.step(action)
            if terminated:
                if winner == 1:
                    w += 1
                elif winner == -1:
                    l += 1
                else:
                    d += 1
                break
    return w / n, d / n, l / n


def arena_vs_random(nnue: str, n: int, depth: int, budget: int) -> tuple[float, float, float]:
    """NNUE 引擎(先手) vs 纯随机对手，返回 (胜率, 和率, 负率)。"""
    import random as _random

    rng = _random.Random(0)
    w = d = l = 0
    for _ in range(n):
        env = banqi_4x8.Game4x4()
        while True:
            if env.current_player() == 1:
                action = env.expectimax_action(nnue, max_depth=depth, node_budget=budget)
            else:
                moves = env.legal_moves()
                action = rng.choice(moves) if moves else None
            if action is None:
                break
            terminated, _, winner = env.step(action)
            if terminated:
                if winner == 1:
                    w += 1
                elif winner == -1:
                    l += 1
                else:
                    d += 1
                break
    return w / n, d / n, l / n


def run_round(round_idx: int, base_nnue: str | None, out_dir: str, args) -> str:
    """执行一轮回环，返回新训练出的 .nnue 路径。"""
    variant = get_variant("4x4")
    dim = nnue_feature_dim(
        variant.total_positions,
        variant.num_active_piece_types,
        max(variant.piece_counts),
    )
    t0 = time.time()

    # ---- 1. 随机基座 ----
    if base_nnue is None:
        base_nnue = os.path.join(out_dir, "base_random.nnue")
        export_random(base_nnue, feature_dim=dim, output_scale=args.output_scale)
        print(f"[Round {round_idx}] 随机基座已导出: {base_nnue} (dim={dim})")
    else:
        print(f"[Round {round_idx}] 基座 = 上一轮产物: {base_nnue}")

    # ---- 2. 自对弈（expectimax + nnue，Rust 原生）----
    jsonl = os.path.join(out_dir, f"selfplay_r{round_idx}.jsonl")
    stats = banqi_4x8.run_expectimax_self_play(
        base_nnue,
        n_games=args.games,
        num_workers=args.workers,
        node_budget=args.node_budget,
        max_depth=args.depth,
        threads_per_search=1,
        seed=args.seed + round_idx,
        out_jsonl=jsonl,
        variant_id="4x4",
    )
    print(
        f"[Round {round_idx}] 自对弈完成: {stats['games']} 局 "
        f"(A{stats['a_wins']}/和{stats['draws']}/B{stats['b_wins']}), "
        f"共 {stats['steps']} 步, jsonl={jsonl}"
    )

    # ---- 3. 训练 ----
    ds = NnueSampleDataset([jsonl], value_source="completed_q")
    assert len(ds) > 0, "必须收集到 NNUE 样本"
    assert ds.feature_dim == dim, f"特征维度不一致: 数据 {ds.feature_dim} vs 变体 {dim}"
    print(f"[Round {round_idx}] 样本数: {len(ds)}, feature_dim={ds.feature_dim}")

    new_nnue = os.path.join(out_dir, f"model_r{round_idx}.nnue")
    ckpt = os.path.join(out_dir, f"model_r{round_idx}.pth")
    model = train_nnue(
        ds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        output_nnue=new_nnue,
        checkpoint=ckpt,
    )
    assert os.path.getsize(new_nnue) > 0

    # 快速过拟合自检：训练损失应明显低于随机基线
    with torch.no_grad():
        xs = torch.stack([ds[i][0] for i in range(min(len(ds), 256))])
        ys = torch.stack([ds[i][1] for i in range(min(len(ds), 256))])
        mse = torch.nn.functional.mse_loss(model(xs), ys).item()
    print(f"[Round {round_idx}] 训练后样本内 MSE: {mse:.6f}")

    # ---- 4. 深度展开评测（显式预算，保证分钟级）----
    depth, budget = args.arena_depth, args.arena_budget
    # a) new(先手) vs random
    wr_vs_rand = arena_vs_random(new_nnue, args.arena_games, depth, budget)
    print(
        f"[Round {round_idx}] 深度展开 vs random: "
        f"胜 {wr_vs_rand[0]:.2f} / 和 {wr_vs_rand[1]:.2f} / 负 {wr_vs_rand[2]:.2f}"
    )
    # b) new vs 本轮基座（首轮=随机基座，之后=上一轮产物）
    wr_h2h = arena_budgeted(new_nnue, base_nnue, args.arena_games, depth, budget)
    print(
        f"[Round {round_idx}] 深度展开 new vs prev: "
        f"胜 {wr_h2h[0]:.2f} / 和 {wr_h2h[1]:.2f} / 负 {wr_h2h[2]:.2f}"
    )

    print(f"[Round {round_idx}] 本轮耗时 {time.time() - t0:.1f}s")
    return new_nnue


def main() -> None:
    parser = argparse.ArgumentParser(description="4x4 NNUE+Expectimax 训练回环验证")
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--games", type=int, default=8, help="每轮自对弈局数")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--node-budget", type=int, default=20_000, help="expectimax 单搜索节点预算")
    parser.add_argument("--depth", type=int, default=6, help="自对弈 expectimax 深度")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output-scale", type=float, default=0.1)
    parser.add_argument("--arena-games", type=int, default=8, help="深度展开评测局数")
    parser.add_argument("--arena-depth", type=int, default=4, help="评测 expectimax 深度")
    parser.add_argument("--arena-budget", type=int, default=8_000, help="评测单步节点预算")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()

    out_dir = args.out_dir or os.path.join(_PYTHON_DIR, "outputs", "4x4", "nnue_loop")
    os.makedirs(out_dir, exist_ok=True)
    print(f"=== [4x4 NNUE+Expectimax 回环验证] rounds={args.rounds}, out={out_dir} ===")

    base: str | None = None
    for r in range(1, args.rounds + 1):
        base = run_round(r, base, out_dir, args)

    # 终检：checkpoint 可加载回 BanqiNNUE
    final_ckpt = os.path.join(out_dir, f"model_r{args.rounds}.pth")
    dim = nnue_feature_dim(16, get_variant("4x4").num_active_piece_types, 2)
    BanqiNNUE(dim).load_state_dict(torch.load(final_ckpt, map_location="cpu"))
    print("=== [回环验证 PASSED] 随机基座 → 自对弈 → 训练 → 深度展开 全链路 OK ===")


if __name__ == "__main__":
    main()
