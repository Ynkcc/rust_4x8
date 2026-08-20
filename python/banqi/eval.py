"""banqi/eval.py — 统一评估接口（薄封装，核心实现已下沉到 banqi/training/eval.py）。

本模块仅做别名导出 + CLI 入口，评估核心工具（常量、选手解析、双选手对战评估、
评估报告）统一收敛在 `banqi/training/eval.py`，消除跨文件重复代码。新增评估逻辑
请直接维护 `banqi/training/eval.py`。
"""

from __future__ import annotations

import argparse

from banqi.training.eval import (
    EVAL_SIMS,
    EVAL_MAX_ACTIONS,
    EVAL_C_SCALE,
    EVAL_GUMBEL_SCALE,
    HM_SIMS,
    MINIMAX_DEPTH,
    OPP_HEURISTIC128,
    OPP_MINIMAX3,
    OPP_HEURISTIC64,
    OPPONENTS,
    _resolve_player_spec,
    play_match,
    play_match_stats,
    report,
)

__all__ = [
    "EVAL_SIMS",
    "EVAL_MAX_ACTIONS",
    "EVAL_C_SCALE",
    "EVAL_GUMBEL_SCALE",
    "HM_SIMS",
    "MINIMAX_DEPTH",
    "OPP_HEURISTIC128",
    "OPP_MINIMAX3",
    "OPP_HEURISTIC64",
    "OPPONENTS",
    "_resolve_player_spec",
    "play_match",
    "play_match_stats",
    "report",
]


def main():
    ap = argparse.ArgumentParser(description="暗棋双选手统一评估（Rust 原生下沉引擎）")
    ap.add_argument("player_a", help="选手 A (格式: random / mcts128 / minimax3 / .pt路径)")
    ap.add_argument("player_b", help="选手 B (格式: random / mcts128 / minimax3 / .pt路径)")
    ap.add_argument("n", nargs="?", type=int, default=100, help="评估局数（默认 100）")
    ap.add_argument("--variant", default="4x4", choices=("4x2", "4x4", "4x8"), help="棋盘变体（默认 4x4）")
    ap.add_argument("--seed", type=int, default=None, help="固定随机种子 (RNG Seed)")
    ap.add_argument("--heuristic-sims", type=int, default=None, help="启发式 MCTS 对手的模拟数")
    ap.add_argument("--model-sims", type=int, default=EVAL_SIMS, help=f"模型 MCTS 模拟数（默认 {EVAL_SIMS}）")
    ap.add_argument("-j", "--num-threads", type=int, default=4, help="Rust 侧并发线程数（默认 4）")

    args = ap.parse_args()

    report(
        args.player_a,
        args.player_b,
        tag="main",
        n=args.n,
        variant_id=args.variant,
        model_sims=args.model_sims,
        heuristic_sims=args.heuristic_sims,
        seed=args.seed,
        num_threads=args.num_threads,
    )


if __name__ == "__main__":
    main()
