"""banqi/eval.py — 统一评估接口（全下沉至 Rust 原生并发引擎）

支持显式指定对战双方（Player A vs Player B，任意组合：模型/规则/MCTS/Minimax/随机模型），
支持固定随机种子 (RNG Seed) 控制开局洗牌与决策。
"""

from __future__ import annotations

import argparse
import os
import tempfile
from typing import List, Optional, Tuple, Union

import numpy as np
import torch

try:
    import banqi_4x8
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "无法导入 banqi_4x8。请先执行: maturin develop --features pyo3,torch,onnx"
    ) from exc

# 评估常量
EVAL_SIMS = 64
EVAL_MAX_ACTIONS = 16
EVAL_C_SCALE = 0.25
EVAL_GUMBEL_SCALE = 1.0
HM_SIMS = 128
MINIMAX_DEPTH = 3

# 规则对手预设
OPP_HEURISTIC128 = "heuristic128"
OPP_MINIMAX3 = "minimax3"
OPP_HEURISTIC64 = "heuristic64"
OPPONENTS = (OPP_HEURISTIC128, OPP_MINIMAX3, OPP_HEURISTIC64)


def _resolve_player_spec(
    spec_or_path: Union[str, torch.nn.Module, object],
    variant_id: str = "4x4",
    seed: Optional[int] = None,
) -> str:
    """解析选手标识为 Rust 可读的格式（.pt / .onnx / 规则标识符）。"""
    if isinstance(spec_or_path, str):
        path = spec_or_path
        if path == "random" or path.startswith("random:"):
            # 随机初始化模型：受 seed 确定性驱动
            if ":" in path:
                r_seed = int(path.split(":")[1])
            else:
                r_seed = seed if seed is not None else 42
            torch.manual_seed(r_seed)
            from banqi.variant import get_variant
            from banqi.nn_model import BanqiNet
            from banqi.checkpoint import export_torchscript

            v = get_variant(variant_id)
            model = BanqiNet(v).to("cpu").eval()
            temp_dir = tempfile.gettempdir()
            pt_path = os.path.join(temp_dir, f"banqi_random_{variant_id}_seed{r_seed}.pt")
            export_torchscript(model, pt_path, v, torch.device("cpu"))
            return pt_path

        if path.endswith(".ckpt") or path.endswith(".pth"):
            pt_path = os.path.splitext(path)[0] + ".pt"
            if not os.path.exists(pt_path) or os.path.getmtime(path) > os.path.getmtime(pt_path):
                from banqi.tools.export_ckpt import export_checkpoint_file
                export_checkpoint_file(path, variant_id)
            return pt_path
        return path

    if hasattr(spec_or_path, "model"):
        return _resolve_player_spec(getattr(spec_or_path, "model"), variant_id, seed)

    if isinstance(spec_or_path, torch.nn.Module):
        from banqi.variant import get_variant
        from banqi.checkpoint import export_torchscript
        temp_dir = tempfile.gettempdir()
        v = get_variant(variant_id)
        r_seed = seed if seed is not None else 42
        pt_path = os.path.join(temp_dir, f"banqi_eval_temp_{variant_id}_seed{r_seed}.pt")
        export_torchscript(spec_or_path, pt_path, v, torch.device("cpu"))
        return pt_path

    raise TypeError(f"无法识别的选手标识格式: {type(spec_or_path)}")


def play_match(
    player_a,
    player_b,
    n: int = 100,
    model_sims: int = EVAL_SIMS,
    variant_id: str = "4x4",
    heuristic_sims: Optional[int] = None,
    seed: Optional[int] = None,
    num_threads: int = 4,
) -> Tuple[int, int, int, List[float]]:
    """双选手对战评估（调用 Rust 侧原生并发引擎）。"""
    spec_a = _resolve_player_spec(player_a, variant_id, seed)
    spec_b = _resolve_player_spec(player_b, variant_id, seed)

    wins, draws, losses, block_wr, _avg_moves = banqi_4x8.run_eval_match(
        player_a=spec_a,
        player_b=spec_b,
        n=n,
        variant_id=variant_id,
        model_sims=model_sims,
        heuristic_sims=heuristic_sims,
        seed=seed,
        num_threads=num_threads,
    )
    return wins, draws, losses, block_wr


def play_match_stats(
    player_a,
    player_b,
    n: int = 100,
    model_sims: int = EVAL_SIMS,
    variant_id: str = "4x4",
    heuristic_sims: Optional[int] = None,
    seed: Optional[int] = None,
    num_threads: int = 4,
) -> Tuple[int, int, int, float]:
    """双选手对战评估并统计平均步数。"""
    spec_a = _resolve_player_spec(player_a, variant_id, seed)
    spec_b = _resolve_player_spec(player_b, variant_id, seed)

    wins, draws, losses, _block_wr, avg_moves = banqi_4x8.run_eval_match(
        player_a=spec_a,
        player_b=spec_b,
        n=n,
        variant_id=variant_id,
        model_sims=model_sims,
        heuristic_sims=heuristic_sims,
        seed=seed,
        num_threads=num_threads,
    )
    return wins, draws, losses, avg_moves


def report(
    player_a,
    player_b,
    tag: str = "main",
    n: int = 100,
    model_sims: int = EVAL_SIMS,
    variant_id: str = "4x4",
    heuristic_sims: Optional[int] = None,
    seed: Optional[int] = None,
    num_threads: int = 4,
) -> Tuple[int, int, int, List[float]]:
    """统一打印评估报告。"""
    wins, draws, losses, blk = play_match(
        player_a,
        player_b,
        n=n,
        model_sims=model_sims,
        variant_id=variant_id,
        heuristic_sims=heuristic_sims,
        seed=seed,
        num_threads=num_threads,
    )
    mean = float(np.mean(blk)) if blk else 0.0
    std = float(np.std(blk)) if blk else 0.0
    print(
        f"[Eval:{tag}] [{player_a}] vs [{player_b}] | 胜{wins} 平{draws} 负{losses} "
        f"(n={n}, 块均胜率={mean:.1f}±{std:.1f}%, seed={seed})",
        flush=True,
    )
    return wins, draws, losses, blk


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
