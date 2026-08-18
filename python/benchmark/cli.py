"""benchmark/cli.py — 基准命令行入口。"""

from __future__ import annotations

import argparse
from typing import List

from .runner import run_all, print_summary


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python benchmark_production.py",
        description="暗棋自对弈吞吐基准（串行/并行/批量/多进程对比）",
    )
    p.add_argument("--variants", nargs="+", default=["4x8", "4x4mini", "4x4"],
                   help="要基准的变种 id 列表")
    p.add_argument("--schemes", nargs="+", default=["serial", "parallel", "batched", "multiproc"],
                   choices=["serial", "parallel", "batched", "multiproc"],
                   help="运行方案列表")
    p.add_argument("--games", type=int, default=64, help="每种方案的对局数")
    p.add_argument("--concurrency", type=int, default=4, help="并行/batch/多进程的并发度")
    p.add_argument("--model-path", default=None, help="模型权重路径（默认用 config.MODEL_PATH）")
    p.add_argument("--device", default=None, help="推理设备（默认 auto: CUDA>CPU）")
    p.add_argument("--simulated", action="store_true",
                   help="用模拟推理延迟（不加载真实模型，快速验证流程）")
    return p.parse_args(argv)


def main(argv: List[str] | None = None) -> None:
    import time
    args = parse_args(argv)
    print(f"🔧 基准配置: variants={args.variants}, schemes={args.schemes}, "
          f"games={args.games}, concurrency={args.concurrency}, "
          f"device={args.device or 'auto'}, simulated={args.simulated}")
    t0 = time.time()
    results = run_all(
        variant_ids=args.variants,
        schemes=args.schemes,
        model_path=args.model_path,
        device=args.device,
        games=args.games,
        concurrency=args.concurrency,
        simulated=args.simulated,
    )
    print_summary(results, args.schemes, args.variants)
    print(f"⏱️ 全部基准耗时: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
