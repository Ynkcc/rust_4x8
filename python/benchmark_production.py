"""
benchmark_production.py — 向后兼容入口。

原单文件实现已拆分为 benchmark/ 子包（results.py / predictors.py / schemes.py /
runner.py / cli.py）。本文件仅做 re-export 并保留直接运行能力，避免改动现有调用方。
新增代码请直接 `from benchmark import main`。
"""

from benchmark import (
    BenchResult,
    CountingPredictor,
    SimulatedPredictor,
    benchmark_cell,
    run_all,
    print_summary,
    parse_args,
    main,
)

__all__ = [
    "BenchResult",
    "CountingPredictor",
    "SimulatedPredictor",
    "benchmark_cell",
    "run_all",
    "print_summary",
    "parse_args",
    "main",
]


if __name__ == "__main__":
    main()
