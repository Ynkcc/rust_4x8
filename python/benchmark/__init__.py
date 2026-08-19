"""benchmark — 暗棋自对弈吞吐基准子包。

拆分自原单文件 benchmark_production.py，按职责分层：
  results.py     : BenchResult 结果容器与格式化打印
  predictors.py  : CountingPredictor / SimulatedPredictor（统计 / 模拟推理）
  schemes.py     : 各运行方案调度（serial/parallel/batched/multiproc）
  runner.py      : benchmark_cell / run_all / print_summary
  cli.py         : argparse 与 main 入口

benchmark_production.py 保留为向后兼容的 re-export 入口。
"""

from .results import BenchResult
from .predictors import CountingPredictor, SimulatedPredictor
from .schemes import (
    _VARIANT_MAP,
    build_self_play_config,
    _run_scheme,
    _benchmark_multiproc,
    _mp_bench_worker_main,
)
from .runner import (
    benchmark_cell,
    run_all,
    print_summary,
    _resolve_device,
    _resolve_model_path,
    _build_predictor,
)
from .cli import parse_args, main

__all__ = [
    "BenchResult",
    "CountingPredictor",
    "SimulatedPredictor",
    "build_self_play_config",
    "_run_scheme",
    "_benchmark_multiproc",
    "_mp_bench_worker_main",
    "benchmark_cell",
    "run_all",
    "print_summary",
    "_resolve_device",
    "_resolve_model_path",
    "_build_predictor",
    "parse_args",
    "main",
]
