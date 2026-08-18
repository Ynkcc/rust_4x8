"""benchmark/runner.py — 基准调度核心与结果汇总。"""

from __future__ import annotations

import os
import time
from typing import List, Optional

import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:  # pragma: no cover
    HAS_TORCH = False

import banqi_4x8
from banqi.config import make_config
from banqi.variant import Variant

from .predictors import CountingPredictor
from .results import BenchResult
from .schemes import _run_scheme

# 默认用于基准的 MCTS 模拟次数（覆盖 config.MCTS_SIMS，使基准更具代表性的真实负载）
BENCH_MCTS_SIMS = int(os.getenv("BENCH_MCTS_SIMS", "200"))


def _resolve_device(device: Optional[str]) -> str:
    """解析推理设备：None/auto 时 CUDA 可用则用 cuda，否则 cpu。"""
    if device in (None, "auto", ""):
        if HAS_TORCH and torch.cuda.is_available():
            return "cuda"
        return "cpu"
    return device


def _resolve_model_path(model_path: Optional[str]) -> Optional[str]:
    """解析模型路径：None 时用 config 的 MODEL_PATH（可能不存在 -> 退化初始化）。"""
    if model_path is not None:
        if os.path.exists(model_path):
            return model_path
        print(f"⚠️ 指定模型路径不存在: {model_path}，使用退化初始化网络")
        return None
    return make_config("4x8").MODEL_PATH if os.path.exists(make_config("4x8").MODEL_PATH) else None


def _build_predictor(variant: Variant, model_path: Optional[str], device: str):
    """构建真实推理 Predictor（来自 banqi.selfplay）。"""
    from banqi.selfplay import build_predictor as _bp
    return _bp(variant, model_path, device)


def benchmark_cell(variant_id: str, scheme: str,
                   model_path: Optional[str] = None,
                   device: Optional[str] = None,
                   games: int = 64,
                   concurrency: int = 1,
                   simulated: bool = False) -> BenchResult:
    """基准单格 (变种 × 方案)，返回 BenchResult。

    scheme: "serial" | "parallel" | "batched" | "multiproc"
    若 simulated=True，不加载真实模型，用模拟推理延迟（需 BENCH_SIMULATED=1 供多进程子进程读取）。
    """
    variant = __import__("banqi.variant", fromlist=["get_variant"]).get_variant(variant_id)
    cfg = make_config(variant_id)
    device = _resolve_device(device)

    if simulated:
        os.environ["BENCH_SIMULATED"] = "1"
        from .predictors import SimulatedPredictor
        from banqi.constants import build_constants
        action_space = build_constants(variant).ACTION_SPACE_SIZE
        raw = SimulatedPredictor(action_space)
        # 临时覆盖 MCTS 模拟数（模拟负载更接近真实），通过 env 注入 config 已读取，故直接改 cfg 拷贝无效；
        # 这里改为直接修改 sp_cfg 的 mcts_sims 在 schemes.build_self_play_config 内读取 config，
        # 故需经 env 传递：
        os.environ[f"{variant.env_prefix}MCTS_SIMS"] = str(BENCH_MCTS_SIMS)
        predictor = CountingPredictor(raw)
    else:
        os.environ.pop("BENCH_SIMULATED", None)
        raw, _ = _build_predictor(variant, _resolve_model_path(model_path), device)
        predictor = CountingPredictor(raw)
        # 真实基准同样用代表性模拟数（必要时覆盖）
        os.environ[f"{variant.env_prefix}MCTS_SIMS"] = str(BENCH_MCTS_SIMS)

    return _run_scheme(variant_id, scheme, predictor, games, concurrency)


def run_all(variant_ids: List[str], schemes: List[str],
            model_path: Optional[str] = None,
            device: Optional[str] = None,
            games: int = 64,
            concurrency: int = 1,
            simulated: bool = False) -> List[BenchResult]:
    """运行所有 (变种 × 方案) 组合，返回结果列表（不打印，归总交给 print_summary）。"""
    results: List[BenchResult] = []
    for vid in variant_ids:
        for scheme in schemes:
            try:
                results.append(
                    benchmark_cell(vid, scheme, model_path=model_path, device=device,
                                   games=games, concurrency=concurrency, simulated=simulated)
                )
            except Exception as exc:  # pragma: no cover
                print(f"❌ 基准失败 (variant={vid}, scheme={scheme}): {exc}")
    return results


def print_summary(results: List[BenchResult],
                  schemes: List[str], variant_ids: List[str]) -> None:
    """打印 (变种 × 方案) 吞吐对比表。"""
    from .results import _SCHEME_LABELS

    bar = "=" * 92
    print(f"\n{bar}")
    print("  📊 自对弈吞吐基准汇总")
    print(f"{bar}")
    header = f"  {'变种':<8} | " + " | ".join(
        f"{_SCHEME_LABELS.get(s, s):>22}" for s in schemes
    )
    print(header)
    print(f"  {'-' * (8 + 3 + 25 * len(schemes))}")

    # 取各组合的样本/s，缺失填 '-'
    lookup = {(r.variant_id, r.scheme): r for r in results}
    for vid in variant_ids:
        row = f"  {vid:<8} | "
        cells = []
        for s in schemes:
            r = lookup.get((vid, s))
            cells.append(f"{r.samples_per_second:>20.1f} sps" if r else f"{'—':>22}")
        row += " | ".join(cells)
        print(row)

    print(f"\n  legend: sps = samples/秒（越高越好）")
    print(f"{bar}")
    # 各结果独立详细打印
    for r in results:
        r.print()
