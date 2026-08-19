"""benchmark/schemes.py — 各运行方案的调度与多进程执行。"""

from __future__ import annotations

import os
import queue
import time
from typing import Any, Dict

import numpy as np

try:
    import banqi_4x8
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "无法导入 banqi_4x8。请先执行: maturin develop --features pyo3"
    ) from exc

from banqi.config import make_config
from banqi.variant import Variant, get_variant

from .predictors import CountingPredictor
from .results import BenchResult

# 变体 -> 统一入口 run_python_match 的 variant_id。
# 旧的 serial/parallel/batched 多入口已彻底移除，统一走 run_python_match
# （Python 推理单线程）。scheme 标签仅保留用于结果标记。
_VARIANT_MAP: Dict[str, str] = {
    "": "4x8",        # 4x8
    "mini": "4x2",    # 4x2
    "game4x4": "4x4", # 4x4
}


def build_self_play_config(variant: Variant) -> "banqi_4x8.SelfPlayConfig":
    """构造 SelfPlayConfig（c_scale/gumbel_scale 支持 env 覆盖）。"""
    cfg = make_config(variant.id)
    c_scale = float(os.getenv("C_SCALE", "1.0"))
    gumbel_scale = float(os.getenv("GUMBEL_SCALE", "1.0"))
    return banqi_4x8.SelfPlayConfig(
        mcts_sims=cfg.MCTS_SIMS,
        max_considered_actions=cfg.MAX_CONSIDERED_ACTIONS,
        temperature_steps=cfg.TEMPERATURE_STEPS,
        c_scale=c_scale,
        gumbel_scale=gumbel_scale,
    )


def _run_scheme(variant_id: str, scheme: str, predictor: Any,
                games: int, concurrency: int, worker_id: int = 0) -> BenchResult:
    """对单个 (变种, 方案) 运行基准；返回 BenchResult（已完成对局/步数/样本数/调用次数）。"""
    variant = get_variant(variant_id)
    cfg = make_config(variant_id)
    sp_cfg = build_self_play_config(variant)
    vid = _VARIANT_MAP[variant.rust_prefix]

    if scheme in ("serial", "parallel", "batched"):
        t0 = time.time()
        # 统一入口 run_python_match（单线程）；concurrency 参数忽略（保持兼容）。
        episodes = banqi_4x8.run_python_match(
            predict_fn=predictor, config=sp_cfg, num_games=games,
            concurrency=concurrency, worker_id=worker_id, variant_id=vid,
        )
        duration = time.time() - t0

        completed_games = len(episodes)
        total_steps = sum(int(ep.game_length) for ep in episodes)
        total_samples = sum(int(ep.num_samples) for ep in episodes)
        return BenchResult(
            variant_id=variant_id, scheme=scheme, device=str(cfg.INFER_DEVICE),
            duration_s=duration, completed_games=completed_games,
            total_steps=total_steps, total_samples=total_samples,
            predictor_calls=predictor.calls, predictor_samples=predictor.samples,
        )
    elif scheme == "multiproc":
        return _benchmark_multiproc(variant_id, predictor, games, concurrency)
    else:
        raise ValueError(f"未知方案: {scheme}")


def _benchmark_multiproc(variant_id: str, predictor: Any, games: int,
                         concurrency: int) -> BenchResult:
    """多进程方案：spawn N 个自对弈进程并行生成，聚合计数（需独立进程，无法直接复用 predictor 统计）。"""
    import multiprocessing as mp

    variant = get_variant(variant_id)
    cfg = make_config(variant_id)
    sp_cfg = build_self_play_config(variant)
    vid = _VARIANT_MAP[variant.rust_prefix]

    n_proc = max(1, concurrency)
    games_per_proc = max(1, -(-games // n_proc))
    result_q: "mp.Queue" = mp.Queue()
    procs = [
        mp.Process(
            target=_mp_bench_worker_main,
            args=(variant_id, vid, sp_cfg, games_per_proc, cfg.BATCH_CONCURRENCY,
                  i, result_q),
            name=f"BenchMP-{i}",
        )
        for i in range(n_proc)
    ]
    for p in procs:
        p.start()
    t0 = time.time()
    agg = {"games": 0, "steps": 0, "samples": 0, "calls": 0, "predictor_samples": 0}
    for _ in procs:
        r = result_q.get()
        agg["games"] += r["games"]
        agg["steps"] += r["steps"]
        agg["samples"] += r["samples"]
        agg["calls"] += r["calls"]
        agg["predictor_samples"] += r["predictor_samples"]
    for p in procs:
        p.join()
    result_q.close()
    result_q.cancel_join_thread()
    duration = time.time() - t0
    return BenchResult(
        variant_id=variant_id, scheme="multiproc", device=str(cfg.INFER_DEVICE),
        duration_s=duration, completed_games=agg["games"],
        total_steps=agg["steps"], total_samples=agg["samples"],
        predictor_calls=agg["calls"], predictor_samples=agg["predictor_samples"],
    )


def _mp_bench_worker_main(variant_id: str, vid: str, sp_cfg, games: int,
                          concurrency: int, worker_id: int, result_q: "queue.Queue"):
    """多进程 worker：独立 predictor（真实模型或模拟），统计本进程计数。"""
    from .predictors import SimulatedPredictor
    from banqi.constants import build_constants

    variant = get_variant(variant_id)
    action_space = build_constants(variant).ACTION_SPACE_SIZE
    if os.getenv("BENCH_SIMULATED") == "1":
        predictor = SimulatedPredictor(action_space)
    else:
        from .runner import _build_predictor
        predictor, _ = _build_predictor(variant, make_config(variant_id).MODEL_PATH,
                                        make_config(variant_id).INFER_DEVICE)
    counting = CountingPredictor(predictor)
    episodes = banqi_4x8.run_python_match(
        predict_fn=counting, config=sp_cfg, num_games=games,
        concurrency=concurrency, worker_id=worker_id, variant_id=vid,
    )
    result_q.put({
        "games": len(episodes),
        "steps": sum(int(ep.game_length) for ep in episodes),
        "samples": sum(int(ep.num_samples) for ep in episodes),
        "calls": counting.calls,
        "predictor_samples": counting.samples,
    })
