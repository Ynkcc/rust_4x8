"""
benchmark_throughput.py — 吞吐量基准测试

场景：模拟 predictor (predictor(batch=128, 每批模拟 ~ 0.01s）
统计 10 秒内能完成多少局自对弈游戏；对比 baseline vs 优化后。

用法：
    # 1) 先 maturin develop --features pyo3
    python python/benchmark_throughput.py  # 默认 baseline: 串行
    python python/benchmark_throughput.py --parallel   # 使用 Rust 并行版 (如果可用
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import threading
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

# 确保可以跳过真实模型，只用纯随机数据返回
import banqi_4x8

from constant import (
    ACTION_SPACE_SIZE,
    BOARD_CHANNELS,
    BOARD_COLS,
    BOARD_ROWS,
    SCALAR_FEATURE_COUNT,
)


# ============================================================================
# 模拟 Predictor: 每批 ≥1 个样本，模拟 0.01s 延迟（并按batch 128 的粒度来凑整）
# ============================================================================

class SimulatedPredictor:
    """
    纯随机返回 policy (均匀logits) + value (0)，模拟：
    - 每当调用 predict(boards, scalars)
    - batch_size 向上取整到 128 的块数 × 0.01s，模拟推理延迟

    （相当于：1 块（<=128样本）0.01s；129~256 样本 0.02s，以此类推）
    """

    BATCH_SIZE = 128
    PER_BATCH_SECONDS = 0.01

    def __init__(self, name: str = "sim") -> None:
        self.name = name
        self.call_count = 0
        self.total_samples = 0
        self.total_slept = 0.0
        self._lock = threading.Lock()

    def __call__(self, board: np.ndarray, scalars: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        batch = board.shape[0]
        n_blocks = (batch + self.BATCH_SIZE - 1) // self.BATCH_SIZE
        sleep_s = n_blocks * self.PER_BATCH_SECONDS

        time.sleep(sleep_s)

        rng = np.random.default_rng()
        policy_logits = rng.standard_normal((batch, ACTION_SPACE_SIZE)).astype(np.float32)
        values = rng.uniform(-1.0, 1.0, size=batch).astype(np.float32)

        with self._lock:
            self.call_count += 1
            self.total_samples += batch
            self.total_slept += sleep_s

        return policy_logits, values

    def reset(self) -> None:
        self.call_count = 0
        self.total_samples = 0
        self.total_slept = 0.0

    def stats(self) -> Dict[str, float]:
        return {
            "calls": self.call_count,
            "samples": self.total_samples,
            "slept_s": self.total_slept,
            "avg_batch": (self.total_samples / max(1, self.call_count)),
        }


# ============================================================================
# 基准测试 Harness
# ============================================================================

@dataclass
class BenchResult:
    label: str
    duration_s: float
    completed_games: int = 0
    total_steps: int = 0
    total_samples_processed: int = 0
    predictor_calls: int = 0
    predictor_samples: int = 0
    predictor_slept_s: float = 0.0
    errors: List[str] = field(default_factory=list)

    @property
    def games_per_second(self) -> float:
        return self.completed_games / max(1e-9, self.duration_s)

    @property
    def steps_per_second(self) -> float:
        return self.total_steps / max(1e-9, self.duration_s)

    def print(self) -> None:
        bar = "=" * 78
        print(f"\n{bar}")
        print(f"  🏁 基准结果: {self.label}  (时长 = {self.duration_s:.2f}s")
        print(f"{bar}")
        print(f"  完成局数           : {self.completed_games} 局")
        print(f"  吞吐量 (games/s)   : {self.games_per_second:.3f} 局/秒")
        print(f"  总步数             : {self.total_steps}")
        print(f"  吞吐量 (steps/s)   : {self.steps_per_second:.1f} 步/秒")
        print(f"  -Predictor 调用次数   : {self.predictor_calls} 次")
        print(f"  Predictor 样本总数  : {self.predictor_samples:,} 个")
        print(f"  Predictor 模拟IO等待 : {self.predictor_slept_s:.3f} s")
        if self.errors:
            c = Counter(self.errors)
            print(f"  ⚠️  错误分布:")
            for k, v in c.most_common():
                print(f"      × {v}  {k}")
        print(f"{bar}\n")


def _run_serial(
    predictor: SimulatedPredictor,
    cfg: banqi_4x8.SelfPlayConfig,
    num_games_per_round: int,
    deadline_s: float,
) -> BenchResult:
    label = f"串行 (baseline) — games_per_round={num_games_per_round}"
    result = BenchResult(label=label, duration_s=deadline_s)

    t0 = time.perf_counter()
    round_idx = 0
    while True:
        # 串行执行：直接调用 run_self_play_with_predictor
        try:
            episodes: List[banqi_4x8.GameEpisode] = banqi_4x8.run_self_play_with_predictor(
                predict_fn=predictor,
                config=cfg,
                num_games=num_games_per_round,
                worker_id=0,
            )
        except Exception as exc:  # pragma: no cover
            result.errors.append(str(exc)[:80])
            # 避免无限重试
            if len(result.errors) > 10:
                break
            continue

        for ep in episodes:
            result.completed_games += 1
            result.total_steps += ep.game_length

        round_idx += 1
        elapsed = time.perf_counter() - t0
        if elapsed >= deadline_s:
            break

    result.duration_s = time.perf_counter() - t0
    stats = predictor.stats()
    result.predictor_calls = int(stats["calls"])
    result.predictor_samples = int(stats["samples"])
    result.predictor_slept_s = float(stats["slept_s"])
    return result


def _run_parallel(
    predictor: SimulatedPredictor,
    cfg: banqi_4x8.SelfPlayConfig,
    num_workers: int,
    games_per_worker: int,
    deadline_s: float,
) -> BenchResult:
    label = f"并行 (Rust rayon) — workers={num_workers}, games_per_worker={games_per_worker}"
    result = BenchResult(label=label, duration_s=deadline_s)

    t0 = time.perf_counter()
    round_idx = 0

    # 如果 Rust 端暴露了并行 API 就用；否则退化为串行多线程（Python 并发跑多个串行版
    if hasattr(banqi_4x8, "run_parallel_self_play_with_predictor"):
        runner = banqi_4x8.run_parallel_self_play_with_predictor
    else:
        runner = None

    while True:
        try:
            if runner is not None:
                episodes = runner(
                    predict_fn=predictor,
                    config=cfg,
                    num_workers=num_workers,
                    games_per_worker=games_per_worker,
                    worker_id=0,
                )
            else:
                # 退化方案： Python 端开多线程，每个线程跑 N 局（因为 Python GIL，但 predictor sleep
                # ，但在 time.sleep 是释放 GIL 的，所以实际是并发等待可以重叠）
                episodes: List[banqi_4x8.GameEpisode] = []
                threads = []
                lock = threading.Lock()

                def _worker(wid: int) -> None:
                    try:
                        eps = banqi_4x8.run_self_play_with_predictor(
                            predict_fn=predictor,
                            config=cfg,
                            num_games=games_per_worker,
                            worker_id=wid,
                        )
                        with lock:
                            episodes.extend(eps)
                    except Exception as exc:
                        with lock:
                            result.errors.append(str(exc)[:80])

                for wid in range(num_workers):
                    t = threading.Thread(target=_worker, args=(wid,), daemon=True)
                    threads.append(t)
                    t.start()
                for t in threads:
                    t.join()

        except Exception as exc:  # pragma: no cover
            result.errors.append(str(exc)[:80])
            if len(result.errors) > 10:
                break
            continue

        for ep in episodes:
            result.completed_games += 1
            result.total_steps += ep.game_length

        round_idx += 1
        elapsed = time.perf_counter() - t0
        if elapsed >= deadline_s:
            break

    result.duration_s = time.perf_counter() - t0
    stats = predictor.stats()
    result.predictor_calls = int(stats["calls"])
    result.predictor_samples = int(stats["samples"])
    result.predictor_slept_s = float(stats["slept_s"])
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=10.0,
                        help="基准测试时长(秒)，默认 10s")
    parser.add_argument("--mcts-sims", type=int, default=32,
                        help="MCTS sims (越小越快，默认 32 节省时间)")
    parser.add_argument("--games-per-call", type=int, default=2,
                        help="串行版每次调用 run_self_play 的 num_games")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="并行版 workers (默认 4)")
    parser.add_argument("--games-per-worker", type=int, default=1,
                        help="并行版 每 worker games 数")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="模拟的批大小(默认 128")
    parser.add_argument("--per-batch-sleep", type=float, default=0.01,
                        help="每批模拟延迟秒 (默认 0.01s)")
    parser.add_argument("--parallel-only", action="store_true",
                        help="只跑并行版")
    parser.add_argument("--serial-only", action="store_true",
                        help="只跑串行版")
    args = parser.parse_args()

    SimulatedPredictor.BATCH_SIZE = args.batch_size
    SimulatedPredictor.PER_BATCH_SECONDS = args.per_batch_sleep

    print(f"\n⚙️  配置: sim batch={args.batch_size}, per_batch_sleep={args.per_batch_sleep}s")
    print(f"    mcts_sims={args.mcts_sims}, 时长={args.duration}s")

    cfg = banqi_4x8.SelfPlayConfig(
        mcts_sims=args.mcts_sims,
        max_considered_actions=16,
        dirichlet_alpha=0.3,
        dirichlet_epsilon=0.25,
        temperature_steps=12,
    )

    results: List[BenchResult] = []

    # ---- Baseline: 串行
    if not args.parallel_only:
        print(f"\n▶️  运行 Baseline (串行)...")
        predictor_serial = SimulatedPredictor("serial")
        res_serial = _run_serial(predictor_serial, cfg, args.games_per_call, args.duration)
        res_serial.print()
        results.append(res_serial)

    # ---- 并行
    if not args.serial_only:
        print(f"\n▶️  运行 并行版 (workers={args.num_workers})...")
        predictor_par = SimulatedPredictor("parallel")
        res_par = _run_parallel(
            predictor_par, cfg,
            num_workers=args.num_workers,
            games_per_worker=args.games_per_worker,
            deadline_s=args.duration,
        )
        res_par.print()
        results.append(res_par)

    # ---- 对比
    if len(results) == 2:
        base, opt = results[0], results[1]
        speedup = opt.games_per_second / max(1e-9, base.games_per_second)
        bar = "=" * 78
        print(f"\n{bar}")
        print(f"  📊 吞吐量对比 (s → 并行)")
        print(f"{bar}")
        print(f"  串行基线 games/s : {base.games_per_second:.3f}")
        print(f"  优化版本 games/s : {opt.games_per_second:.3f}")
        print(f"  加速比         : {speedup:.2f}×")
        print(f"  串行总步数/s    : {base.steps_per_second:.1f}")
        print(f"  优化总步数/s    : {opt.steps_per_second:.1f}")
        print(f"{bar}\n")


if __name__ == "__main__":
    main()
