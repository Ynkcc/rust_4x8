"""
benchmark_batched.py — 批量自对弈吞吐基准测试（无 CLI 参数）

对比 baseline（串行单树自对弈，小 batch 推理）与新的批量自对弈
（run_batched_self_play_with_predictor，多局合并成大 batch 推理）。

用法：
    # 先编译绑定：
    #   maturin develop --features pyo3 --release
    # 或先安装到当前 Python 环境后：
    python python/benchmark_batched.py

要点：
    - 复用 benchmark_throughput.py 的 SimulatedPredictor 思想：模拟 GPU 推理
      有「固定每批开销 + 每样本开销」，因此 batch 越大、每样本均摊成本越低。
    - 通过统计相同模拟时长/局数下谁更快，来对比「小 batch 串行」vs「大 batch 并行」。
    - 所有配置复用 config.py / self_play.build_self_play_config()。
"""

from __future__ import annotations

import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

import banqi_4x8

from config import config
from constant import ACTION_SPACE_SIZE
from self_play import build_self_play_config


# ============================================================================
# 模拟 Predictor：模拟 GPU 推理延迟
# ============================================================================
# 真实 GPU 推理延迟 ≈ 固定启动开销 + 每样本线性成本。
# 我们用：
#   delay(batch) = FIXED_OVERHEAD + batch * PER_SAMPLE
# 小 batch 时固定开销占比高、每样本均摊成本大；大 batch 显著摊薄固定开销。
# ============================================================================

class SimulatedPredictor:
    # 调小模拟推理成本，让 MCTS 的 CPU 搜索（走树/展开/回溯）成为主导，
    # 从而能在 top 里观察到 CPU 利用率明显上升。想进一步压榨 CPU 可继续调小。
    FIXED_OVERHEAD_S = 0.02   # 原 0.02：每次推理调用的固定开销（kernel 启动 / CPU→GPU 拷贝）
    PER_SAMPLE_S = 0.0004      # 原 0.0004：每个样本的线性推理成本

    def __init__(self, name: str = "sim") -> None:
        self.name = name
        self.call_count = 0
        self.total_samples = 0
        self.total_slept = 0.0
        self._lock = threading.Lock()

    def __call__(self, board: np.ndarray, scalars: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        batch = board.shape[0]
        delay = self.FIXED_OVERHEAD_S + batch * self.PER_SAMPLE_S
        time.sleep(delay)

        rng = np.random.default_rng()
        policy_logits = rng.standard_normal((batch, ACTION_SPACE_SIZE)).astype(np.float32)
        values = rng.uniform(-1.0, 1.0, size=batch).astype(np.float32)

        with self._lock:
            self.call_count += 1
            self.total_samples += batch
            self.total_slept += delay
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
            "avg_batch": self.total_samples / max(1, self.call_count),
        }


# ============================================================================
# 结果容器
# ============================================================================

@dataclass
class BenchResult:
    label: str
    duration_s: float
    completed_games: int = 0
    total_steps: int = 0
    predictor_calls: int = 0
    predictor_samples: int = 0
    predictor_slept_s: float = 0.0

    @property
    def games_per_second(self) -> float:
        return self.completed_games / max(1e-9, self.duration_s)

    @property
    def steps_per_second(self) -> float:
        return self.total_steps / max(1e-9, self.duration_s)

    def print(self) -> None:
        bar = "=" * 80
        print(f"\n{bar}")
        print(f"  🏁 基准结果: {self.label}  (时长 = {self.duration_s:.2f}s)")
        print(f"{bar}")
        print(f"  完成局数           : {self.completed_games} 局")
        print(f"  吞吐量 (games/s)   : {self.games_per_second:.3f} 局/秒")
        print(f"  总步数             : {self.total_steps}")
        print(f"  吞吐量 (steps/s)   : {self.steps_per_second:.1f} 步/秒")
        print(f"  Predictor 调用次数 : {self.predictor_calls} 次")
        print(f"  Predictor 样本总数 : {self.predictor_samples:,} 个")
        print(f"  Predictor 平均 batch: {self.predictor_samples / max(1, self.predictor_calls):.1f}")
        print(f"  Predictor 模拟IO耗时: {self.predictor_slept_s:.3f} s")
        print(f"{bar}\n")


# ============================================================================
# 运行器
#
# 为获得稳定、可复现的对比，统一采用「跑固定局数、计时」的方式：
#   各方案都生成同样数量的对局，然后比较总耗时 / 吞吐 / CPU 利用率。
# ============================================================================

def _run_serial(predictor: SimulatedPredictor, cfg, num_games: int) -> BenchResult:
    label = f"baseline 串行 (单树, 小 batch)"
    result = BenchResult(label=label, duration_s=0.0)
    t0 = time.perf_counter()
    episodes = banqi_4x8.run_self_play_with_predictor(
        predict_fn=predictor, config=cfg, num_games=num_games, worker_id=0,
    )
    result.duration_s = time.perf_counter() - t0
    for ep in episodes:
        result.completed_games += 1
        result.total_steps += ep.game_length
    s = predictor.stats()
    result.predictor_calls = int(s["calls"])
    result.predictor_samples = int(s["samples"])
    result.predictor_slept_s = float(s["slept_s"])
    return result


def _run_parallel(predictor: SimulatedPredictor, cfg, num_workers: int, num_games: int) -> BenchResult:
    label = f"baseline 并行 rayon (workers={num_workers})"
    result = BenchResult(label=label, duration_s=0.0)
    # games_per_worker 向上取整，保证总对局 >= num_games
    games_per_worker = -(-num_games // num_workers)
    t0 = time.perf_counter()
    episodes = banqi_4x8.run_parallel_self_play_with_predictor(
        predict_fn=predictor, config=cfg,
        num_workers=num_workers, games_per_worker=games_per_worker, worker_id=0,
    )
    result.duration_s = time.perf_counter() - t0
    for ep in episodes:
        result.completed_games += 1
        result.total_steps += ep.game_length
    s = predictor.stats()
    result.predictor_calls = int(s["calls"])
    result.predictor_samples = int(s["samples"])
    result.predictor_slept_s = float(s["slept_s"])
    return result


def _run_batched(predictor: SimulatedPredictor, cfg, num_games: int, concurrency: int) -> BenchResult:
    label = f"batched (并发 {concurrency} 局, 合并大 batch)"
    result = BenchResult(label=label, duration_s=0.0)
    t0 = time.perf_counter()
    episodes = banqi_4x8.run_batched_self_play_with_predictor(
        predict_fn=predictor, config=cfg,
        num_games=num_games, concurrency=concurrency, worker_id=0,
    )
    result.duration_s = time.perf_counter() - t0
    for ep in episodes:
        result.completed_games += 1
        result.total_steps += ep.game_length
    s = predictor.stats()
    result.predictor_calls = int(s["calls"])
    result.predictor_samples = int(s["samples"])
    result.predictor_slept_s = float(s["slept_s"])
    return result


# ============================================================================
# 主流程
# ============================================================================

def override_config() -> None:
    """覆盖基线参数。

    为观察 CPU 利用率随搜索预算上升，这里把 MCTS_SIMS / MAX_CONSIDERED_ACTIONS
    调大，放大 MCTS 的 CPU 搜索量（走树/展开/回溯）。配合 SimulatedPredictor
    把模拟推理成本调小，使 CPU 搜索成为主导，CPU 利用率会明显上升。
    调参旋钮：继续调大 MCTS_SIMS / MAX_CONSIDERED_ACTIONS，或调小
    FIXED_OVERHEAD_S / PER_SAMPLE_S。
    """
    config.MCTS_SIMS = 256
    config.MAX_CONSIDERED_ACTIONS = 32
    config.TEMPERATURE_STEPS = 8
    config.NUM_WORKERS = 2


def main() -> None:
    override_config()

    num_games = 4                 # 每个方案都生成这么多局（保证可比）
    num_workers = config.NUM_WORKERS

    cfg = build_self_play_config()

    print("=" * 80)
    print("  🚀 批量自对弈吞吐基准 (batch 合并 vs 串行小 batch)")
    print("=" * 80)
    print(f"  模拟推理: 固定开销={SimulatedPredictor.FIXED_OVERHEAD_S}s "
          f"+ 每样本={SimulatedPredictor.PER_SAMPLE_S}s")
    print(f"  mcts_sims={cfg.mcts_sims}, max_considered_actions={cfg.max_considered_actions}")
    print(f"  每个方案固定生成 {num_games} 局，计时对比")
    print(f"  (baseline 并行 workers={num_workers})")
    print("=" * 80)

    results: List[BenchResult] = []

    # # ---- baseline: 串行单树（小 batch）
    # print("\n▶️  baseline 串行 ...")
    # p1 = SimulatedPredictor("serial")
    # r1 = _run_serial(p1, cfg, num_games)
    # r1.print()
    # results.append(r1)

    # ---- baseline: 并行 rayon
    print(f"\n▶️  baseline 并行 (workers={num_workers}) ...")
    p2 = SimulatedPredictor("parallel")
    r2 = _run_parallel(p2, cfg, num_workers, num_games)
    r2.print()
    results.append(r2)

    # ---- batched: 多个并发等级
    batched_results: List[BenchResult] = []
    for conc in (2, 4, 8):
        print(f"\n▶️  batched (concurrency={conc}) ...")
        p = SimulatedPredictor(f"batched-{conc}")
        r = _run_batched(p, cfg, num_games, conc)
        r.print()
        batched_results.append(r)

    results.extend(batched_results)

    # ---- 汇总对比
    bar = "=" * 80
    print(f"\n{bar}")
    print("  📊 汇总对比")
    print(f"{bar}")
    print(f"  {'方案':<40}{'games/s':>10}{'steps/s':>10}{'avg_batch':>12}")
    print(f"{'-' * 80}")
    for r in [r1, r2, *batched_results]:
        avg_batch = r.predictor_samples / max(1, r.predictor_calls)
        print(f"  {r.label:<40}{r.games_per_second:>10.3f}{r.steps_per_second:>10.1f}{avg_batch:>12.1f}")



if __name__ == "__main__":
    main()
