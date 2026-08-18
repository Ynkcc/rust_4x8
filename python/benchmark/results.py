"""benchmark/results.py — 基准测试结果容器。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

# 自对弈运行方案标签（与 schemes.py 中的 _SCHEME_FNS 对应）
_SCHEME_LABELS: Dict[str, str] = {
    "serial": "串行 (单树, 小 batch)",
    "parallel": "并行 (rayon, 多树各自推理)",
    "batched": "批量 (并发多局, 合并大 batch)",
    "multiproc": "多进程 (spawn ×N, 独立 GIL)",
}


@dataclass
class BenchResult:
    """单个 (变种 × 方案) 组合的基准结果。"""

    variant_id: str
    scheme: str
    device: str
    duration_s: float = 0.0
    completed_games: int = 0
    total_steps: int = 0
    total_samples: int = 0
    predictor_calls: int = 0
    predictor_samples: int = 0

    @property
    def games_per_second(self) -> float:
        return self.completed_games / max(1e-9, self.duration_s)

    @property
    def steps_per_second(self) -> float:
        return self.total_steps / max(1e-9, self.duration_s)

    @property
    def samples_per_second(self) -> float:
        return self.total_samples / max(1e-9, self.duration_s)

    @property
    def avg_batch(self) -> float:
        return self.predictor_samples / max(1, self.predictor_calls)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "variant": self.variant_id,
            "scheme": self.scheme,
            "device": self.device,
            "duration_s": round(self.duration_s, 3),
            "completed_games": self.completed_games,
            "total_steps": self.total_steps,
            "total_samples": self.total_samples,
            "samples_per_sec": round(self.samples_per_second, 2),
            "games_per_sec": round(self.games_per_second, 3),
            "steps_per_sec": round(self.steps_per_second, 1),
            "predictor_calls": self.predictor_calls,
            "predictor_samples": self.predictor_samples,
            "avg_batch": round(self.avg_batch, 1),
        }

    def print(self) -> None:
        bar = "=" * 80
        print(f"\n{bar}")
        print(f"  🏁 基准结果: 变种={self.variant_id}  "
              f"方案={_SCHEME_LABELS.get(self.scheme, self.scheme)}  设备={self.device}")
        print(f"{bar}")
        print(f"  耗时 (duration)      : {self.duration_s:.2f} s")
        print(f"  完成局数             : {self.completed_games} 局")
        print(f"  样本生产效率 (样本/s): {self.samples_per_second:.1f}")
        print(f"  吞吐量 (games/s)     : {self.games_per_second:.3f} 局/秒")
        print(f"  吞吐量 (steps/s)     : {self.steps_per_second:.1f} 步/秒")
        print(f"  总步数 / 总样本数    : {self.total_steps} 步 / {self.total_samples:,} 样本")
        print(f"  Predictor 调用次数   : {self.predictor_calls} 次")
        print(f"  Predictor 平均 batch : {self.avg_batch:.1f}")
        print(f"{bar}\n")
