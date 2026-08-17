"""
benchmark_production.py — 统一的样本实际生产效率基准测试

在真实环境下驱动 Rust 自对弈（Gumbel MCTS + 神经网络推理），测量各环境变种
在指定推理设备上的「样本实际生产效率」（samples/s），并给出 games/s、steps/s、
平均 batch 等派生指标，用于评估 / 对比：

  - 不同环境变种（4x8 / 4x4 / 4x2）的样本生产效率
  - 是否使用 GPU 推理（--gpu / --no-gpu / --device）对吞吐的影响
  - 不同自对弈运行方案（串行 / rayon 并行 / 批量合并大 batch）的吞吐差异

用法（需先编译绑定：maturin develop --features pyo3 --release）：

    # 全部变种，自动选择设备
    python python/benchmark_production.py

    # 指定变种 + 指定设备
    python python/benchmark_production.py --variants 4x8,4x4 --device cpu
    python python/benchmark_production.py --variants 4x2 --gpu

    # 只测某几种运行方案，每方案跑更多局，结果写入 JSON
    python python/benchmark_production.py --schemes serial,parallel --num-games 16 \
        --json bench_results.json

    # 不依赖 PyTorch / 真实模型：用模拟推理延迟快速验证脚本流程
    python python/benchmark_production.py --simulated --num-games 2

注意：
  - 推理设备解析顺序：--gpu / --no-gpu > --device > 各变体配置的 INFER_DEVICE。
  - 每个 (变种 × 设备 × 方案) 组合独立构建全新 Predictor 并完整计时，
    保证「固定局数、计时」的可复现对比。
  - 4x4 的 batched 方案在 Rust 侧使用启发式评估器（不调用神经网络），
    因此该组合的 predictor 调用计数为 0，属预期行为。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

# Windows 控制台默认 GBK 无法编码 emoji 等字符，强制以 UTF-8 输出避免启动崩溃
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover
    raise SystemExit("缺少 numpy，请先执行: pip install -r python/requirements.txt") from exc

try:
    import torch
    HAS_TORCH = True
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    HAS_TORCH = False

try:
    import banqi_4x8
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "无法导入 banqi_4x8。请先执行: maturin develop --features pyo3 --release"
    ) from exc

from banqi.variant import Variant, get_variant
from banqi.constants import build_constants
from banqi.config import make_config
from banqi.memory_guard import start_memory_guard
from banqi.self_play import build_predictor

# 全部可测环境变种（顺序即打印顺序）
AVAILABLE_VARIANTS: Tuple[str, ...] = ("4x8", "4x4", "4x2")
# 自对弈运行方案
AVAILABLE_SCHEMES: Tuple[str, ...] = ("serial", "parallel", "batched")

# Rust 绑定函数名分发表（与 banqi/self_play.py::_SPLAY_FNS 保持一致）：
#   变体 rust_prefix -> (serial, parallel, batched)
_SCHEME_FNS: Dict[str, Tuple[str, str, str]] = {
    "": ("run_self_play_with_predictor",
         "run_parallel_self_play_with_predictor",
         "run_batched_self_play_with_predictor"),
    "mini": ("run_mini_self_play_with_predictor",
             "run_mini_parallel_self_play_with_predictor",
             "run_mini_batched_self_play_with_predictor"),
    "game4x4": ("run_game4x4_self_play_with_predictor",
                "run_game4x4_parallel_self_play_with_predictor",
                "run_game4x4_batched_self_play_with_predictor"),
}

_SCHEME_LABELS: Dict[str, str] = {
    "serial": "串行 (单树, 小 batch)",
    "parallel": "并行 (rayon, 多树各自推理)",
    "batched": "批量 (并发多局, 合并大 batch)",
}


# ============================================================================
# 结果容器
# ============================================================================

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
        print(f"  🏁 基准结果: 变种={self.variant_id}  方案={_SCHEME_LABELS.get(self.scheme, self.scheme)}  设备={self.device}")
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


# ============================================================================
# Predictor 包装 / 模拟
# ============================================================================

class CountingPredictor:
    """包装任意 callable predictor，统计调用次数与送入的总样本数（线程安全）。

    Rust 侧串行/并行/批量自对弈都可能从多线程调用 predict_fn，因此统计需加锁。
    """

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.calls = 0
        self.samples = 0
        self._lock = threading.Lock()

    def __call__(self, board: np.ndarray, scalars: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        n = int(board.shape[0])
        with self._lock:
            self.calls += 1
            self.samples += n
        return self.inner(board, scalars)


class SimulatedPredictor:
    """模拟推理延迟：delay(batch) = 固定启动开销 + 每样本线性成本。

    不依赖 PyTorch / 真实模型，用于快速验证基准脚本流程（--simulated），
    或对比「小 batch 串行 vs 大 batch 并行」时模拟 GPU 推理的固定开销摊薄效果。
    """

    FIXED_OVERHEAD_S = 0.002   # 每次推理调用的固定开销（kernel 启动 / 拷贝）
    PER_SAMPLE_S = 0.0001      # 每个样本的线性推理成本

    def __init__(self, action_space_size: int) -> None:
        self.action_space_size = action_space_size

    def __call__(self, board: np.ndarray, scalars: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        batch = int(board.shape[0])
        time.sleep(self.FIXED_OVERHEAD_S + batch * self.PER_SAMPLE_S)
        rng = np.random.default_rng()
        policy = rng.standard_normal((batch, self.action_space_size)).astype(np.float32)
        values = rng.uniform(-1.0, 1.0, size=batch).astype(np.float32)
        return policy, values


# ============================================================================
# 参数解析
# ============================================================================

def _parse_variants(raw: Optional[str]) -> List[str]:
    if not raw:
        return list(AVAILABLE_VARIANTS)
    ids: List[str] = []
    for part in raw.split(","):
        vid = part.strip()
        if not vid:
            continue
        if vid not in AVAILABLE_VARIANTS:
            raise SystemExit(f"未知变种 {vid!r}，可选: {', '.join(AVAILABLE_VARIANTS)}")
        if vid not in ids:
            ids.append(vid)
    if not ids:
        raise SystemExit("--variants 未指定任何有效变种")
    return ids


def _parse_schemes(raw: Optional[str]) -> List[str]:
    if not raw:
        return list(AVAILABLE_SCHEMES)
    schemes: List[str] = []
    for part in raw.split(","):
        s = part.strip()
        if not s:
            continue
        if s not in AVAILABLE_SCHEMES:
            raise SystemExit(f"未知方案 {s!r}，可选: {', '.join(AVAILABLE_SCHEMES)}")
        if s not in schemes:
            schemes.append(s)
    if not schemes:
        raise SystemExit("--schemes 未指定任何有效方案")
    return schemes


def _resolve_device(args: argparse.Namespace, variant: Variant) -> str:
    """推理设备解析：--gpu / --no-gpu > --device > 变体配置 INFER_DEVICE。"""
    if args.gpu:
        device = "cuda"
    elif args.no_gpu:
        device = "cpu"
    elif args.device:
        device = args.device
    else:
        device = make_config(variant.id).INFER_DEVICE or "auto"

    if device not in ("auto", "cpu", "cuda"):
        raise SystemExit(f"非法设备 {device!r}，可选: auto / cpu / cuda")

    # 模拟推理不依赖真实 GPU/模型，跳过 CUDA 可用性检查（device 仅作为结果标签）
    if not args.simulated and device == "cuda" and not (HAS_TORCH and torch.cuda.is_available()):
        raise SystemExit(
            "指定了 GPU (cuda) 推理，但当前环境不可用："
            + ("未安装 PyTorch。" if not HAS_TORCH else "torch.cuda.is_available() 为 False。")
            + " 请用 --no-gpu / --device cpu 回退 CPU，或修复 CUDA 环境。"
        )
    if device == "auto" and not HAS_TORCH:
        device = "cpu"
    return device


def _resolve_model_path(variant: Variant, model_flag: Optional[str]) -> Optional[str]:
    if model_flag:
        return model_flag
    path = make_config(variant.id).MODEL_PATH
    return path if path and os.path.exists(path) else None


# ============================================================================
# 基准执行
# ============================================================================

def _build_predictor(
    variant: Variant, device: str, model_path: Optional[str],
    simulated: bool = False,
) -> Any:
    """构建 Predictor（真实模型 / 模拟延迟），并包一层计数统计。

    返回 (counting_predictor, 解析后的设备标签)：
      - simulated=True : 使用 SimulatedPredictor，不依赖 PyTorch / 模型文件；
      - 否则            : 用 banqi.self_play.build_predictor 构建真实 Predictor
                          （模型不存在时使用全新初始化网络）。
    """
    if simulated:
        action_space = build_constants(variant).ACTION_SPACE_SIZE
        return CountingPredictor(SimulatedPredictor(action_space)), device
    if HAS_TORCH:
        infer_device = device if device in ("cpu", "cuda") else "auto"
        predictor, resolved = build_predictor(variant, model_path, device_str=infer_device)
        return CountingPredictor(predictor), str(resolved)
    # 无 torch 时用退化 Predictor：直接借道 build_predictor（内部会退化预测）
    predictor, resolved = build_predictor(variant, model_path, device_str="cpu")
    return CountingPredictor(predictor), str(resolved)


def _run_scheme(
    variant: Variant,
    scheme: str,
    predictor: Any,
    sp_cfg: Any,
    args: argparse.Namespace,
) -> Tuple[List[Any], int, int]:
    """调用 Rust 绑定运行指定方案，返回 (episodes, 实际调用次数, 送入样本数)。"""
    fns = _SCHEME_FNS[variant.rust_prefix]
    cfg = make_config(variant.id)

    if scheme == "serial":
        episodes = getattr(banqi_4x8, fns[0])(
            predict_fn=predictor, config=sp_cfg,
            num_games=args.num_games, worker_id=0,
        )
    elif scheme == "parallel":
        workers = args.num_workers or cfg.NUM_WORKERS
        games_per_worker = max(1, -(-args.num_games // max(1, workers)))
        episodes = getattr(banqi_4x8, fns[1])(
            predict_fn=predictor, config=sp_cfg,
            num_workers=workers, games_per_worker=games_per_worker, worker_id=0,
        )
    elif scheme == "batched":
        conc = args.concurrency or cfg.BATCH_CONCURRENCY
        episodes = getattr(banqi_4x8, fns[2])(
            predict_fn=predictor, config=sp_cfg,
            num_games=args.num_games, concurrency=conc, worker_id=0,
        )
    else:  # pragma: no cover
        raise SystemExit(f"未知方案 {scheme!r}")

    if isinstance(predictor, CountingPredictor):
        calls, samples = predictor.calls, predictor.samples
    else:
        calls, samples = 0, 0
    return episodes, calls, samples


def benchmark_cell(
    variant: Variant,
    scheme: str,
    device: str,
    args: argparse.Namespace,
) -> BenchResult:
    """运行单个 (变种 × 方案) 组合，返回完整结果。"""
    label = f"{variant.id}/{scheme}/{device}"
    model_path = _resolve_model_path(variant, args.model)
    sp_cfg = build_self_play_config(variant, args)

    print(f"\n▶️  开始: {label}  "
          f"(sims={sp_cfg.mcts_sims}, max_actions={sp_cfg.max_considered_actions})")

    predictor, resolved_device = _build_predictor(
        variant, device, model_path, simulated=args.simulated,
    )

    result = BenchResult(
        variant_id=variant.id, scheme=scheme, device=resolved_device,
    )
    t0 = time.perf_counter()
    try:
        episodes, calls, samples = _run_scheme(variant, scheme, predictor, sp_cfg, args)
    except Exception as exc:  # pragma: no cover
        print(f"  ⚠️  运行失败: {label}: {exc}")
        return result
    result.duration_s = time.perf_counter() - t0

    for ep in episodes:
        result.completed_games += 1
        result.total_steps += int(ep.game_length)
        result.total_samples += int(ep.num_samples)

    result.predictor_calls = int(calls)
    result.predictor_samples = int(samples)
    return result


def build_self_play_config(variant: Variant, args: Optional[argparse.Namespace] = None) -> Any:
    """构建 SelfPlayConfig：CLI 覆盖 > 变体配置默认值。

    与 banqi/self_play.py::build_self_play_config 保持同等契约（mcts_sims /
    max_considered_actions / temperature_steps / c_scale / gumbel_scale）。
    """
    cfg = make_config(variant.id)
    c_scale = float(os.getenv(variant.env_prefix + "C_SCALE", os.getenv("C_SCALE", "1.0")))
    gumbel = float(os.getenv(variant.env_prefix + "GUMBEL_SCALE", os.getenv("GUMBEL_SCALE", "1.0")))
    return banqi_4x8.SelfPlayConfig(
        mcts_sims=args.mcts_sims if args and args.mcts_sims else cfg.MCTS_SIMS,
        max_considered_actions=args.max_considered_actions if args and args.max_considered_actions else cfg.MAX_CONSIDERED_ACTIONS,
        temperature_steps=args.temperature_steps if args and args.temperature_steps else cfg.TEMPERATURE_STEPS,
        c_scale=c_scale,
        gumbel_scale=gumbel,
    )


# ============================================================================
# 汇总 / 输出
# ============================================================================

def _print_summary(results: List[BenchResult]) -> None:
    bar = "=" * 88
    print(f"\n{bar}")
    print("  📊 样本实际生产效率 — 汇总对比")
    print(f"{bar}")
    header = f"  {'变种/方案':<38}{'设备':<8}{'样本/s':>10}{'局/s':>9}{'步/s':>9}{'avg_batch':>10}"
    print(header)
    print(f"  {'-' * 86}")
    for r in results:
        name = f"{r.variant_id}/{r.scheme}"
        print(
            f"  {name:<38}{r.device:<8}{r.samples_per_second:>10.1f}"
            f"{r.games_per_second:>9.3f}{r.steps_per_second:>9.1f}{r.avg_batch:>10.1f}"
        )
    print(f"{bar}\n")


def _print_per_variant(results: List[BenchResult]) -> None:
    """按变种分组，横向对比该变种在不同设备/方案下的样本生产效率。"""
    by_variant: Dict[str, List[BenchResult]] = {}
    for r in results:
        by_variant.setdefault(r.variant_id, []).append(r)

    for vid, rs in by_variant.items():
        print(f"\n  [变种 {vid}] 样本生产效率 (samples/s) 排序:")
        for rank, r in enumerate(sorted(rs, key=lambda x: x.samples_per_second, reverse=True), 1):
            print(
                f"    #{rank}  {r.scheme:<12} 设备={r.device:<5}  "
                f"{r.samples_per_second:>9.1f} 样本/s  "
                f"({r.completed_games} 局, {r.duration_s:.1f}s)"
            )


# ============================================================================
# 主流程
# ============================================================================

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="benchmark_production.py",
        description="统一的样本实际生产效率基准测试（支持 GPU/CPU 与多环境变种）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--variants", metavar="ID[,ID...]",
        default=None,
        help=f"要测试的环境变种，逗号分隔，可选: {', '.join(AVAILABLE_VARIANTS)}（默认全部）",
    )
    parser.add_argument(
        "--schemes", metavar="NAME[,NAME...]",
        default=None,
        help=f"要测试的自对弈方案，可选: {', '.join(AVAILABLE_SCHEMES)}（默认全部）",
    )
    device_group = parser.add_mutually_exclusive_group()
    device_group.add_argument(
        "--device", choices=("auto", "cpu", "cuda"),
        default=None,
        help="推理设备（默认取各变体配置的 INFER_DEVICE）",
    )
    device_group.add_argument(
        "--gpu", action="store_true",
        help="使用 GPU (cuda) 推理（等价 --device cuda）",
    )
    device_group.add_argument(
        "--no-gpu", action="store_true",
        help="使用 CPU 推理，禁用 GPU（等价 --device cpu）",
    )
    parser.add_argument(
        "--num-games", type=int, default=8,
        help="每个方案生成的局数（并行方案会向上取整到 worker 倍数）",
    )
    parser.add_argument(
        "--num-workers", type=int, default=None,
        help="parallel 方案的 worker 数（默认取变体配置 NUM_WORKERS）",
    )
    parser.add_argument(
        "--concurrency", type=int, default=None,
        help="batched 方案的并发局数（默认取变体配置 BATCH_CONCURRENCY）",
    )
    parser.add_argument(
        "--mcts-sims", type=int, default=None,
        help="覆盖 MCTS 模拟数（默认取变体配置 MCTS_SIMS）",
    )
    parser.add_argument(
        "--max-considered-actions", type=int, default=None,
        help="覆盖最大考虑动作数（默认取变体配置 MAX_CONSIDERED_ACTIONS）",
    )
    parser.add_argument(
        "--temperature-steps", type=int, default=None,
        help="覆盖温度退火步数（默认取变体配置 TEMPERATURE_STEPS）",
    )
    parser.add_argument(
        "--torch-threads", type=int, default=None,
        help="设置 torch 推理线程数（CPU 推理优化；默认不修改进程级设置）",
    )
    parser.add_argument(
        "--model", metavar="PATH", default=None,
        help="指定模型权重路径；默认取变体配置 MODEL_PATH（不存在则用全新初始化网络）",
    )
    parser.add_argument(
        "--simulated", action="store_true",
        help="使用模拟推理延迟 Predictor，不依赖 PyTorch / 真实模型（流程自检用）",
    )
    parser.add_argument(
        "--json", metavar="PATH", default=None,
        help="将全部结果以 JSON 写入指定文件",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    variant_ids = _parse_variants(args.variants)
    schemes = _parse_schemes(args.schemes)

    if HAS_TORCH and args.torch_threads:
        torch.set_num_threads(args.torch_threads)
        print(f"[Main] torch.set_num_threads({args.torch_threads})")

    print("=" * 80)
    print("  🚀 样本实际生产效率基准 (自对弈 + 神经网络推理)")
    print("=" * 80)
    print(f"  环境变种 : {', '.join(variant_ids)}")
    print(f"  方案     : {', '.join(schemes)}")
    print(f"  局数/方案: {args.num_games}")
    if args.mcts_sims:
        print(f"  MCTS_SIMS= {args.mcts_sims}")
    if args.simulated:
        print(f"  模拟推理 : 固定开销={SimulatedPredictor.FIXED_OVERHEAD_S}s "
              f"+ 每样本={SimulatedPredictor.PER_SAMPLE_S}s")
    print(f"  PyTorch  : {'可用' if HAS_TORCH else '不可用（退化预测）'}"
          f"{' | CUDA 可用' if HAS_TORCH and torch.cuda.is_available() else ''}")
    print("=" * 80)

    # ---- 内存看门狗守护线程（超限主动终止，防止长时间卡死 / 拖垮整机）----
    start_memory_guard()

    results: List[BenchResult] = []

    for vid in variant_ids:
        variant = get_variant(vid)
        device = _resolve_device(args, variant)
        print(f"\n{'#' * 80}")
        print(f"  # 变种 {vid}  (rust_prefix={variant.rust_prefix!r}, "
              f"action_space={build_constants(variant).ACTION_SPACE_SIZE}, "
              f"board={variant.board_rows}x{variant.board_cols})")
        print(f"{'#' * 80}")
        print(f"  推理设备: {device}")

        for scheme in schemes:
            result = benchmark_cell(variant, scheme, device, args)
            result.print()
            results.append(result)

    _print_summary(results)
    _print_per_variant(results)

    if args.json:
        payload = {
            "meta": {
                "tool": "benchmark_production.py",
                "time": datetime.now(timezone.utc).isoformat(),
                "variants": variant_ids,
                "schemes": schemes,
                "num_games": args.num_games,
                "simulated": args.simulated,
                "device_flag": {
                    "gpu": args.gpu,
                    "no_gpu": args.no_gpu,
                    "device": args.device,
                },
                "torch_available": HAS_TORCH,
                "cuda_available": HAS_TORCH and torch.cuda.is_available(),
            },
            "results": [r.to_dict() for r in results],
        }
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"\n[Main] ✅ 结果已写入 {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
