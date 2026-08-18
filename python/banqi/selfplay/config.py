"""banqi/selfplay/config.py — 自对弈相关工厂（模型/Predictor/配置/收集器）。

build_predictor        : 构建单设备 Predictor（device=auto 时 CUDA 优先）
build_mixed_predictor  : 构建 GPU+CPU 混合推理 MultiDevicePredictor
build_self_play_config : 构建 banqi_4x8.SelfPlayConfig（c_scale/gumbel_scale 支持 env 覆盖）
build_rust_collector   : 构建 Rust 侧持有模型的收集器（免 GIL，可选）
rust_collector_run_batch: 用 Rust 收集器批量生成 episode
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:  # pragma: no cover
    HAS_TORCH = False

try:
    import banqi_4x8
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "无法导入 banqi_4x8。请先执行: maturin develop --features pyo3"
    ) from exc

from banqi.config import make_config
from banqi.constants import build_constants
from banqi.nn_model import BanqiNet
from banqi.variant import Variant

from .predictor import OnnxPredictor, Predictor, MultiDevicePredictor


def _resolve_device(device_str: str) -> "torch.device":
    if not HAS_TORCH:
        return torch.device("cpu")
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def build_predictor(variant: Variant, model_path: Optional[str],
                    device_str: str = "auto") -> Tuple[Predictor, "torch.device"]:
    """构建 Predictor。model_path 为 None 时使用全新初始化网络。

    当 config.MODEL_BACKEND == "onnx" 时优先构建 `OnnxPredictor`（onnxruntime）；
    若 ONNX 推理不可用（缺文件 / 未装 onnxruntime），回退 torch 推理。
    """
    cfg = make_config(variant.id)
    backend = (cfg.MODEL_BACKEND or "torchscript").strip().lower()
    if backend == "onnx":
        onnx_predictor = build_onnx_predictor(variant, model_path or cfg.ONNX_PATH)
        if onnx_predictor is not None:
            return onnx_predictor, torch.device("cpu")
        print(f"[SP-{variant.id}] ⚠️ ONNX 推理不可用，回退 torch 推理")

    if not HAS_TORCH:
        print(f"[SP-{variant.id}] 警告：未安装 PyTorch，将使用退化预测（均匀 logits）")
        device = torch.device("cpu")
    else:
        device = _resolve_device(device_str)
    print(f"[SP-{variant.id}] device = {device}")

    model = BanqiNet(variant)
    if HAS_TORCH:
        model = model.to(device)
    predictor = Predictor(model, device, model_path, variant)

    if model_path and os.path.exists(model_path):
        print(f"[SP-{variant.id}] 使用模型权重: {model_path}")
    else:
        print(f"[SP-{variant.id}] 未指定有效模型路径，使用全新初始化网络")
    return predictor, device


def build_onnx_predictor(
    variant: Variant, model_path: Optional[str] = None
) -> Optional[OnnxPredictor]:
    """构建 Python onnxruntime Predictor（MODEL_BACKEND="onnx" 时的回退推理）。

    仅在 Rust 绑定 `RustOnnxCollector` 不可用时被调用；若 onnx 文件缺失或
    onnxruntime 未安装，返回 None（调用方回退 torch 推理）。
    """
    cfg = make_config(variant.id)
    path = model_path or cfg.ONNX_PATH
    if not path or not os.path.exists(path):
        print(f"[SP-{variant.id}] ⚠️ ONNX 模型不存在: {path}")
        return None
    providers = [p.strip() for p in cfg.ONNX_PROVIDERS.split(",") if p.strip()] or [
        "CPUExecutionProvider"
    ]
    try:
        return OnnxPredictor(
            path,
            build_constants(variant).ACTION_SPACE_SIZE,
            providers=providers,
            variant=variant,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[SP-{variant.id}] ⚠️ ONNX Predictor 构建失败: {exc}")
        return None


def build_mixed_predictor(
    variant: Variant,
    model_path: Optional[str],
    device_str: str = "auto",
    cpu_workers: int = 1,
    cpu_fraction: float = 0.3,
    min_split_batch: int = 16,
) -> Tuple[Predictor, "torch.device"]:
    """构建 GPU + CPU 混合推理 Predictor（MultiDevicePredictor）。

    主设备（INFER_DEVICE）必须解析为 CUDA 才有混合意义。若主设备不是 CUDA
    （例如 INFER_DEVICE=cpu），回退到普通单设备 Predictor。
    """
    if not HAS_TORCH:
        print(f"[SP-{variant.id}] 警告：未安装 PyTorch，无法启用 GPU+CPU 混合推理，回退单设备")
        return build_predictor(variant, model_path, device_str)

    device = _resolve_device(device_str)
    if device.type != "cuda":
        print(f"[SP-{variant.id}] 主推理设备 = {device}（非 CUDA），跳过 GPU+CPU 混合推理")
        return build_predictor(variant, model_path, device_str)

    print(
        f"[SP-{variant.id}] 启用 GPU+CPU 混合推理: GPU={device}, "
        f"CPU线程={cpu_workers}, CPU比例={cpu_fraction:.2f}"
    )
    gpu_model = BanqiNet(variant).to(device)
    gpu_predictor = Predictor(gpu_model, device, model_path, variant)

    cpu_device = torch.device("cpu")
    cpu_model = BanqiNet(variant).to(cpu_device)
    cpu_predictor = Predictor(cpu_model, cpu_device, model_path, variant)

    return (
        MultiDevicePredictor(
            gpu_predictor,
            cpu_predictor,
            cpu_fraction=cpu_fraction,
            cpu_workers=cpu_workers,
            min_split_batch=min_split_batch,
        ),
        device,
    )


def build_self_play_config(variant: Variant) -> "banqi_4x8.SelfPlayConfig":
    """构建 SelfPlayConfig（与 py/mod.rs 契约一致），c_scale/gumbel_scale 支持 env 覆盖。"""
    cfg = make_config(variant.id)
    p = variant.env_prefix
    c_scale = float(os.getenv(p + "C_SCALE", os.getenv("C_SCALE", "1.0")))
    gumbel_scale = float(os.getenv(p + "GUMBEL_SCALE", os.getenv("GUMBEL_SCALE", "1.0")))
    return banqi_4x8.SelfPlayConfig(
        mcts_sims=cfg.MCTS_SIMS,
        max_considered_actions=cfg.MAX_CONSIDERED_ACTIONS,
        temperature_steps=cfg.TEMPERATURE_STEPS,
        c_scale=c_scale,
        gumbel_scale=gumbel_scale,
    )


# ============================================================================
# Rust 持有模型的收集器（可选，需 maturin 以对应 feature 构建）
# ============================================================================
#
# 背景：`run_*_self_play_with_predictor` 把 Python `predict_fn` 传给 Rust，MCTS 评估
# 时通过 GIL 调 Python 推理 → 即使 Rust 侧多线程，推理仍被 GIL 串行化；若改用
# multiprocessing(spawn) 绕开 GIL，每个子进程又重复加载一份 libtorch + 权重。
#
# 解法：收集器在 Rust 侧一次性加载模型（模型留在 Rust，推理不经过 GIL），
# 跨线程共享单份模型。这里提供便捷工厂与一个「生产一批 episode」的辅助函数，
# 作为 python 回调方案的替代。
#
# 两个后端：
#   - `RustTorchCollector`（RustTorchCollector）：TorchScript .pt，tch-rs 推理。
#     需同时启用 torch + pyo3 feature 构建（Cargo.toml `rust-torch-collector`）。
#   - `RustOnnxCollector`（RustOnnxCollector）：ONNX .onnx，ONNX Runtime 推理。
#     需同时启用 onnx + pyo3 feature 构建（Cargo.toml `rust-onnx-collector`），
#     不依赖 libtorch。
#
# 按 config.MODEL_BACKEND 自动选择：MODEL_BACKEND="onnx" 时优先使用 ONNX 后端，
# 否则回退 Torch 后端。

_RUST_TORCH_COLLECTOR_AVAILABLE = hasattr(banqi_4x8, "RustTorchCollector")
_RUST_ONNX_COLLECTOR_AVAILABLE = hasattr(banqi_4x8, "RustOnnxCollector")


def build_rust_collector(
    variant: Variant,
    model_path: Optional[str] = None,
    device: Optional[str] = None,
    backend: Optional[str] = None,
):
    """构建 Rust 侧持有模型的收集器（模型只加载一份，推理不经过 GIL）。

    backend：
      - "torchscript"：RustTorchCollector（.pt）
      - "onnx"：       RustOnnxCollector（.onnx）
      - None：按 config.MODEL_BACKEND 自动选择（默认）。

    返回对应 pyclass 实例；若当前 wheel 未启用对应绑定，返回 None。
    """
    cfg = make_config(variant.id)
    backend = (backend or cfg.MODEL_BACKEND or "torchscript").strip().lower()
    if backend == "onnx":
        return build_onnx_collector(variant, model_path, device)
    if not _RUST_TORCH_COLLECTOR_AVAILABLE:
        print(
            f"[SP-{variant.id}] 未检测到 RustTorchCollector。若需要 Rust 持有模型、"
            f"免 GIL 的数据收集，请用 maturin build --features torch,pyo3-extension 构建。"
        )
        return None
    path = model_path or cfg.MODEL_PATH
    dev = device or cfg.INFER_DEVICE
    print(f"[SP-{variant.id}] 构建 RustTorchCollector: model={path} device={dev}")
    return banqi_4x8.RustTorchCollector(path, variant.id, dev)


def build_onnx_collector(
    variant: Variant,
    model_path: Optional[str] = None,
    device: Optional[str] = None,
):
    """构建 Rust 侧持有 ONNX 模型的收集器（`banqi_4x8.RustOnnxCollector`）。

    推理由 ONNX Runtime 完成（不经过 GIL、不依赖 libtorch）。
    若当前 wheel 未启用 onnx+pyo3 绑定（`rust-onnx-collector` feature 未开），
    返回 None（调用方可回退到 Python onnxruntime 推理）。
    """
    if not _RUST_ONNX_COLLECTOR_AVAILABLE:
        print(
            f"[SP-{variant.id}] 未检测到 RustOnnxCollector。若需要 Rust 持有 ONNX 模型、"
            f"免 GIL 的数据收集，请用 maturin build --features onnx,pyo3-extension 构建。"
        )
        return None
    cfg = make_config(variant.id)
    path = model_path or cfg.ONNX_PATH or cfg.MODEL_PATH
    dev = device or cfg.INFER_DEVICE
    if not path or not os.path.exists(path):
        print(f"[SP-{variant.id}] ⚠️ ONNX 模型不存在: {path}（无法构建 RustOnnxCollector）")
        return None
    print(f"[SP-{variant.id}] 构建 RustOnnxCollector: model={path} device={dev}")
    return banqi_4x8.RustOnnxCollector(path, variant.id, dev)


def rust_collector_run_batch(
    collector,
    variant: Variant,
    sp_cfg,
    num_games: int,
    concurrency: int,
    worker_id: int = 0,
) -> list:
    """用 Rust 持有模型的收集器批量生成一局，返回 episode 对象列表。

    `collector`：`build_rust_collector` 的返回值（None 时回退空列表）。
    """
    if collector is None:
        return []
    return list(
        collector.run_batched(
            config=sp_cfg,
            num_games=num_games,
            concurrency=concurrency,
            worker_id=worker_id,
        )
    )
