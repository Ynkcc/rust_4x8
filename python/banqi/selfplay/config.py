"""banqi/selfplay/config.py — 自对弈相关工厂（模型/Predictor/配置）。

build_predictor        : 构建单设备 Predictor（device=auto 时 CUDA 优先）
build_mixed_predictor  : 构建 GPU+CPU 混合推理 MultiDevicePredictor
build_self_play_config : 构建 banqi_4x8.SelfPlayConfig（c_scale/gumbel_scale 支持 env 覆盖）

Rust 持有模型的收集器（RustTorchCollector / RustOnnxCollector）已彻底移除，其能力
统一由 `banqi_4x8.run_native_match`（record_episodes=True）承载。
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

    enable_health = bool(getattr(cfg, "HEALTH_VALUE_HEAD_ENABLED", False))
    model = BanqiNet(variant, enable_health=enable_health)
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
    enable_health = bool(getattr(make_config(variant.id), "HEALTH_VALUE_HEAD_ENABLED", False))
    gpu_model = BanqiNet(variant, enable_health=enable_health).to(device)
    gpu_predictor = Predictor(gpu_model, device, model_path, variant)

    cpu_device = torch.device("cpu")
    cpu_model = BanqiNet(variant, enable_health=enable_health).to(cpu_device)
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
    c_scale = float(os.getenv("C_SCALE", "1.0"))
    gumbel_scale = float(os.getenv("GUMBEL_SCALE", "1.0"))
    return banqi_4x8.SelfPlayConfig(
        mcts_sims=cfg.MCTS_SIMS,
        max_considered_actions=cfg.MAX_CONSIDERED_ACTIONS,
        c_scale=c_scale,
        gumbel_scale=gumbel_scale,
        playout_cap_random_enabled=getattr(cfg, "PLAYOUT_CAP_RANDOM_ENABLED", True),
        fast_mcts_sims=getattr(cfg, "FAST_MCTS_SIMS", 16),
        full_search_prob=float(getattr(cfg, "FULL_SEARCH_PROB", 0.25)),
    )
