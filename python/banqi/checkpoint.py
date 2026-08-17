"""banqi/checkpoint.py — 通用 checkpoint 保存 / 恢复（含 variant 元数据）

面向新格式：保存时写入 variant 关键维度到 model_config；
恢复时校验维度匹配（不兼容旧文件，见项目决策「不保留旧版本兼容」）。
"""

from __future__ import annotations

import os
from typing import Optional

import torch

from banqi.constants import Constants, build_constants
from banqi.variant import Variant


def _model_config(c: Constants) -> dict:
    return {
        "variant": c.variant.id,
        "input_channels": c.TOTAL_INPUT_CHANNELS,
        "board_rows": c.BOARD_ROWS,
        "board_cols": c.BOARD_COLS,
        "scalar_features": c.SCALAR_FEATURE_COUNT,
        "action_space": c.ACTION_SPACE_SIZE,
    }


def save_checkpoint(
    model,
    optimizer,
    scheduler,
    model_path: str,
    state_dict_path: str,
    device: torch.device,
    variant: Variant,
) -> None:
    """保存完整训练状态：.pth（可恢复）+ .pt（TorchScript 供推理）。

    先写临时文件再原子 replace，避免进程中断留下半成品。
    """
    c = build_constants(variant)
    pt_temp = model_path + ".tmp"
    pth_temp = state_dict_path + ".tmp"
    trace_model = getattr(model, "_orig_mod", model)
    try:
        model.eval()
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "model_config": _model_config(c),
            },
            pth_temp,
        )
        os.replace(pth_temp, state_dict_path)

        with torch.inference_mode():
            example_board = torch.randn(
                1, c.TOTAL_INPUT_CHANNELS, c.BOARD_ROWS, c.BOARD_COLS, device=device
            )
            example_scalars = torch.randn(1, c.SCALAR_FEATURE_COUNT, device=device)
            traced = torch.jit.trace(trace_model, (example_board, example_scalars))
            traced.save(pt_temp)
        os.replace(pt_temp, model_path)
        print(f"[checkpoint] ✅ 保存成功: {state_dict_path} + {model_path}")
    except Exception as exc:  # noqa: BLE001
        print(f"[checkpoint] ❌ 保存失败: {exc}")
        for tmp in (pt_temp, pth_temp):
            if os.path.exists(tmp):
                os.remove(tmp)


def load_checkpoint(
    model,
    optimizer,
    scheduler,
    model_path: str,
    state_dict_path: str,
    device: torch.device,
    variant: Variant,
) -> bool:
    """从 .pth 恢复完整训练状态；失败回退 .pt 权重。返回是否恢复成功。"""
    c = build_constants(variant)
    state_loaded = False
    if os.path.exists(state_dict_path):
        try:
            checkpoint = torch.load(state_dict_path, map_location=device)
            _check_dimensions(checkpoint.get("model_config"), c)
            model.load_state_dict(checkpoint["model_state_dict"])
            if "optimizer_state_dict" in checkpoint:
                try:
                    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                except Exception as e_opt:  # noqa: BLE001
                    print(f"[checkpoint] ⚠️ Optimizer 状态加载失败 ({e_opt})，保持新初始化")
            if "scheduler_state_dict" in checkpoint:
                try:
                    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                except Exception as e_sch:  # noqa: BLE001
                    print(f"[checkpoint] ⚠️ Scheduler 状态加载失败 ({e_sch})，保持新初始化")
            print(f"[checkpoint] ✅ 从 {state_dict_path} 恢复完整训练状态")
            state_loaded = True
        except Exception as exc:  # noqa: BLE001
            print(f"[checkpoint] ⚠️ 完整 .pth 加载失败 ({exc})，尝试仅加载权重...")

    if not state_loaded and os.path.exists(model_path):
        try:
            jit_model = torch.jit.load(model_path, map_location=device)
            model.load_state_dict(jit_model.state_dict())
            print(f"[checkpoint] ✅ 从 {model_path} 加载模型权重 (TorchScript 回退)")
            state_loaded = True
        except Exception as e2:  # noqa: BLE001
            print(f"[checkpoint] ⚠️ 权重加载失败 ({e2})，使用全新模型")

    if not state_loaded and not os.path.exists(model_path) and not os.path.exists(state_dict_path):
        print("[checkpoint] 📝 初始化全新模型（无 checkpoint）")
    return state_loaded


def _check_dimensions(cfg: Optional[dict], c: Constants) -> None:
    """校验 checkpoint 的维度与当前 variant 一致；不一致抛错（明确失败）。"""
    if not cfg:
        return
    expect = {
        "input_channels": c.TOTAL_INPUT_CHANNELS,
        "board_rows": c.BOARD_ROWS,
        "board_cols": c.BOARD_COLS,
        "scalar_features": c.SCALAR_FEATURE_COUNT,
        "action_space": c.ACTION_SPACE_SIZE,
    }
    for k, val in expect.items():
        got = cfg.get(k)
        if got is not None and got != val:
            raise ValueError(
                f"checkpoint 维度 {k}={got} 与当前变体({c.variant.id}) 预期 {val} 不一致"
            )
