"""
validate_predictor.py — 验证 self_play.Predictor 推理端（纯 CPU）。

检查项：
  1. Predictor 输出形状：policy (N, 352)，value (N,)
  2. 分块推理（batch > PREDICT_BATCH）与整批推理结果一致
  3. Predictor 输出与直接调用模型一致
  4. 热重载：model_path 文件 mtime 变化后 Predictor 重载新权重
  5. 输入 board 形状 (N,16,4,8)、scalars (N,35)

运行：python python/validate/validate_predictor.py
"""

from __future__ import annotations

import os
import time

import numpy as np
import torch

import validate_common  # noqa: F401
from validate_common import DEVICE, Reporter, run_part

from config import config
from constant import ACTION_SPACE_SIZE, BOARD_ROWS, BOARD_COLS, SCALAR_FEATURE_COUNT, TOTAL_INPUT_CHANNELS
from nn_model import BanqiNet
from self_play import Predictor


def _dummy_inputs(batch: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    board = rng.random((batch, TOTAL_INPUT_CHANNELS, BOARD_ROWS, BOARD_COLS), dtype=np.float32)
    scalars = rng.random((batch, SCALAR_FEATURE_COUNT), dtype=np.float32)
    return board, scalars


def test_shapes_and_range() -> None:
    rep = Reporter("Predictor shapes/range")
    model = BanqiNet().to(DEVICE)
    pred = Predictor(model, DEVICE, model_path=None)
    for batch in [1, 4, 33, 64]:
        board, scalars = _dummy_inputs(batch)
        logits, values = pred(board, scalars)
        rep.check(tuple(logits.shape) == (batch, ACTION_SPACE_SIZE),
                  f"batch={batch} policy shape {tuple(logits.shape)}")
        rep.check(tuple(values.shape) == (batch,),
                  f"batch={batch} value shape {tuple(values.shape)}")
        rep.check(np.isfinite(logits).all(), f"batch={batch} logits finite")
        rep.check(np.isfinite(values).all(), f"batch={batch} values finite")
    rep.summary()


def test_chunk_consistency() -> None:
    rep = Reporter("Predictor chunk consistency")
    model = BanqiNet().to(DEVICE)
    pred = Predictor(model, DEVICE, model_path=None)
    batch = 70  # > PREDICT_BATCH=32，且不能被整除
    board, scalars = _dummy_inputs(batch, seed=1)

    logits_chunk, values_chunk = pred(board, scalars)

    # 与整批直接调用模型对比
    model.eval()
    with torch.no_grad():
        b = torch.from_numpy(board).to(DEVICE)
        s = torch.from_numpy(scalars).to(DEVICE)
        full_logits, full_value = model(b, s)
    full_logits = full_logits.cpu().numpy()
    full_value = full_value.cpu().numpy().reshape(-1)

    rep.check(np.allclose(logits_chunk, full_logits, atol=1e-5),
              "chunked policy == full-batch policy")
    rep.check(np.allclose(values_chunk, full_value, atol=1e-5),
              "chunked value == full-batch value")
    rep.summary()


def test_hot_reload() -> None:
    """验证 mtime 驱动的热重载：更换模型文件内容后 Predictor 自动更新。"""
    rep = Reporter("Predictor hot reload")
    # 用临时模型文件
    import tempfile
    tmp_dir = tempfile.mkdtemp(prefix="validate_pred_")
    model_path = os.path.join(tmp_dir, "weights.pth")
    try:
        # 初始权重
        model_a = BanqiNet().to(DEVICE)
        with torch.no_grad():
            for p in model_a.parameters():
                p.fill_(0.01)
        torch.save(model_a.state_dict(), model_path)

        # 构建 Predictor，加载模型 a
        pred_model = BanqiNet().to(DEVICE)
        pred = Predictor(pred_model, DEVICE, model_path=model_path)
        board, scalars = _dummy_inputs(4, seed=3)

        # 更新文件（改变权重）并强制 mtime 不同
        time.sleep(0.05)
        model_b = BanqiNet().to(DEVICE)
        with torch.no_grad():
            for p in model_b.parameters():
                p.fill_(0.99)
        torch.save(model_b.state_dict(), model_path)

        # 重新调用应触发热重载
        logits_reloaded, _ = pred(board, scalars)

        # 对比 model_b 的输出
        model_b.eval()
        with torch.no_grad():
            b = torch.from_numpy(board).to(DEVICE)
            s = torch.from_numpy(scalars).to(DEVICE)
            exp_logits, _ = model_b(b, s)
        exp_logits = exp_logits.cpu().numpy()
        rep.check(np.allclose(logits_reloaded, exp_logits, atol=1e-5),
                  "hot-reloaded weights == new model output")
    finally:
        # 清理临时文件
        if os.path.exists(model_path):
            os.remove(model_path)
        try:
            os.rmdir(tmp_dir)
        except OSError:
            pass
    rep.summary()


def test_hot_reload_torchscript() -> None:
    """回归：Predictor 必须能加载 TorchScript .pt（training_service trace 产物）。

    曾因 PyTorch 2.6+ torch.load(weights_only=True) 无法加载 TorchScript 归档，
    异常被吞掉导致 Predictor 一直用随机权重（训练闭环断裂）。
    """
    rep = Reporter("Predictor hot reload (TorchScript .pt)")
    import tempfile
    tmp_dir = tempfile.mkdtemp(prefix="validate_pred_ts_")
    model_path = os.path.join(tmp_dir, "weights.pt")
    try:
        # 用与 training_service 相同的方式生成 TorchScript 归档
        model_a = BanqiNet().to(DEVICE)
        with torch.no_grad():
            for p in model_a.parameters():
                p.fill_(0.01)
        model_a.eval()
        ex_board = torch.randn(1, TOTAL_INPUT_CHANNELS, BOARD_ROWS, BOARD_COLS, device=DEVICE)
        ex_scalars = torch.randn(1, SCALAR_FEATURE_COUNT, device=DEVICE)
        traced = torch.jit.trace(model_a, (ex_board, ex_scalars))
        traced.save(model_path)

        pred_model = BanqiNet().to(DEVICE)
        pred = Predictor(pred_model, DEVICE, model_path=model_path)
        board, scalars = _dummy_inputs(4, seed=7)
        logits, _ = pred(board, scalars)

        with torch.no_grad():
            b = torch.from_numpy(board).to(DEVICE)
            s = torch.from_numpy(scalars).to(DEVICE)
            exp_logits, _ = model_a(b, s)
        exp_logits = exp_logits.cpu().numpy()
        rep.check(np.allclose(logits, exp_logits, atol=1e-5),
                  "TorchScript .pt loaded and matches traced model")
    finally:
        if os.path.exists(model_path):
            os.remove(model_path)
        try:
            os.rmdir(tmp_dir)
        except OSError:
            pass
    rep.summary()


def test_degraded_no_torch() -> None:
    """验证无 torch 时的退化路径（均匀 logits + 0 值）——通过 monkeypatch 模拟。"""
    rep = Reporter("Predictor degraded (no torch)")
    import unittest.mock as mock
    model = BanqiNet().to(DEVICE)
    pred = Predictor(model, DEVICE, model_path=None)
    board, scalars = _dummy_inputs(4, seed=4)
    with mock.patch("self_play.HAS_TORCH", False):
        logits, values = pred(board, scalars)
    rep.check(np.all(logits == 0.0), "degraded logits all zero")
    rep.check(np.all(values == 0.0), "degraded values all zero")
    rep.summary()


def main() -> None:
    run_part("predictor: shapes/range", test_shapes_and_range)
    run_part("predictor: chunk consistency", test_chunk_consistency)
    run_part("predictor: hot reload", test_hot_reload)
    run_part("predictor: hot reload TorchScript", test_hot_reload_torchscript)
    run_part("predictor: degraded no-torch", test_degraded_no_torch)


if __name__ == "__main__":
    main()
