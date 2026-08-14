"""
validate_model.py — 验证 BanqiNet 前向 / 反向传播（纯 CPU）。

检查项：
  1. 输出形状：policy (N, 352)，value (N, 1)
  2. value ∈ [-1, 1]（tanh 激活）
  3. scalar 拼接维度：policy_fc_input == 4*4*8 + 35，value_fc_input == 4*4*8 + 35
  4. 单步 forward + backward 后所有参数梯度存在
  5. 不同 batch 大小（含 > PREDICT_BATCH 的分块推理一致性）形状稳定

运行：python python/validate/validate_model.py
"""

from __future__ import annotations

import sys

import numpy as np
import torch

# 先 import validate_common 以设置 sys.path（加入 python/ 父目录）
import validate_common  # noqa: F401
from validate_common import DEVICE, Reporter, run_part

from constant import (
    ACTION_SPACE_SIZE,
    BOARD_ROWS,
    BOARD_COLS,
    SCALAR_FEATURE_COUNT,
    TOTAL_INPUT_CHANNELS,
)
from nn_model import BanqiNet


def test_shapes() -> None:
    rep = Reporter("model shapes")
    model = BanqiNet().to(DEVICE)
    model.eval()

    for batch in [1, 4, 33, 64]:  # 含 > PREDICT_BATCH=32 的 batch
        rng = np.random.default_rng(0)
        board = torch.from_numpy(
            rng.random((batch, TOTAL_INPUT_CHANNELS, BOARD_ROWS, BOARD_COLS),
                       dtype=np.float32)
        ).to(DEVICE)
        scalars = torch.from_numpy(
            rng.random((batch, SCALAR_FEATURE_COUNT), dtype=np.float32)
        ).to(DEVICE)
        with torch.no_grad():
            logits, value = model(board, scalars)
        rep.check(tuple(logits.shape) == (batch, ACTION_SPACE_SIZE),
                  f"batch={batch} policy shape {tuple(logits.shape)}")
        rep.check(tuple(value.shape) == (batch, 1),
                  f"batch={batch} value shape {tuple(value.shape)}")
        rep.check(bool(torch.isfinite(logits).all()),
                  f"batch={batch} logits finite")
        rep.check(bool(torch.isfinite(value).all()),
                  f"batch={batch} value finite")

    rep.summary()


def test_value_range() -> None:
    rep = Reporter("model value range")
    model = BanqiNet().to(DEVICE)
    model.eval()
    rng = np.random.default_rng(1)
    board = torch.from_numpy(
        rng.random((16, TOTAL_INPUT_CHANNELS, BOARD_ROWS, BOARD_COLS),
                   dtype=np.float32)
    ).to(DEVICE)
    scalars = torch.from_numpy(
        rng.random((16, SCALAR_FEATURE_COUNT), dtype=np.float32)
    ).to(DEVICE)
    with torch.no_grad():
        _, value = model(board, scalars)
    rep.check(bool((value >= -1.0).all()) and bool((value <= 1.0).all()),
              f"value within [-1,1]: min={value.min().item():.4f}, "
              f"max={value.max().item():.4f}")
    rep.summary()


def test_scalar_concat() -> None:
    rep = Reporter("scalar concat dimension")
    model = BanqiNet()
    policy_flat = model.policy_flat_size
    value_flat = model.value_flat_size
    rep.check(policy_flat == 4 * BOARD_ROWS * BOARD_COLS,
              f"policy_flat_size == {4 * BOARD_ROWS * BOARD_COLS} (got {policy_flat})")
    rep.check(value_flat == 4 * BOARD_ROWS * BOARD_COLS,
              f"value_flat_size == {4 * BOARD_ROWS * BOARD_COLS} (got {value_flat})")
    rep.check(model.policy_fc_input == policy_flat + SCALAR_FEATURE_COUNT,
              f"policy_fc_input == {policy_flat} + {SCALAR_FEATURE_COUNT}")
    rep.check(model.value_fc_input == value_flat + SCALAR_FEATURE_COUNT,
              f"value_fc_input == {value_flat} + {SCALAR_FEATURE_COUNT}")
    # 验证 forward 时拼接确实是 (flat+scalar)
    rep.check(model.policy_fc1.in_features == policy_flat + SCALAR_FEATURE_COUNT,
              "policy_fc1.in_features matches")
    rep.check(model.value_fc1.in_features == value_flat + SCALAR_FEATURE_COUNT,
              "value_fc1.in_features matches")
    rep.summary()


def test_backward_gradients() -> None:
    rep = Reporter("backward gradients")
    model = BanqiNet().to(DEVICE)
    model.train()
    rng = np.random.default_rng(2)
    batch = 8
    board = torch.from_numpy(
        rng.random((batch, TOTAL_INPUT_CHANNELS, BOARD_ROWS, BOARD_COLS),
                   dtype=np.float32)
    ).to(DEVICE)
    scalars = torch.from_numpy(
        rng.random((batch, SCALAR_FEATURE_COUNT), dtype=np.float32)
    ).to(DEVICE)

    logits, value = model(board, scalars)
    # 用均匀策略目标 + 0 值目标合成一个可导损失
    target = torch.full((batch, ACTION_SPACE_SIZE), 1.0 / ACTION_SPACE_SIZE,
                        device=DEVICE)
    loss = torch.nn.functional.cross_entropy(logits, target) + torch.nn.functional.mse_loss(
        value, torch.zeros_like(value)
    )
    loss.backward()

    n_params = 0
    n_with_grad = 0
    for name, p in model.named_parameters():
        n_params += 1
        if p.grad is not None and bool(torch.isfinite(p.grad).all()):
            n_with_grad += 1
        else:
            print(f"      missing/bad grad: {name}")
    rep.check(n_with_grad == n_params,
              f"all {n_params} params have finite grad ({n_with_grad})")
    rep.check(bool(torch.isfinite(loss)), f"loss finite: {loss.item():.6f}")
    rep.summary()


def test_batch_chunk_consistency() -> None:
    """分块推理结果与整批推理一致（对应 Predictor 的 chunk 逻辑）。"""
    rep = Reporter("batch chunk consistency")
    model = BanqiNet().to(DEVICE)
    model.eval()
    rng = np.random.default_rng(3)
    batch = 37
    board = rng.random((batch, TOTAL_INPUT_CHANNELS, BOARD_ROWS, BOARD_COLS),
                       dtype=np.float32)
    scalars = rng.random((batch, SCALAR_FEATURE_COUNT), dtype=np.float32)

    with torch.no_grad():
        b = torch.from_numpy(board).to(DEVICE)
        s = torch.from_numpy(scalars).to(DEVICE)
        logits_full, value_full = model(b, s)
        logits_full = logits_full.cpu().numpy()
        value_full = value_full.cpu().numpy().reshape(-1)

    # 手工分块 32
    chunk = 32
    pieces_p, pieces_v = [], []
    with torch.no_grad():
        for i in range(0, batch, chunk):
            lp, lv = model(torch.from_numpy(board[i:i + chunk]).to(DEVICE),
                           torch.from_numpy(scalars[i:i + chunk]).to(DEVICE))
            pieces_p.append(lp.cpu().numpy())
            pieces_v.append(lv.cpu().numpy().reshape(-1))
    logits_chunk = np.concatenate(pieces_p, axis=0)
    value_chunk = np.concatenate(pieces_v, axis=0)

    rep.check(np.allclose(logits_full, logits_chunk, atol=1e-5),
              "chunked policy == full-batch policy")
    rep.check(np.allclose(value_full, value_chunk, atol=1e-5),
              "chunked value == full-batch value")
    rep.summary()


def main() -> None:
    run_part("model: shapes", test_shapes)
    run_part("model: value range", test_value_range)
    run_part("model: scalar concat", test_scalar_concat)
    run_part("model: backward gradients", test_backward_gradients)
    run_part("model: batch chunk consistency", test_batch_chunk_consistency)


if __name__ == "__main__":
    main()
