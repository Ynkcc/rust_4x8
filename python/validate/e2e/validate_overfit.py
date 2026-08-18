"""
validate_overfit.py — 过拟合回归（纯 CPU）。

核心：用一份固定的小数据集反复训练，验证 total loss 能持续下降并逼近 0。
这同时验证了反向传播、优化器、调度器以及掩码/损失计算的端到端正确性——
如果这些环节有 bug，loss 将无法在固定数据上收敛。

用 train_step 逐 batch 更新（模拟真实训练），记录 loss 历史，
再用 validate_common.assert_converged 断言收敛。

运行：python python/validate/validate_overfit.py
"""

from __future__ import annotations

import numpy as np
import torch

import os
import sys

_VALIDATE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PYTHON_DIR = os.path.dirname(_VALIDATE_DIR)
_BANQI_DIR = os.path.join(_PYTHON_DIR, "banqi")
for _d in (_PYTHON_DIR, _BANQI_DIR, _VALIDATE_DIR):
    if _d not in sys.path:
        sys.path.insert(0, _d)

import validate_common  # noqa: F401
from validate_common import DEVICE, Reporter, assert_converged, run_part

from banqi.variant import get_variant
from banqi.constants import build_constants
from banqi.nn_model import BanqiNet

VARIANT = get_variant("4x8")
C = build_constants(VARIANT)
ACTION_SPACE_SIZE = C.ACTION_SPACE_SIZE
BOARD_ROWS = C.BOARD_ROWS
BOARD_COLS = C.BOARD_COLS
SCALAR_FEATURE_COUNT = C.SCALAR_FEATURE_COUNT
TOTAL_INPUT_CHANNELS = C.TOTAL_INPUT_CHANNELS
from banqi.training_service import DataBuffer, episode_to_samples, train_step


def _build_fixed_dataset(n_episodes: int = 8, steps_per_ep: int = 4, seed: int = 7):
    """
    构造固定小数据集（确定性、可学习）。

    注意：过拟合回归要求 policy 目标是与输入相关的**可学习映射**，否则（纯随机
    目标）网络无法记忆，loss 无法收敛。因此这里把每个样本的 policy 目标设为
    确定性 one-hot（每步固定在某个合法动作上），value 目标设为确定性 ±1。
    这样网络有足够容量即可在固定数据上把 loss 记忆到接近 0，从而验证
    反向传播 + 优化器 + 调度器端到端正确。
    """
    from validate_common import make_episode
    rng = np.random.default_rng(seed)
    buf = DataBuffer(capacity=100000)
    # 确定性 one-hot 动作选择：每个样本选 (i*7+j) % 合法数 作为唯一合法动作
    for _ in range(n_episodes):
        ep = make_episode(steps_per_ep, winner=1, rng=rng)
        samples = episode_to_samples(ep)
        for s_idx, s in enumerate(samples):
            mask = np.asarray(s["action_mask"])
            legal = np.where(mask > 0)[0]
            chosen = legal[(s_idx * 7 + _) % max(1, len(legal))]
            onehot = np.zeros_like(mask)
            onehot[chosen] = 1.0
            s["policy_probs"] = onehot
            s["game_result_value"] = 1.0 if _ % 2 == 0 else -1.0
        buf.add_samples(samples)
    return buf


def _full_batch_metrics(model, buf):
    """在整个固定数据集上计算 (total, policy, value) loss 均值。"""
    import torch.nn.functional as F
    idx = list(range(len(buf)))
    boards, scalars, probs, vals, masks = buf.get_batch(idx)
    boards = boards.to(DEVICE); scalars = scalars.to(DEVICE)
    probs = probs.to(DEVICE); vals = vals.to(DEVICE).view(-1, 1); masks = masks.to(DEVICE)
    model.eval()
    with torch.no_grad():
        logits, values = model(boards, scalars)
        masked = logits + (masks - 1.0) * 1e9
        pl = -torch.sum(probs * F.log_softmax(masked, dim=1), dim=1).mean()
        vl = F.mse_loss(values, vals)
    return pl.item() + vl.item(), pl.item(), vl.item()


def test_overfit() -> None:
    rep = Reporter("overfit regression")
    rng = np.random.default_rng(123)
    buf = _build_fixed_dataset()

    model = BanqiNet(VARIANT).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)

    initial_total, _, _ = _full_batch_metrics(model, buf)
    rep.check(np.isfinite(initial_total), f"initial loss finite: {initial_total:.4f}")

    # 全量索引，打乱成 mini-batch 训练多个 step
    indices = list(range(len(buf)))
    num_steps = 300
    loss_history: list = []
    for step in range(num_steps):
        rng.shuffle(indices)
        tl_sum, cnt = 0.0, 0
        for i in range(0, len(indices), 8):
            batch_idx = indices[i:i + 8]
            batch = buf.get_batch(batch_idx)
            tl, _, _ = train_step(model, optimizer, batch, DEVICE)
            tl_sum += tl
            cnt += 1
        if cnt:
            loss_history.append(tl_sum / cnt)

    rep.check(len(loss_history) >= 20, f"collected {len(loss_history)} epoch-avg losses")
    # 输出趋势
    for i, v in enumerate(loss_history):
        if i % 25 == 0:
            print(f"      step~{i * 8}: loss={v:.4f}")
    final_total, _, _ = _full_batch_metrics(model, buf)
    print(f"      full-set total: initial={initial_total:.4f} -> final={final_total:.4f}")

    assert_converged(loss_history, threshold=1e-3, at_fraction=0.2)
    rep.check(final_total < 1e-2,
              f"full-set total loss < 1e-2 ({final_total:.4f})")
    rep.summary()


def main() -> None:
    run_part("overfit: convergence regression", test_overfit)


if __name__ == "__main__":
    main()
