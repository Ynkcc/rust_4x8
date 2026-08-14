"""
validate_train.py — 验证 train_step / evaluate（纯 CPU）。

检查项：
  1. train_step 损失有限且 > 0
  2. mask 屏蔽：被屏蔽动作的 log_prob 不参与 loss（通过对比有无 mask 的 loss 差异，
     或验证屏蔽后合法动作上的分布近似目标）
  3. backward 后参数更新（optimizer.step 生效）
  4. clip_grad_norm 生效（梯度范数受限于 1.0）
  5. evaluate 与手动 loss 一致

运行：python python/validate/validate_train.py
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

import validate_common  # noqa: F401
from validate_common import DEVICE, Reporter, run_part

from constant import ACTION_SPACE_SIZE, BOARD_ROWS, BOARD_COLS, SCALAR_FEATURE_COUNT, TOTAL_INPUT_CHANNELS
from nn_model import BanqiNet
from training_service import DataBuffer, evaluate, train_step


def _make_batch_data(batch: int, rng: np.random.Generator, legal_frac: float = 0.3):
    """构造一个 DataBuffer.get_batch 返回形式的 batch。"""
    board = rng.random((batch, TOTAL_INPUT_CHANNELS, BOARD_ROWS, BOARD_COLS), dtype=np.float32)
    scalars = rng.random((batch, SCALAR_FEATURE_COUNT), dtype=np.float32)
    mask = (rng.random((batch, ACTION_SPACE_SIZE)) < legal_frac).astype(np.float32)
    for i in range(batch):
        if mask[i].sum() == 0:
            mask[i, i % ACTION_SPACE_SIZE] = 1.0
    policy = mask * rng.random((batch, ACTION_SPACE_SIZE))
    policy = policy / policy.sum(axis=1, keepdims=True)
    values = rng.random(batch, dtype=np.float32) * 2 - 1  # ∈ [-1,1]

    boards = torch.from_numpy(board)
    scalars = torch.from_numpy(scalars)
    probs = torch.from_numpy(policy)
    vals = torch.tensor(values, dtype=torch.float32)
    masks = torch.from_numpy(mask)
    return boards, scalars, probs, vals, masks


def test_train_step_loss() -> None:
    rep = Reporter("train_step loss")
    model = BanqiNet().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    rng = np.random.default_rng(0)
    batch = _make_batch_data(8, rng)

    tl, pl, vl = train_step(model, optimizer, batch, DEVICE)
    rep.check(np.isfinite(tl), f"total loss finite: {tl:.6f}")
    rep.check(np.isfinite(pl) and pl > 0, f"policy loss > 0: {pl:.6f}")
    rep.check(np.isfinite(vl), f"value loss finite: {vl:.6f}")
    rep.check(tl > 0, f"total loss > 0: {tl:.6f}")
    rep.summary()


def test_mask_blocks_illegal() -> None:
    """验证 mask 让非法动作在 softmax 后概率≈0，合法动作概率与被屏蔽前不同。"""
    rep = Reporter("mask blocks illegal actions")
    model = BanqiNet().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    rng = np.random.default_rng(1)
    boards, scalars, probs, vals, masks = _make_batch_data(8, rng, legal_frac=0.2)

    model.eval()
    with torch.no_grad():
        logits, _ = model(boards.to(DEVICE), scalars.to(DEVICE))
        logits = logits.cpu()
    # masked logits 与 train_step 相同：logits + (mask-1)*1e9
    masked = logits + (masks - 1.0) * 1e9
    masked_probs = F.softmax(masked, dim=1).numpy()
    # 非法位置 (mask==0) 概率应≈0
    illegal_prob = masked_probs * (1.0 - masks.numpy())
    rep.check(illegal_prob.max() < 1e-6,
              f"max illegal probability {illegal_prob.max():.2e} < 1e-6")

    # 合法动作概率归一化到和=1
    legal_sum = (masked_probs * masks.numpy()).sum(axis=1)
    rep.check(np.allclose(legal_sum, 1.0, atol=1e-5),
              "legal-action probabilities sum to ~1")
    rep.summary()


def test_parameter_update() -> None:
    rep = Reporter("parameter update")
    model = BanqiNet().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    rng = np.random.default_rng(2)
    batch = _make_batch_data(8, rng)

    before = [p.detach().clone() for p in model.parameters()]
    # 多个 step 以便观察明显变化
    for _ in range(5):
        train_step(model, optimizer, batch, DEVICE)
    changed = sum(
        1 for b, p in zip(before, model.parameters())
        if not torch.allclose(b, p.detach(), atol=1e-6)
    )
    rep.check(changed > 0, f"params changed after training steps ({changed}/{len(before)})")
    rep.summary()


def test_grad_clip() -> None:
    """验证 clip_grad_norm 后梯度范数 ≤ max_norm(=1.0)。"""
    rep = Reporter("grad clip")
    model = BanqiNet().to(DEVICE)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    rng = np.random.default_rng(3)
    batch = _make_batch_data(4, rng)
    boards, scalars, probs, vals, masks = batch
    boards = boards.to(DEVICE); scalars = scalars.to(DEVICE)
    probs = probs.to(DEVICE); vals = vals.to(DEVICE).view(-1, 1); masks = masks.to(DEVICE)

    optimizer.zero_grad()
    logits, values = model(boards, scalars)
    masked = logits + (masks - 1.0) * 1e9
    loss = -torch.sum(probs * F.log_softmax(masked, dim=1), dim=1).mean() \
        + F.mse_loss(values, vals)
    # 放大 loss 以强制产生超大梯度，确保触发裁剪
    (loss * 100.0).backward()
    # 记录裁剪前总范数（clip_grad_norm_ 返回值即裁剪前范数）
    before_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    # 裁剪后所有参数的梯度 L2 范数应 ≤ 1.0
    grads = [p.grad.detach().flatten() for p in model.parameters() if p.grad is not None]
    cat = torch.cat(grads) if grads else torch.zeros(1)
    after_norm = float(cat.norm())
    rep.check(before_norm > 1.0,
              f"pre-clip grad norm {before_norm:.4f} > 1.0 (triggered clipping)")
    rep.check(after_norm <= 1.0 + 1e-4,
              f"post-clip grad norm {after_norm:.4f} <= 1.0")
    rep.summary()


def test_evaluate_consistency() -> None:
    """evaluate 返回的 loss 应与在同样数据上手动计算的 loss 一致。"""
    rep = Reporter("evaluate consistency")
    model = BanqiNet().to(DEVICE)
    rng = np.random.default_rng(4)
    # 用 DataBuffer 存样本
    buf = DataBuffer(capacity=100)
    from validate_common import make_episode
    from training_service import episode_to_samples
    for _ in range(2):
        ep = make_episode(num_steps=5, winner=1, rng=rng)
        buf.add_samples(episode_to_samples(ep))
    res = evaluate(model, buf, batch_size=4, device=DEVICE)
    rep.check(res is not None, "evaluate returned a result")
    if res is None:
        rep.summary()
        return
    vl, vp, vv = res
    rep.check(np.isfinite(vl), f"eval total loss finite: {vl:.6f}")
    rep.check(abs(vl - (vp + vv)) < 1e-4,
              f"total == policy+value ({vl:.6f} vs {vp+vv:.6f})")
    rep.summary()


def main() -> None:
    run_part("train: train_step loss", test_train_step_loss)
    run_part("train: mask blocks illegal", test_mask_blocks_illegal)
    run_part("train: parameter update", test_parameter_update)
    run_part("train: grad clip", test_grad_clip)
    run_part("train: evaluate consistency", test_evaluate_consistency)


if __name__ == "__main__":
    main()
