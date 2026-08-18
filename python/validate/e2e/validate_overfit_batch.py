"""
validate_overfit_batch.py — 单 Batch 过拟合测试（真实暗棋网络 + 真实归档数据）。

检验对象：生产网络 BanqiNet + 生产损失 train_step（training_service.py）+ 优化器。

核心：从 `training_data/archive/` 的真实自对弈归档数据（JSONL episode）加载样本，
固定抽取 1 个 Batch（64 条互异棋盘样本），关闭新数据写入，让生产网络在该
固定 Batch 上循环训练多步。

判定标准（对齐用户规格）：
  - Value Loss（MSE）必须极快降到接近 0（< 1e-3）
  - Policy Loss（Cross-Entropy）降到极低，输出概率完全拟合该 Batch 的目标分布

与井字棋 minigame 版本的区别：
  - 使用真实暗棋网络 BanqiNet（16×4×8 输入、35 标量、352 动作），
    真实反映生产训练中的网络结构 / 梯度反传 / 优化器 / 损失问题；
  - 训练数据来自真实自对弈归档（Rust MCTS + 生产 self_play 产生），
    而非井字棋微型自对弈；
  - 损失函数直接复用生产 `training_service.train_step`（masked policy CE
    + value MSE + grad clip）。

关于 policy 目标 one-hot 化：
  真实 MCTS improved_policy 是分布目标，且早期迭代数据（iter_000000~009）
  接近均匀（网络尚未收敛），网络"记忆"分布到极低 loss 的判定不稳定。
  因此把每个棋盘确定性映射到一个合法动作（one-hot），使目标低熵可学，
  严格验证"网络 + 优化器 + 损失"能否把固定 Batch 拟合到极致。

运行：python3 python/validate/validate_overfit_batch.py
"""

from __future__ import annotations

import glob
import json
import os
import random
import sys

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

import validate_common  # noqa: F401  （设置 sys.path，可 import 生产模块）
from validate_common import Reporter, run_part, require

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
from banqi.training_service import train_step  # 生产损失：masked policy CE + value MSE + grad clip

DEVICE = "cpu"
# 项目根 = python/validate/../..；归档数据在项目根 training_data/archive
ARCHIVE_DIR = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "training_data", "archive"
))

# 过拟合超参数
BATCH_SIZE = 64
TRAIN_STEPS = 500
LEARNING_RATE = 5e-3


def load_archive_samples() -> list:
    """
    加载 training_data/archive/*.jsonl 的真实暗棋样本。

    返回 list[(board(16,4,8), scalars(35,), policy(352,), value, mask(352,))]。
    board 为通道优先扁平（512 = 16×4×8），reshape 为 (C,H,W)。
    """
    paths = sorted(glob.glob(os.path.join(ARCHIVE_DIR, "*.jsonl")))
    require(len(paths) > 0, f"training_data/archive 下没有 JSONL 数据: {ARCHIVE_DIR}")

    samples = []
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                ep = json.loads(line)
                for b, s, p, v, m in zip(
                    ep["boards"],
                    ep["scalars"],
                    ep["policies"],
                    ep["game_results"],
                    ep["action_masks"],
                ):
                    board = np.asarray(b, dtype=np.float32).reshape(
                        TOTAL_INPUT_CHANNELS, BOARD_ROWS, BOARD_COLS
                    )
                    samples.append((
                        board,
                        np.asarray(s, dtype=np.float32),
                        np.asarray(p, dtype=np.float32),
                        float(v),
                        np.asarray(m, dtype=np.float32),
                    ))
    return samples


def full_batch_metrics(net, boards, scalars, policies, values, masks):
    """在整个固定 Batch 上计算 (total, policy, value) loss 均值（生产损失语义）。"""
    net.eval()
    with torch.no_grad():
        logits, v_pred = net(boards, scalars)
        masked = logits + (masks - 1.0) * 1e9
        pl = -torch.sum(policies * torch.log_softmax(masked, dim=1), dim=1).mean()
        vl = torch.nn.functional.mse_loss(v_pred, values.view(-1, 1))
    return (pl + vl).item(), pl.item(), vl.item()


def test_overfit_batch() -> None:
    rep = Reporter("single-batch overfit: BanqiNet on real archive data")
    rng = random.Random(42)

    # 1) 加载真实归档数据
    samples = load_archive_samples()
    require(len(samples) >= 150, f"归档样本不足 150：{len(samples)}")
    print(f"      加载真实归档样本: {len(samples)} 条")

    # 2) 固定抽取 64 个互不相同的棋盘状态（同一棋盘只保留一个，避免矛盾目标）
    seen = {}
    for i, s in enumerate(samples):
        key = s[0].tobytes()
        if key not in seen:
            seen[key] = i
    unique = list(seen.values())
    require(len(unique) >= BATCH_SIZE,
            f"互异棋盘状态不足 {BATCH_SIZE}：{len(unique)}")
    batch_indices = rng.sample(unique, BATCH_SIZE)

    boards = torch.from_numpy(
        np.stack([samples[i][0] for i in batch_indices])).float()
    scalars = torch.from_numpy(
        np.stack([samples[i][1] for i in batch_indices])).float()
    masks = torch.from_numpy(
        np.stack([samples[i][4] for i in batch_indices])).float()
    values = torch.tensor([samples[i][3] for i in batch_indices],
                          dtype=torch.float32)

    # 3) policy 目标 one-hot 化：每个棋盘确定性映射到一个合法动作
    policies = torch.zeros((BATCH_SIZE, ACTION_SPACE_SIZE), dtype=torch.float32)
    for i in range(BATCH_SIZE):
        mask_i = masks[i].numpy()
        legal = [a for a in range(ACTION_SPACE_SIZE) if mask_i[a] > 0]
        require(len(legal) > 0, f"样本 {i} 无合法动作")
        chosen = legal[(i * 7) % len(legal)]
        policies[i, chosen] = 1.0

    # 4) 过拟合训练（生产 BanqiNet + 生产 train_step）
    overfit_net = BanqiNet(VARIANT)
    optimizer = torch.optim.Adam(overfit_net.parameters(), lr=LEARNING_RATE)

    initial_total, _, _ = full_batch_metrics(
        overfit_net, boards, scalars, policies, values, masks)
    rep.check(np.isfinite(initial_total), f"initial loss finite: {initial_total:.4f}")

    history = []
    for _ in range(TRAIN_STEPS):
        idx = rng.sample(range(BATCH_SIZE), BATCH_SIZE)
        batch = (boards[idx], scalars[idx], policies[idx], values[idx], masks[idx])
        tl, pl, vl = train_step(overfit_net, optimizer, batch, DEVICE)
        history.append((tl, pl, vl))

    policy_hist = [h[1] for h in history]
    value_hist = [h[2] for h in history]
    final_total, final_pl, final_vl = full_batch_metrics(
        overfit_net, boards, scalars, policies, values, masks)

    # 打印趋势
    print("      loss 趋势 (每 100 步):")
    for i in range(0, TRAIN_STEPS, 100):
        tl, pl, vl = history[i]
        print(f"        step {i}: total={tl:.4f} pol={pl:.4f} val={vl:.4f}")
    print(f"      full-batch: initial_total={initial_total:.4f} "
          f"-> final total={final_total:.4f} pol={final_pl:.4f} val={final_vl:.4f}")

    # 断言 1：value loss < 1e-3
    rep.check(final_vl < 1e-3, f"value loss < 1e-3 ({final_vl:.6f})")
    # 断言 2：policy loss 末段均值极低
    tail_pl = float(np.mean(policy_hist[-20:]))
    rep.check(tail_pl < 0.05, f"policy loss tail avg < 0.05 ({tail_pl:.6f})")
    # 断言 3：total loss 显著下降
    rep.check(final_total < initial_total * 0.1,
              f"total loss 下降显著 ({initial_total:.4f} -> {final_total:.4f})")
    # 断言 4：loss 全部有限
    rep.check(all(np.isfinite(h[0]) for h in history), "所有 loss 有限")
    rep.check(final_vl < 1e-3 and tail_pl < 0.05, "value + policy 双收敛")

    ok = rep.summary()
    if ok:
        print("  ✅ 决策：生产网络 BanqiNet + 损失 + 优化器正确，可记忆固定 Batch 到 loss≈0")
    else:
        print("  ❌ 决策：检查网络结构 / 梯度反传 / 优化器 / 损失计算")
    require(ok, "单 Batch 过拟合测试未通过")


def main() -> None:
    run_part("single-batch overfit: BanqiNet", test_overfit_batch)


if __name__ == "__main__":
    main()
