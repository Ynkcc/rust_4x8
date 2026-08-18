"""
validate_checkpoint.py — 验证 save_checkpoint / load_checkpoint 持久化（纯 CPU）。

检查项：
  1. save_checkpoint 生成 .pth 与 .pt 两个文件
  2. .pth 包含 model/optimizer/scheduler 状态
  3. load_checkpoint 能恢复权重（前后一致）
  4. .pt 是 TorchScript，可被 torch.jit.load 加载，且输出与 Python 模型一致
  5. optimizer/scheduler 状态能恢复

注意：按用户要求，checkpoint 输出到**当前目录**（config 默认路径
banqi_model_latest.pt / .pth）。脚本会覆盖这些文件。可用环境变量
MODEL_PATH / STATE_DICT_PATH 覆盖。

运行：python python/validate/validate_checkpoint.py
"""

from __future__ import annotations

import os

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
from validate_common import DEVICE, Reporter, run_part

from banqi.config import make_config
from banqi.variant import get_variant
from banqi.constants import build_constants
from banqi.nn_model import BanqiNet
from banqi.checkpoint import load_checkpoint, save_checkpoint

VARIANT = get_variant("4x8")
config = make_config(VARIANT.id)
C = build_constants(VARIANT)
BOARD_ROWS = C.BOARD_ROWS
BOARD_COLS = C.BOARD_COLS
SCALAR_FEATURE_COUNT = C.SCALAR_FEATURE_COUNT
TOTAL_INPUT_CHANNELS = C.TOTAL_INPUT_CHANNELS
def save_ckpt(model, opt, sched):
    save_checkpoint(model, opt, sched, config.MODEL_PATH, config.STATE_DICT_PATH, torch.device(DEVICE), VARIANT)

def load_ckpt(model, opt, sched):
    return load_checkpoint(model, opt, sched, config.MODEL_PATH, config.STATE_DICT_PATH, torch.device(DEVICE), VARIANT)


def _rand_weights(rng: np.random.Generator):
    """为模型随机初始化一个确定性的权重种子差异（用于对比前后）。"""
    return rng.random(5, dtype=np.float32)


def _tensorish_equal(a, b) -> bool:
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        return torch.allclose(a, b)
    return a == b


def _optimizer_state_equal(sa, sb) -> bool:
    """递归比较两个 optimizer.state_dict() 是否数值等价。"""
    if type(sa) is not type(sb):
        return False
    if isinstance(sa, dict):
        if set(sa.keys()) != set(sb.keys()):
            return False
        return all(_optimizer_state_equal(sa[k], sb[k]) for k in sa)
    if isinstance(sa, list):
        if len(sa) != len(sb):
            return False
        return all(_optimizer_state_equal(x, y) for x, y in zip(sa, sb))
    return _tensorish_equal(sa, sb)


def test_files_created() -> None:
    rep = Reporter("checkpoint files created")
    pt, pth = config.MODEL_PATH, config.STATE_DICT_PATH
    print(f"      .pt  -> {os.path.abspath(pt)}")
    print(f"      .pth -> {os.path.abspath(pth)}")
    # 清理旧文件确保是本次生成
    for f in (pt, pth):
        if os.path.exists(f):
            os.remove(f)

    model = BanqiNet(VARIANT).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100, eta_min=1e-6)
    save_ckpt(model, opt, sched)

    rep.check(os.path.exists(pth), f".pth exists: {pth}")
    rep.check(os.path.exists(pt), f".pt exists: {pt}")
    rep.check(os.path.getsize(pth) > 0, ".pth non-empty")
    rep.check(os.path.getsize(pt) > 0, ".pt non-empty")
    rep.summary()


def test_state_dict_roundtrip() -> None:
    rep = Reporter("state_dict roundtrip")
    model = BanqiNet(VARIANT).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100, eta_min=1e-6)

    # 随机扰动一次参数，制造有区分度的权重
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p) * 0.1)
    before = {k: v.detach().clone() for k, v in model.state_dict().items()}

    save_ckpt(model, opt, sched)

    # 新模型 + 清空优化器后加载
    model2 = BanqiNet(VARIANT).to(DEVICE)
    opt2 = torch.optim.Adam(model2.parameters(), lr=config.LEARNING_RATE)
    sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=100, eta_min=1e-6)
    load_ckpt(model2, opt2, sched2)

    after = model2.state_dict()
    keys_equal = list(before.keys()) == list(after.keys())
    rep.check(keys_equal, "state_dict keys identical")
    vals_equal = all(
        torch.allclose(before[k], after[k]) for k in before
    )
    rep.check(vals_equal, "all weights restored exactly")
    rep.summary()


def test_torchscript_loadable() -> None:
    rep = Reporter("TorchScript loadable")
    pt = config.MODEL_PATH
    # 加载参考 Python 模型
    ref_model = BanqiNet(VARIANT).to(DEVICE)
    load_ckpt(ref_model, torch.optim.Adam(ref_model.parameters()),
                    torch.optim.lr_scheduler.CosineAnnealingLR(
                        torch.optim.Adam(ref_model.parameters()), T_max=100))
    ref_model.eval()

    jit_model = torch.jit.load(pt, map_location=DEVICE)
    jit_model.eval()

    rng = np.random.default_rng(0)
    batch = 4
    board = torch.from_numpy(
        rng.random((batch, TOTAL_INPUT_CHANNELS, BOARD_ROWS, BOARD_COLS), dtype=np.float32)
    ).to(DEVICE)
    scalars = torch.from_numpy(
        rng.random((batch, SCALAR_FEATURE_COUNT), dtype=np.float32)
    ).to(DEVICE)

    with torch.no_grad():
        ref_logits, ref_value = ref_model(board, scalars)
        jit_logits, jit_value = jit_model(board, scalars)

    rep.check(tuple(jit_logits.shape) == (batch, 352), "jit policy shape")
    rep.check(tuple(jit_value.shape) == (batch, 1), "jit value shape")
    rep.check(torch.allclose(jit_logits, ref_logits, atol=1e-5),
              "jit logits == python model logits")
    rep.check(torch.allclose(jit_value, ref_value, atol=1e-5),
              "jit value == python model value")
    rep.summary()


def test_optimizer_scheduler_restored() -> None:
    rep = Reporter("optimizer/scheduler restored")
    # 构造一个已训练的模型 + 优化器状态
    model = BanqiNet(VARIANT).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100, eta_min=1e-6)
    # 走几步让 optimizer/scheduler 内部状态变化
    rng = np.random.default_rng(1)
    b = torch.from_numpy(rng.random((4, TOTAL_INPUT_CHANNELS, BOARD_ROWS, BOARD_COLS),
                                    dtype=np.float32)).to(DEVICE)
    s = torch.from_numpy(rng.random((4, SCALAR_FEATURE_COUNT), dtype=np.float32)).to(DEVICE)
    for _ in range(5):
        opt.zero_grad()
        logits, value = model(b, s)
        loss = logits.mean() + value.mean()
        loss.backward()
        opt.step()
        sched.step()

    opt_state_before = opt.state_dict()
    save_ckpt(model, opt, sched)

    # 恢复
    model2 = BanqiNet(VARIANT).to(DEVICE)
    opt2 = torch.optim.Adam(model2.parameters(), lr=config.LEARNING_RATE)
    sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=100, eta_min=1e-6)
    load_ckpt(model2, opt2, sched2)
    opt_state_after = opt2.state_dict()

    state_equal = _optimizer_state_equal(opt_state_before, opt_state_after)
    rep.check(state_equal, "optimizer state restored")
    rep.check(abs(opt2.param_groups[0]['lr'] - opt.param_groups[0]['lr']) < 1e-12,
              "optimizer lr restored")
    rep.summary()


def main() -> None:
    print("[validate_checkpoint] 将覆盖 config 默认路径的 checkpoint 文件："
          f"{config.STATE_DICT_PATH} / {config.MODEL_PATH}")
    run_part("checkpoint: files created", test_files_created)
    run_part("checkpoint: state_dict roundtrip", test_state_dict_roundtrip)
    run_part("checkpoint: TorchScript loadable", test_torchscript_loadable)
    run_part("checkpoint: optimizer/scheduler restored", test_optimizer_scheduler_restored)


if __name__ == "__main__":
    main()
