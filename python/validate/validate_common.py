"""
validate_common.py — 训练逻辑验证的共享工具（纯 CPU）。

提供：
  - 确定性合成数据构造（board / scalars / policy / mask / episode）
  - 合成 episode dict 工厂（与 Rust episode_to_dict 契约一致）
  - 收敛性断言（loss 单调下降逼近阈值）
  - 统一 PASS/FAIL 输出封装与检查计数

仅被 python/validate/*.py 引用，不改动生产代码。
"""

from __future__ import annotations

import sys
import traceback
from typing import Callable, List, Optional, Tuple

import numpy as np

# 确保可从 python/ 目录 import 生产模块（config/constant/nn_model 等）。
import os

_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_BANQI_DIR = os.path.join(_PARENT, "banqi")
for _d in (_PARENT, _BANQI_DIR):
    if _d not in sys.path:
        sys.path.insert(0, _d)

from banqi.variant import get_variant  # noqa: E402
from banqi.constants import build_constants  # noqa: E402

VARIANT = get_variant("4x8")  # noqa: E402
C = build_constants(VARIANT)  # noqa: E402
ACTION_SPACE_SIZE = C.ACTION_SPACE_SIZE
BOARD_ROWS = C.BOARD_ROWS
BOARD_COLS = C.BOARD_COLS
SCALAR_FEATURE_COUNT = C.SCALAR_FEATURE_COUNT
TOTAL_INPUT_CHANNELS = C.TOTAL_INPUT_CHANNELS

# 强制 CPU，保证可复现
DEVICE = "cpu"


# ============================================================================
# 输出封装
# ============================================================================

class Reporter:
    """累计断言结果并统一打印 PASS / FAIL 摘要。"""

    def __init__(self, name: str) -> None:
        self.name = name
        self._checks: List[Tuple[bool, str]] = []

    def check(self, cond: bool, msg: str) -> bool:
        tag = "PASS" if cond else "FAIL"
        print(f"    [{tag}] {msg}")
        self._checks.append((cond, msg))
        return cond

    def summary(self) -> bool:
        passed = sum(1 for ok, _ in self._checks if ok)
        failed = len(self._checks) - passed
        print(f"  == {self.name}: {passed} PASS, {failed} FAIL ==")
        return failed == 0


def run_part(label: str, fn: Callable[[], None]) -> None:
    """运行一个分区函数，捕获异常并打印 PASS/FAIL。"""
    print(f"\n=== {label} ===")
    try:
        fn()
        print(f"  >>> {label} OK")
    except AssertionError as exc:  # noqa: F841
        print(f"  >>> {label} FAIL (assertion): {exc}")
        traceback.print_exc()
        sys.exit(1)
    except Exception as exc:  # noqa: BLE001
        print(f"  >>> {label} FAIL (exception): {exc}")
        traceback.print_exc()
        sys.exit(1)


def require(cond: bool, msg: str) -> None:
    """硬断言：失败直接抛 AssertionError 终止脚本。"""
    if not cond:
        raise AssertionError(msg)


# ============================================================================
# 确定性合成数据
# ============================================================================

def make_observation(
    batch: int,
    rng: np.random.Generator,
    seed_offset: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """合成一个 batch 的 (board, scalars)。"""
    _ = seed_offset
    board = rng.random(
        (batch, TOTAL_INPUT_CHANNELS, BOARD_ROWS, BOARD_COLS), dtype=np.float32
    )
    scalars = rng.random((batch, SCALAR_FEATURE_COUNT), dtype=np.float32)
    return board, scalars


def make_policy_mask(
    batch: int,
    rng: np.random.Generator,
    legal_frac: float = 0.3,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    合成 (policy_probs, action_mask)。
    policy 仅在合法动作上有正概率且归一化到和为 1；mask 0/1 与之对应。
    """
    mask = (rng.random((batch, ACTION_SPACE_SIZE)) < legal_frac).astype(np.float32)
    # 保证每行至少一个合法动作
    for i in range(batch):
        if mask[i].sum() == 0:
            mask[i, i % ACTION_SPACE_SIZE] = 1.0
    policy = mask * rng.random((batch, ACTION_SPACE_SIZE))
    policy = policy / policy.sum(axis=1, keepdims=True)
    return policy.astype(np.float32), mask.astype(np.float32)


def make_episode(
    num_steps: int,
    winner: int = 1,
    rng: Optional[np.random.Generator] = None,
) -> dict:
    """
    合成一个与 Rust `episode_to_dict` 契约一致的 episode dict。
    键与 src/py/mod.rs::episode_to_dict 保持一致：
      boards, scalars, policies, mcts_values, completed_qs, root_visits,
      game_results, action_masks, game_length, winner, num_samples
    外加 self_play.py 补充的 iteration / worker_id。
    """
    rng = rng or np.random.default_rng(0)
    boards, scalars = [], []
    policies, mcts_values, completed_qs, root_visits = [], [], [], []
    game_results, action_masks = [], []
    for step in range(num_steps):
        b = rng.random(
            (TOTAL_INPUT_CHANNELS, BOARD_ROWS, BOARD_COLS), dtype=np.float32
        )
        s = rng.random(SCALAR_FEATURE_COUNT, dtype=np.float32)
        p, m = make_policy_mask(1, rng, legal_frac=0.3)
        boards.append(b)
        scalars.append(s)
        policies.append(p[0])
        action_masks.append(m[0])
        mcts_values.append(float(rng.random()))
        completed_qs.append(float(rng.random()))
        root_visits.append(int(rng.integers(1, 100)))
        # 步进角度：value 逐步逼近最终结果（模拟折现/结果传播）
        game_results.append(float(winner) if step == num_steps - 1 else 0.0)

    return {
        "game_length": num_steps,
        "winner": winner,
        "num_samples": num_steps,
        "boards": boards,
        "scalars": scalars,
        "policies": policies,
        "mcts_values": mcts_values,
        "completed_qs": completed_qs,
        "root_visits": root_visits,
        "game_results": game_results,
        "action_masks": action_masks,
        "iteration": 0,
        "worker_id": 0,
    }


# ============================================================================
# 收敛断言
# ============================================================================

def assert_converged(
    loss_history: List[float],
    threshold: float = 1e-3,
    at_fraction: float = 0.2,
) -> None:
    """
    断言训练 loss 收敛：
      - loss 序列有限且非负
      - 最终(后 at_fraction 段平均) loss 低于 threshold
      - 整体趋势明显下降（末段均值显著小于初段均值）
    """
    require(len(loss_history) >= 10, f"loss 历史过短: {len(loss_history)}")
    require(all(np.isfinite(v) and v >= 0 for v in loss_history),
            "loss 出现 NaN / Inf / 负值")
    n = len(loss_history)
    k = max(1, int(n * at_fraction))
    final_avg = float(np.mean(loss_history[-k:]))
    head_avg = float(np.mean(loss_history[:k]))
    print(f"      loss: head_avg={head_avg:.5f}, final_avg={final_avg:.5f}")
    require(final_avg < threshold,
            f"loss 未收敛到 <{threshold}: final_avg={final_avg:.5f}")
    require(final_avg <= head_avg * 0.5 + 1e-6,
            f"loss 未显著下降: head_avg={head_avg:.5f}, final_avg={final_avg:.5f}")


if __name__ == "__main__":
    rep = Reporter("validate_common smoke")
    rep.check(ACTION_SPACE_SIZE == 352, "ACTION_SPACE_SIZE == 352")
    ep = make_episode(5, winner=-1)
    rep.check(len(ep["boards"]) == 5, "episode boards length == 5")
    rep.check(ep["boards"][0].shape == (TOTAL_INPUT_CHANNELS, BOARD_ROWS, BOARD_COLS),
              "board step shape OK")
    p, m = make_policy_mask(3, np.random.default_rng(1))
    rep.check(p.shape == (3, ACTION_SPACE_SIZE), "policy shape OK")
    rep.check(np.allclose(p.sum(axis=1), 1.0), "policy sums to 1")
    assert_converged([10.0, 5.0, 2.0, 1.0, 0.5, 0.1, 0.05, 0.01, 0.0005, 0.0001])
    rep.check(True, "converged assertion OK")
    ok = rep.summary()
    sys.exit(0 if ok else 1)
