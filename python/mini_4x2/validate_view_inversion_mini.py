"""
validate_view_inversion_mini.py — 4x2 迷你暗棋模型 视角与标签反转测试。

与 validate_view_inversion.py（4x8 暗棋）完全同构，但针对迷你环境/网络：
  - 环境：Rust `banqi_4x8.MiniDarkChess`（board 10 通道 / 4x2 / scalars 11）
  - 网络：`MiniBanqiNet`，直接加载已训练权重 `banqi_mini_model_latest.pt`
    （不重新训练）

Part A：Rust 环境 `switch_player()` 视角切换编码正确性（确定性断言）。
Part B：真实网络 value 视角对称性（标签反转）`v(视角B) ≈ -v(视角A)`
       （收集统计，暂不断言）。

关于 policy / mask 的说明：
  `flip_player` 仅切换 current_player（不改变棋盘），因此合法动作集随之变为
  对方的动作集（各自动自己的子），mask / policy 分布作用域不同，不能跨视角
  直接比较（连翻棋动作"翻格子 sq"的对象也从"我的子"变成"对方的子"，
  属不同绝对事件）。网络级唯一的视角对称性（标签反转）是 value 标量：
  同一绝对局面的胜率期望从两视角互为相反数。

运行：python3 python/mini_4x2/validate_view_inversion_mini.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

import banqi_4x8 as b  # pyo3 绑定（Rust 迷你暗棋环境）

# 本脚本位于 mini_4x2/ 目录，导入共享的 validate_common（位于 python/validate/）。
_HERE = os.path.dirname(os.path.abspath(__file__))
_VALIDATE_DIR = os.path.abspath(os.path.join(_HERE, "..", "validate"))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
if _VALIDATE_DIR not in sys.path:
    sys.path.insert(0, _VALIDATE_DIR)

import validate_common  # noqa: F401, E402
from validate_common import Reporter, run_part, require  # noqa: E402

from constant_mini import (  # noqa: E402
    BOARD_ROWS,
    BOARD_COLS,
    SCALAR_FEATURE_COUNT,
    TOTAL_INPUT_CHANNELS,
)
from nn_model_mini import MiniBanqiNet, load_model_weights  # noqa: E402

DEVICE = "cpu"
MODEL_PATH = os.path.join(_HERE, "banqi_mini_model_latest.pt")

# 随机走步数（生成评估局面）与局面数量
WALK_STEPS = 20
NUM_POSITIONS = 30


def load_real_model() -> MiniBanqiNet:
    """加载真实迷你暗棋网络权重到 MiniBanqiNet（TorchScript / state_dict 自动识别）。"""
    require(os.path.exists(MODEL_PATH), f"模型文件不存在: {MODEL_PATH}")
    model = MiniBanqiNet()
    load_model_weights(model, MODEL_PATH, torch.device(DEVICE))
    model.eval()
    return model


def _encode_obs(board_flat, scalars_flat):
    """观测扁平数组 → 网络输入 Tensor。"""
    board = torch.from_numpy(
        np.asarray(board_flat, dtype=np.float32)).reshape(1, TOTAL_INPUT_CHANNELS, BOARD_ROWS, BOARD_COLS)
    scalars = torch.from_numpy(np.asarray(scalars_flat, dtype=np.float32)).reshape(1, SCALAR_FEATURE_COUNT)
    return board, scalars


# ============================================================================
# Part A：Rust 环境视角切换正确性（确定性断言）
# ============================================================================

def test_env_switch_view() -> None:
    """验证：switch_player 后 my↔opp 通道互换、hidden/empty 不变、scalars 视角互换。"""
    rep = Reporter("view: mini Rust env switch_player encoding")
    ok_all = True
    for _ in range(20):
        env = b.MiniDarkChess()
        env.random_steps(20)
        b1, s1 = env.observation()
        env.switch_player()
        b2, s2 = env.observation()
        a = np.asarray(b1).reshape(TOTAL_INPUT_CHANNELS, BOARD_ROWS, BOARD_COLS)
        bb = np.asarray(b2).reshape(TOTAL_INPUT_CHANNELS, BOARD_ROWS, BOARD_COLS)
        s1a = np.asarray(s1)
        s2a = np.asarray(s2)

        # board: my(0-3) ↔ opp(4-7) 互换；hidden(8)/empty(9) 不变
        ok = (
            np.allclose(a[4:8], bb[0:4]) and np.allclose(a[0:4], bb[4:8])
            and np.allclose(a[8], bb[8]) and np.allclose(a[9], bb[9])
        )
        # scalars: move_count(0) 不变；my_hp(1)↔opp_hp(2)；
        # my_survival(3-6)↔opp_survival(7-10) 互换
        ok = ok and abs(s1a[0] - s2a[0]) < 1e-5
        ok = ok and abs(s1a[1] - s2a[2]) < 1e-5 and abs(s1a[2] - s2a[1]) < 1e-5
        ok = ok and np.allclose(s1a[3:7], s2a[7:11]) and np.allclose(s1a[7:11], s2a[3:7])
        if not ok:
            rep.check(False, f"第 {_} 个随机局面视角切换编码错误")
            ok_all = False
            break
    rep.check(ok_all, "20 个随机局面的视角切换编码全部正确（my/opp 互换，hidden/empty 不变）")
    rep.summary()


# ============================================================================
# Part B：真实网络 value 视角对称性（收集统计，暂不断言）
# ============================================================================

def test_network_value_symmetry() -> None:
    """加载真实模型，随机 20 步局面两视角推理，收集 value 对称性统计（暂不断言）。"""
    rep = Reporter("view: mini real network value symmetry (collect only)")
    model = load_real_model()
    print(f"      加载真实迷你模型: {MODEL_PATH}")

    v1_list, v2_list = [], []
    for i in range(NUM_POSITIONS):
        env = b.MiniDarkChess()
        env.random_steps(WALK_STEPS)
        b1, s1 = env.observation()
        env.switch_player()
        b2, s2 = env.observation()
        env.switch_player()  # 切回（不改变环境语义）

        with torch.no_grad():
            _, v1 = model(*_encode_obs(b1, s1))
            _, v2 = model(*_encode_obs(b2, s2))
        v1_list.append(float(v1[0, 0]))
        v2_list.append(float(v2[0, 0]))

    v1a = np.array(v1_list)
    v2a = np.array(v2_list)
    # 理想视角对称：v2 ≈ -v1
    sums = v1a + v2a
    sign_opp = np.mean(np.sign(v1a) == -np.sign(v2a)) if np.all(v1a != 0) else np.nan
    corr = np.corrcoef(v1a, -v2a)[0, 1] if len(v1a) > 2 else np.nan
    print(f"      value 统计: mean(v1)={v1a.mean():.4f} mean(v2)={v2a.mean():.4f}")
    print(f"      mean(v1+v2)={sums.mean():.4f} (理想 0)  |  std={sums.std():.4f}")
    print(f"      符号相反比例={sign_opp:.3f} (随机≈0.5, 理想≈1.0)")
    print(f"      corr(v1, -v2)={corr:.4f} (理想→1.0)")
    print(f"      |v1| mean={np.abs(v1a).mean():.4f}  v 值分布: min={v1a.min():.4f} max={v1a.max():.4f}")

    # 暂不设定断言：先观察真实迷你模型（可能未充分训练）的对称性表现
    rep.check(True, f"收集 {NUM_POSITIONS} 个局面的 value 视角对称性统计（暂不断言）")
    rep.summary()


def main() -> None:
    run_part("view: mini Rust env switch_player encoding", test_env_switch_view)
    run_part("view: mini real network value symmetry", test_network_value_symmetry)


if __name__ == "__main__":
    main()
