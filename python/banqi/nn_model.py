"""banqi/nn_model.py — 参数化策略-价值网络（AlphaZero 风格）

一份 `BanqiNet` 服务 4x2 / 4x4 / 4x8 三个变体：所有维度（输入通道、棋盘尺寸、
标量维度、动作空间、残差块数、头尺寸）由 `Variant` / `Constants` 派生。

网络类名统一为 `BanqiNet`（旧 Banqi4x4Net / MiniBanqiNet 不再需要）。
"""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from banqi.constants import Constants, build_constants
from banqi.variant import Variant


class BasicBlock(nn.Module):
    """标准残差块：Conv -> BN -> ReLU -> Conv -> BN -> (+Input) -> ReLU。"""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        return F.relu(out)


class BanqiNet(nn.Module):
    """AlphaZero 策略-价值网络，结构由 variant 决定。

    输入: board (N, C, R, C)，scalars (N, S)
    输出: policy_logits (N, A)，value (N, 1) [tanh]
    """

    def __init__(self, variant: Variant) -> None:
        super().__init__()
        self.variant_id = variant.id
        c: Constants = build_constants(variant)
        hidden = c.HIDDEN_CHANNELS
        rows, cols = c.BOARD_ROWS, c.BOARD_COLS
        scalar = c.SCALAR_FEATURE_COUNT

        # 1. 输入卷积
        self.conv_input = nn.Conv2d(
            c.TOTAL_INPUT_CHANNELS, hidden, kernel_size=3, padding=1, bias=False
        )
        self.bn_input = nn.BatchNorm2d(hidden)

        # 2. 残差塔
        self.res_tower = nn.ModuleList(
            [BasicBlock(hidden) for _ in range(c.NUM_RES_BLOCKS)]
        )

        # 3. 策略头
        self.policy_channels = c.POLICY_HEAD_CHANNELS
        self.policy_conv = nn.Conv2d(
            hidden, self.policy_channels, kernel_size=1, bias=False
        )
        self.policy_bn = nn.BatchNorm2d(self.policy_channels)
        self.policy_flat_size = self.policy_channels * rows * cols
        self.policy_fc_input = self.policy_flat_size + scalar
        self.policy_fc1 = nn.Linear(self.policy_fc_input, c.POLICY_FC1_HIDDEN)
        self.policy_fc2 = nn.Linear(c.POLICY_FC1_HIDDEN, c.ACTION_SPACE_SIZE)

        # 4. 价值头
        self.value_channels = c.VALUE_HEAD_CHANNELS
        self.value_conv = nn.Conv2d(
            hidden, self.value_channels, kernel_size=1, bias=False
        )
        self.value_bn = nn.BatchNorm2d(self.value_channels)
        self.value_flat_size = self.value_channels * rows * cols
        self.value_fc_input = self.value_flat_size + scalar
        self.value_fc1 = nn.Linear(self.value_fc_input, c.VALUE_FC1_HIDDEN)
        self.value_fc2 = nn.Linear(c.VALUE_FC1_HIDDEN, 1)

    def forward(
        self, board: torch.Tensor, scalars: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.conv_input(board)
        x = self.bn_input(x)
        x = F.relu(x)
        for block in self.res_tower:
            x = block(x)

        # 策略头
        p = self.policy_conv(x)
        p = self.policy_bn(p)
        p = F.relu(p)
        p = p.view(p.size(0), -1)
        policy_logits = self.policy_fc2(F.relu(self.policy_fc1(torch.cat([p, scalars], dim=1))))

        # 价值头
        v = self.value_conv(x)
        v = self.value_bn(v)
        v = F.relu(v)
        v = v.view(v.size(0), -1)
        value = torch.tanh(self.value_fc2(F.relu(self.value_fc1(torch.cat([v, scalars], dim=1)))))

        return policy_logits, value


def load_model_weights(model: nn.Module, path: str, device: torch.device) -> None:
    """从 path 加载权重，兼容 TorchScript(.pt) / state_dict(.pth) / checkpoint dict。"""
    try:
        jit_model = torch.jit.load(path, map_location=device)
        model.load_state_dict(jit_model.state_dict())
        return
    except Exception:
        pass
    try:
        state = torch.load(path, map_location=device, weights_only=True)
    except Exception:
        state = torch.load(path, map_location=device)
    if hasattr(state, "state_dict"):
        model.load_state_dict(state.state_dict())
    elif isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _dummy_shapes(variant: Variant) -> Dict[str, Any]:
    c = build_constants(variant)
    return {
        "board": (c.TOTAL_INPUT_CHANNELS, c.BOARD_ROWS, c.BOARD_COLS),
        "scalar": c.SCALAR_FEATURE_COUNT,
        "action": c.ACTION_SPACE_SIZE,
    }


if __name__ == "__main__":
    from banqi.variant import VARIANTS
    for vid, v in VARIANTS.items():
        c = build_constants(v)
        model = BanqiNet(v).eval()
        batch = 2
        board = torch.randn(batch, c.TOTAL_INPUT_CHANNELS, c.BOARD_ROWS, c.BOARD_COLS)
        scalars = torch.randn(batch, c.SCALAR_FEATURE_COUNT)
        with torch.inference_mode():
            logits, value = model(board, scalars)
        assert logits.shape == (batch, c.ACTION_SPACE_SIZE), f"{vid} logits shape"
        assert value.shape == (batch, 1), f"{vid} value shape"
        print(f"[banqi.nn_model] {vid}: input={tuple(board.shape[1:])} "
              f"scalar={c.SCALAR_FEATURE_COUNT} action={c.ACTION_SPACE_SIZE} "
              f"params={count_params(model)} -> logits={tuple(logits.shape)} value={tuple(value.shape)}")
    print("[banqi.nn_model] all OK")
