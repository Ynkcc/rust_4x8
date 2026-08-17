"""banqi/nn_model_health.py — 带「终局血量差异头」的策略-价值网络（实验对照）

与 BanqiNet 的关系：
  - 共享 torso（输入卷积 + 残差塔）、策略头、价值头的**模块命名完全一致**，
    因此可以 `load_state_dict` 直接加载 BanqiNet 的权重（warm start 可选）。
  - 新增第 3 个输出头 health：预测当前局面下的**终局归一化血量差异**
    （样本视角，红方为正，归一化 [-1,1]，标签来自冷存储 `health_diff`）。
  - health 头结构与价值头同构：1x1 conv -> BN -> ReLU -> fc -> tanh。

用途：
  - 作为多任务辅助信号（auxiliary loss）让共享特征携带更细粒度的终局
    强度信息（胜负 ±1 之外的"大胜 vs 险胜"）。
  - 与 BanqiNet 用同一份冷存储数据、同一套超参公平对比，验证新头增益。
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from banqi.constants import Constants, build_constants
from banqi.variant import Variant


class BanqiNetHealth(nn.Module):
    """策略-价值-血量差异网络（AlphaZero 风格 + 血量辅助头）。

    输入: board (N, C, R, C)，scalars (N, S)
    输出: policy_logits (N, A)，value (N, 1) [tanh]，health (N, 1) [tanh]
    """

    def __init__(self, variant: Variant) -> None:
        super().__init__()
        self.variant_id = variant.id
        c: Constants = build_constants(variant)
        hidden = c.HIDDEN_CHANNELS
        rows, cols = c.BOARD_ROWS, c.BOARD_COLS
        scalar = c.SCALAR_FEATURE_COUNT

        # ---- 共享 torso（命名与 BanqiNet 一致，可加载现有权重）----
        self.conv_input = nn.Conv2d(
            c.TOTAL_INPUT_CHANNELS, hidden, kernel_size=3, padding=1, bias=False
        )
        self.bn_input = nn.BatchNorm2d(hidden)
        self.res_tower = nn.ModuleList(
            [self._make_block(hidden) for _ in range(c.NUM_RES_BLOCKS)]
        )

        # ---- 策略头（与 BanqiNet 一致）----
        self.policy_channels = c.POLICY_HEAD_CHANNELS
        self.policy_conv = nn.Conv2d(hidden, self.policy_channels, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(self.policy_channels)
        self.policy_flat_size = self.policy_channels * rows * cols
        self.policy_fc_input = self.policy_flat_size + scalar
        self.policy_fc1 = nn.Linear(self.policy_fc_input, c.POLICY_FC1_HIDDEN)
        self.policy_fc2 = nn.Linear(c.POLICY_FC1_HIDDEN, c.ACTION_SPACE_SIZE)

        # ---- 价值头（与 BanqiNet 一致）----
        self.value_channels = c.VALUE_HEAD_CHANNELS
        self.value_conv = nn.Conv2d(hidden, self.value_channels, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(self.value_channels)
        self.value_flat_size = self.value_channels * rows * cols
        self.value_fc_input = self.value_flat_size + scalar
        self.value_fc1 = nn.Linear(self.value_fc_input, c.VALUE_FC1_HIDDEN)
        self.value_fc2 = nn.Linear(c.VALUE_FC1_HIDDEN, 1)

        # ---- 血量差异头（新增）----
        # 结构与价值头同构；输出 tanh 匹配归一化 [-1,1] 的终局血量差标签。
        self.health_channels = c.VALUE_HEAD_CHANNELS
        self.health_conv = nn.Conv2d(hidden, self.health_channels, kernel_size=1, bias=False)
        self.health_bn = nn.BatchNorm2d(self.health_channels)
        self.health_flat_size = self.health_channels * rows * cols
        self.health_fc_input = self.health_flat_size + scalar
        self.health_fc1 = nn.Linear(self.health_fc_input, c.VALUE_FC1_HIDDEN)
        self.health_fc2 = nn.Linear(c.VALUE_FC1_HIDDEN, 1)

    @staticmethod
    def _make_block(channels: int) -> nn.Module:
        """标准残差块（与 nn_model.BasicBlock 结构一致）。"""
        import torch.nn.functional as _F

        class _BasicBlock(nn.Module):
            def __init__(self, ch: int) -> None:
                super().__init__()
                self.conv1 = nn.Conv2d(ch, ch, kernel_size=3, padding=1, bias=False)
                self.bn1 = nn.BatchNorm2d(ch)
                self.conv2 = nn.Conv2d(ch, ch, kernel_size=3, padding=1, bias=False)
                self.bn2 = nn.BatchNorm2d(ch)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                residual = x
                out = self.conv1(x)
                out = self.bn1(out)
                out = _F.relu(out)
                out = self.conv2(out)
                out = self.bn2(out)
                out = out + residual
                return _F.relu(out)

        return _BasicBlock(channels)

    def forward(
        self, board: torch.Tensor, scalars: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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

        # 血量差异头
        h = self.health_conv(x)
        h = self.health_bn(h)
        h = F.relu(h)
        h = h.view(h.size(0), -1)
        health = torch.tanh(self.health_fc2(F.relu(self.health_fc1(torch.cat([h, scalars], dim=1)))))

        return policy_logits, value, health


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    from banqi.nn_model import BanqiNet
    from banqi.variant import VARIANTS

    for vid, v in VARIANTS.items():
        c = build_constants(v)
        model = BanqiNetHealth(v).eval()
        batch = 2
        board = torch.randn(batch, c.TOTAL_INPUT_CHANNELS, c.BOARD_ROWS, c.BOARD_COLS)
        scalars = torch.randn(batch, c.SCALAR_FEATURE_COUNT)
        with torch.inference_mode():
            logits, value, health = model(board, scalars)
        assert logits.shape == (batch, c.ACTION_SPACE_SIZE), f"{vid} logits shape"
        assert value.shape == (batch, 1), f"{vid} value shape"
        assert health.shape == (batch, 1), f"{vid} health shape"
        # 验证与 BanqiNet 共享部分权重可互通：能加载 BanqiNet 的 state_dict
        base = BanqiNet(v).eval()
        compat = {k: p for k, p in base.state_dict().items() if k in model.state_dict()}
        missing = set(model.state_dict()) - set(compat)
        extra = set(base.state_dict()) - set(compat)
        print(f"[banqi.nn_model_health] {vid}: params={count_params(model)} "
              f"-> logits={tuple(logits.shape)} value={tuple(value.shape)} health={tuple(health.shape)} "
              f"| 与 BanqiNet 共享参数={len(compat)} 新头参数={len(missing)} 无法复用={len(extra)}")
    print("[banqi.nn_model_health] all OK")
