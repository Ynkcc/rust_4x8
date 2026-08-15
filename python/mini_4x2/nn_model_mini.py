# nn_model_mini.py — 4x2 迷你暗棋的极小网络
#
# 与 nn_model.py 结构相似（AlphaZero 风格），但问题规模很小：
#   - 输入棋盘 (10, 4, 2)，标量 11
#   - hidden_channels=16，仅 1 个残差块
#   - 动作空间 40
# 参数量约 3 万，CPU 训练即可快速收敛（预期 20 分钟内）。
import torch
import torch.nn as nn
import torch.nn.functional as F

from constant_mini import (
    TOTAL_INPUT_CHANNELS,
    HIDDEN_CHANNELS,
    NUM_RES_BLOCKS,
    BOARD_ROWS,
    BOARD_COLS,
    SCALAR_FEATURE_COUNT,
    ACTION_SPACE_SIZE,
)


class MiniBasicBlock(nn.Module):
    """标准残差块：Conv -> BN -> ReLU -> Conv -> BN -> (+Input) -> ReLU"""

    def __init__(self, channels):
        super(MiniBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = F.relu(out)
        return out


class MiniBanqiNet(nn.Module):
    """
    AlphaZero-style policy-value network for 4x2 mini Dark Chess.

    Input:
      board   - (N, 10, 4, 2) float32
      scalars - (N, 11) float32 (no action_mask — masks handled at loss level)

    Architecture:
      Input conv(10→16, 3×3)
      1 × ResidualBlock (16 ch)
      Policy head: 1×1 conv(16→2) -> flatten(16) -> +scalars(11) -> FC1(27→64) -> FC2(64→40)
      Value head:  1×1 conv(16→2) -> flatten(16) -> +scalars(11) -> FC1(27→32) -> FC2(32→1) -> tanh

    Total params ~32K.
    """

    def __init__(self, num_res_blocks=NUM_RES_BLOCKS, hidden_channels=HIDDEN_CHANNELS,
                 policy_channels=2, value_channels=2):
        super(MiniBanqiNet, self).__init__()

        # 1. 输入卷积
        self.conv_input = nn.Conv2d(
            TOTAL_INPUT_CHANNELS,
            hidden_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.bn_input = nn.BatchNorm2d(hidden_channels)

        # 2. 残差塔
        self.res_tower = nn.ModuleList(
            [MiniBasicBlock(hidden_channels) for _ in range(num_res_blocks)]
        )

        # 3. 策略头
        self.policy_channels = policy_channels
        self.policy_conv = nn.Conv2d(hidden_channels, policy_channels, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(policy_channels)
        self.policy_flat_size = policy_channels * BOARD_ROWS * BOARD_COLS
        self.policy_fc_input = self.policy_flat_size + SCALAR_FEATURE_COUNT
        self.policy_fc1 = nn.Linear(self.policy_fc_input, 64)
        self.policy_fc2 = nn.Linear(64, ACTION_SPACE_SIZE)

        # 4. 价值头
        self.value_channels = value_channels
        self.value_conv = nn.Conv2d(hidden_channels, value_channels, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(value_channels)
        self.value_flat_size = value_channels * BOARD_ROWS * BOARD_COLS
        self.value_fc_input = self.value_flat_size + SCALAR_FEATURE_COUNT
        self.value_fc1 = nn.Linear(self.value_fc_input, 32)
        self.value_fc2 = nn.Linear(32, 1)

    def forward(self, board, scalars):
        # 1. 输入卷积
        x = self.conv_input(board)
        x = self.bn_input(x)
        x = F.relu(x)

        # 2. 残差塔
        for block in self.res_tower:
            x = block(x)

        # 3. 策略头
        p = self.policy_conv(x)
        p = self.policy_bn(p)
        p = F.relu(p)
        p = p.view(p.size(0), -1)
        p_combined = torch.cat([p, scalars], dim=1)
        p_out = F.relu(self.policy_fc1(p_combined))
        policy_logits = self.policy_fc2(p_out)

        # 4. 价值头
        v = self.value_conv(x)
        v = self.value_bn(v)
        v = F.relu(v)
        v = v.view(v.size(0), -1)
        v_combined = torch.cat([v, scalars], dim=1)
        v_out = F.relu(self.value_fc1(v_combined))
        value = torch.tanh(self.value_fc2(v_out))

        return policy_logits, value


def load_model_weights(model: "nn.Module", path: str, device: "torch.device") -> None:
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


if __name__ == "__main__":
    batch_size = 4
    dummy_board = torch.randn(batch_size, TOTAL_INPUT_CHANNELS, BOARD_ROWS, BOARD_COLS)
    dummy_scalars = torch.randn(batch_size, SCALAR_FEATURE_COUNT)
    model = MiniBanqiNet()
    n_params = sum(p.numel() for p in model.parameters())
    policy, value = model(dummy_board, dummy_scalars)
    print(f"Input Board: {dummy_board.shape}")
    print(f"Input Scalars: {dummy_scalars.shape}")
    print(f"Output Policy: {policy.shape} (Expected [{batch_size}, {ACTION_SPACE_SIZE}])")
    print(f"Output Value: {value.shape} (Expected [{batch_size}, 1])")
    print(f"Total params: {n_params}")
