# nn_model.py — 4x4 暗棋的策略-价值网络（AlphaZero 风格）
#
# 输入：棋盘 (16, 4, 4) + 标量 19；动作空间 112。
# 规模略大于 4x2 迷你（隐藏 24 通道、2 残差块），参数量约 6 万，CPU 可训练。
import torch
import torch.nn as nn
import torch.nn.functional as F

from constant import (
    TOTAL_INPUT_CHANNELS,
    HIDDEN_CHANNELS,
    NUM_RES_BLOCKS,
    BOARD_ROWS,
    BOARD_COLS,
    SCALAR_FEATURE_COUNT,
    ACTION_SPACE_SIZE,
)


class BasicBlock(nn.Module):
    """标准残差块：Conv -> BN -> ReLU -> Conv -> BN -> (+Input) -> ReLU"""

    def __init__(self, channels):
        super(BasicBlock, self).__init__()
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


class Banqi4x4Net(nn.Module):
    """
    AlphaZero-style policy-value network for 4x4 Dark Chess.

    Input:
      board   - (N, 16, 4, 4) float32
      scalars - (N, 19) float32

    Architecture:
      Input conv(16→24, 3×3)
      2 × ResidualBlock (24 ch)
      Policy head: 1×1 conv(24→2) -> flatten(32) -> +scalars(19) -> FC(51→128) -> FC(128→112)
      Value head:  1×1 conv(24→2) -> flatten(32) -> +scalars(19) -> FC(51→64)  -> FC(64→1) -> tanh
    """

    def __init__(self, num_res_blocks=NUM_RES_BLOCKS, hidden_channels=HIDDEN_CHANNELS,
                 policy_channels=2, value_channels=2):
        super(Banqi4x4Net, self).__init__()

        self.conv_input = nn.Conv2d(
            TOTAL_INPUT_CHANNELS, hidden_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn_input = nn.BatchNorm2d(hidden_channels)

        self.res_tower = nn.ModuleList(
            [BasicBlock(hidden_channels) for _ in range(num_res_blocks)]
        )

        self.policy_channels = policy_channels
        self.policy_conv = nn.Conv2d(hidden_channels, policy_channels, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(policy_channels)
        self.policy_flat_size = policy_channels * BOARD_ROWS * BOARD_COLS
        self.policy_fc_input = self.policy_flat_size + SCALAR_FEATURE_COUNT
        self.policy_fc1 = nn.Linear(self.policy_fc_input, 128)
        self.policy_fc2 = nn.Linear(128, ACTION_SPACE_SIZE)

        self.value_channels = value_channels
        self.value_conv = nn.Conv2d(hidden_channels, value_channels, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(value_channels)
        self.value_flat_size = value_channels * BOARD_ROWS * BOARD_COLS
        self.value_fc_input = self.value_flat_size + SCALAR_FEATURE_COUNT
        self.value_fc1 = nn.Linear(self.value_fc_input, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, board, scalars):
        x = self.conv_input(board)
        x = self.bn_input(x)
        x = F.relu(x)

        for block in self.res_tower:
            x = block(x)

        p = self.policy_conv(x)
        p = self.policy_bn(p)
        p = F.relu(p)
        p = p.view(p.size(0), -1)
        p_combined = torch.cat([p, scalars], dim=1)
        p_out = F.relu(self.policy_fc1(p_combined))
        policy_logits = self.policy_fc2(p_out)

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
    model = Banqi4x4Net()
    n_params = sum(p.numel() for p in model.parameters())
    policy, value = model(dummy_board, dummy_scalars)
    print(f"Input Board: {dummy_board.shape}")
    print(f"Input Scalars: {dummy_scalars.shape}")
    print(f"Output Policy: {policy.shape} (Expected [{batch_size}, {ACTION_SPACE_SIZE}])")
    print(f"Output Value: {value.shape} (Expected [{batch_size}, 1])")
    print(f"Total params: {n_params}")
