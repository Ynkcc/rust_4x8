# nn_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from constant import (
    TOTAL_INPUT_CHANNELS,
    HIDDEN_CHANNELS,
    BOARD_ROWS,
    BOARD_COLS,
    SCALAR_FEATURE_COUNT,
    ACTION_SPACE_SIZE
)

class BasicBlock(nn.Module):
    """
    标准残差块，参考 nn_model.rs
    结构: Conv -> BN -> ReLU -> Conv -> BN -> (+Input) -> ReLU
    """
    def __init__(self, channels):
        super(BasicBlock, self).__init__()
        # Rust: padding: 1, kernel_size: 3
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
        
        # 残差连接
        out += residual
        out = F.relu(out)
        
        return out

class BanqiNet(nn.Module):
    """
    改进版 Banqi 策略-价值网络（接近 AlphaZero 标准架构）
    1. 动态深度残差塔（默认 8 层）
    2. 双头 1x1 卷积降维，显著减少全连接参数量
    """
    def __init__(self, num_res_blocks=8):
        super(BanqiNet, self).__init__()
        
        # 1. 输入卷积
        # Rust: nn::conv2d(..., TOTAL_CHANNELS, HIDDEN_CHANNELS, 3, conv_cfg)
        self.conv_input = nn.Conv2d(
            TOTAL_INPUT_CHANNELS, 
            HIDDEN_CHANNELS, 
            kernel_size=3, 
            padding=1,
            bias=False
        )
        self.bn_input = nn.BatchNorm2d(HIDDEN_CHANNELS)
        
        # 2. 残差塔（加深网络）
        self.res_tower = nn.ModuleList(
            [BasicBlock(HIDDEN_CHANNELS) for _ in range(num_res_blocks)]
        )

        # 3. 策略头
        policy_channels = 4
        self.policy_conv = nn.Conv2d(HIDDEN_CHANNELS, policy_channels, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(policy_channels)
        self.policy_flat_size = policy_channels * BOARD_ROWS * BOARD_COLS
        self.policy_fc_input = self.policy_flat_size + SCALAR_FEATURE_COUNT
        self.policy_fc1 = nn.Linear(self.policy_fc_input, 512)
        self.policy_fc2 = nn.Linear(512, ACTION_SPACE_SIZE)

        # 4. 价值头
        value_channels = 2
        self.value_conv = nn.Conv2d(HIDDEN_CHANNELS, value_channels, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(value_channels)
        self.value_flat_size = value_channels * BOARD_ROWS * BOARD_COLS
        self.value_fc_input = self.value_flat_size + SCALAR_FEATURE_COUNT
        self.value_fc1 = nn.Linear(self.value_fc_input, 256)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, board, scalars):
        """
        Args:
            board: Tensor，形状 (Batch, Channels, H, W)
            scalars: Tensor，形状 (Batch, Scalar_Features)
        Returns:
            policy_logits: (Batch, Action_Size)
            value: (Batch, 1) - Tanh 激活
        """
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

if __name__ == "__main__":
    # 简单测试：验证维度是否匹配
    batch_size = 4
    dummy_board = torch.randn(batch_size, TOTAL_INPUT_CHANNELS, BOARD_ROWS, BOARD_COLS)
    dummy_scalars = torch.randn(batch_size, SCALAR_FEATURE_COUNT)
    
    model = BanqiNet()
    policy, value = model(dummy_board, dummy_scalars)
    
    print(f"Model Structure: {model}")
    print(f"Input Board: {dummy_board.shape}")
    print(f"Input Scalars: {dummy_scalars.shape}")
    print(f"Output Policy: {policy.shape} (Expected: [{batch_size}, {ACTION_SPACE_SIZE}])")
    print(f"Output Value: {value.shape} (Expected: [{batch_size}, 1])")