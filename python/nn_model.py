# nn_model.py
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
    AlphaZero-style policy-value network for 4x8 Dark Chess.

    Input:
      board   - (N, 16, 4, 8) float32
      scalars - (N, 35) float32 (no action_mask — masks handled at loss level)

    Architecture:
      Input conv(16→64, 3×3)
      6 × ResidualBlock (64 ch)
      Policy head: 1×1 conv(64→4) → flatten(128) → +scalars(35) → FC1(163→512) → FC2(512→352)
      Value head:  1×1 conv(64→4) → flatten(128) → +scalars(35) → FC1(163→256) → FC2(256→1) → tanh

    Total params ~1.03M (down from ~2.94M).
    """
    def __init__(self, num_res_blocks=NUM_RES_BLOCKS, hidden_channels=HIDDEN_CHANNELS,
                 policy_channels=4, value_channels=4):
        super(BanqiNet, self).__init__()
        
        # 1. 输入卷积
        self.conv_input = nn.Conv2d(
            TOTAL_INPUT_CHANNELS,
            hidden_channels,
            kernel_size=3,
            padding=1,
            bias=False
        )
        self.bn_input = nn.BatchNorm2d(hidden_channels)
        
        # 2. 残差塔
        self.res_tower = nn.ModuleList(
            [BasicBlock(hidden_channels) for _ in range(num_res_blocks)]
        )

        # 3. 策略头
        self.policy_channels = policy_channels
        self.policy_conv = nn.Conv2d(hidden_channels, policy_channels, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(policy_channels)
        self.policy_flat_size = policy_channels * BOARD_ROWS * BOARD_COLS
        self.policy_fc_input = self.policy_flat_size + SCALAR_FEATURE_COUNT
        self.policy_fc1 = nn.Linear(self.policy_fc_input, 512)
        self.policy_fc2 = nn.Linear(512, ACTION_SPACE_SIZE)

        # 4. 价值头
        self.value_channels = value_channels
        self.value_conv = nn.Conv2d(hidden_channels, value_channels, kernel_size=1, bias=False)
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

def load_model_weights(model: "nn.Module", path: str, device: "torch.device") -> None:
    """
    把 path 指向的权重文件加载进 model。自动识别三种格式：
      1. TorchScript 归档 (.pt, training_service 中 torch.jit.trace 产出) —— torch.jit.load
      2. 普通 state_dict (.pth)
      3. 完整 checkpoint dict（含 model_state_dict / optimizer_state_dict 等）

    注意：PyTorch 2.6+ 中 torch.load(weights_only=True) 无法加载 TorchScript
    归档（抛 "Cannot use weights_only=True with TorchScript archives"），
    因此必须先尝试 torch.jit.load。全部格式都失败时抛出最后一个异常，
    由调用方决定如何处理（如保留旧模型）。
    """
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