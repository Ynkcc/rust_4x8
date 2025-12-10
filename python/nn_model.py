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
    Standard Residual Block derived from nn_model.rs
    Structure: Conv -> BN -> ReLU -> Conv -> BN -> (+Input) -> ReLU
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
        
        # Residual connection
        out += residual
        out = F.relu(out)
        
        return out

class BanqiNet(nn.Module):
    """
    Banqi Strategy-Value Network
    Architecture matches the Rust implementation:
    1. Input Conv
    2. Residual Tower (3 Blocks)
    3. Flatten & Concat with Scalars
    4. Shared FC Layers
    5. Policy Head & Value Head
    """
    def __init__(self):
        super(BanqiNet, self).__init__()
        
        # 1. Input Conv
        # Maps board state tensor to hidden feature space
        # Rust: nn::conv2d(..., TOTAL_CHANNELS, HIDDEN_CHANNELS, 3, conv_cfg)
        self.conv_input = nn.Conv2d(
            TOTAL_INPUT_CHANNELS, 
            HIDDEN_CHANNELS, 
            kernel_size=3, 
            padding=1,
            bias=False
        )
        self.bn_input = nn.BatchNorm2d(HIDDEN_CHANNELS)
        
        # 2. Residual Tower
        # 3 consecutive residual blocks
        self.res_block1 = BasicBlock(HIDDEN_CHANNELS)
        self.res_block2 = BasicBlock(HIDDEN_CHANNELS)
        self.res_block3 = BasicBlock(HIDDEN_CHANNELS)
        
        # 3. Calculate FC Input Dimension
        # Spatial flatten size
        self.flat_size = HIDDEN_CHANNELS * BOARD_ROWS * BOARD_COLS
        # Total input = Flattened Spatial + Scalar Features
        self.total_fc_input = self.flat_size + SCALAR_FEATURE_COUNT
        
        # 4. Shared Fully Connected Layers
        self.fc1 = nn.Linear(self.total_fc_input, 512)
        self.fc2 = nn.Linear(512, 256)
        
        # 5. Output Heads
        # Policy: Logits (Action Space Size)
        # Value: Scalar [-1, 1]
        self.policy_head = nn.Linear(256, ACTION_SPACE_SIZE)
        self.value_head = nn.Linear(256, 1)

    def forward(self, board, scalars):
        """
        Args:
            board: Tensor of shape (Batch, Channels, H, W)
            scalars: Tensor of shape (Batch, Scalar_Features)
        Returns:
            policy_logits: (Batch, Action_Size)
            value: (Batch, 1) - Tanh activated
        """
        # 1. Input Conv
        x = self.conv_input(board)
        x = self.bn_input(x)
        x = F.relu(x)
        
        # 2. Residual Tower
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        
        # 3. Flatten
        # Rust: x.flatten(1, -1) -> flattens starting from dim 1
        x = x.view(x.size(0), -1) 
        
        # 4. Concatenate
        # Combine spatial features with global scalars
        combined = torch.cat([x, scalars], dim=1)
        
        # 5. Shared Dense Layers
        shared = F.relu(self.fc1(combined))
        shared = F.relu(self.fc2(shared))
        
        # 6. Output Calculation
        policy_logits = self.policy_head(shared)
        value = torch.tanh(self.value_head(shared))
        
        return policy_logits, value

if __name__ == "__main__":
    # Simple test to verify dimensions
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