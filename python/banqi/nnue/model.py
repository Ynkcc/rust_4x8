"""python/banqi/nnue/model.py — 神经网络评估器 (BanqiNNUE)

模型包含:
1. Feature Transformer: (feature_dim -> 256) 稀疏输入到 256 维隐向量
2. Clipped ReLU 激活 (0.0 .. 1.0)
3. 隐层 1: (256 -> 32)
4. 输出层: (32 -> 1), Tanh 归一化 [-1, 1]

feature_dim 由变体 config 推导（与 Rust `GameConfig::nnue_feature_dim` 一致）:
    total_positions * (2 + 2*num_active) + num_active * (max_piece_count + 1) + 1
"""

from __future__ import annotations

import struct
import torch
import torch.nn as nn

TRANSFORMER_OUT_DIM = 256
FC1_OUT_DIM = 32


def nnue_feature_dim(
    total_positions: int,
    num_active: int,
    max_piece_count: int,
    scalar_buckets: int = 1,
) -> int:
    """由变体布局参数推导 NNUE 输入维度（与 Rust config 推导公式对齐）。"""
    return total_positions * (2 + 2 * num_active) + num_active * (max_piece_count + 1) + scalar_buckets


class BanqiNNUE(nn.Module):
    """Banqi NNUE 评估网络（按变体维度参数化）"""

    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        # Feature Transformer (W0: [256, feature_dim], B0: [256])
        self.ft = nn.Linear(feature_dim, TRANSFORMER_OUT_DIM)
        # 隐层 1 (W1: [32, 256], B1: [32])
        self.fc1 = nn.Linear(TRANSFORMER_OUT_DIM, FC1_OUT_DIM)
        # 输出层 (W2: [1, 32], B2: [1])
        self.fc2 = nn.Linear(FC1_OUT_DIM, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """前向传播
        features: (N, FEATURE_DIM) 浮点特征或 0/1 张量
        返回: (N, 1) [-1.0, 1.0] 的评估得分
        """
        # Feature Transformer
        h0 = torch.clamp(self.ft(features), 0.0, 1.0)
        # 隐层 1
        h1 = torch.clamp(self.fc1(h0), 0.0, 1.0)
        # 输出层
        out = torch.tanh(self.fc2(h1))
        return out

    def export_nnue_binary(self, filepath: str) -> None:
        """导出为 Rust 引擎可读取的二进制 .nnue 格式小端 float32 文件"""
        self.eval()
        with torch.no_grad():
            # PyTorch Linear weight 形状为 [out_features, in_features]
            # Rust 累加器查找偏移为 feat_idx * TRANSFORMER_OUT_DIM，因此需要转置为 [in_features, out_features]
            w0 = self.ft.weight.detach().cpu().t().flatten().tolist()  # [feature_dim * 256]
            b0 = self.ft.bias.detach().cpu().flatten().tolist()    # [256]
            w1 = self.fc1.weight.detach().cpu().flatten().tolist() # [32*256]
            b1 = self.fc1.bias.detach().cpu().flatten().tolist()   # [32]
            w2 = self.fc2.weight.detach().cpu().flatten().tolist() # [1*32]
            b2 = self.fc2.bias.detach().cpu().flatten().tolist()   # [1]

            all_params = w0 + b0 + w1 + b1 + w2 + b2

        with open(filepath, "wb") as f:
            for val in all_params:
                f.write(struct.pack("<f", float(val)))


if __name__ == "__main__":
    dim = nnue_feature_dim(total_positions=32, num_active=7, max_piece_count=5)
    assert dim == 555, f"4x8 变体维度应为 555, 实际为 {dim}"
    net = BanqiNNUE(dim)
    dummy_input = torch.randn(4, dim)
    out = net(dummy_input)
    assert out.shape == (4, 1), f"输出维度应为 (4, 1), 实际为 {out.shape}"
    print(f"[BanqiNNUE] Test OK. Dummy output: {out.squeeze().tolist()}")
