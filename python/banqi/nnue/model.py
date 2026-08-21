"""python/banqi/nnue/model.py — 神经网络评估器 (BanqiNNUE)

模型包含:
1. Feature Transformer: (562 -> 256) 稀疏输入到 256 维隐向量
2. Clipped ReLU 激活 (0.0 .. 1.0)
3. 隐层 1: (256 -> 32)
4. 输出层: (32 -> 1), Tanh 归一化 [-1, 1]
"""

from __future__ import annotations

import struct
import torch
import torch.nn as nn
import torch.nn.functional as F

FEATURE_DIM = 562
TRANSFORMER_OUT_DIM = 256
FC1_OUT_DIM = 32


class BanqiNNUE(nn.Module):
    """Banqi 4x8 NNUE 评估网络"""

    def __init__(self) -> None:
        super().__init__()
        # Feature Transformer (W0: [256, 562], B0: [256])
        self.ft = nn.Linear(FEATURE_DIM, TRANSFORMER_OUT_DIM)
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
            w0 = self.ft.weight.detach().cpu().flatten().tolist()  # [256*562]
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
    net = BanqiNNUE()
    dummy_input = torch.randn(4, FEATURE_DIM)
    out = net(dummy_input)
    assert out.shape == (4, 1), f"输出维度应为 (4, 1), 实际为 {out.shape}"
    print(f"[BanqiNNUE] Test OK. Dummy output: {out.squeeze().tolist()}")
