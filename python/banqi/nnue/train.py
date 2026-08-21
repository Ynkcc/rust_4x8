"""python/banqi/nnue/train.py — Banqi NNUE 浅层网络训练脚本

使用 PyTorch 拟合 self-play 生成的暗棋评估样本或对弈价值，训练完成后自动导出二进制 .nnue 格式模型。
"""

from __future__ import annotations

import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from banqi.nnue.model import BanqiNNUE, FEATURE_DIM


class SyntheticBanqiDataset(Dataset):
    """随机合成/模拟暗棋训练数据集（真实训练可用实际 self-play 样本替换）"""

    def __init__(self, num_samples: int = 1000) -> None:
        super().__init__()
        self.features = torch.randn(num_samples, FEATURE_DIM)
        # 激活稀疏特征 (模拟 40 个 one-hot 激活项)
        self.features = (self.features > 1.5).float()
        self.targets = torch.tanh(torch.randn(num_samples, 1))

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.targets[idx]


def train_nnue(
    epochs: int = 5,
    batch_size: int = 64,
    lr: float = 1e-3,
    output_nnue: str = "banqi_model.nnue",
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[NNUE Train] 训练设备: {device}")

    dataset = SyntheticBanqiDataset(num_samples=2000)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = BanqiNNUE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for features, targets in dataloader:
            features = features.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            preds = model(features)
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(features)

        avg_loss = total_loss / len(dataset)
        print(f"[NNUE Train] Epoch {epoch}/{epochs} Loss: {avg_loss:.6f}")

    # 导出 .nnue 权重
    model.cpu()
    model.export_nnue_binary(output_nnue)
    print(f"[NNUE Train] 训练完毕，成功导出模型至: {output_nnue}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Banqi NNUE Trainer")
    parser.add_argument("--epochs", type=int, default=5, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=64, help="批次大小")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--output", type=str, default="banqi_model.nnue", help="输出二进制 .nnue 路径")
    args = parser.parse_args()

    train_nnue(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, output_nnue=args.output)


if __name__ == "__main__":
    main()
