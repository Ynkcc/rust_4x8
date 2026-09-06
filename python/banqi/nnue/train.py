"""python/banqi/nnue/train.py — Banqi NNUE 浅层网络训练脚本

从 self-play 导出的 episode JSONL 读取 NNUE 稀疏特征样本进行训练。
标签采用「搜索价值 + 终局回报」混合：y = w * value + (1 - w) * game_result。
训练完成后导出二进制 .nnue 格式模型（维度由数据集 nnue_meta 推导，按变体自适应）。

样本解析/过滤逻辑统一在 banqi.nnue.samples.NnueSampleBuffer：
- CLI（main）路径：JSONL 批量装载 → to_dataset()；
- 主闭环蒸馏路径：NnueDistillWorker 流式 add_episode → to_dataset()。
"""

from __future__ import annotations

import argparse
import logging
from typing import List, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from banqi.nnue.model import BanqiNNUE
from banqi.nnue.samples import NnueSampleBuffer

logger = logging.getLogger(__name__)


class NnueSampleDataset(Dataset):
    """NNUE 训练数据集（稀疏特征索引 + 混合价值标签）。

    两种构造方式：
    1. NnueSampleDataset(jsonl_paths=[...], ...) — 兼容旧 CLI 签名，内部经
       NnueSampleBuffer 解析（契约与 PyO3 episode_to_dict / serialize.rs 一致）；
    2. NnueSampleDataset(jsonl_paths=None, features=..., targets=..., feature_dim=...)
       — 供 NnueSampleBuffer.to_dataset() 流式物化使用。
    """

    def __init__(
        self,
        jsonl_paths: Optional[List[str]] = None,
        value_source: str = "completed_q",
        value_weight: float = 0.7,
        full_only: bool = False,
        features: Optional[List[List[int]]] = None,
        targets: Optional[torch.Tensor] = None,
        feature_dim: Optional[int] = None,
    ) -> None:
        if jsonl_paths is not None:
            buffer = NnueSampleBuffer(
                value_source=value_source,
                value_weight=value_weight,
                full_only=full_only,
            )
            for path in jsonl_paths:
                buffer.ingest_jsonl(path)
            features, targets, feature_dim = buffer.to_tensors()
            if features is None:
                raise ValueError(
                    "数据集中没有任何 NNUE 样本。请确认 JSONL 由开启 collect_nnue_features 的 "
                    "self-play 生成（episode dict 需含 nnue_meta / nnue_features 字段）"
                )

        if not features or feature_dim is None or targets is None:
            raise ValueError("NnueSampleDataset 需要非空 features/targets/feature_dim")
        self.features = features
        self.targets = targets
        self.feature_dim = int(feature_dim)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.zeros(self.feature_dim, dtype=torch.float32)
        x[torch.tensor(self.features[idx], dtype=torch.long)] = 1.0
        return x, self.targets[idx]


def train_nnue(
    dataset: NnueSampleDataset,
    epochs: int = 20,
    batch_size: int = 256,
    lr: float = 1e-3,
    output_nnue: str = "banqi_model.nnue",
    checkpoint: str | None = None,
) -> BanqiNNUE:
    if dataset.feature_dim is None:
        raise ValueError("dataset 未初始化 feature_dim")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("训练设备: %s, 样本数: %d, feature_dim: %d", device, len(dataset), dataset.feature_dim)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = BanqiNNUE(dataset.feature_dim).to(device)
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
        logger.info("Epoch %d/%d Loss: %.6f", epoch, epochs, avg_loss)

    model.cpu()
    model.export_nnue_binary(output_nnue)
    logger.info("训练完毕，已导出模型至: %s", output_nnue)

    if checkpoint:
        torch.save(model.state_dict(), checkpoint)
        logger.info("checkpoint 已保存至: %s", checkpoint)
    return model


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[NNUE Train] %(message)s")
    parser = argparse.ArgumentParser(description="Banqi NNUE Trainer (self-play JSONL)")
    parser.add_argument("--data", type=str, nargs="+", required=True, help="episode JSONL 路径（可多个）")
    parser.add_argument("--epochs", type=int, default=20, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=256, help="批次大小")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--value-source", choices=["completed_q", "mcts_value"], default="completed_q", help="搜索价值来源")
    parser.add_argument("--value-weight", type=float, default=0.7, help="混合标签中搜索价值权重（终局回报权重 = 1 - w）")
    parser.add_argument("--full-only", action="store_true", help="仅使用 Full Search 样本")
    parser.add_argument("--output", type=str, default="banqi_model.nnue", help="输出二进制 .nnue 路径")
    parser.add_argument("--checkpoint", type=str, default=None, help="可选 checkpoint (.pth) 输出路径")
    args = parser.parse_args()

    dataset = NnueSampleDataset(
        jsonl_paths=args.data,
        value_source=args.value_source,
        value_weight=args.value_weight,
        full_only=args.full_only,
    )
    train_nnue(
        dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        output_nnue=args.output,
        checkpoint=args.checkpoint,
    )


if __name__ == "__main__":
    main()
