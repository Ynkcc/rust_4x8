"""python/banqi/nnue/train.py — Banqi NNUE 浅层网络训练脚本

从 self-play 导出的 episode JSONL 读取 NNUE 稀疏特征样本进行训练。
标签采用「搜索价值 + 终局回报」混合：y = w * value + (1 - w) * game_result。
训练完成后导出二进制 .nnue 格式模型（维度由数据集 nnue_meta 推导，按变体自适应）。
"""

from __future__ import annotations

import argparse
import json
import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from banqi.nnue.model import BanqiNNUE

logger = logging.getLogger(__name__)


class NnueSampleDataset(Dataset):
    """NNUE 训练数据集。

    输入为 self-play episode dict（PyO3 episode_to_dict / serialize.rs JSON 契约）：
    - nnue_meta:    特征布局元信息（feature_dim 等，随变体推导）
    - nnue_features: {"mover": [[索引...]每步], "opponent": [[索引...]每步]}
    - completed_qs / mcts_values: 搜索价值（行棋方视角）
    - game_results: 终局回报（行棋方视角）
    - is_full_search: 算力随机化标记

    默认仅使用行棋方（mover）视角样本；dual_perspective=True 时同时纳入对方视角，
    其搜索价值取反（零和对局近似）。
    """

    def __init__(
        self,
        jsonl_paths: list[str],
        value_source: str = "completed_q",
        value_weight: float = 0.7,
        full_only: bool = False,
        dual_perspective: bool = False,
    ) -> None:
        if value_source not in ("completed_q", "mcts_value"):
            raise ValueError(f"未知 value_source: {value_source}")
        # episode dict 字段为复数形式（与 serialize.rs / episode.rs 契约一致）
        value_key = value_source + "s"
        self.feature_dim: int | None = None
        features: list[list[int]] = []
        targets: list[float] = []

        for path in jsonl_paths:
            n_eps = n_steps = 0
            with open(path, encoding="utf-8") as f:
                for line_no, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        ep = json.loads(line)
                    except json.JSONDecodeError as e:
                        logger.warning("跳过非法 JSON 行 %s:%d: %s", path, line_no, e)
                        continue
                    nnue = ep.get("nnue_features")
                    meta = ep.get("nnue_meta")
                    if nnue is None or meta is None:
                        continue
                    dim = int(meta["feature_dim"])
                    if self.feature_dim is None:
                        self.feature_dim = dim
                    elif self.feature_dim != dim:
                        raise ValueError(
                            f"{path}:{line_no} 特征维度不一致: {dim} != {self.feature_dim}，"
                            "数据集中混入了不同变体的样本"
                        )

                    n_eps += 1
                    movers: list[list[int]] = nnue["mover"]
                    opponents: list[list[int]] = nnue.get("opponent") or []
                    values = ep[value_key]
                    results = ep["game_results"]
                    full_flags = ep.get("is_full_search") or [True] * len(movers)

                    for i, feats in enumerate(movers):
                        if full_only and not full_flags[i]:
                            continue
                        features.append(feats)
                        targets.append(value_weight * values[i] + (1.0 - value_weight) * results[i])
                        n_steps += 1
                        if dual_perspective and i < len(opponents):
                            features.append(opponents[i])
                            targets.append(
                                value_weight * (-values[i]) + (1.0 - value_weight) * (-results[i])
                            )
                            n_steps += 1
            logger.info("加载 %s: %d 个 episode, 累计 %d 个 NNUE 样本", path, n_eps, n_steps)

        if not features:
            raise ValueError(
                "数据集中没有任何 NNUE 样本。请确认 JSONL 由开启 collect_nnue_features 的 "
                "self-play 生成（episode dict 需含 nnue_meta / nnue_features 字段）"
            )
        assert self.feature_dim is not None
        self.features = features
        self.targets = torch.tensor(targets, dtype=torch.float32).unsqueeze(1)

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
    parser.add_argument("--dual-perspective", action="store_true", help="同时使用对方视角样本（价值取反）")
    parser.add_argument("--output", type=str, default="banqi_model.nnue", help="输出二进制 .nnue 路径")
    parser.add_argument("--checkpoint", type=str, default=None, help="可选 checkpoint (.pth) 输出路径")
    args = parser.parse_args()

    dataset = NnueSampleDataset(
        jsonl_paths=args.data,
        value_source=args.value_source,
        value_weight=args.value_weight,
        full_only=args.full_only,
        dual_perspective=args.dual_perspective,
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
