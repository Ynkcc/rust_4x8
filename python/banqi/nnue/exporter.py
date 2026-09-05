"""python/banqi/nnue/exporter.py — NNUE 权重量化导出 CLI

从 PyTorch checkpoint (.pth/.pt) 加载 BanqiNNUE 权重，量化/序列化导出为二进制 .nnue 格式文件。
"""

from __future__ import annotations

import argparse
import sys
import torch

from banqi.nnue.model import BanqiNNUE


def export_checkpoint(checkpoint_path: str, output_path: str, feature_dim: int = 555) -> None:
    model = BanqiNNUE(feature_dim)
    state = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    elif isinstance(state, dict) and "state_dict" in state:
        model.load_state_dict(state["state_dict"])
    elif isinstance(state, dict):
        model.load_state_dict(state)
    else:
        model = state

    model.export_nnue_binary(output_path)
    print(f"[NNUE Exporter] 成功从 {checkpoint_path} 导出二进制 NNUE 权重至 {output_path} (feature_dim={feature_dim})")


def export_random(output_path: str, feature_dim: int = 555, output_scale: float = 0.1) -> None:
    """导出随机初始化的基座 .nnue（冷启动用）。

    输出层权重乘以 output_scale，使初始评估接近 0 但非恒 0——
    足够打破 expectimax 无评估导致的截断死锁，又不被随机大分数带偏早期搜索。
    """
    model = BanqiNNUE(feature_dim)
    with torch.no_grad():
        model.fc2.weight.mul_(output_scale)
        model.fc2.bias.mul_(output_scale)
    model.export_nnue_binary(output_path)
    print(f"[NNUE Exporter] 已导出随机基座 NNUE 至 {output_path} (feature_dim={feature_dim}, output_scale={output_scale})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Banqi NNUE Weight Exporter")
    parser.add_argument("--input", "-i", type=str, default=None, help="PyTorch checkpoint (.pth/.pt) 路径（--random 时可省略）")
    parser.add_argument("--output", "-o", type=str, required=True, help="输出二进制 .nnue 路径")
    parser.add_argument("--feature-dim", type=int, default=555, help="NNUE 输入特征维度（4x8=555）")
    parser.add_argument("--random", action="store_true", help="导出随机初始化基座模型（冷启动），忽略 --input")
    parser.add_argument("--output-scale", type=float, default=0.1, help="--random 时输出层缩放系数")
    args = parser.parse_args()

    if args.random:
        export_random(args.output, args.feature_dim, args.output_scale)
        return
    if args.input is None:
        parser.error("必须提供 --input（或使用 --random 导出随机基座）")
    export_checkpoint(args.input, args.output, args.feature_dim)


if __name__ == "__main__":
    main()
