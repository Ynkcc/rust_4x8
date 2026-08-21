"""python/banqi/nnue/exporter.py — NNUE 权重量化导出 CLI

从 PyTorch checkpoint (.pth/.pt) 加载 BanqiNNUE 权重，量化/序列化导出为二进制 .nnue 格式文件。
"""

from __future__ import annotations

import argparse
import sys
import torch

from banqi.nnue.model import BanqiNNUE


def export_checkpoint(checkpoint_path: str, output_path: str) -> None:
    model = BanqiNNUE()
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
    print(f"[NNUE Exporter] 成功从 {checkpoint_path} 导出二进制 NNUE 权重至 {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Banqi NNUE Weight Exporter")
    parser.add_argument("--input", "-i", type=str, required=True, help="PyTorch checkpoint (.pth/.pt) 路径")
    parser.add_argument("--output", "-o", type=str, required=True, help="输出二进制 .nnue 路径")
    args = parser.parse_args()

    export_checkpoint(args.input, args.output)


if __name__ == "__main__":
    main()
