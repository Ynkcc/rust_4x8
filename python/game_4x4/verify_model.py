"""
verify_model.py — 指定模型文件验证 vs 启发式 MCTS（统一协议）

复用 eval_common（官方协议：c_scale=0.25, gumbel_scale=1.0, hm=64, 交替先后手），
输出胜/平/负 + 分块均值±std。

用法：
    python verify_model.py <模型.pt 或 .pth 路径> [games] [模型sims]
"""
from __future__ import annotations

import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # python/
import torch

torch.set_num_threads(int(os.getenv("G4X4_TORCH_THREADS", "2")))

from banqi.eval import load_predictor, report


def main():
    model_path = sys.argv[1] if len(sys.argv) > 1 else None
    games = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    sims = int(sys.argv[3]) if len(sys.argv) > 3 else 64

    predictor = load_predictor(model_path)
    print(f"[Verify] 模型={model_path or '(默认MODEL_PATH)'} 对局={games} "
          f"模型sims={sims} 启发式sims=64 协议=c_scale0.25/gumbel1.0", flush=True)
    report(predictor, "model", n=games, model_sims=sims)


if __name__ == "__main__":
    main()
