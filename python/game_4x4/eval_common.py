"""
eval_common.py — 统一评估协议（单一入口）

所有 diag/verify/train 脚本复用本模块的评估函数，避免"同一模型同一协议
测出 35%/60%"这类 n 太小导致的不可信结论。

协议固定（与 verify_vs_heuristic_mcts.py 默认完全一致）：
  - 模型 MCTS：num_simulations=64, max_considered_actions=16,
               c_scale=0.25, gumbel_scale=1.0（Rust 签名
               mcts_search_action(predict_fn, sims, max_acts, c_scale, gumbel_scale)）
  - 对手：启发式 MCTS 64 sims
  - 交替先后手
  - 分块统计：n 局分成 k 块（默认 5×20），输出每块胜率 + 均值±std，
    便于评估差异是否超过噪声（n=20 单块 σ≈11%）。

警告：此前的 diag/verify 脚本曾误用 c_scale=1.0, gumbel_scale=0.25（与官方
verify_vs_heuristic_mcts.py 的 0.25/1.0 相反），导致同一模型测出 35% vs 60%
的假差异。所有评估必须统一走本模块。
"""
from __future__ import annotations

import os
import sys
from typing import Callable, List, Optional, Tuple

import numpy as np

# 评估常量（唯一权威定义处，与 verify_vs_heuristic_mcts.py 默认一致）
EVAL_SIMS = 64
EVAL_MAX_ACTIONS = 16
EVAL_C_SCALE = 0.25
EVAL_GUMBEL_SCALE = 1.0
HM_SIMS = 64


def model_mcts_action(env, predictor, sims: int = EVAL_SIMS,
                      max_actions: int = EVAL_MAX_ACTIONS,
                      c_scale: float = EVAL_C_SCALE,
                      gumbel_scale: float = EVAL_GUMBEL_SCALE) -> int:
    """模型 MCTS 动作（与官方验证协议完全一致的参数）。"""
    return env.mcts_search_action(predictor, sims, max_actions, c_scale, gumbel_scale)


def heuristic_action(env, sims: int = HM_SIMS) -> int:
    return env.heuristic_mcts_action(sims)


def play_one(predictor, model_is_red: bool, max_moves: int = 400,
             model_sims: int = EVAL_SIMS) -> int:
    """单局：模型 vs 启发式64。返回 (红视角) 1/0/-1。"""
    import banqi_4x8
    env = banqi_4x8.Game4x4()
    moves = 0
    while not env.terminated():
        if env.winner() is not None:
            break
        cur = env.current_player()
        if (cur == 1) == model_is_red:
            a = model_mcts_action(env, predictor, model_sims)
        else:
            a = heuristic_action(env)
        if a is None:
            break
        env.step(a)
        moves += 1
        if moves > max_moves:
            break
    w = env.winner()
    return 1 if w == 1 else (-1 if w == -1 else 0)


def play_match(predictor, n: int = 100, model_sims: int = EVAL_SIMS,
               progress: bool = False) -> Tuple[int, int, int, List[float]]:
    """n 局分块对战，返回 (wins, draws, losses, block_wr)。

    wins/draws/losses 均为模型视角（交替先后手，模型既当红也当黑）。
    block_wr 是每块（默认 20 局）的胜率，用于估计均值±std。
    """
    block = 20
    wins = draws = losses = 0
    block_wr: List[float] = []
    model_is_red = True
    blk_w = blk_tot = 0
    for i in range(n):
        r = play_one(predictor, model_is_red, model_sims=model_sims)
        if r == 0:
            draws += 1
        elif r == 1:
            wins += 1
            blk_w += 1
        else:
            losses += 1
        blk_tot += 1
        if (i + 1) % block == 0:
            block_wr.append(100.0 * blk_w / blk_tot)
            blk_w = blk_tot = 0
        model_is_red = not model_is_red
        if progress and (i + 1) % 20 == 0:
            print(f"    ... {i+1}/{n} 局", flush=True)
    return wins, draws, losses, block_wr


def report(predictor, tag: str, n: int = 100, model_sims: int = EVAL_SIMS) -> None:
    """统一打印评估报告：胜/平/负 + 分块均值±std。"""
    wins, draws, losses, blk = play_match(predictor, n=n, model_sims=model_sims)
    mean = float(np.mean(blk)) if blk else 0.0
    std = float(np.std(blk)) if blk else 0.0
    print(f"[Eval:{tag}] 胜{wins} 平{draws} 负{losses} "
          f"(n={n}, 块均胜率={mean:.1f}±{std:.1f}%)", flush=True)
    return wins, draws, losses, blk


# 便捷：构建 Predictor（兼容 verify_vs_heuristic_mcts.ModelPredictor）
def load_predictor(model_path: Optional[str] = None, device=None):
    import torch
    from config import config
    from nn_model import Banqi4x4Net, load_model_weights
    from verify_vs_heuristic_mcts import ModelPredictor
    if device is None:
        device = torch.device("cpu")
    model = Banqi4x4Net().to(device).eval()
    load_model_weights(model, model_path or config.MODEL_PATH, device)
    return ModelPredictor(model, device)


if __name__ == "__main__":
    # CLI: python eval_common.py [model.pt] [n]
    p = load_predictor(sys.argv[1] if len(sys.argv) > 1 else None)
    report(p, "main", n=int(sys.argv[2]) if len(sys.argv) > 2 else 100)
