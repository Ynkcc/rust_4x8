"""banqi/eval.py — 统一评估协议（单一入口，4x2 / 4x4 / 4x8 公共）

由原 python/game_4x4/eval_common.py 拆分而来，参数化为任意变体。
所有 diag/verify/train 脚本复用本模块的评估函数，避免"同一模型同一协议
测出 35%/60%"这类 n 太小导致的不可信结论。

协议固定（与 verify_vs_heuristic_mcts.py 默认完全一致）：
  - 模型 MCTS：num_simulations=64, max_considered_actions=16,
               c_scale=0.25, gumbel_scale=1.0（Rust 签名
               mcts_search_action(predict_fn, sims, max_acts, c_scale, gumbel_scale)）
  - 对手（主）：minimax(depth=3)（expectiminimax + alpha-beta，纯规则搜索）。
    依据：minimax(d3) vs 启发式64 = 66.25%（80 局双边，~3σ），是更强的对手；
    "vs 启发式64 五成"是低天花板假象，主评估锚定 minimax(d3)。
  - 对手（次）：启发式 MCTS 64 sims，保留作历史可比性参照。
  - 交替先后手
  - 分块统计：n 局分成 k 块（默认 5×20），输出每块胜率 + 均值±std，
    便于评估差异是否超过噪声（n=20 单块 σ≈11%）。

判定规则：主指标 vs minimax(d3) 取 n=100（SE≈3.4%），差异 <10pp 不下结论。

警告：所有评估必须统一走本模块。
"""
from __future__ import annotations

import argparse
from typing import List, Optional, Tuple

import numpy as np

from banqi.variant import Variant, get_variant

# 评估常量（唯一权威定义处，与 verify_vs_heuristic_mcts.py 默认一致）
EVAL_SIMS = 64
EVAL_MAX_ACTIONS = 16
EVAL_C_SCALE = 0.25
EVAL_GUMBEL_SCALE = 1.0
HM_SIMS = 64
# 主对手：minimax 搜索深度（Rust 侧 Game4x4.minimax_action 实现 expectiminimax）
MINIMAX_DEPTH = 3

# 对手枚举
OPP_MINIMAX3 = "minimax3"
OPP_HEURISTIC64 = "heuristic64"
OPPONENTS = (OPP_MINIMAX3, OPP_HEURISTIC64)


# ---------------------------------------------------------------------------
# Rust 绑定环境类分派（按 variant.rust_prefix）
# ---------------------------------------------------------------------------

# rust_prefix -> banqi_4x8 模块中的 pyclass 名
_ENV_CLASS_NAMES = {
    "": "DarkChess",          # 4x8
    "game4x4": "Game4x4",     # 4x4
    "mini": "MiniDarkChess",  # 4x2
}

_env_class_cache: dict = {}


def get_env_class(variant_id: str = "4x4"):
    """返回指定变体的 Rust 绑定环境类（延迟 import banqi_4x8）。"""
    if variant_id in _env_class_cache:
        return _env_class_cache[variant_id]
    variant: Variant = get_variant(variant_id)
    class_name = _ENV_CLASS_NAMES.get(variant.rust_prefix)
    if class_name is None:
        raise ValueError(
            f"未知 rust_prefix {variant.rust_prefix!r}（变体 {variant_id}），"
            f"可选: {sorted(_ENV_CLASS_NAMES)}"
        )
    import banqi_4x8  # 延迟导入，避免评估前强制加载 Rust 绑定

    cls = getattr(banqi_4x8, class_name)
    _env_class_cache[variant_id] = cls
    return cls


# ---------------------------------------------------------------------------
# 动作选择
# ---------------------------------------------------------------------------

def model_mcts_action(env, predictor, sims: int = EVAL_SIMS,
                      max_actions: int = EVAL_MAX_ACTIONS,
                      c_scale: float = EVAL_C_SCALE,
                      gumbel_scale: float = EVAL_GUMBEL_SCALE) -> int:
    """模型 MCTS 动作（与官方验证协议完全一致的参数）。"""
    return env.mcts_search_action(predictor, sims, max_actions, c_scale, gumbel_scale)


def heuristic_action(env, sims: int = HM_SIMS) -> int:
    return env.heuristic_mcts_action(sims)


def minimax_action(env, depth: int = MINIMAX_DEPTH) -> Optional[int]:
    """minimax(depth) 动作（纯规则搜索，不依赖网络）。"""
    return env.minimax_action(depth)


def opponent_action(env, opponent: str, sims: int = HM_SIMS,
                    depth: int = MINIMAX_DEPTH) -> Optional[int]:
    """按对手类型取动作。"""
    if opponent == OPP_MINIMAX3:
        return minimax_action(env, depth)
    if opponent == OPP_HEURISTIC64:
        return heuristic_action(env, sims)
    raise ValueError(f"未知对手: {opponent}（可选 {OPPONENTS}）")


# ---------------------------------------------------------------------------
# 单局 / 多局对战
# ---------------------------------------------------------------------------

def play_one(predictor, model_is_red: bool, max_moves: int = 400,
             model_sims: int = EVAL_SIMS,
             opponent: str = OPP_MINIMAX3,
             variant_id: str = "4x4",
             heuristic_sims: Optional[int] = None) -> int:
    """单局：模型 vs 指定对手。返回 (红视角) 1/0/-1。

    heuristic_sims：非 None 时覆盖启发式 MCTS 对手的模拟数（仅
    opponent 为启发式时生效），默认用 HM_SIMS(=64)。
    """
    env = get_env_class(variant_id)()
    moves = 0
    while not env.terminated():
        if env.winner() is not None:
            break
        cur = env.current_player()
        if (cur == 1) == model_is_red:
            a = model_mcts_action(env, predictor, model_sims)
        else:
            a = opponent_action(env, opponent,
                                sims=heuristic_sims if heuristic_sims is not None else HM_SIMS)
        if a is None:
            break
        env.step(a)
        moves += 1
        if moves > max_moves:
            break
    w = env.winner()
    return 1 if w == 1 else (-1 if w == -1 else 0)


def play_match(predictor, n: int = 100, model_sims: int = EVAL_SIMS,
               progress: bool = False,
               opponent: str = OPP_MINIMAX3,
               variant_id: str = "4x4",
               heuristic_sims: Optional[int] = None) -> Tuple[int, int, int, List[float]]:
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
        r = play_one(predictor, model_is_red, model_sims=model_sims,
                     opponent=opponent, variant_id=variant_id,
                     heuristic_sims=heuristic_sims)
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


def report(predictor, tag: str, n: int = 100, model_sims: int = EVAL_SIMS,
           opponent: str = OPP_MINIMAX3,
           variant_id: str = "4x4",
           heuristic_sims: Optional[int] = None) -> Tuple[int, int, int, List[float]]:
    """统一打印评估报告：胜/平/负 + 分块均值±std。

    heuristic_sims 非 None 时显示实际模拟数（如 heuristic64(sims=300)）。
    """
    wins, draws, losses, blk = play_match(predictor, n=n, model_sims=model_sims,
                                          opponent=opponent, variant_id=variant_id,
                                          heuristic_sims=heuristic_sims)
    mean = float(np.mean(blk)) if blk else 0.0
    std = float(np.std(blk)) if blk else 0.0
    opp_disp = f"{opponent}(sims={heuristic_sims})" if heuristic_sims is not None else opponent
    print(f"[Eval:{tag}] 对手={opp_disp} 胜{wins} 平{draws} 负{losses} "
          f"(n={n}, 块均胜率={mean:.1f}±{std:.1f}%)", flush=True)
    return wins, draws, losses, blk


# ---------------------------------------------------------------------------
# 便捷：构建 Predictor（兼容 verify_vs_heuristic_mcts.ModelPredictor）
# ---------------------------------------------------------------------------

class ModelPredictor:
    """包装 BanqiNet 为 eval 约定：__call__(boards, scalars) -> (logits, values)。"""

    def __init__(self, model, device) -> None:
        self.model = model.to(device).eval()
        self.device = device

    def __call__(self, boards: np.ndarray, scalars: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        import torch

        with torch.inference_mode():
            b = torch.from_numpy(np.ascontiguousarray(boards)).to(self.device)
            s = torch.from_numpy(np.ascontiguousarray(scalars)).to(self.device)
            logits, value = self.model(b, s)
            return (
                logits.cpu().numpy().astype(np.float32),
                value.cpu().numpy().reshape(-1).astype(np.float32),
            )


def load_predictor(model_path: Optional[str] = None, device=None,
                   variant_id: str = "4x4") -> ModelPredictor:
    import torch
    from banqi.config import make_config
    from banqi.nn_model import BanqiNet, load_model_weights

    if device is None:
        device = torch.device("cpu")
    model = BanqiNet(get_variant(variant_id)).to(device).eval()
    if model_path is None:
        model_path = make_config(variant_id).MODEL_PATH
    load_model_weights(model, model_path, device)
    return ModelPredictor(model, device)


# ---------------------------------------------------------------------------
# 对头评估：模型 A vs 模型 B
# ---------------------------------------------------------------------------

def play_one_vs_model(predictor_a, predictor_b, model_a_is_red: bool, max_moves: int = 400,
                      model_sims: int = EVAL_SIMS,
                      variant_id: str = "4x4") -> int:
    """单局对头：模型 A vs 模型 B。返回 A 视角 1/0/-1。"""
    env = get_env_class(variant_id)()
    moves = 0
    while not env.terminated():
        if env.winner() is not None:
            break
        cur = env.current_player()
        if (cur == 1) == model_a_is_red:
            a = model_mcts_action(env, predictor_a, model_sims)
        else:
            a = model_mcts_action(env, predictor_b, model_sims)
        if a is None:
            break
        env.step(a)
        moves += 1
        if moves > max_moves:
            break
    w = env.winner()
    return 1 if w == 1 else (-1 if w == -1 else 0)


def play_match_vs(predictor_a, predictor_b, n: int = 50, model_sims: int = EVAL_SIMS,
                  progress: bool = False,
                  variant_id: str = "4x4") -> Tuple[int, int, int, List[float]]:
    """n 局对头分块对战（模型 A vs 模型 B），返回 A 视角 (wins, draws, losses, block_wr)。"""
    block = 10
    wins = draws = losses = 0
    block_wr: List[float] = []
    a_is_red = True
    blk_w = blk_tot = 0
    for i in range(n):
        r = play_one_vs_model(predictor_a, predictor_b, a_is_red,
                              model_sims=model_sims, variant_id=variant_id)
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
        a_is_red = not a_is_red
        if progress and (i + 1) % 20 == 0:
            print(f"    ... {i+1}/{n} 局", flush=True)
    return wins, draws, losses, block_wr


def report_vs(predictor_a, predictor_b, tag: str, n: int = 50,
              model_sims: int = EVAL_SIMS,
              variant_id: str = "4x4") -> Tuple[int, int, int, List[float]]:
    """对头评估报告：A 视角 胜/平/负 + 分块均值±std。"""
    wins, draws, losses, blk = play_match_vs(
        predictor_a, predictor_b, n=n, model_sims=model_sims, variant_id=variant_id
    )
    mean = float(np.mean(blk)) if blk else 0.0
    std = float(np.std(blk)) if blk else 0.0
    print(f"[EvalVs:{tag}] 胜{wins} 平{draws} 负{losses} "
          f"(n={n}, 块均胜率={mean:.1f}±{std:.1f}%)", flush=True)
    return wins, draws, losses, blk


if __name__ == "__main__":
    # CLI: python -m banqi.eval [model.pt] [n] [--opponent minimax3|heuristic64]
    #      python -m banqi.eval [model.pt] [n] --vs <ckpt.pt>   (对头评估)
    ap = argparse.ArgumentParser(description="暗棋模型统一评估（主对手 minimax(d3)）")
    ap.add_argument("model_path", nargs="?", default=None, help="模型权重路径（默认 config.MODEL_PATH）")
    ap.add_argument("n", nargs="?", type=int, default=100, help="评估局数（默认 100）")
    ap.add_argument("--variant", default="4x4", choices=("4x2", "4x4", "4x8"),
                    help="棋盘变体（默认 4x4）")
    ap.add_argument("--opponent", choices=OPPONENTS, default=OPP_MINIMAX3,
                    help=f"对手类型（默认 {OPP_MINIMAX3}；与 --vs 互斥）")
    ap.add_argument("--heuristic-sims", type=int, default=None,
                    help="启发式 MCTS 对手的模拟数（默认 HM_SIMS=64；"
                         "仅 --opponent heuristic64 时生效）")
    ap.add_argument("--vs", default=None, help="对头评估：对方模型权重路径（覆盖 --opponent）")
    args = ap.parse_args()
    p = load_predictor(args.model_path, variant_id=args.variant)
    if args.vs:
        pb = load_predictor(args.vs, variant_id=args.variant)
        report_vs(p, pb, "main", n=args.n, variant_id=args.variant)
    else:
        report(p, "main", n=args.n, opponent=args.opponent,
               variant_id=args.variant, heuristic_sims=args.heuristic_sims)
