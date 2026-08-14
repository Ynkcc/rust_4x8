"""
predictor_entry.py — PyO3 独立 Rust bin (banqi-py-collector) 使用的入口脚本。

Rust bin 会：
1. 以 Python 嵌入模式启动
2. import 本模块
3. 调用 `predict(board, scalars)` 做神经网络推理（内部按 PREDICT_BATCH=32 分块）
4. 可选调用 `save_episodes(episode_dicts)` 保存整局记录

环境变量 (由 Rust bin 读取)：
    PY_PREDICTOR_MODULE = ./python/predictor_entry.py   (默认)
    PY_PREDICT_FUNC    = predict                         (默认)
    PY_SAVE_FUNC       = save_episodes                   (默认)
    OUTPUT_DIR         = ./training_data/py_collected    (默认，若没有 PY_SAVE_FUNC 则写入 JSON 到这里)
    MCTS_SIMS          = 64
    GAMES_PER_ITERATION = 100
    WORKER_ID          = CLI argv[1]
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Tuple

import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:  # pragma: no cover
    HAS_TORCH = False

from constant import (
    ACTION_SPACE_SIZE,
    BOARD_CHANNELS,
    BOARD_ROWS,
    BOARD_COLS,
    SCALAR_FEATURE_COUNT,
    TOTAL_INPUT_CHANNELS,
)
from nn_model import BanqiNet
from storage import to_json_safe


# ---------------------------------------------------------------------------
# 全局模型实例（带简易热重载）
# ---------------------------------------------------------------------------

_MODEL: "BanqiNet | None" = None
_MODEL_PATH: str | None = None
_MODEL_MTIME: float = 0.0
_DEVICE = None


def _reload_if_updated() -> None:
    """若 MODEL_PATH 指向的文件有更新则重载；没有模型就新建一个。"""
    global _MODEL, _MODEL_PATH, _MODEL_MTIME, _DEVICE

    if not HAS_TORCH:
        return

    model_path = os.environ.get("MODEL_PATH")
    if _DEVICE is None:
        _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    need_load = _MODEL is None
    if model_path and os.path.exists(model_path):
        mtime = os.path.getmtime(model_path)
        if _MODEL_PATH != model_path or mtime > _MODEL_MTIME:
            need_load = True
            _MODEL_PATH = model_path
            _MODEL_MTIME = mtime

    if _MODEL is None:
        _MODEL = BanqiNet().to(_DEVICE)
        print(f"[predictor_entry] Initialized new BanqiNet on {_DEVICE}")

    if need_load and model_path and os.path.exists(model_path):
        try:
            st = torch.load(model_path, map_location=_DEVICE, weights_only=True)
            if hasattr(st, "state_dict"):
                _MODEL.load_state_dict(st.state_dict())
            elif isinstance(st, dict) and "model_state_dict" in st:
                _MODEL.load_state_dict(st["model_state_dict"])
            else:
                _MODEL.load_state_dict(st)
            print(f"[predictor_entry] Loaded weights from {model_path}")
        except Exception as exc:  # pragma: no cover
            print(f"[predictor_entry] Failed to load {model_path}: {exc}")

    _MODEL.eval()


def _infer_batch(board: np.ndarray, scalars: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """单次推理，输入已是完整 batch（不拆分）。"""
    from config import config

    with torch.no_grad():
        b = torch.from_numpy(np.ascontiguousarray(board)).to(_DEVICE)
        s = torch.from_numpy(np.ascontiguousarray(scalars)).to(_DEVICE)
        logits, value = _MODEL(b, s)  # type: ignore[misc]
        # 忽略未使用的 import，避免仅用于类型提示时的告警
        _ = config
        return (
            logits.detach().cpu().numpy().astype(np.float32),
            value.detach().cpu().numpy().reshape(-1).astype(np.float32),
        )


# ---------------------------------------------------------------------------
# Rust 回调：预测接口 (输入 numpy，返回 numpy；内部按 PREDICT_BATCH=32 分块)
# ---------------------------------------------------------------------------

def predict(board: np.ndarray, scalars: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rust MCTS 每次评估都会调用这里。
    参数:
        board:  (N, BOARD_CHANNELS, BOARD_ROWS, BOARD_COLS) float32
        scalars:(N, SCALAR_FEATURE_COUNT) float32
    返回:
        policy_logits: (N, ACTION_SPACE_SIZE) float32
        values:        (N,) float32

    注意：Rust 侧 envs.len() 可能任意，这里按 PREDICT_BATCH 分块送入模型，
    再拼接结果，避免一次性推理过大 batch 导致显存/内存峰值。
    """
    from config import config

    batch = board.shape[0]

    if not HAS_TORCH:
        # 无 torch 时的退化：均匀 logits + 0 值
        return (
            np.zeros((batch, ACTION_SPACE_SIZE), dtype=np.float32),
            np.zeros(batch, dtype=np.float32),
        )

    _reload_if_updated()

    if batch == 0:
        return (
            np.zeros((0, ACTION_SPACE_SIZE), dtype=np.float32),
            np.zeros(0, dtype=np.float32),
        )

    chunk = config.PREDICT_BATCH
    if batch <= chunk:
        return _infer_batch(board, scalars)

    policy_list: List[np.ndarray] = []
    value_list: List[np.ndarray] = []
    for i in range(0, batch, chunk):
        pl, vl = _infer_batch(board[i : i + chunk], scalars[i : i + chunk])
        policy_list.append(pl)
        value_list.append(vl)

    return (
        np.concatenate(policy_list, axis=0).astype(np.float32),
        np.concatenate(value_list, axis=0).astype(np.float32),
    )


# ---------------------------------------------------------------------------
# Rust 回调（可选）：把一局或多局完整数据交给 Python 处理/保存
# ---------------------------------------------------------------------------

def save_episodes(episodes: List[Dict[str, Any]]) -> None:
    """
    episodes: 每局一个 dict, 包含:
        game_length, winner, iteration, worker_id,
        boards, scalars, policies, mcts_values, completed_qs,
        root_visits, game_results, action_masks
    默认实现：追加写 jsonl；若要存到数据库 / 训练样本池，可在此修改。
    """
    out_dir = os.environ.get("OUTPUT_DIR", "./training_data/py_collected")
    os.makedirs(out_dir, exist_ok=True)

    worker_id = episodes[0].get("worker_id", 0) if episodes else 0
    iteration = episodes[0].get("iteration", 0) if episodes else 0
    jsonl_path = os.path.join(
        out_dir, f"iter_{iteration:06d}_worker_{worker_id:03d}.jsonl"
    )
    with open(jsonl_path, "a", encoding="utf-8") as fp:
        for ep in episodes:
            fp.write(json.dumps(to_json_safe(ep), ensure_ascii=False))
            fp.write("\n")

    print(
        f"[predictor_entry] append {len(episodes)} episodes -> {jsonl_path}"
        f" (total now {_count_lines(jsonl_path)} lines)"
    )


def _count_lines(path: str) -> int:
    try:
        with open(path, "rb") as fp:
            return sum(1 for _ in fp)
    except OSError:
        return 0


# ============================================================================
# 内存估算模块：估计挂起单局游戏 (游戏状态 + MCTS) 需要预留的内存大小
# ============================================================================

from dataclasses import dataclass, field
from typing import List, Tuple

from constant import NUM_PIECE_TYPES, TOTAL_PIECES_PER_PLAYER


MAX_STEPS_PER_EPISODE = 100
REVEAL_PROBABILITY_SIZE = 2 * NUM_PIECE_TYPES
TOTAL_POSITIONS = BOARD_ROWS * BOARD_COLS


def _sizeof_fmt(num_bytes: int) -> str:
    """人类可读的字节大小格式"""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:,.1f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:,.1f} PB"


@dataclass
class MemoryBreakdown:
    item: str
    size_bytes: int
    note: str = ""


@dataclass
class MemoryEstimate:
    breakdown: List[MemoryBreakdown] = field(default_factory=list)
    total_bytes: int = 0

    def add(self, item: str, size_bytes: int, note: str = "") -> None:
        self.breakdown.append(MemoryBreakdown(item, size_bytes, note))
        self.total_bytes += size_bytes

    @property
    def total_kb(self) -> float:
        return self.total_bytes / 1024.0

    @property
    def total_mb(self) -> float:
        return self.total_kb / 1024.0

    def print_report(self, title: str) -> None:
        sep = "=" * 88
        print(f"\n{sep}")
        print(f"  {title}")
        print(f"{sep}")
        header = f"  {'项目':<46} {'字节 (B)':>14} {'KB':>10}  说明"
        print(header)
        print(f"  {'-' * 86}")
        for b in self.breakdown:
            kb = b.size_bytes / 1024.0
            print(
                f"  {b.item:<46} {b.size_bytes:>14,} {kb:>10.2f}  {b.note}"
            )
        print(f"  {'-' * 86}")
        print(
            f"  {'TOTAL':<46} {self.total_bytes:>14,} {self.total_kb:>10.2f}"
            f"  ({self.total_mb:.2f} MB)"
        )
        print(f"{sep}\n")


def estimate_dark_chess_env() -> int:
    """估算 DarkChessEnv (游戏环境) 的内存大小 (字节)"""
    # [Slot; 32] - 枚举，约 2B/个 (tag + Piece)
    board_size = TOTAL_POSITIONS * 2
    current_player = 1
    counters = 8 * 2
    piece_bbs = 2 * NUM_PIECE_TYPES * 8
    revealed_bbs = 2 * 8
    hidden_empty_bbs = 2 * 8
    dead_pool = 2 * TOTAL_PIECES_PER_PLAYER * 1
    dead_count = 2 * 8
    scores = 2 * 4
    last_action = 4
    hidden_pool = TOTAL_POSITIONS * 2
    hidden_count = 8
    reveal_probs = REVEAL_PROBABILITY_SIZE * 4
    opts = (1 + 8) + (1 + TOTAL_POSITIONS * 2)
    return (
        board_size + current_player + counters + piece_bbs
        + revealed_bbs + hidden_empty_bbs + dead_pool + dead_count
        + scores + last_action + hidden_pool + hidden_count
        + reveal_probs + opts
    )


def estimate_observation() -> int:
    """估算 Observation (NN输入) = board tensor + scalars tensor"""
    board_data = BOARD_CHANNELS * BOARD_ROWS * BOARD_COLS * 4
    ndarray_overhead = 56
    scalar_data = SCALAR_FEATURE_COUNT * 4
    return board_data + scalar_data + 2 * ndarray_overhead


def estimate_game_state_suspended() -> MemoryEstimate:
    """估算单个挂起的游戏状态占用"""
    est = MemoryEstimate()
    est.add("DarkChessEnv (游戏环境)", estimate_dark_chess_env(),
            "4x8棋盘+位棋盘+棋子池+概率表")
    est.add("Box<DarkChessEnv> 指针", 8, "Box 指针占用")
    est.add("Observation (NN输入)", estimate_observation(),
            "board(16,4,8)f32 + scalars(35,)f32")
    est.add("Option<Observation> tag", 1, "Option 判别位")
    est.add("Action Mask (动作掩码)", ACTION_SPACE_SIZE * 4,
            f"{ACTION_SPACE_SIZE} 个动作的 i32 掩码")
    est.add("Policy π (策略分布)", ACTION_SPACE_SIZE * 4,
            f"{ACTION_SPACE_SIZE} 个动作的 f32 概率")
    est.add("杂项标量 (value/Q/N 等)", 4 * 4 + 2 * 4,
            "MCTS value, Q, visit count 等")
    return est


def estimate_mcts_node(
    avg_children: int,
    avg_possible_states: int,
    has_env: bool,
    has_state: bool,
) -> Tuple[int, MemoryEstimate]:
    """估算单个 MctsNode 及其动态分配的大小"""
    est = MemoryEstimate()

    fixed_size = (
        4                       # visit_count: u32
        + 4 * 4                 # value_sum, prior, logit, initial_value: f32
        + 24                    # children Vec (ptr+len+cap)
        + 4                     # 4 bools
        + 24                    # possible_states Vec
        + 9                     # Option<Box<Env>> (tag + ptr)
        + 1                     # Player enum
        + 16                    # Option<Observation> approx
    )
    est.add("MctsNode 固定字段", fixed_size,
            "不含 Vec/Box/Option 的内部数据")

    children_data = avg_children * (8 * 2)
    est.add(
        f"children Vec (avg {avg_children} entries)",
        children_data + 24,
        "(action_idx, node_idx) 对",
    )

    possible_data = avg_possible_states * (8 + 4 + 8)
    est.add(
        f"possible_states Vec (avg {avg_possible_states})",
        possible_data + 24,
        "机会节点 (outcome, prob, node) 三元组",
    )

    if has_env:
        env_sz = estimate_dark_chess_env()
        est.add("Option<Box<DarkChessEnv>>", 8 + 1 + env_sz,
                "部分节点保存完整环境副本")

    if has_state:
        obs_sz = estimate_observation()
        est.add("Option<Observation>", obs_sz + 1,
                "部分节点缓存 NN 输入特征")

    total = (fixed_size + children_data + 24
             + possible_data + 24
             + (8 + 1 + estimate_dark_chess_env() if has_env else 0)
             + (estimate_observation() + 1 if has_state else 0))
    return total, est


def estimate_mcts_tree(mcts_sims: int) -> MemoryEstimate:
    """估算一棵 MCTS 搜索树的总内存"""
    est = MemoryEstimate()

    avg_branching_factor = 16.0
    total_nodes = int(mcts_sims * avg_branching_factor * 0.6)
    env_coverage = 0.15
    state_coverage = 0.35
    chance_node_ratio = 0.20

    est.add(
        f"MCTS 总节点数 (估算)",
        total_nodes,
        f"sims={mcts_sims}, avg_branch={avg_branching_factor:.1f}, factor=0.6",
    )

    regular_size, _ = estimate_mcts_node(8, 0, False, False)
    regular_count = int(total_nodes * (1.0 - chance_node_ratio))
    est.add(
        f"普通决策节点 × {regular_count:,}",
        regular_size * regular_count,
        "avg 8 children, 无 env/state",
    )

    chance_size, _ = estimate_mcts_node(2, REVEAL_PROBABILITY_SIZE, False, False)
    chance_count = total_nodes - regular_count
    est.add(
        f"机会节点 × {chance_count:,}",
        chance_size * chance_count,
        f"possible_states ≈ {REVEAL_PROBABILITY_SIZE} outcomes",
    )

    env_only_size = estimate_dark_chess_env() + 8 + 1
    env_nodes = int(total_nodes * env_coverage)
    est.add(
        f"带 Env 副本的节点 × {env_nodes:,}",
        env_only_size * env_nodes,
        f"{env_coverage*100:.0f}% 节点保存完整游戏环境",
    )

    state_only_size = estimate_observation() + 1
    state_nodes = int(total_nodes * state_coverage)
    est.add(
        f"带 Observation 缓存的节点 × {state_nodes:,}",
        state_only_size * state_nodes,
        f"{state_coverage*100:.0f}% 节点缓存 NN 输入特征",
    )

    slab_overhead = 32 + total_nodes
    est.add("Slab<MctsNode> 内存池开销", slab_overhead,
            "Slab 元数据 + 空洞 (~1B/节点)")

    return est


def estimate_episode_storage(game_length: int) -> MemoryEstimate:
    """估算一局游戏的训练数据 GameEpisode 大小"""
    est = MemoryEstimate()
    est.add(f"GameEpisode: {game_length} 步样本", 0,
            "单局游戏完整训练数据 (含训练样本)")

    step_board = BOARD_CHANNELS * BOARD_ROWS * BOARD_COLS * 4
    step_scalars = SCALAR_FEATURE_COUNT * 4
    step_policy = ACTION_SPACE_SIZE * 4
    step_mask = ACTION_SPACE_SIZE * 4
    step_scalar_fields = 3 * 4 + 4 + 4
    per_step = step_board + step_scalars + step_policy + step_mask + step_scalar_fields

    est.add(
        "  每步 Observation+Policy+Mask+Scalars",
        per_step,
        "board(16,4,8) + scalars(35) + policy(352) + mask(352) + value/Q/N/result",
    )
    est.add(
        f"  {game_length} 步样本数据小计",
        per_step * game_length,
        "samples Vec 数据内容",
    )
    vec_meta = game_length * (8 + 24 + 24 + 24)
    est.add("  samples Vec 元数据 (tuple+Vec 开销)",
            vec_meta, "嵌套 Vec 结构的分配器开销")
    est.add("  Episode 头部字段", 8 + 1 + 24,
            "game_length + winner + samples Vec 元")
    return est


def estimate_single_game_suspended(
    mcts_sims: int,
    expected_total_steps: int = MAX_STEPS_PER_EPISODE,
) -> MemoryEstimate:
    """估算挂起单局游戏的总内存 = 游戏状态 + MCTS 树 + (可选) Episode 存储"""
    est = MemoryEstimate()

    est.add("=== 挂起单局游戏: 运行时内存 (MCTS 决策峰值) ===", 0,
            "实际 MCTS 搜索时的峰值内存占用")

    state_est = estimate_game_state_suspended()
    for b in state_est.breakdown:
        est.add(b.item, b.size_bytes, b.note)

    tree_est = estimate_mcts_tree(mcts_sims)
    for b in tree_est.breakdown:
        est.add(b.item, b.size_bytes, b.note)

    est.add("--- 子项小计: 游戏状态 + MCTS 树 (运行时) ---",
            state_est.total_bytes + tree_est.total_bytes,
            "当次 MCTS 决策时的峰值占用")

    est.add("\n=== 挂起单局游戏: 训练数据存储 (GameEpisode) ===".strip(),
            0, "整局结束后保存的训练数据")
    ep_est = estimate_episode_storage(expected_total_steps)
    for b in ep_est.breakdown:
        est.add(b.item, b.size_bytes, b.note)

    return est


def print_memory_estimate_report(
    mcts_sims: int,
    games_per_iter: int = 1,
    num_workers: int = 1,
) -> None:
    """打印完整的内存估算报告（Python 侧入口函数）"""
    sep = "=" * 72
    print(f"\n{sep}")
    print(f"  🧮 Python 内存估算报告")
    print(f"     mcts_sims={mcts_sims}, games_per_iter={games_per_iter}, "
          f"workers={num_workers}")
    print(f"{sep}")

    env_sz = estimate_dark_chess_env()
    obs_sz = estimate_observation()
    node_sz, _ = estimate_mcts_node(8, 0, False, False)
    print(f"\n  [基础结构]  DarkChessEnv = {env_sz:,} B ({env_sz/1024:.1f} KB)")
    print(f"  [基础结构]  Observation  = {obs_sz:,} B ({obs_sz/1024:.1f} KB)")
    print(f"  [基础结构]  MctsNode(普通) ≈ {node_sz:,} B\n")

    estimate_game_state_suspended().print_report(
        "① 单个游戏状态 (Suspended Game State)")

    estimate_mcts_tree(mcts_sims).print_report(
        f"② 单次 MCTS 搜索树 (sims={mcts_sims})")

    estimate_episode_storage(MAX_STEPS_PER_EPISODE).print_report(
        f"③ 单局训练数据 GameEpisode (max {MAX_STEPS_PER_EPISODE} 步)")

    single = estimate_single_game_suspended(mcts_sims)
    single.print_report(
        f"④ 挂起单局游戏 = 状态 + MCTS 树 (sims={mcts_sims})")

    safety_low = single.total_mb * 1.5
    safety_high = single.total_mb * 2.0
    parallel = games_per_iter * num_workers
    print("  ⚠️  安全余量建议: 理论估算值 × (1.5 ~ 2.0) 以覆盖实际分配开销")
    print(f"     - 单局挂起 (sims={mcts_sims}) 建议预留: "
          f"{safety_low:.0f} ~ {safety_high:.0f} MB")
    if parallel > 1:
        print(f"     - {parallel} 路并行 (workers×games) 建议预留: "
              f"{safety_low * parallel:.0f} ~ {safety_high * parallel:.0f} MB")
    print(f"{sep}\n")


def estimate_memory_bytes(
    mcts_sims: int,
    expected_game_length: int = MAX_STEPS_PER_EPISODE,
    include_episode_storage: bool = True,
    safety_factor: float = 1.5,
) -> int:
    """
    程序化 API：返回挂起单局游戏需要预留的内存字节数
    """
    state_bytes = estimate_game_state_suspended().total_bytes
    tree_bytes = estimate_mcts_tree(mcts_sims).total_bytes
    total = state_bytes + tree_bytes
    if include_episode_storage:
        total += estimate_episode_storage(expected_game_length).total_bytes
    return int(total * safety_factor)


if __name__ == "__main__":
    bs = 4
    dummy_board = np.random.randn(bs, TOTAL_INPUT_CHANNELS, BOARD_ROWS, BOARD_COLS).astype(np.float32)
    dummy_scalars = np.random.randn(bs, SCALAR_FEATURE_COUNT).astype(np.float32)
    pl, vl = predict(dummy_board, dummy_scalars)
    print("predict() output shapes:", pl.shape, vl.shape)
    print("expected:               ", (bs, ACTION_SPACE_SIZE), (bs,))
    assert pl.shape == (bs, ACTION_SPACE_SIZE)
    assert vl.shape == (bs,)
    print("OK")
