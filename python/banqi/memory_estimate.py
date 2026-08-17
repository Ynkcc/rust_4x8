"""banqi/memory_estimate.py — 挂起单局游戏 (游戏状态 + MCTS 树) 内存占用估算（公共）

由原 python/predictor_entry.py 的内存估算部分拆分而来，参数化为任意变体
（4x2 / 4x4 / 4x8），估算 Rust `DarkChessEnv` 环境、NN 输入、MCTS 节点、
搜索树与整局训练数据（GameEpisode）的字节占用。

所有估算函数接受 `variant_id` 参数（默认 "4x8"），尺寸常量经
`banqi.constants.build_constants` 按变体派生，公式对三个变体通用。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import List, Tuple

from banqi.constants import Constants, build_constants
from banqi.variant import get_variant

MAX_STEPS_PER_EPISODE = 100


@lru_cache(maxsize=None)
def _constants_for(variant_id: str) -> Constants:
    return build_constants(get_variant(variant_id))


def _reveal_probability_size(variant_id: str) -> int:
    return 2 * _constants_for(variant_id).NUM_PIECE_TYPES


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


def estimate_dark_chess_env(variant_id: str = "4x8") -> int:
    """估算 DarkChessEnv (游戏环境) 的内存大小 (字节)"""
    c = _constants_for(variant_id)
    total_positions = c.TOTAL_POSITIONS
    # [Slot; 32] - 枚举，约 2B/个 (tag + Piece)
    board_size = total_positions * 2
    current_player = 1
    counters = 8 * 2
    piece_bbs = 2 * c.NUM_PIECE_TYPES * 8
    revealed_bbs = 2 * 8
    hidden_empty_bbs = 2 * 8
    dead_pool = 2 * c.TOTAL_PIECES_PER_PLAYER * 1
    dead_count = 2 * 8
    scores = 2 * 4
    last_action = 4
    hidden_pool = total_positions * 2
    hidden_count = 8
    reveal_probs = _reveal_probability_size(variant_id) * 4
    opts = (1 + 8) + (1 + total_positions * 2)
    return (
        board_size + current_player + counters + piece_bbs
        + revealed_bbs + hidden_empty_bbs + dead_pool + dead_count
        + scores + last_action + hidden_pool + hidden_count
        + reveal_probs + opts
    )


def estimate_observation(variant_id: str = "4x8") -> int:
    """估算 Observation (NN输入) = board tensor + scalars tensor"""
    c = _constants_for(variant_id)
    board_data = c.BOARD_CHANNELS * c.BOARD_ROWS * c.BOARD_COLS * 4
    ndarray_overhead = 56
    scalar_data = c.SCALAR_FEATURE_COUNT * 4
    return board_data + scalar_data + 2 * ndarray_overhead


def estimate_game_state_suspended(variant_id: str = "4x8") -> MemoryEstimate:
    """估算单个挂起的游戏状态占用"""
    c = _constants_for(variant_id)
    est = MemoryEstimate()
    est.add("DarkChessEnv (游戏环境)", estimate_dark_chess_env(variant_id),
            f"{c.BOARD_ROWS}x{c.BOARD_COLS}棋盘+位棋盘+棋子池+概率表")
    est.add("Box<DarkChessEnv> 指针", 8, "Box 指针占用")
    est.add("Observation (NN输入)", estimate_observation(variant_id),
            f"board({c.BOARD_CHANNELS},{c.BOARD_ROWS},{c.BOARD_COLS})f32 + "
            f"scalars({c.SCALAR_FEATURE_COUNT},)f32")
    est.add("Option<Observation> tag", 1, "Option 判别位")
    est.add("Action Mask (动作掩码)", c.ACTION_SPACE_SIZE * 4,
            f"{c.ACTION_SPACE_SIZE} 个动作的 i32 掩码")
    est.add("Policy π (策略分布)", c.ACTION_SPACE_SIZE * 4,
            f"{c.ACTION_SPACE_SIZE} 个动作的 f32 概率")
    est.add("杂项标量 (value/Q/N 等)", 4 * 4 + 2 * 4,
            "MCTS value, Q, visit count 等")
    return est


def estimate_mcts_node(
    avg_children: int,
    avg_possible_states: int,
    has_env: bool,
    has_state: bool,
    variant_id: str = "4x8",
) -> Tuple[int, MemoryEstimate]:
    """估算单个 MctsNode 及其动态分配的大小"""
    c = _constants_for(variant_id)
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
        env_sz = estimate_dark_chess_env(variant_id)
        est.add("Option<Box<DarkChessEnv>>", 8 + 1 + env_sz,
                "部分节点保存完整环境副本")

    if has_state:
        obs_sz = estimate_observation(variant_id)
        est.add("Option<Observation>", obs_sz + 1,
                "部分节点缓存 NN 输入特征")

    total = (fixed_size + children_data + 24
             + possible_data + 24
             + (8 + 1 + estimate_dark_chess_env(variant_id) if has_env else 0)
             + (estimate_observation(variant_id) + 1 if has_state else 0))
    return total, est


def estimate_mcts_tree(mcts_sims: int, variant_id: str = "4x8") -> MemoryEstimate:
    """估算一棵 MCTS 搜索树的总内存"""
    c = _constants_for(variant_id)
    est = MemoryEstimate()
    reveal_prob = _reveal_probability_size(variant_id)

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

    regular_size, _ = estimate_mcts_node(8, 0, False, False, variant_id)
    regular_count = int(total_nodes * (1.0 - chance_node_ratio))
    est.add(
        f"普通决策节点 × {regular_count:,}",
        regular_size * regular_count,
        "avg 8 children, 无 env/state",
    )

    chance_size, _ = estimate_mcts_node(2, reveal_prob, False, False, variant_id)
    chance_count = total_nodes - regular_count
    est.add(
        f"机会节点 × {chance_count:,}",
        chance_size * chance_count,
        f"possible_states ≈ {reveal_prob} outcomes",
    )

    env_only_size = estimate_dark_chess_env(variant_id) + 8 + 1
    env_nodes = int(total_nodes * env_coverage)
    est.add(
        f"带 Env 副本的节点 × {env_nodes:,}",
        env_only_size * env_nodes,
        f"{env_coverage*100:.0f}% 节点保存完整游戏环境",
    )

    state_only_size = estimate_observation(variant_id) + 1
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


def estimate_episode_storage(game_length: int, variant_id: str = "4x8") -> MemoryEstimate:
    """估算一局游戏的训练数据 GameEpisode 大小"""
    c = _constants_for(variant_id)
    est = MemoryEstimate()
    est.add(f"GameEpisode: {game_length} 步样本", 0,
            "单局游戏完整训练数据 (含训练样本)")

    step_board = c.BOARD_CHANNELS * c.BOARD_ROWS * c.BOARD_COLS * 4
    step_scalars = c.SCALAR_FEATURE_COUNT * 4
    step_policy = c.ACTION_SPACE_SIZE * 4
    step_mask = c.ACTION_SPACE_SIZE * 4
    step_scalar_fields = 3 * 4 + 4 + 4
    per_step = step_board + step_scalars + step_policy + step_mask + step_scalar_fields

    est.add(
        "  每步 Observation+Policy+Mask+Scalars",
        per_step,
        f"board({c.BOARD_CHANNELS},{c.BOARD_ROWS},{c.BOARD_COLS}) + "
        f"scalars({c.SCALAR_FEATURE_COUNT}) + policy({c.ACTION_SPACE_SIZE}) + "
        f"mask({c.ACTION_SPACE_SIZE}) + value/Q/N/result",
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
    variant_id: str = "4x8",
) -> MemoryEstimate:
    """估算挂起单局游戏的总内存 = 游戏状态 + MCTS 树 + (可选) Episode 存储"""
    est = MemoryEstimate()

    est.add("=== 挂起单局游戏: 运行时内存 (MCTS 决策峰值) ===", 0,
            "实际 MCTS 搜索时的峰值内存占用")

    state_est = estimate_game_state_suspended(variant_id)
    for b in state_est.breakdown:
        est.add(b.item, b.size_bytes, b.note)

    tree_est = estimate_mcts_tree(mcts_sims, variant_id)
    for b in tree_est.breakdown:
        est.add(b.item, b.size_bytes, b.note)

    est.add("--- 子项小计: 游戏状态 + MCTS 树 (运行时) ---",
            state_est.total_bytes + tree_est.total_bytes,
            "当次 MCTS 决策时的峰值占用")

    est.add("\n=== 挂起单局游戏: 训练数据存储 (GameEpisode) ===".strip(),
            0, "整局结束后保存的训练数据")
    ep_est = estimate_episode_storage(expected_total_steps, variant_id)
    for b in ep_est.breakdown:
        est.add(b.item, b.size_bytes, b.note)

    return est


def print_memory_estimate_report(
    mcts_sims: int,
    games_per_iter: int = 1,
    num_workers: int = 1,
    variant_id: str = "4x8",
) -> None:
    """打印完整的内存估算报告（Python 侧入口函数）"""
    sep = "=" * 72
    print(f"\n{sep}")
    print(f"  🧮 Python 内存估算报告 ({variant_id})")
    print(f"     mcts_sims={mcts_sims}, games_per_iter={games_per_iter}, "
          f"workers={num_workers}")
    print(f"{sep}")

    env_sz = estimate_dark_chess_env(variant_id)
    obs_sz = estimate_observation(variant_id)
    node_sz, _ = estimate_mcts_node(8, 0, False, False, variant_id)
    print(f"\n  [基础结构]  DarkChessEnv = {env_sz:,} B ({env_sz/1024:.1f} KB)")
    print(f"  [基础结构]  Observation  = {obs_sz:,} B ({obs_sz/1024:.1f} KB)")
    print(f"  [基础结构]  MctsNode(普通) ≈ {node_sz:,} B\n")

    estimate_game_state_suspended(variant_id).print_report(
        "① 单个游戏状态 (Suspended Game State)")

    estimate_mcts_tree(mcts_sims, variant_id).print_report(
        f"② 单次 MCTS 搜索树 (sims={mcts_sims})")

    estimate_episode_storage(MAX_STEPS_PER_EPISODE, variant_id).print_report(
        f"③ 单局训练数据 GameEpisode (max {MAX_STEPS_PER_EPISODE} 步)")

    single = estimate_single_game_suspended(mcts_sims, variant_id=variant_id)
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
    variant_id: str = "4x8",
) -> int:
    """
    程序化 API：返回挂起单局游戏需要预留的内存字节数
    """
    state_bytes = estimate_game_state_suspended(variant_id).total_bytes
    tree_bytes = estimate_mcts_tree(mcts_sims, variant_id).total_bytes
    total = state_bytes + tree_bytes
    if include_episode_storage:
        total += estimate_episode_storage(
            expected_game_length, variant_id
        ).total_bytes
    return int(total * safety_factor)
