"""
memory_estimate.py — ReplayBuffer 最大内存占用估算（无 CLI 参数）

从 run_self_play_and_train.py 精简而来，专注于「估算 ReplayBuffer 最大内存占用」。

补全说明（对比早期仅调用 estimate_episode_storage 的版本）：
- ReplayBuffer 的真实实现是 training_service.DataBuffer，其容量单位是【样本数】，
  由 config.MAX_SAMPLE_BUFFER_SIZE 决定（而非"局数"）。本脚本据此做精确估算。
- DataBuffer 每个字段用 Python list 存储，元素为独立 numpy 数组
  (boards / scalars / probs / masks) 或 Python 标量 (values / root_visits)，
  因此必须计入 numpy 数组对象头、Python 浮点/整型对象与 list 指针开销。
- 复用 predictor_entry._sizeof_fmt 做人类可读格式化；基础维度取自 constant。

用法：
    python python/memory_estimate.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from config import config
from banqi.variant import get_variant
from banqi.constants import build_constants

VARIANT = get_variant("4x8")
C = build_constants(VARIANT)
ACTION_SPACE_SIZE = C.ACTION_SPACE_SIZE
BOARD_COLS = C.BOARD_COLS
BOARD_ROWS = C.BOARD_ROWS
SCALAR_FEATURE_COUNT = C.SCALAR_FEATURE_COUNT
TOTAL_INPUT_CHANNELS = C.TOTAL_INPUT_CHANNELS
from predictor_entry import (
    MAX_STEPS_PER_EPISODE,
    _sizeof_fmt,
    estimate_memory_bytes,
    estimate_single_game_suspended,
)

# 数据字段对应 DataBuffer 的存储维度（float32 numpy 数组）
BOARD_SHAPE = (TOTAL_INPUT_CHANNELS, BOARD_ROWS, BOARD_COLS)
SCALAR_SHAPE = (SCALAR_FEATURE_COUNT,)
POLICY_SHAPE = (ACTION_SPACE_SIZE,)
MASK_SHAPE = (ACTION_SPACE_SIZE,)

# 单个 numpy 数组对象头开销（ndarray.base + shape + strides + dtype 等，经验值）
NDARRAY_OBJECT_BYTES = 112
# Python list 每个元素一个指针
LIST_PTR_BYTES = 8
# Python float / int 对象（CPython 29/32 位小对象，经验值）
PY_FLOAT_BYTES = 28
PY_INT_BYTES = 28


@dataclass
class _FieldEstimate:
    name: str
    per_sample_bytes: int
    array_object_bytes: int   # 每个数组的 numpy 对象头（values/root_visits 用标量对象）
    note: str


def _estimate_field(
    name: str, data_bytes: int, count: int, note: str,
    per_object_bytes: int = NDARRAY_OBJECT_BYTES,
) -> _FieldEstimate:
    """单个字段估算：数据字节 + numpy/Python 对象头 + list 指针。"""
    array_objects = count * per_object_bytes
    list_ptrs = count * LIST_PTR_BYTES
    total = data_bytes + array_objects + list_ptrs
    return _FieldEstimate(name, total, array_objects, note)


@dataclass
class BufferEstimate:
    """一个 DataBuffer（train 或 val）的完整内存估算。"""
    label: str
    capacity: int              # 样本容量
    field_estimates: List[_FieldEstimate]
    total_bytes: int

    def summary(self) -> str:
        per_sample = sum(f.per_sample_bytes for f in self.field_estimates)
        return (
            f"{self.label}: 容量 {self.capacity:,} 样本 ≈ "
            f"{_sizeof_fmt(self.total_bytes)}"
            f"（每样本约 {per_sample} B / {_sizeof_fmt(per_sample)}）"
        )


def build_buffer_field_estimates() -> List[_FieldEstimate]:
    """估算单个样本的每个字段大小（对应 DataBuffer.boards/scalars/probs/...）。"""
    board_data = int(np.prod(BOARD_SHAPE)) * np.dtype(np.float32).itemsize
    scalar_data = int(np.prod(SCALAR_SHAPE)) * np.dtype(np.float32).itemsize
    policy_data = int(np.prod(POLICY_SHAPE)) * np.dtype(np.float32).itemsize
    mask_data = int(np.prod(MASK_SHAPE)) * np.dtype(np.float32).itemsize

    fields: List[_FieldEstimate] = []
    # 每个数组字段：数据 + numpy 对象头 + list 指针
    for name, data_bytes, shape in [
        ("boards", board_data, BOARD_SHAPE),
        ("scalars", scalar_data, SCALAR_SHAPE),
        ("probs (policy_probs)", policy_data, POLICY_SHAPE),
        ("masks (action_mask)", mask_data, MASK_SHAPE),
    ]:
        est = _estimate_field(
            name, data_bytes, 1,
            f"{shape} float32 数组 + ndarray 对象头 + list 指针",
        )
        fields.append(est)

    # 标量字段：Python float/int 对象 + list 指针（数据本身含在对象里）
    fields.append(_estimate_field(
        "values (game_result_value)", PY_FLOAT_BYTES, 1,
        "Python float 对象 + list 指针", per_object_bytes=0,
    ))
    fields.append(_estimate_field(
        "root_visits", PY_INT_BYTES, 1,
        "Python int 对象 + list 指针", per_object_bytes=0,
    ))
    return fields


def estimate_buffer(capacity: int, label: str) -> BufferEstimate:
    """估算一个 DataBuffer 在装满 capacity 个样本时的最大内存。"""
    fields = build_buffer_field_estimates()
    total = 0
    for f in fields:
        # f 是按单样本估算，乘上容量得到整个 buffer 的该字段总大小
        total += f.per_sample_bytes * capacity
    return BufferEstimate(label, capacity, fields, total)


def estimate_replay_buffer_bytes(
    train_capacity: int | None = None,
) -> dict:
    """
    程序化 API：返回 ReplayBuffer 整体内存估算。

    返回 dict:
        train_buffer: BufferEstimate
        total_bytes:  int
    """
    train_capacity = config.MAX_SAMPLE_BUFFER_SIZE if train_capacity is None else train_capacity

    train = estimate_buffer(train_capacity, "train buffer")
    total = train.total_bytes
    return {"train_buffer": train, "total_bytes": total}


def _print_field_breakdown(est: BufferEstimate) -> None:
    """打印单个 buffer 的字段级明细。"""
    sep = "-" * 78
    header = f"  {'字段':<24} {'单样本(B)':>10} {'满容量总字节':>16} {'说明'}"
    print(header)
    print(sep)
    for f in est.field_estimates:
        print(
            f"  {f.name:<24} {f.per_sample_bytes:>10} "
            f"{f.per_sample_bytes * est.capacity:>16,}  {f.note}"
        )
    print(sep)
    per_sample = sum(f.per_sample_bytes for f in est.field_estimates)
    print(
        f"  合计（每样本 {per_sample} B × {est.capacity:,} 样本）"
        f" = {_sizeof_fmt(est.total_bytes)}"
    )


def estimate_suspended_game(
    mcts_sims: int | None = None,
    expected_total_steps: int = MAX_STEPS_PER_EPISODE,
    include_episode_storage: bool = True,
    safety_factor: float = 1.5,
) -> dict:
    """
    估算挂起一个 self_play 对局所需预留的内存。

    一个挂起的对局 = 游戏环境状态 + 单次 MCTS 搜索树 (+ 整局训练数据存储)。
    复用 predictor_entry.estimate_single_game_suspended（明细）与
    estimate_memory_bytes（含安全余量的字节数）。

    返回 dict:
        mcts_sims:  int          实际使用的 MCTS 模拟数
        est:        MemoryEstimate  明细（可 .print_report()）
        raw_bytes:  int          理论字节数（无安全余量）
        bytes:      int          含 safety_factor 安全余量的字节数
        total_steps:int         单局预估步数
        include_episode_storage: bool
    """
    mcts_sims = config.MCTS_SIMS if mcts_sims is None else mcts_sims
    est = estimate_single_game_suspended(mcts_sims, expected_total_steps)
    raw = est.total_bytes
    if not include_episode_storage:
        # 仅保留"游戏状态 + MCTS 树"部分（est 中含 episode 存储小计）
        raw = est.total_bytes - estimate_episode_storage_for_suspended(expected_total_steps)
    total = int(raw * safety_factor)
    return {
        "mcts_sims": mcts_sims,
        "est": est,
        "raw_bytes": raw,
        "bytes": total,
        "total_steps": expected_total_steps,
        "include_episode_storage": include_episode_storage,
    }


def estimate_episode_storage_for_suspended(steps: int) -> int:
    """挂起对局中 GameEpisode 训练数据存储部分的字节数（复用 predictor_entry）。"""
    from predictor_entry import estimate_episode_storage
    return estimate_episode_storage(steps).total_bytes


def main() -> None:
    result = estimate_replay_buffer_bytes()
    train: BufferEstimate = result["train_buffer"]
    total_bytes: int = result["total_bytes"]

    print("\n" + "=" * 78)
    print("  ReplayBuffer（DataBuffer）最大内存占用估算")
    print("=" * 78)
    print(
        f"  容量来源: config.MAX_SAMPLE_BUFFER_SIZE = {config.MAX_SAMPLE_BUFFER_SIZE:,} 样本"
        f"（训练 buffer）"
    )
    print(
        f"  注: 每样本字段对应 DataBuffer 存储结构 "
        f"(boards/scalars/probs/masks 为 numpy 数组, values/root_visits 为 Python 标量)\n"
    )

    print(f"[1] {train.label}")
    _print_field_breakdown(train)

    print("\n" + "=" * 78)
    print(f"  TOTAL: {_sizeof_fmt(total_bytes)}")
    print(f"    train buffer: {_sizeof_fmt(train.total_bytes)}")
    print(f"  Python list 扩容/GC 预留建议 ×1.2: {_sizeof_fmt(int(total_bytes * 1.2))}")
    print("=" * 78 + "\n")

    # ================= 挂起一个 self_play 对局的预留内存 =================
    print("=" * 78)
    print("  挂起一个 self_play 对局所需预留内存")
    print("=" * 78)
    print(
        f"  配置: mcts_sims = config.MCTS_SIMS = {config.MCTS_SIMS}, "
        f"单局预估步数 = {MAX_STEPS_PER_EPISODE}"
    )
    print(
        "  挂起对局 = 游戏环境状态 + 单次 MCTS 搜索树 (+ 整局训练数据存储)，"
        "明细复用 predictor_entry\n"
    )
    suspended = estimate_suspended_game()
    suspended["est"].print_report(
        f"① 挂起单局对局明细 (mcts_sims={suspended['mcts_sims']})"
    )
    print(
        f"  单局挂起预留: 理论 {_sizeof_fmt(suspended['raw_bytes'])} "
        f"× 1.5 安全余量 ≈ {_sizeof_fmt(suspended['bytes'])}"
    )

    # 多 worker 并行时的同时挂起对局数（每个 worker 同一时刻挂起一局）
    suspended_games = max(config.NUM_WORKERS, 1)
    parallel_bytes = suspended["bytes"] * suspended_games
    print(
        f"  {config.NUM_WORKERS} 个 worker 并行 (同时挂起 {suspended_games} 局): "
        f"≈ {_sizeof_fmt(parallel_bytes)}"
    )
    print("=" * 78 + "\n")


if __name__ == "__main__":
    main()
