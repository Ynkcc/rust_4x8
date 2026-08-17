"""banqi/constants.py — 由 Variant 派生全部维度常量

所有「棋盘尺寸 / 通道数 / 标量维度 / 动作空间 / 子力配置」常量都由
variant 唯一声明源计算，不再手写。

对外暴露统一入口：
    consts = build_constants(variant)        # -> Constants（含全部命名常量）
    d = consts.module()                      # -> 模块级字典（供薄壳 globals().update）
"""

from __future__ import annotations

from typing import Any, Dict

from banqi.variant import Variant


class Constants:
    """一个变体的全部派生常量。全部字段只读。"""

    def __init__(self, v: Variant) -> None:
        self.variant = v
        # ---- 棋盘 / 特征 ----
        self.BOARD_ROWS = v.board_rows
        self.BOARD_COLS = v.board_cols
        self.TOTAL_POSITIONS = v.total_positions
        self.NUM_PIECE_TYPES = 7
        self.NUM_ACTIVE_PIECE_TYPES = v.num_active_piece_types
        self.BOARD_CHANNELS = v.board_channels
        self.TOTAL_INPUT_CHANNELS = v.board_channels
        self.TOTAL_PIECES_PER_PLAYER = v.total_pieces_per_player
        self.SURVIVAL_VECTOR_SIZE = v.total_pieces_per_player
        self.SCALAR_FEATURE_COUNT = v.scalar_feature_count
        # ---- 子力 ----
        self.PIECE_COUNTS = v.piece_counts
        self.PIECE_VALUES = v.piece_values
        self.INITIAL_HEALTH = v.initial_health
        self.SOLDIERS_COUNT, self.CANNONS_COUNT, self.HORSES_COUNT, \
            self.CHARIOTS_COUNT, self.ELEPHANTS_COUNT, self.ADVISORS_COUNT, \
            self.GENERALS_COUNT = v.piece_counts
        # ---- 动作空间 ----
        self.REVEAL_ACTIONS_COUNT, self.REGULAR_MOVE_ACTIONS_COUNT, \
            self.CANNON_ATTACK_ACTIONS_COUNT = v.action_counts
        self.ACTION_SPACE_SIZE = v.action_space_size
        # ---- 网络 ----
        self.HIDDEN_CHANNELS = v.hidden_channels
        self.NUM_RES_BLOCKS = v.num_res_blocks
        self.POLICY_HEAD_CHANNELS = v.policy_head_channels
        self.VALUE_HEAD_CHANNELS = v.value_head_channels
        self.POLICY_FC1_HIDDEN = v.policy_fc1_hidden
        self.VALUE_FC1_HIDDEN = v.value_fc1_hidden

    def as_dict(self) -> Dict[str, Any]:
        return {k: getattr(self, k) for k in dir(self)
                if k.isupper() and not k.startswith("_")}

    def module(self) -> Dict[str, Any]:
        """生成适合 `globals().update(...)` 的模块级字典（含 variant）。"""
        d = self.as_dict()
        d["variant"] = self.variant
        return d


_cache: Dict[str, Constants] = {}


def build_constants(variant: Variant) -> Constants:
    """构造（并缓存）一个变体的 Constants。"""
    if variant.id not in _cache:
        _cache[variant.id] = Constants(variant)
    return _cache[variant.id]


def verify_against_bindings(variant: Variant) -> Dict[str, int]:
    """（可选）与已编译的 banqi_4x8 绑定核对维度一致性，返回 {常量名: 绑定值}。

    找不到绑定（未 maturin develop）时返回空 dict 并静默跳过；
    找到但值不一致时抛 AssertionError —— 用于在训练/增强前拦截 Rust/Python 维度脱节。
    """
    try:
        import banqi_4x8  # type: ignore
    except Exception:
        return {}
    p = variant.env_const_prefix
    key = p + "BOARD_ROWS"
    if not hasattr(banqi_4x8, key):
        return {}
    c = build_constants(variant)
    expected = {
        key: c.BOARD_ROWS,
        p + "BOARD_COLS": c.BOARD_COLS,
        p + "BOARD_CHANNELS": c.BOARD_CHANNELS,
        p + "SCALAR_FEATURE_COUNT": c.SCALAR_FEATURE_COUNT,
        p + "ACTION_SPACE_SIZE": c.ACTION_SPACE_SIZE,
    }
    bound: Dict[str, int] = {}
    for name, val in expected.items():
        if hasattr(banqi_4x8, name):
            got = getattr(banqi_4x8, name)
            bound[name] = got
            assert got == val, f"{name}: Python 派生 {val} != Rust 绑定 {got}"
    return bound


if __name__ == "__main__":
    from banqi.variant import VARIANTS
    for vid, v in VARIANTS.items():
        c = build_constants(v)
        verify_against_bindings(v)
        print(f"[banqi.constants] {vid}: ch={c.TOTAL_INPUT_CHANNELS} "
              f"scalar={c.SCALAR_FEATURE_COUNT} action={c.ACTION_SPACE_SIZE} "
              f"pieces={c.TOTAL_PIECES_PER_PLAYER} health={c.INITIAL_HEALTH}")
    print("[banqi.constants] all OK")
