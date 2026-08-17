"""banqi/data_augmentation.py — 参数化空间对称增强引擎

对 4x2 / 4x4 / 4x8 三变体统一实现「训练数据空间对称增强」：
  - 变换集由 variant.symmetries 限定（4x8 / 4x2 用 hflip/vflip/rot180 4 个；
    4x4 方盘用全部 8 个 D4 对称）。
  - board 特征张量沿空间轴翻转；策略/动作掩码按动作索引置换
    （置换表由 banqi.actions 动作表 + _sq_map 推导，与 Rust 一致）。
  - scalar / 血量 / 存活等全局量不变。

⚠️ 仅作用于训练侧 replay buffer；冷存储归档必须保存原始数据。

用法（显式传入 variant）：
    from banqi.data_augmentation import make_augmentor
    aug = make_augmentor(variant)          # 绑定变体，函数签名与旧版一致
    aug.transform_episode(ep, "hflip")
    aug.augment_samples(samples, keep_original=True)
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from banqi.actions import build_action_tables
from banqi.constants import build_constants
from banqi.variant import Variant


class Augmentor:
    """绑定某个变体的对称增强工具。"""

    def __init__(self, variant: Variant) -> None:
        self.variant = variant
        c = build_constants(variant)
        self.rows = c.BOARD_ROWS
        self.cols = c.BOARD_COLS
        self.board_channels = c.TOTAL_INPUT_CHANNELS
        self.action_space = c.ACTION_SPACE_SIZE
        self.SYMMETRY_TRANSFORMS = tuple(variant.symmetries)
        self.NON_IDENTITY_TRANSFORMS = tuple(variant.non_identity_transforms)
        self.INVOLUTIONS = tuple(variant.involutions)

        self._action_to_coords, self._coords_to_action = build_action_tables(
            self.rows, self.cols
        )
        assert len(self._action_to_coords) == self.action_space, "动作表尺寸异常"
        self._perm_cache: Dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------ #
    # 坐标 / 动作映射
    # ------------------------------------------------------------------ #

    def _sq_map(self, transform: str) -> np.ndarray:
        """返回 map[i] = 变换后位置 i 对应的原格子索引。"""
        r, c, rows, cols = (
            np.arange(self.rows, dtype=np.int64)[:, None],
            np.arange(self.cols, dtype=np.int64)[None, :],
            self.rows,
            self.cols,
        )
        if transform == "identity":
            pr, pc = r, c
        elif transform == "rot90":       # 顺时针 90°
            pr, pc = cols - 1 - c, r
        elif transform == "rot180":
            pr, pc = rows - 1 - r, cols - 1 - c
        elif transform == "rot270":      # 逆时针 90°
            pr, pc = c, rows - 1 - r
        elif transform == "hflip":
            pr, pc = r, cols - 1 - c
        elif transform == "vflip":
            pr, pc = rows - 1 - r, c
        elif transform == "diag":        # 主对角线镜像（转置）
            pr, pc = c, r
        else:                            # anti_diag：反对角线镜像
            pr, pc = cols - 1 - c, rows - 1 - r
        return (pr * cols + pc).reshape(-1)

    def action_permutation(self, transform: str) -> np.ndarray:
        """返回置换表 perm，满足 new_policy = old_policy[perm]。"""
        if transform in self._perm_cache:
            return self._perm_cache[transform]
        if transform not in self.SYMMETRY_TRANSFORMS:
            raise ValueError(
                f"未知对称变换 {transform!r}，可选: {self.SYMMETRY_TRANSFORMS}"
            )
        m = self._sq_map(transform)
        perm = np.empty(self.action_space, dtype=np.int64)
        for a, coords in enumerate(self._action_to_coords):
            mapped = tuple(int(m[sq]) for sq in coords)
            perm[self._coords_to_action[mapped]] = a
        self._perm_cache[transform] = perm
        return perm

    # ------------------------------------------------------------------ #
    # 单对象变换
    # ------------------------------------------------------------------ #

    def transform_board(self, board: np.ndarray, transform: str) -> np.ndarray:
        if transform == "identity":
            return board
        arr = np.asarray(board)
        flat_in = arr.reshape(self.board_channels, self.rows, self.cols)
        m = self._sq_map(transform)
        out = flat_in[:, m // self.cols, m % self.cols]
        return out.reshape(arr.shape)

    def transform_policy(self, policy: np.ndarray, transform: str) -> np.ndarray:
        if transform == "identity":
            return policy
        return np.asarray(policy)[self.action_permutation(transform)]

    def transform_action(self, action: int, transform: str) -> int:
        if transform == "identity":
            return action
        return int(self.action_permutation(transform)[action])

    # ------------------------------------------------------------------ #
    # 样本级增强
    # ------------------------------------------------------------------ #

    def transform_sample(self, sample: Dict, transform: str) -> Dict:
        out = dict(sample)
        if transform != "identity":
            out["board_state"] = self.transform_board(out["board_state"], transform)
            out["policy_probs"] = self.transform_policy(out["policy_probs"], transform)
            out["action_mask"] = self.transform_policy(out["action_mask"], transform)
        return out

    def augment_samples(
        self,
        samples: Sequence[Dict],
        transforms: Optional[Sequence[str]] = None,
        keep_original: bool = True,
        rng: Optional[random.Random] = None,
    ) -> List[Dict]:
        if transforms is None:
            transforms = self.NON_IDENTITY_TRANSFORMS
        if not transforms:
            return list(samples)
        rng = rng or random
        out: List[Dict] = []
        for s in samples:
            if keep_original:
                out.append(s)
            t = transforms[rng.randrange(len(transforms))]
            out.append(self.transform_sample(s, t))
        return out

    # ------------------------------------------------------------------ #
    # 局级增强
    # ------------------------------------------------------------------ #

    def transform_episode(self, episode: Dict, transform: str) -> Dict:
        out = dict(episode)
        if transform == "identity":
            return out
        out["boards"] = [self.transform_board(b, transform) for b in episode["boards"]]
        out["policies"] = [self.transform_policy(p, transform) for p in episode["policies"]]
        out["action_masks"] = [
            self.transform_policy(m, transform) for m in episode["action_masks"]
        ]
        if episode.get("actions"):
            perm = self.action_permutation(transform)
            out["actions"] = [int(perm[a]) for a in episode["actions"]]
        return out

    def augment_episode(
        self,
        episode: Dict,
        transforms: Optional[Sequence[str]] = None,
        keep_original: bool = True,
        rng: Optional[random.Random] = None,
    ) -> List[Dict]:
        if transforms is None:
            transforms = self.NON_IDENTITY_TRANSFORMS
        if not transforms:
            return [episode]
        rng = rng or random
        out: List[Dict] = [episode] if keep_original else []
        t = transforms[rng.randrange(len(transforms))]
        out.append(self.transform_episode(episode, t))
        return out


_augmentor_cache: Dict[str, Augmentor] = {}


def make_augmentor(variant: Variant) -> Augmentor:
    """构造（并缓存）一个变体的 Augmentor。"""
    if variant.id not in _augmentor_cache:
        _augmentor_cache[variant.id] = Augmentor(variant)
    return _augmentor_cache[variant.id]


def self_check(variant: Variant) -> None:
    """对给定变体跑一遍增强自检（置换合法性 / 对合 / board 一致性 / 求和不变）。"""
    aug = make_augmentor(variant)
    rows, cols = aug.rows, aug.cols
    ch, A = aug.board_channels, aug.action_space
    pos = rows * cols
    rng = random.Random(42)
    board = np.random.RandomState(0).rand(ch * pos).astype(np.float32)
    policy = np.random.RandomState(1).rand(A).astype(np.float32)
    mask = np.random.RandomState(2).randint(0, 2, A).astype(np.float32)

    for t in aug.SYMMETRY_TRANSFORMS:
        perm = aug.action_permutation(t)
        assert sorted(perm.tolist()) == list(range(A)), f"{t} 非排列"
        if t in aug.INVOLUTIONS:
            assert (perm[perm] == np.arange(A)).all(), f"{t} 非对合"
        else:
            four = perm[perm][perm][perm]
            assert (four == np.arange(A)).all(), f"{t} 非 4 次还原"
        # board 变换与坐标映射一致
        mapped = aug.transform_board(board, t)
        b_flat = board.reshape(ch, rows, cols)
        m = aug._sq_map(t)
        expect = b_flat[:, m // cols, m % cols].reshape(-1)
        assert np.allclose(mapped, expect), f"{t} board 映射不一致"
        # policy 还原
        if t in aug.INVOLUTIONS:
            twice = aug.transform_policy(aug.transform_policy(policy, t), t)
            assert np.allclose(twice, policy), f"{t} policy 非对合"
        else:
            four = aug.transform_policy(aug.transform_policy(
                aug.transform_policy(aug.transform_policy(policy, t), t), t), t)
            assert np.allclose(four, policy), f"{t} policy 非 4 次还原"
        # mask 求和不变
        assert abs(mask[perm].sum() - mask.sum()) < 1e-5, f"{t} mask 和变化"

    # 翻棋动作映射抽查（hflip 下 sq -> 水平镜像；rot180 -> 中心对称）
    def sq(r, c):
        return r * cols + c
    # 选一个所有变换都可用的中心格子（避免棋盘过小越界）
    r0, c0 = rows // 2, cols // 2
    if rows >= 2 and cols >= 2 and "hflip" in aug.SYMMETRY_TRANSFORMS:
        assert aug.transform_action(sq(0, c0), "hflip") == sq(0, cols - 1 - c0)
    if "rot180" in aug.SYMMETRY_TRANSFORMS:
        assert aug.transform_action(sq(r0, c0), "rot180") == sq(rows - 1 - r0, cols - 1 - c0)

    print(f"[banqi.data_augmentation] {variant.id} self-check OK "
          f"(action={A}, transforms={aug.SYMMETRY_TRANSFORMS})")


if __name__ == "__main__":
    from banqi.variant import VARIANTS
    for vid, v in VARIANTS.items():
        self_check(v)
    print("[banqi.data_augmentation] all OK")
