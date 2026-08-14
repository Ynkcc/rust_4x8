"""
data_augmentation.py — 4x8 暗棋训练数据的空间对称增强（独立模块，以导入方式使用）

针对 4x8 棋盘提供 4 个空间对称变换：
    identity（恒等）/ hflip（水平翻转）/ vflip（垂直翻转）/ rot180（180° 旋转）

变换时同步映射：
  - board 特征张量（(16, 4, 8) 或扁平 512 维）沿对应空间轴翻转；
  - 策略分布 / 动作掩码（352 维）按动作索引置换（与 src/game_env/actions.rs 的动作表一致）；
  - scalar / mcts_value / completed_q / root_visit / game_result 保持不变
    （纯空间变换不改变「当前玩家视角」编码，也不改变血量、存活等全局量）。

⚠️ 重要约定：本模块只应作用于「训练侧」数据。冷存储（MongoDB / JSONL 归档）
保存的必须是原始数据，不得对归档数据应用任何增强。运行链路见 run_training.py：
data_q（训练）与 archive_q（归档）各自独立，本模块仅在训练消费端调用。

用法示例：
    from data_augmentation import augment_samples, transform_episode

    # 样本级增强（episode_to_samples 输出格式）
    aug_samples = augment_samples(samples, keep_original=True)   # 数据量 ×2

    # 局级增强（self_play 队列格式，同局所有步共用同一变换，保持一致性）
    ep2 = transform_episode(episode, "hflip")
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# ============================================================================
# 常量（与 python/constant.py、src/game_env/constants.rs 保持一致）
# ============================================================================
BOARD_ROWS = 4
BOARD_COLS = 8
TOTAL_POSITIONS = BOARD_ROWS * BOARD_COLS
BOARD_CHANNELS = 16  # my(7) + opp(7) + hidden(1) + empty(1)

REVEAL_ACTIONS_COUNT = 32
REGULAR_MOVE_ACTIONS_COUNT = 104
CANNON_ATTACK_ACTIONS_COUNT = 216
ACTION_SPACE_SIZE = (
    REVEAL_ACTIONS_COUNT + REGULAR_MOVE_ACTIONS_COUNT + CANNON_ATTACK_ACTIONS_COUNT
)  # 352

# 全部对称变换（4x8 长方形棋盘只有这 4 个空间自同构；
# 90° 旋转 / 对角线镜像会改变棋盘形状，不适用）
SYMMETRY_TRANSFORMS: Tuple[str, ...] = ("identity", "hflip", "vflip", "rot180")
# 默认增强所用变换（排除恒等，避免无效复制）
NON_IDENTITY_TRANSFORMS: Tuple[str, ...] = ("hflip", "vflip", "rot180")


# ============================================================================
# 动作查找表（与 Rust src/game_env/actions.rs::build_action_lookup_tables 逐条一致）
# ============================================================================
_ACTION_TO_COORDS: Optional[List[Tuple[int, ...]]] = None
_COORDS_TO_ACTION: Optional[Dict[Tuple[int, ...], int]] = None


def _build_action_tables() -> Tuple[List[Tuple[int, ...]], Dict[Tuple[int, ...], int]]:
    """惰性重建动作 -> 坐标表，顺序与 Rust 侧完全一致（依赖划分边界）。"""
    global _ACTION_TO_COORDS, _COORDS_TO_ACTION
    if _ACTION_TO_COORDS is not None:
        return _ACTION_TO_COORDS, _COORDS_TO_ACTION

    action_to_coords: List[Tuple[int, ...]] = []
    coords_to_action: Dict[Tuple[int, ...], int] = {}
    idx = 0

    # 1. 翻棋：action == sq
    for sq in range(TOTAL_POSITIONS):
        coords = (sq,)
        action_to_coords.append(coords)
        coords_to_action[coords] = idx
        idx += 1

    # 2. 常规移动：四方向各 1 步（顺序：上/下/左/右）
    moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for r1 in range(BOARD_ROWS):
        for c1 in range(BOARD_COLS):
            from_sq = r1 * BOARD_COLS + c1
            for dr, dc in moves:
                r2, c2 = r1 + dr, c1 + dc
                if 0 <= r2 < BOARD_ROWS and 0 <= c2 < BOARD_COLS:
                    coords = (from_sq, r2 * BOARD_COLS + c2)
                    action_to_coords.append(coords)
                    coords_to_action[coords] = idx
                    idx += 1

    # 3. 炮击：同行隔子（水平）+ 同列隔子（垂直），已在表中的对跳过
    for r1 in range(BOARD_ROWS):
        for c1 in range(BOARD_COLS):
            from_sq = r1 * BOARD_COLS + c1
            # 水平
            for c2 in range(BOARD_COLS):
                if abs(c1 - c2) > 1:
                    coords = (from_sq, r1 * BOARD_COLS + c2)
                    if coords not in coords_to_action:
                        action_to_coords.append(coords)
                        coords_to_action[coords] = idx
                        idx += 1
            # 垂直
            for r2 in range(BOARD_ROWS):
                if abs(r1 - r2) > 1:
                    coords = (from_sq, r2 * BOARD_COLS + c1)
                    if coords not in coords_to_action:
                        action_to_coords.append(coords)
                        coords_to_action[coords] = idx
                        idx += 1

    assert idx == ACTION_SPACE_SIZE, "动作表尺寸异常"
    _ACTION_TO_COORDS, _COORDS_TO_ACTION = action_to_coords, coords_to_action
    return action_to_coords, coords_to_action


# ============================================================================
# 坐标 / 动作索引映射
# ============================================================================

def _sq_map(transform: str) -> np.ndarray:
    """返回 32 元素映射表 map，满足 board_new.flat[i] = board_flat[map[i]]。

    即 map[i] 表示「变换后棋盘位置 i 对应的原格子索引」。
    """
    if transform not in SYMMETRY_TRANSFORMS:
        raise ValueError("未知对称变换 %r，可选: %r" % (transform, SYMMETRY_TRANSFORMS))
    rows = np.arange(BOARD_ROWS, dtype=np.int64)[:, None] * BOARD_COLS
    cols = np.arange(BOARD_COLS, dtype=np.int64)[None, :]
    sq = rows + cols  # (4, 8)，sq[r, c] = r*8 + c
    if transform == "identity":
        return np.arange(TOTAL_POSITIONS, dtype=np.int64)
    if transform == "hflip":
        return sq[:, ::-1].reshape(-1)
    if transform == "vflip":
        return sq[::-1, :].reshape(-1)
    return sq[::-1, ::-1].reshape(-1)  # rot180


_PERM_CACHE: Dict[str, np.ndarray] = {}


def action_permutation(transform: str) -> np.ndarray:
    """返回动作索引置换表 perm（长度 352），满足 new_policy = old_policy[perm]。

    等价地：对原动作 a 施加变换后得到的动作索引为 perm[a]（perm 即变换函数）。
    所有对称均为对合（involution），即 perm[perm] == identity。
    """
    if transform in _PERM_CACHE:
        return _PERM_CACHE[transform]
    if transform not in SYMMETRY_TRANSFORMS:
        raise ValueError("未知对称变换 %r，可选: %r" % (transform, SYMMETRY_TRANSFORMS))

    action_to_coords, coords_to_action = _build_action_tables()
    m = _sq_map(transform)
    perm = np.empty(ACTION_SPACE_SIZE, dtype=np.int64)
    for a, coords in enumerate(action_to_coords):
        mapped = tuple(int(m[sq]) for sq in coords)
        perm[coords_to_action[mapped]] = a
    _PERM_CACHE[transform] = perm
    return perm


# ============================================================================
# 单对象变换
# ============================================================================

def transform_board(board: np.ndarray, transform: str) -> np.ndarray:
    """对棋盘张量应用空间翻转。支持 (16,4,8) 或扁平 (512,) 输入，形状不变。"""
    if transform == "identity":
        return board
    arr = np.asarray(board)
    flat_in = arr.reshape(BOARD_CHANNELS, BOARD_ROWS, BOARD_COLS)
    if transform == "hflip":
        out = flat_in[:, :, ::-1]
    elif transform == "vflip":
        out = flat_in[:, ::-1, :]
    else:  # rot180
        out = flat_in[:, ::-1, ::-1]
    return out.reshape(arr.shape)


def transform_policy(policy: np.ndarray, transform: str) -> np.ndarray:
    """对策略分布 / 动作掩码（352 维）按动作索引置换。"""
    if transform == "identity":
        return policy
    arr = np.asarray(policy)
    return arr[action_permutation(transform)]


def transform_action(action: int, transform: str) -> int:
    """把实际选择的动作索引映射到变换后的动作索引。"""
    if transform == "identity":
        return action
    return int(action_permutation(transform)[action])


# ============================================================================
# 样本级增强（episode_to_samples 输出格式）
# ============================================================================

def transform_sample(sample: Dict, transform: str) -> Dict:
    """对单个 sample dict 应用对称变换，返回新 dict（不修改原对象）。

    处理字段：board_state / policy_probs / action_mask；
    保持字段：scalar_state / mcts_value / completed_q / root_visit_count /
              game_result_value / step_in_game（空间变换不改变这些量）。
    """
    out = dict(sample)
    if transform != "identity":
        out["board_state"] = transform_board(out["board_state"], transform)
        out["policy_probs"] = transform_policy(out["policy_probs"], transform)
        out["action_mask"] = transform_policy(out["action_mask"], transform)
    return out


def augment_samples(
    samples: Sequence[Dict],
    transforms: Sequence[str] = NON_IDENTITY_TRANSFORMS,
    keep_original: bool = True,
    rng: Optional[random.Random] = None,
) -> List[Dict]:
    """对一批样本做对称增强：每条样本随机等概率选一个变换生成增强样本。

    参数：
      samples       : episode_to_samples 输出的样本 dict 列表
      transforms    : 候选变换集合（默认非恒等三个）
      keep_original : True 时原始样本与增强样本一起保留（数据量 ×2）
      rng           : 可注入随机源（默认全局 random）
    """
    if not transforms:
        return list(samples)
    rng = rng or random
    out: List[Dict] = []
    for s in samples:
        if keep_original:
            out.append(s)
        t = transforms[rng.randrange(len(transforms))]
        out.append(transform_sample(s, t))
    return out


# ============================================================================
# 局级增强（self_play 队列格式：episode dict）
# ============================================================================

def transform_episode(episode: Dict, transform: str) -> Dict:
    """对整局 episode 应用同一对称变换，返回新 episode dict（不修改原对象）。

    同一局所有步共用同一个变换（保证整局动作链自洽），
    boards / policies / action_masks / actions 同步映射，
    scalars / mcts_values / completed_qs / root_visits / game_results /
    winner / iteration / worker_id 等保持不变。
    """
    out = dict(episode)
    if transform == "identity":
        return out
    out["boards"] = [transform_board(b, transform) for b in episode["boards"]]
    out["policies"] = [transform_policy(p, transform) for p in episode["policies"]]
    out["action_masks"] = [
        transform_policy(m, transform) for m in episode["action_masks"]
    ]
    if episode.get("actions"):
        perm = action_permutation(transform)
        out["actions"] = [int(perm[a]) for a in episode["actions"]]
    return out


def augment_episode(
    episode: Dict,
    transforms: Sequence[str] = NON_IDENTITY_TRANSFORMS,
    keep_original: bool = True,
    rng: Optional[random.Random] = None,
) -> List[Dict]:
    """对单局 episode 做对称增强：随机选一个变换生成增强局。

    返回列表（keep_original=True 时含原始局 + 增强局），供调用方展平使用。
    """
    if not transforms:
        return [episode]
    rng = rng or random
    out: List[Dict] = [episode] if keep_original else []
    t = transforms[rng.randrange(len(transforms))]
    out.append(transform_episode(episode, t))
    return out


# ============================================================================
# 自检（python python/data_augmentation.py）
# ============================================================================
if __name__ == "__main__":
    atc, cta = _build_action_tables()
    assert len(atc) == ACTION_SPACE_SIZE, "动作表长度错误"
    n_reveal = sum(1 for c in atc if len(c) == 1)
    n_move = sum(1 for c in atc if len(c) == 2)
    assert n_reveal == REVEAL_ACTIONS_COUNT, "翻棋数错误"
    assert n_move == REGULAR_MOVE_ACTIONS_COUNT + CANNON_ATTACK_ACTIONS_COUNT, "移动+炮击数错误"

    rng = random.Random(42)
    board = np.random.RandomState(0).rand(BOARD_CHANNELS * TOTAL_POSITIONS).astype(np.float32)
    policy = np.random.RandomState(1).rand(ACTION_SPACE_SIZE).astype(np.float32)
    mask = np.random.RandomState(2).randint(0, 2, ACTION_SPACE_SIZE).astype(np.float32)

    for t in SYMMETRY_TRANSFORMS:
        perm = action_permutation(t)
        # 1) 置换表必须是合法排列
        assert sorted(perm.tolist()) == list(range(ACTION_SPACE_SIZE)), t + " 非排列"
        # 2) 对合性：应用两次还原
        assert (perm[perm] == np.arange(ACTION_SPACE_SIZE)).all(), t + " 非对合"
        # 3) board 变换与坐标映射一致
        mapped = transform_board(board, t)
        b_flat = board.reshape(BOARD_CHANNELS, BOARD_ROWS, BOARD_COLS)
        m = _sq_map(t)
        expect = b_flat[:, m // BOARD_COLS, m % BOARD_COLS].reshape(-1)
        assert np.allclose(mapped, expect), t + " board 映射不一致"
        # 4) policy 再变一次可还原
        twice = transform_policy(transform_policy(policy, t), t)
        assert np.allclose(twice, policy), t + " policy 非对合"
        # 5) mask 变换保持求和不变
        assert abs(mask[perm].sum() - mask.sum()) < 1e-5, t + " mask 和变化"

    # 6) 翻棋动作映射抽查：hflip 下 sq=5 (r0,c5) -> sq'=2 (r0,c2)；rot180 下 sq=1 -> sq'=30
    assert transform_action(5, "hflip") == 2
    assert transform_action(1, "rot180") == (3 - 0) * 8 + (7 - 1)

    # 7) 样本级增强（episode_to_samples 格式）
    sample = {
        "board_state": board.copy(),
        "scalar_state": np.random.rand(35).astype(np.float32),
        "policy_probs": policy.copy(),
        "mcts_value": 0.3,
        "completed_q": 0.2,
        "root_visit_count": 10,
        "game_result_value": 1.0,
        "action_mask": mask.copy(),
        "step_in_game": 3,
    }
    aug_list = augment_samples([sample], keep_original=True, rng=rng)
    assert len(aug_list) == 2
    for s in aug_list:
        assert np.allclose(s["scalar_state"], sample["scalar_state"])
        assert s["mcts_value"] == sample["mcts_value"]
        assert s["game_result_value"] == sample["game_result_value"]
        assert np.abs(s["policy_probs"].sum() - policy.sum()) < 1e-5
        assert np.abs(s["action_mask"].sum() - mask.sum()) < 1e-5
    assert np.array_equal(aug_list[0]["board_state"], board)  # 原始保留

    # 8) 局级增强（episode 格式）
    episode = {
        "game_length": 2,
        "winner": 1,
        "num_samples": 2,
        "iteration": 7,
        "worker_id": 0,
        "boards": [board.copy(), board.copy()],
        "scalars": [np.random.rand(35).astype(np.float32)] * 2,
        "policies": [policy.copy(), policy.copy()],
        "mcts_values": [0.3, -0.1],
        "completed_qs": [0.2, 0.0],
        "root_visits": [10, 5],
        "game_results": [1.0, 1.0],
        "action_masks": [mask.copy(), mask.copy()],
        "actions": [0, 5],
    }
    ep_t = transform_episode(episode, "hflip")
    assert np.array_equal(ep_t["boards"][0], transform_board(board, "hflip")), "episode board 变换失败"
    assert np.array_equal(ep_t["action_masks"][1], transform_policy(mask, "hflip")), "episode mask 变换失败"
    assert ep_t["actions"][1] == transform_action(5, "hflip")
    assert ep_t["game_results"] == episode["game_results"]
    assert ep_t["winner"] == 1 and ep_t["iteration"] == 7
    # 原 episode 未被修改
    assert np.array_equal(episode["boards"][0], board)

    print("data_augmentation self-check OK (action table size = %d, transforms = %s)"
          % (len(atc), SYMMETRY_TRANSFORMS))
