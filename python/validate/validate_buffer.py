"""
validate_buffer.py — 验证 DataBuffer 与 episode_to_samples（纯 CPU）。

检查项：
  1. episode_to_samples 字段映射完整：8 个并行序列一一对应，step_in_game 递增
  2. 转换后的 sample dict 字段与 Rust episode_to_dict 契约一致
  3. DataBuffer.add_samples 形状正确（board/scalar/policy/mask reshape）
  4. DataBuffer 超容量时 FIFO 裁剪正确（先进先出）
  5. DataBuffer.get_batch 返回形状与索引对应值一致
  6. get_batch 的 target_values 优先取 game_result_value（回退 mcts_value）

运行：python python/validate/validate_buffer.py
"""

from __future__ import annotations

import numpy as np

import validate_common  # noqa: F401  (设置 sys.path)
from validate_common import Reporter, make_episode, run_part

from constant import ACTION_SPACE_SIZE, BOARD_ROWS, BOARD_COLS, SCALAR_FEATURE_COUNT, TOTAL_INPUT_CHANNELS
from training_service import DataBuffer, episode_to_samples


def test_episode_to_samples_mapping() -> None:
    rep = Reporter("episode_to_samples mapping")
    rng = np.random.default_rng(0)
    ep = make_episode(num_steps=4, winner=1, rng=rng)
    samples = episode_to_samples(ep)

    rep.check(len(samples) == ep["num_samples"] == 4,
              f"sample count == 4 (got {len(samples)})")
    # 字段键齐全
    required_keys = {
        "board_state", "scalar_state", "policy_probs", "mcts_value",
        "completed_q", "root_visit_count", "game_result_value", "action_mask",
        "step_in_game",
    }
    rep.check(required_keys <= set(samples[0].keys()),
              f"all required keys present: {sorted(required_keys)}")

    # 逐样本核对与源 episode 的一致性
    ok = True
    for idx, s in enumerate(samples):
        ok &= np.array_equal(s["board_state"], ep["boards"][idx])
        ok &= np.array_equal(s["scalar_state"], ep["scalars"][idx])
        ok &= np.array_equal(s["policy_probs"], ep["policies"][idx])
        ok &= np.array_equal(s["action_mask"], ep["action_masks"][idx])
        ok &= (abs(s["mcts_value"] - ep["mcts_values"][idx]) < 1e-9)
        ok &= (abs(s["completed_q"] - ep["completed_qs"][idx]) < 1e-9)
        ok &= (s["root_visit_count"] == ep["root_visits"][idx])
        ok &= (abs(s["game_result_value"] - ep["game_results"][idx]) < 1e-9)
        ok &= (s["step_in_game"] == idx)
    rep.check(ok, "each sample field matches source episode")
    # 形状
    rep.check(np.asarray(samples[0]["board_state"]).shape ==
              (TOTAL_INPUT_CHANNELS, BOARD_ROWS, BOARD_COLS), "board_state shape")
    rep.check(np.asarray(samples[0]["scalar_state"]).shape[0] == SCALAR_FEATURE_COUNT,
              "scalar_state length")
    rep.check(len(samples[0]["policy_probs"]) == ACTION_SPACE_SIZE, "policy length")
    rep.check(len(samples[0]["action_mask"]) == ACTION_SPACE_SIZE, "mask length")
    rep.summary()


def test_data_buffer_shapes() -> None:
    rep = Reporter("DataBuffer add/get shapes")
    rng = np.random.default_rng(1)
    ep = make_episode(num_steps=3, winner=-1, rng=rng)
    buf = DataBuffer(capacity=100)
    buf.add_samples(episode_to_samples(ep))

    rep.check(len(buf) == 3, f"buffer len == 3 (got {len(buf)})")
    indices = [0, 1, 2]
    boards, scalars, probs, values, masks = buf.get_batch(indices)
    rep.check(tuple(boards.shape) == (3, TOTAL_INPUT_CHANNELS, BOARD_ROWS, BOARD_COLS),
              f"boards shape {tuple(boards.shape)}")
    rep.check(tuple(scalars.shape) == (3, SCALAR_FEATURE_COUNT),
              f"scalars shape {tuple(scalars.shape)}")
    rep.check(tuple(probs.shape) == (3, ACTION_SPACE_SIZE),
              f"probs shape {tuple(probs.shape)}")
    rep.check(tuple(values.shape) == (3,), f"values shape {tuple(values.shape)}")
    rep.check(tuple(masks.shape) == (3, ACTION_SPACE_SIZE),
              f"masks shape {tuple(masks.shape)}")
    rep.summary()


def test_data_buffer_fifo_trim() -> None:
    rep = Reporter("DataBuffer FIFO trim")
    rng = np.random.default_rng(2)
    cap = 4
    buf = DataBuffer(capacity=cap)
    # 添加 3 局各 2 步 = 6 样本，超过 cap=4
    ep = make_episode(num_steps=2, winner=1, rng=rng)
    buf.add_samples(episode_to_samples(ep))  # 2
    ep2 = make_episode(num_steps=2, winner=-1, rng=rng)
    buf.add_samples(episode_to_samples(ep2))  # 4
    ep3 = make_episode(num_steps=2, winner=1, rng=rng)
    buf.add_samples(episode_to_samples(ep3))  # 6 → 裁剪到 4

    rep.check(len(buf) == cap, f"buffer trimmed to capacity {cap} (got {len(buf)})")

    # FIFO：最早加入的样本应被淘汰。首局 ep 的两步 step_in_game=0,1 应最先被移除。
    # 裁剪 excess=2，淘汰 ep 的 step0 和 step1。
    boards, scalars, probs, values, masks = buf.get_batch(list(range(cap)))
    # 剩余应为：ep2 step0, ep2 step1, ep3 step0, ep3 step1
    rep.check(np.allclose(boards[0].cpu().numpy(), np.asarray(ep2["boards"][0], dtype=np.float32)),
              "FIFO: first retained = ep2 step0")
    rep.check(np.allclose(boards[2].cpu().numpy(), np.asarray(ep3["boards"][0], dtype=np.float32)),
              "FIFO: third retained = ep3 step0")
    # 首局 ep 的数据不应再出现
    ep0_board = np.asarray(ep["boards"][0], dtype=np.float32)
    matched_old = any(
        np.allclose(boards[i].cpu().numpy(), ep0_board) for i in range(cap)
    )
    rep.check(not matched_old, "oldest episode step was evicted")
    rep.summary()


def test_get_batch_index_correctness() -> None:
    rep = Reporter("get_batch index correctness")
    rng = np.random.default_rng(3)
    buf = DataBuffer(capacity=50)
    for _ in range(3):
        ep = make_episode(num_steps=2, winner=1, rng=rng)
        buf.add_samples(episode_to_samples(ep))
    # 乱序索引
    idx = [5, 0, 3, 1]
    boards, scalars, probs, values, masks = buf.get_batch(idx)
    # 校验 boards 顺序与 idx 一致：从单独 get_batch 逐索引对比
    for j, i in enumerate(idx):
        single_b, _, _, _, _ = buf.get_batch([i])
        rep.check(np.array_equal(boards[j].cpu().numpy(), single_b[0].cpu().numpy()),
                  f"boards[order {j}] == buffer[{i}]")
    rep.summary()


def test_value_priority() -> None:
    """target value 应优先取 game_result_value，回退 mcts_value。"""
    rep = Reporter("target value priority")
    rng = np.random.default_rng(4)
    ep = make_episode(num_steps=1, winner=1, rng=rng)
    samples = episode_to_samples(ep)
    # 人为设置 game_result_value 与 mcts_value 不同，验证取 game_result_value
    samples[0]["game_result_value"] = 0.7
    samples[0]["mcts_value"] = 0.2
    buf = DataBuffer(capacity=10)
    buf.add_samples(samples)
    _, _, _, values, _ = buf.get_batch([0])
    rep.check(abs(values[0].item() - 0.7) < 1e-6,
              f"target value uses game_result_value (got {values[0].item():.4f})")
    rep.summary()


def main() -> None:
    run_part("buffer: episode_to_samples mapping", test_episode_to_samples_mapping)
    run_part("buffer: DataBuffer add/get shapes", test_data_buffer_shapes)
    run_part("buffer: FIFO trim", test_data_buffer_fifo_trim)
    run_part("buffer: get_batch index", test_get_batch_index_correctness)
    run_part("buffer: value priority", test_value_priority)


if __name__ == "__main__":
    main()
