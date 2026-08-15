# -*- coding: utf-8 -*-
"""
smoke_test_rust_ttt.py
=====================
验证 Python 可通过 pyo3 绑定驱动 Rust 井字棋环境 + 泛型 Gumbel MCTS。

验证点：
1. banqi_4x8 的 TicTacToe 环境类（构造/合法动作/落子/胜负/编码）
2. ttt_mcts_search 单步搜索：空棋盘选合法动作
3. 「一步可赢」局面：MCTS 必须选中致胜动作
4. 「一步必防」局面：MCTS 必须选中防守动作
5. run_ttt_self_play_with_predictor：复用 Rust 泛型 self_play 生成完整对局样本

运行方式（需先构建扩展）：
    maturin develop --features pyo3   # 或 --release
    python smoke_test_rust_ttt.py
"""
from __future__ import annotations

import sys

import numpy as np

import banqi_4x8 as b

# ---------------------------------------------------------------------------
# 模拟 predict_fn：返回均匀 policy logits + 零 value。
# 约定与暗棋一致：predict_fn(boards_np, scalars_np) -> (policy_logits, values)
# ---------------------------------------------------------------------------
def uniform_predict(boards: np.ndarray, scalars: np.ndarray):
    batch = boards.shape[0]
    logits = np.zeros((batch, b.TTT_ACTION_SPACE_SIZE), dtype=np.float32)
    values = np.zeros(batch, dtype=np.float32)
    return logits, values


def nested_values_predict(boards: np.ndarray, scalars: np.ndarray):
    """返回 values 为 (batch, 1)（PyTorch 网络默认形状），验证弹性提取。"""
    batch = boards.shape[0]
    logits = np.zeros((batch, b.TTT_ACTION_SPACE_SIZE), dtype=np.float32)
    values = np.zeros((batch, 1), dtype=np.float32)
    return logits, values


def show_board(cells) -> str:
    rows = []
    for r in range(3):
        row = " ".join("X" if cells[r * 3 + c] == 1 else "O" if cells[r * 3 + c] == -1 else "."
                       for c in range(3))
        rows.append(row)
    return "\n".join(rows)


def check(name: str, cond: bool, detail: str = "") -> None:
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        sys.exit(1)


def test_env_class() -> None:
    e = b.TicTacToe()
    check("TicTacToe() 构造", e.cells == [0] * 9 and e.to_play == 1)
    check("空棋盘合法动作", e.legal_moves() == list(range(9)))
    check("空棋盘无胜者", e.winner() is None)
    enc = e.encode()
    check("编码长度 = 2*9", len(enc) == 18)

    e.step(4)
    check("落子后轮到对手", e.to_play == -1)
    check("占用格不在合法动作", 4 not in e.legal_moves())
    e.step(0)
    check("第二步仍合法", e.to_play == 1)

    # 构造一步致胜局面：X 在 0,4，轮 X，走 8 即对角胜
    w_env = b.TicTacToe(cells=[1, -1, 0, 0, 1, 0, 0, 0, 0], player=1)
    check("致胜局面胜者 None", w_env.winner() is None)
    term, trunc, winner = w_env.step(8)
    check("走 8 致胜", term and winner == 1)


def test_ttt_mcts_search_initial() -> None:
    res = b.ttt_mcts_search(
        predict_fn=uniform_predict,
        cells=[0] * 9,
        player=1,
        num_simulations=96,
        max_considered_actions=9,
    )
    check("空棋盘搜索返回合法动作", not res["game_over"] and 0 <= res["action"] < 9)
    check("improved_policy 归一化", abs(sum(res["policy"]) - 1.0) < 1e-3)
    check("action_mask 全 1", res["action_mask"] == [1] * 9)
    check("board 编码长度", len(res["board"]) == 18)
    check("player 正确", res["player"] == 1)
    print("  空棋盘搜索 → action =", res["action"])


def test_ttt_mcts_search_win() -> None:
    # X 在 0,4；轮 X；走 8 致胜（0,4,8 对角）
    cells = [1, -1, 0, 0, 1, 0, 0, 0, 0]
    res = b.ttt_mcts_search(
        predict_fn=uniform_predict,
        cells=cells,
        player=1,
        num_simulations=160,
        max_considered_actions=9,
    )
    print(f"  致胜局面选 action={res['action']}（期望 8）")
    check("一步致胜局面选中 8", res["action"] == 8)


def test_ttt_mcts_search_block() -> None:
    # O 在 4,5；轮 X；X 必须走 3 堵住 O 的 3,4,5 行，否则 O 一步取胜
    cells = [0, 0, 1, 0, -1, -1, 0, 0, 0]
    res = b.ttt_mcts_search(
        predict_fn=uniform_predict,
        cells=cells,
        player=1,
        num_simulations=160,
        max_considered_actions=9,
    )
    print(f"  防守局面选 action={res['action']}（期望 3）")
    check("一步必防局面选中 3", res["action"] == 3)


def test_nested_values_shape() -> None:
    """验证 predict_fn 返回 values 形状 (batch, 1) 时仍可正常工作。"""
    res = b.ttt_mcts_search(
        predict_fn=nested_values_predict,
        cells=[1, -1, 0, 0, 1, 0, 0, 0, 0],
        player=1,
        num_simulations=96,
        max_considered_actions=9,
    )
    check("(N,1) values 兼容：一步致胜局面选中 8", res["action"] == 8)


def test_ttt_self_play() -> None:
    episodes = b.run_ttt_self_play_with_predictor(
        predict_fn=uniform_predict,
        mcts_sims=48,
        max_considered_actions=9,
        temperature_steps=6,
        num_games=1,
    )
    check("自对弈返回 1 局", len(episodes) == 1)
    ep = episodes[0]
    n = len(ep["boards"])
    check("有训练样本", n >= 1, detail=f"steps={n}")
    if n > 0:
        # 每步 board 为通道优先的扁平 list：2 通道 × 3 × 3 = 18
        check("boards 元素为 2*3*3 扁平", len(ep["boards"][0]) == 18)
        check("scalars 为空", len(ep["scalars"][0]) == 0)
        check("policies 形状", len(ep["policies"][0]) == 9)
        check("game_results 形状", len(ep["game_results"]) == n)
        check("actions 形状", len(ep["actions"]) == n)
        check("action_masks 形状", len(ep["action_masks"][0]) == 9)
        check("winner 合法", ep["winner"] in (0, 1, -1))
    print(f"  自对弈 {n} 步，winner={ep['winner']}")


def main() -> None:
    print("=" * 60)
    print("Rust 井字棋环境 + 泛型 MCTS 绑定 smoke test")
    print("=" * 60)
    print(f"banqi_4x8: {b.__file__}")
    test_env_class()
    test_ttt_mcts_search_initial()
    test_ttt_mcts_search_win()
    test_ttt_mcts_search_block()
    test_nested_values_shape()
    test_ttt_self_play()
    print("=" * 60)
    print("全部通过 ✔")
    print("=" * 60)


if __name__ == "__main__":
    main()
