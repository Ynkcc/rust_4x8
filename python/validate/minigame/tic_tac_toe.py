"""
tic_tac_toe.py — 井字棋环境（微型验证引擎）。

特征编码对齐 Rust `src/game_env/features.rs` 的视角约定：
  通道 0 = 当前行动方（current player）的棋子
  通道 1 = 对手（opponent）的棋子
即"通道 0 恒为当前走子方"，这是视角反转测试的关键断言点。

价值标签语义对齐 Rust `src/self_play.rs::finalize_episode`：
  game_result_val = player == Red ? reward_red : -reward_red
（此处 Red=+1 当前方，Black=-1 对手）
"""

from __future__ import annotations

import numpy as np

# 井字棋常量
BOARD_ROWS = 3
BOARD_COLS = 3
N_CELLS = BOARD_ROWS * BOARD_COLS          # 9
CHANNELS = 2                               # 通道0=当前方, 通道1=对手

# 玩家编码（对齐 Rust Player::Red=1 / Black=-1）
CURRENT = 1   # 当前行动方（对应 Red）
OPPONENT = -1 # 对手（对应 Black）


def _lines() -> list[list[int]]:
    """所有可能连成三子的线（3 行 + 3 列 + 2 对角线）。"""
    rows = [[r * BOARD_COLS + c for c in range(BOARD_COLS)] for r in range(BOARD_ROWS)]
    cols = [[r * BOARD_COLS + c for r in range(BOARD_ROWS)] for c in range(BOARD_COLS)]
    diag1 = [r * BOARD_COLS + c for r in range(BOARD_ROWS) for c in range(BOARD_COLS)
             if r == c]
    diag2 = [r * BOARD_COLS + c for r in range(BOARD_ROWS) for c in range(BOARD_COLS)
             if r + c == BOARD_ROWS - 1]
    return rows + cols + [diag1, diag2]


LINES = _lines()


class TicTacToe:
    """
    井字棋环境。
    内部状态：cells 长度为 9 的数组，取值 0=空, 1=当前方, -1=对手。
    视角约定：`to_play` 始终表示"下一步轮到谁"（+1 或 -1）。
    特征编码按"当前行动方"重排：通道0 = 轮到方棋子，通道1 = 另一方棋子。
    """

    __slots__ = ("cells", "to_play", "moves_done", "_winner_cache")

    def __init__(self, cells=None, to_play=CURRENT):
        # cells 以"绝对玩家"存储：1=红方(+1), -1=黑方(-1)
        if cells is None:
            self.cells = [0] * N_CELLS
        else:
            self.cells = list(cells)
        self.to_play = to_play           # 当前行动方（下一步轮到谁）
        self.moves_done = sum(1 for x in self.cells if x != 0)
        self._winner_cache = None

    # ------------------------------------------------------------------
    # 基本查询
    # ------------------------------------------------------------------
    def clone(self) -> "TicTacToe":
        return TicTacToe(self.cells, self.to_play)

    def legal_actions(self) -> list[int]:
        """返回所有空位（合法落子位置）。"""
        return [i for i in range(N_CELLS) if self.cells[i] == 0]

    def is_terminal(self) -> bool:
        return self.winner() is not None

    def winner(self):
        """
        返回终局结果（绝对玩家视角）：
          +1 = 红方(+1) 胜, -1 = 黑方(-1) 胜, 0 = 平局, None = 未终局
        """
        if self._winner_cache is not None:
            return self._winner_cache
        for line in LINES:
            vals = [self.cells[i] for i in line]
            if vals[0] != 0 and vals[0] == vals[1] == vals[2]:
                self._winner_cache = vals[0]   # 绝对玩家 ±1
                return self._winner_cache
        if all(x != 0 for x in self.cells):
            self._winner_cache = 0             # 平局
            return 0
        return None

    # ------------------------------------------------------------------
    # 动作
    # ------------------------------------------------------------------
    def step(self, action: int):
        """
        在当前行动方落子。返回 (new_env, terminated, winner)。
        winner 为绝对玩家（±1/0），供价值标签换算。
        """
        if self.cells[action] != 0:
            raise ValueError(f"非法落子: 位置 {action} 已被占用")
        new_cells = list(self.cells)
        new_cells[action] = self.to_play      # 当前行动方落子
        new_env = TicTacToe(new_cells, -self.to_play)   # 换手
        new_env._winner_cache = new_env.winner()
        terminated = new_env._winner_cache is not None
        return new_env, terminated, new_env._winner_cache

    # ------------------------------------------------------------------
    # 神经网络输入
    # ------------------------------------------------------------------
    def encode(self) -> np.ndarray:
        """
        编码为 (CHANNELS, ROWS, COLS) 特征张量。
        对齐 features.rs：通道0 = 当前行动方棋子，通道1 = 对手棋子。
        """
        feat = np.zeros((CHANNELS, BOARD_ROWS, BOARD_COLS), dtype=np.float32)
        for i, v in enumerate(self.cells):
            if v == 0:
                continue
            r, c = divmod(i, BOARD_COLS)
            # v 是绝对玩家，self.to_play 是当前行动方
            if v == self.to_play:
                feat[0, r, c] = 1.0
            else:
                feat[1, r, c] = 1.0
        return feat

    # ------------------------------------------------------------------
    # 价值标签（对齐 finalize_episode）
    # ------------------------------------------------------------------
    def result_from_perspective(self, player: int) -> float:
        """
        返回绝对胜者 winner 从 `player` 视角看到的结果。
        等价于 `player * winner`（winner 为 ±1/0）。
        """
        winner = self.winner()
        if winner is None:
            raise ValueError("未终局无法取结果")
        if winner == 0:
            return 0.0
        return float(player * winner)

    def __repr__(self) -> str:
        return (f"TicTacToe(cells={self.cells}, to_play={self.to_play}, "
                f"winner={self.winner()})")


def make_reward(winner: int) -> float:
    """对齐 finalize_episode 的 reward_red：winner=1→1.0, -1→-1.0, 0→0.0。"""
    if winner == 0:
        return 0.0
    return float(winner)


def value_label_for_player(player: int, winner: int) -> float:
    """
    复刻 finalize_episode：给定某步的 `player`（绝对玩家 ±1）与终局 `winner`，
    返回该玩家视角的最终收益标签。
      value_label_for_player(1, 1)  -> +1.0  红方胜
      value_label_for_player(-1, 1) -> -1.0  黑方负
      value_label_for_player(-1,-1) -> +1.0  黑方胜
      value_label_for_player(1, 0)  ->  0.0  平局
    """
    if winner == 0:
        return 0.0
    return float(player * winner)


# 简单的确定性局面构造辅助（供测试构造特定局面）
def board_from_string(rows) -> "TicTacToe":
    """
    从字符串棋盘构造局面，'X'=红方(+1), 'O'=黑方(-1), '.'=空。
    例如: ["X.O", ".X.", "O.."]
    返回 (env, next_player)。next_player 由 X/O 数量差决定。
    """
    assert len(rows) == BOARD_ROWS and all(len(r) == BOARD_COLS for r in rows), \
        "棋盘必须是 3x3"
    cells = [0] * N_CELLS
    n_x = 0
    n_o = 0
    for r in range(BOARD_ROWS):
        for c in range(BOARD_COLS):
            ch = rows[r][c]
            idx = r * BOARD_COLS + c
            if ch == "X":
                cells[idx] = 1
                n_x += 1
            elif ch == "O":
                cells[idx] = -1
                n_o += 1
    # X 先手（当前方=红=+1）；若 X 多一子则轮到 O（-1）
    if n_x == n_o:
        to_play = 1
    elif n_x == n_o + 1:
        to_play = -1
    else:
        raise ValueError("非法的 X/O 数量")
    env = TicTacToe(cells, to_play)
    env._winner_cache = env.winner()
    return env
