# constant.py — 4x4 暗棋维度常量
#
# 与 Rust `game_4x4_config()` 保持一致：
#   - 棋盘 4x4（16 格）
#   - 7 类棋子全激活，每方：兵2 炮1 马1 车1 象1 士1 将1（共 8 子）
#   - 分值：兵4 / 炮10 / 马10 / 车10 / 象10 / 士20 / 将30
#   - 血量上限 = 60（变体指定）
#   - 动作空间 = 翻16 + 常规48 + 炮48 = 112

# --- 棋盘 ---
BOARD_ROWS = 4
BOARD_COLS = 4
NUM_POSITIONS = BOARD_ROWS * BOARD_COLS  # 16

# --- 特征维度（与 Rust GameEnv 关联常量一致） ---
# 7 类棋子全激活 → 我方 7 通道 + 对方 7 通道 + hidden 1 + empty 1 = 16
TOTAL_INPUT_CHANNELS = 16
# 标量 = move_count + my_hp + opp_hp + my_survival(8) + opp_survival(8) = 19
SCALAR_FEATURE_COUNT = 19

# --- 动作空间 ---
# 翻棋 16 + 常规移动 48 + 炮击 48 = 112
ACTION_SPACE_SIZE = 112
REVEAL_ACTIONS_COUNT = 16
REGULAR_MOVE_ACTIONS_COUNT = 48
CANNON_ATTACK_ACTIONS_COUNT = 48

# --- 子力配置 ---
# 每方棋子数量（按 PieceType 索引：兵/炮/马/车/象/士/将）
PIECE_COUNTS = [2, 1, 1, 1, 1, 1, 1]
TOTAL_PIECES_PER_PLAYER = sum(PIECE_COUNTS)  # 8

# 每方棋子分值（按 PieceType 索引）
PIECE_VALUES = [4, 10, 10, 10, 10, 20, 30]
INITIAL_HEALTH = 60

# --- 网络 ---
HIDDEN_CHANNELS = 24
NUM_RES_BLOCKS = 2
