# constant_mini.py — 4x2 迷你暗棋的常量
#
# 仅使用 兵(Soldier)/炮(Cannon)/士(Advisor)/将(General) 四种棋子，每方各 1 子。
# 棋盘 4 行 × 2 列 = 8 格（双方共 8 子填满）。
# 血量上限 = 单方棋子价值总和 = 兵2 + 炮5 + 士10 + 将30 = 47，全灭敌方即判胜。
#
# 这些数值必须与 Rust 侧 `game_env/config.rs::mini_config()` 严格一致。

# ==============================================================================
# --- Board Dimensions ---
# ==============================================================================
BOARD_ROWS = 4
BOARD_COLS = 2
TOTAL_POSITIONS = BOARD_ROWS * BOARD_COLS

# ==============================================================================
# --- Game Constants (4x2 迷你) ---
# ==============================================================================
# 激活的棋子类型数（仅 4 种）
NUM_ACTIVE_PIECE_TYPES = 4

# 每方棋子数量（按实际 PieceType 索引：兵0/炮1/马2/车3/象4/士5/将6）
SOLDIERS_COUNT = 1
CANNONS_COUNT = 1
HORSES_COUNT = 0
CHARIOTS_COUNT = 0
ELEPHANTS_COUNT = 0
ADVISORS_COUNT = 1
GENERALS_COUNT = 1

TOTAL_PIECES_PER_PLAYER = (
    SOLDIERS_COUNT + CANNONS_COUNT + HORSES_COUNT
    + CHARIOTS_COUNT + ELEPHANTS_COUNT + ADVISORS_COUNT + GENERALS_COUNT
)  # 4

# 血量上限 = 单方棋子价值总和 = 47
INITIAL_HEALTH_POINTS = 2 + 5 + 10 + 30

# ==============================================================================
# --- Action Space (与 mini_config 一致) ---
# ==============================================================================
REVEAL_ACTIONS_COUNT = 8     # 每格一个翻棋动作
REGULAR_MOVE_ACTIONS_COUNT = 20
CANNON_ATTACK_ACTIONS_COUNT = 12

ACTION_SPACE_SIZE = (
    REVEAL_ACTIONS_COUNT + REGULAR_MOVE_ACTIONS_COUNT + CANNON_ATTACK_ACTIONS_COUNT
)  # 40

# ==============================================================================
# --- Feature Dimensions ---
# ==============================================================================
# 己方 4 种 + 敌方 4 种 + Hidden + Empty = 10 通道
BOARD_CHANNELS = 2 * NUM_ACTIVE_PIECE_TYPES + 2
TOTAL_INPUT_CHANNELS = BOARD_CHANNELS

# 标量：3 全局（MoveCount, MyHP, OppHP）+ 2 × 4 存活向量 = 11
SURVIVAL_VECTOR_SIZE = TOTAL_PIECES_PER_PLAYER
SCALAR_FEATURE_COUNT = 3 + (2 * SURVIVAL_VECTOR_SIZE)  # 11

# ==============================================================================
# --- Model Hyperparameters (极小网络) ---
# ==============================================================================
HIDDEN_CHANNELS = 16
NUM_RES_BLOCKS = 1
