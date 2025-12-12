# constant.py

# ==============================================================================
# --- Board Dimensions ---
# ==============================================================================
BOARD_ROWS = 4
BOARD_COLS = 8
TOTAL_POSITIONS = BOARD_ROWS * BOARD_COLS

# ==============================================================================
# --- Game Constants ---
# ==============================================================================
NUM_PIECE_TYPES = 7
STATE_STACK_SIZE = 1  # From game_env.rs

# Piece Counts (Per Player)
SOLDIERS_COUNT = 5
CANNONS_COUNT = 2
HORSES_COUNT = 2
CHARIOTS_COUNT = 2
ELEPHANTS_COUNT = 2
ADVISORS_COUNT = 2
GENERALS_COUNT = 1

TOTAL_PIECES_PER_PLAYER = (
    SOLDIERS_COUNT + CANNONS_COUNT + HORSES_COUNT + 
    CHARIOTS_COUNT + ELEPHANTS_COUNT + ADVISORS_COUNT + GENERALS_COUNT
) # 16

# ==============================================================================
# --- Action Space ---
# ==============================================================================
REVEAL_ACTIONS_COUNT = 32
REGULAR_MOVE_ACTIONS_COUNT = 104
CANNON_ATTACK_ACTIONS_COUNT = 216

ACTION_SPACE_SIZE = (
    REVEAL_ACTIONS_COUNT + 
    REGULAR_MOVE_ACTIONS_COUNT + 
    CANNON_ATTACK_ACTIONS_COUNT
) # Total: 352

# ==============================================================================
# --- Feature Dimensions ---
# ==============================================================================
# Board Channels: 
# Own(7 types) + Enemy(7 types) + Hidden(1) + Empty(1) = 16
BOARD_CHANNELS = 2 * NUM_PIECE_TYPES + 2

# Input Channels for the Conv Layer (taking stack size into account)
TOTAL_INPUT_CHANNELS = BOARD_CHANNELS * STATE_STACK_SIZE

# Scalar Feature Count calculation:
# 3 Global (MoveCount, RedHP, BlackHP) + 
# 2 Survival Vectors (16 each) + 
# Action Mask (ACTION_SPACE_SIZE)
SURVIVAL_VECTOR_SIZE = TOTAL_PIECES_PER_PLAYER
SCALAR_FEATURE_COUNT = 3 + (2 * SURVIVAL_VECTOR_SIZE) + ACTION_SPACE_SIZE

# ==============================================================================
# --- Model Hyperparameters ---
# ==============================================================================
HIDDEN_CHANNELS = 128
