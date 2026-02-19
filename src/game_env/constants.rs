// ==============================================================================
// --- 常量定义 ---
// ==============================================================================

/// 棋盘行数 (4)
pub const BOARD_ROWS: usize = 4;
/// 棋盘列数 (8)
pub const BOARD_COLS: usize = 8;
/// 总位置数 (32)
pub const TOTAL_POSITIONS: usize = BOARD_ROWS * BOARD_COLS;
/// 棋子种类数 (7种: 帅/将, 仕/士, 相/象, 俥/车, 傌/马, 炮/包, 兵/卒)
pub const NUM_PIECE_TYPES: usize = 7;
/// 判和步数限制：连续多少步无吃子则判和
pub const MAX_CONSECUTIVE_MOVES_FOR_DRAW: usize = 24;

/// 每局游戏最大总步数限制 (防止死循环)
pub const MAX_STEPS_PER_EPISODE: usize = 100;
/// 初始随机翻开的棋子数量 (用于加速开局)
pub const INITIAL_REVEALED_PIECES: usize = 4;

/// 初始血量 (减分制，每方从60分开始，吃子扣对方分)
pub const INITIAL_HEALTH_POINTS: i32 = 60;

// --- 棋子数量定义（每方） ---
pub const SOLDIERS_COUNT: usize = 5;
pub const CANNONS_COUNT: usize = 2;
pub const HORSES_COUNT: usize = 2;
pub const CHARIOTS_COUNT: usize = 2;
pub const ELEPHANTS_COUNT: usize = 2;
pub const ADVISORS_COUNT: usize = 2;
pub const GENERALS_COUNT: usize = 1;
pub const TOTAL_PIECES_PER_PLAYER: usize = SOLDIERS_COUNT
    + CANNONS_COUNT
    + HORSES_COUNT
    + CHARIOTS_COUNT
    + ELEPHANTS_COUNT
    + ADVISORS_COUNT
    + GENERALS_COUNT; // 16

/// 预定义的各棋子最大数量 (用于Scalar特征编码)
pub const PIECE_MAX_COUNTS: [usize; NUM_PIECE_TYPES] = [
    SOLDIERS_COUNT,
    CANNONS_COUNT,
    HORSES_COUNT,
    CHARIOTS_COUNT,
    ELEPHANTS_COUNT,
    ADVISORS_COUNT,
    GENERALS_COUNT,
];

/// 存活向量大小: 包含双方所有可能的棋子
pub const SURVIVAL_VECTOR_SIZE: usize = TOTAL_PIECES_PER_PLAYER;

/// Scalar 特征数量: 
/// 3个全局标量 (MoveCount, RedHP, BlackHP) + 2个存活向量(各16) + 动作掩码长度
pub const SCALAR_FEATURE_COUNT: usize = 3 + 2 * SURVIVAL_VECTOR_SIZE + ACTION_SPACE_SIZE;

/// 翻棋概率表大小: 2个玩家 * 7种棋子 = 14
pub const REVEAL_PROBABILITY_SIZE: usize = 2 * NUM_PIECE_TYPES;

// --- 方向常量 ---
pub const DIRECTION_UP: usize = 0;
pub const DIRECTION_DOWN: usize = 1;
pub const DIRECTION_LEFT: usize = 2;
pub const DIRECTION_RIGHT: usize = 3;
pub const NUM_DIRECTIONS: usize = 4;

/// MSB (Most Significant Bit) 位数 (64位整数的最高位索引基数)
pub const U64_BITS: usize = 64;

/// 棋盘状态张量的通道数: 
/// 己方7种 + 敌方7种 + 暗子1种 + 空位1种 = 16
pub const BOARD_CHANNELS: usize = 2 * NUM_PIECE_TYPES + 2; 

// --- 动作空间定义 ---
pub const REVEAL_ACTIONS_COUNT: usize = 32;
pub const REGULAR_MOVE_ACTIONS_COUNT: usize = 104;
pub const CANNON_ATTACK_ACTIONS_COUNT: usize = 216;
pub const ACTION_SPACE_SIZE: usize =
    REVEAL_ACTIONS_COUNT + REGULAR_MOVE_ACTIONS_COUNT + CANNON_ATTACK_ACTIONS_COUNT;
