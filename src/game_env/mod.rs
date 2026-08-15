// ==============================================================================
// --- 模块声明 ---
// ==============================================================================

pub mod actions;
pub mod bitboard;
pub mod board;
pub mod config;
pub mod constants;
pub mod features;
pub mod mini_darkchess;
pub mod rules;
pub mod tic_tac_toe;
pub mod traits;
pub mod types;

// ==============================================================================
// --- 公共 API 重导出 ---
// ==============================================================================

// 从 constants.rs 导出所有常量
pub use constants::*;

// 从 config.rs 导出配置与预设
pub use config::{
    GameConfig, MAX_PIECES_PER_PLAYER, MAX_POSITIONS, MAX_REVEAL_PROBABILITY_SIZE,
    NUM_PIECE_TYPES_MAX, compute_action_counts, darkchess_config, mini_config,
};

// 从 types.rs 导出所有数据类型
pub use types::{Observation, Piece, PieceType, Player, Slot};

// 从 board.rs 导出主要的环境结构体
pub use board::DarkChessEnv;

// 从 mini_darkchess.rs 导出 4x2 迷你环境与常量
pub use mini_darkchess::{MINI_ACTION_SPACE_SIZE, MiniDarkChessEnv};

// 从 tic_tac_toe.rs 导出井字棋环境与常量
pub use tic_tac_toe::{
    TTT_ACTION_SPACE_SIZE, TTT_BOARD_CHANNELS, TTT_BOARD_COLS, TTT_BOARD_ROWS,
    TTT_SCALAR_FEATURE_COUNT, TicTacToeEnv,
};

// 从 traits.rs 导出泛型游戏环境抽象
pub use traits::GameEnv;

// 从 bitboard.rs 导出部分工具函数 (如果外部需要)
pub use bitboard::ull;

// 从 actions.rs 导出动作查找表访问器（需要传 config）
pub use actions::action_lookup_tables;
