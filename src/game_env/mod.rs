// ==============================================================================
// --- 模块声明 ---
// ==============================================================================

pub mod actions;
pub mod bitboard;
pub mod board;
pub mod constants;
pub mod features;
pub mod rules;
pub mod types;

// ==============================================================================
// --- 公共 API 重导出 ---
// ==============================================================================

// 从 constants.rs 导出所有常量
pub use constants::*;

// 从 types.rs 导出所有数据类型
pub use types::{Observation, Piece, PieceType, Player, Slot};

// 从 board.rs 导出主要的环境结构体
pub use board::DarkChessEnv;

// 从 bitboard.rs 导出部分工具函数 (如果外部需要)
pub use bitboard::{BOARD_MASK, ull};

// 从 actions.rs 导出动作查找表访问器
pub use actions::action_lookup_tables;
