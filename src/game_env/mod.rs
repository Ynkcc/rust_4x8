// ==============================================================================
// --- 模块声明 ---
// ==============================================================================

pub mod constants;
pub mod types;
pub mod bitboard;
pub mod actions;
pub mod board;
pub mod rules;
pub mod features;

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
pub use bitboard::{ull, BOARD_MASK};

// 从 actions.rs 导出动作查找表访问器
pub use actions::action_lookup_tables;
