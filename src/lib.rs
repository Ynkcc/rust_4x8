//! # Banqi 4x8 - 迷你暗棋游戏库
//!
//! 这是一个用于强化学习的 4x8 暗棋游戏环境实现。
//!
//! ## 模块
//! - `game_env`: 核心游戏逻辑和环境实现
//!
//! ## 使用示例
//! ```rust
//! use banqi_4x8::DarkChessEnv;
//!
//! let mut env = DarkChessEnv::new();
//! let obs = env.reset();
//! // 进行游戏...
//! ```

pub mod game_env;
pub mod mcts;
pub mod mongodb_storage;
pub mod self_play;

// gRPC 模块
pub mod rpc;

// tch 相关模块 - 仅在启用 torch 特性时编译
#[cfg(feature = "torch")]
pub mod ai;
#[cfg(feature = "torch")]
pub mod nn_model;
#[cfg(feature = "torch")]
pub mod inference;
#[cfg(feature = "torch")]
pub mod training;

// 重新导出核心类型，方便外部使用
pub use game_env::{DarkChessEnv, Observation, Piece, PieceType, Player, Slot};

// 导出常量
pub use game_env::{
    ACTION_SPACE_SIZE, BOARD_COLS, BOARD_ROWS, NUM_PIECE_TYPES, REGULAR_MOVE_ACTIONS_COUNT,
    REVEAL_ACTIONS_COUNT, STATE_STACK_SIZE, TOTAL_POSITIONS,
};
