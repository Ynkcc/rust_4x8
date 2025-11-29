//! # Banqi 3x4 - 迷你暗棋游戏库
//!
//! 这是一个用于强化学习的 3x4 暗棋游戏环境实现。
//!
//! ## 模块
//! - `game_env`: 核心游戏逻辑和环境实现
//!
//! ## 使用示例
//! ```rust
//! use banqi_3x4::DarkChessEnv;
//!
//! let mut env = DarkChessEnv::new();
//! let obs = env.reset();
//! // 进行游戏...
//! ```

pub mod game_env;
pub mod ai;
pub mod mcts;
pub mod nn_model;

// 并行训练相关模块
pub mod inference;
pub mod training_log;
pub mod database;
pub mod self_play;
pub mod scenario_validation;
pub mod training;
pub mod lr_finder;

// 重新导出核心类型，方便外部使用
pub use game_env::{
    DarkChessEnv,
    Observation,
    PieceType,
    Player,
    Piece,
    Slot,
};

// 导出常量
pub use game_env::{
    ACTION_SPACE_SIZE,
    REVEAL_ACTIONS_COUNT,
    REGULAR_MOVE_ACTIONS_COUNT,
    BOARD_ROWS,
    BOARD_COLS,
    TOTAL_POSITIONS,
    NUM_PIECE_TYPES,
    STATE_STACK_SIZE,
};
