//! 游戏变体环境子模块
//! 包含 4x4 暗棋、4x2 迷你暗棋及井字棋环境。

pub mod game4x4;
pub mod mini_darkchess;
pub mod tic_tac_toe;

pub use game4x4::{
    GAME4X4_ACTION_SPACE_SIZE, GAME4X4_BOARD_CHANNELS, GAME4X4_BOARD_COLS, GAME4X4_BOARD_ROWS,
    GAME4X4_SCALAR_FEATURE_COUNT, Game4x4Env,
};
pub use mini_darkchess::{
    MINI_ACTION_SPACE_SIZE, MINI_BOARD_CHANNELS, MINI_BOARD_COLS, MINI_BOARD_ROWS,
    MINI_SCALAR_FEATURE_COUNT, MiniDarkChessEnv,
};
pub use tic_tac_toe::{
    TTT_ACTION_SPACE_SIZE, TTT_BOARD_CHANNELS, TTT_BOARD_COLS, TTT_BOARD_ROWS,
    TTT_SCALAR_FEATURE_COUNT, TicTacToeEnv,
};
