// ==============================================================================
// --- 模块声明 ---
// ==============================================================================

pub mod actions;
pub mod bitboard;
pub mod board;
pub mod config;
pub mod constants;
pub mod features;
pub mod rules;
pub mod symmetry;
pub mod traits;
pub mod types;
pub mod variants;

// ==============================================================================
// --- 公共 API 导出 ---
// ==============================================================================

pub use constants::*;
pub use config::{
    GameConfig, MAX_PIECES_PER_PLAYER, MAX_POSITIONS, MAX_REVEAL_PROBABILITY_SIZE,
    NUM_PIECE_TYPES_MAX, compute_action_counts, darkchess_config, game_4x4_config, mini_config,
};
pub use symmetry::{Symmetry, action_permutation, sq_map, transform_action, transform_board_flat};
pub use types::{ResNetObservation, Piece, PieceType, Player, Slot};
pub use board::DarkChessEnv;
pub use traits::GameEnv;
pub use bitboard::ull;
pub use actions::action_lookup_tables;

pub use variants::{
    GAME4X4_ACTION_SPACE_SIZE, GAME4X4_RESNET_BOARD_CHANNELS, GAME4X4_BOARD_COLS, GAME4X4_BOARD_ROWS,
    GAME4X4_RESNET_SCALAR_FEATURE_COUNT, Game4x4Env, MINI_ACTION_SPACE_SIZE, MINI_RESNET_BOARD_CHANNELS,
    MINI_BOARD_COLS, MINI_BOARD_ROWS, MINI_RESNET_SCALAR_FEATURE_COUNT, MiniDarkChessEnv,
    TTT_ACTION_SPACE_SIZE, TTT_RESNET_BOARD_CHANNELS, TTT_BOARD_COLS, TTT_BOARD_ROWS,
    TTT_RESNET_SCALAR_FEATURE_COUNT, TicTacToeEnv,
};
