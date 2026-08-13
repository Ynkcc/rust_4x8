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

pub mod py;

// gRPC 模块 (已禁用 - 现在使用本地模型推理)
// pub mod rpc;

// AI 策略模块（基础策略不依赖 torch）
pub mod ai;
// 注意: nn_model 已移至 Python 侧，Rust 通过 TorchScript 加载模型

// 本地 TorchScript 评估器（仅在 torch feature 启用时编译）
#[cfg(feature = "torch")]
pub mod local_evaluator;

// 重新导出核心类型，方便外部使用
pub use game_env::{DarkChessEnv, Observation, Piece, PieceType, Player, Slot};

// 导出常量
pub use game_env::{
    ACTION_SPACE_SIZE, BOARD_COLS, BOARD_ROWS, NUM_PIECE_TYPES, REGULAR_MOVE_ACTIONS_COUNT,
    REVEAL_ACTIONS_COUNT, TOTAL_POSITIONS,
};

// ============================================================================
// PyO3 模块导出（仅在 pyo3 feature 启用时编译）
// ============================================================================

#[cfg(feature = "pyo3")]
use pyo3::prelude::*;

#[cfg(feature = "pyo3")]
use crate::game_env::{BOARD_CHANNELS, SCALAR_FEATURE_COUNT};
#[cfg(feature = "pyo3")]
use crate::py::{
    PyGameEpisode, PySelfPlayConfig, _run_parallel_self_play_with_predictor,
    _run_self_play_with_predictor,
};
#[cfg(feature = "pyo3")]
use crate::self_play::SelfPlayConfig;

#[cfg(feature = "pyo3")]
#[pyfunction]
#[pyo3(signature = (predict_fn, config=None, num_games=1, worker_id=0))]
fn run_self_play_with_predictor(
    _py: Python<'_>,
    predict_fn: PyObject,
    config: Option<PyRef<PySelfPlayConfig>>,
    num_games: usize,
    worker_id: usize,
) -> PyResult<Vec<PyGameEpisode>> {
    let cfg: SelfPlayConfig = match config {
        Some(c) => c.inner.clone(),
        None => SelfPlayConfig::default(),
    };
    Ok(_run_self_play_with_predictor(predict_fn, cfg, num_games, worker_id))
}

#[cfg(feature = "pyo3")]
#[pyfunction]
#[pyo3(signature = (predict_fn, config=None, num_workers=4, games_per_worker=1, worker_id=0))]
fn run_parallel_self_play_with_predictor(
    _py: Python<'_>,
    predict_fn: PyObject,
    config: Option<PyRef<PySelfPlayConfig>>,
    num_workers: usize,
    games_per_worker: usize,
    worker_id: usize,
) -> PyResult<Vec<PyGameEpisode>> {
    let cfg: SelfPlayConfig = match config {
        Some(c) => c.inner.clone(),
        None => SelfPlayConfig::default(),
    };
    Ok(_run_parallel_self_play_with_predictor(
        predict_fn,
        cfg,
        num_workers,
        games_per_worker,
        worker_id,
    ))
}

#[cfg(feature = "pyo3")]
#[pymodule]
fn banqi_4x8(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyGameEpisode>()?;
    m.add_class::<PySelfPlayConfig>()?;
    m.add_function(wrap_pyfunction!(run_self_play_with_predictor, m)?)?;
    m.add_function(wrap_pyfunction!(run_parallel_self_play_with_predictor, m)?)?;

    m.add("BOARD_ROWS", BOARD_ROWS)?;
    m.add("BOARD_COLS", BOARD_COLS)?;
    m.add("BOARD_CHANNELS", BOARD_CHANNELS)?;
    m.add("SCALAR_FEATURE_COUNT", SCALAR_FEATURE_COUNT)?;
    m.add("ACTION_SPACE_SIZE", ACTION_SPACE_SIZE)?;

    Ok(())
}
