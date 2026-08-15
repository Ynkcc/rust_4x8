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
pub mod replay;
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

// 井字棋环境（用于验证逻辑复用）
pub use game_env::{
    TTT_ACTION_SPACE_SIZE, TTT_BOARD_CHANNELS, TTT_BOARD_COLS, TTT_BOARD_ROWS,
    TTT_SCALAR_FEATURE_COUNT, TicTacToeEnv,
};

// 4x2 迷你暗棋环境（快速验证训练逻辑，仅 兵/将/士/炮，血量上限=47）
pub use game_env::{MINI_ACTION_SPACE_SIZE, MiniDarkChessEnv};

// 4x4 暗棋环境（7 类棋子全激活，每方 8 子，HP=60，分值 4/10/10/10/10/20/30）
pub use game_env::{GAME4X4_ACTION_SPACE_SIZE, Game4x4Env};

/// 4x2 迷你暗棋：棋盘通道数 = 2*4(激活类型) + 2 = 10
pub const MINI_BOARD_CHANNELS: usize = 10;
/// 4x2 迷你暗棋：棋盘行数
pub const MINI_BOARD_ROWS: usize = 4;
/// 4x2 迷你暗棋：棋盘列数
pub const MINI_BOARD_COLS: usize = 2;
/// 4x2 迷你暗棋：标量特征数 = 3 + 2*4 = 11
pub const MINI_SCALAR_FEATURE_COUNT: usize = 11;

/// 4x4 暗棋：棋盘通道数 = 2*7(全激活) + 2 = 16
pub const GAME4X4_BOARD_CHANNELS: usize = 16;
/// 4x4 暗棋：棋盘行数
pub const GAME4X4_BOARD_ROWS: usize = 4;
/// 4x4 暗棋：棋盘列数
pub const GAME4X4_BOARD_COLS: usize = 4;
/// 4x4 暗棋：标量特征数 = 3 + 2*8 = 19
pub const GAME4X4_SCALAR_FEATURE_COUNT: usize = 19;

// 泛型游戏环境抽象
pub use game_env::GameEnv;

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
    PyGameEpisode, PySelfPlayConfig, run_batched_self_play_with_predictor_impl,
    run_game4x4_batched_self_play_with_predictor_impl,
    run_game4x4_parallel_self_play_with_predictor_impl,
    run_game4x4_self_play_with_predictor_impl, run_mini_batched_self_play_with_predictor_impl,
    run_mini_parallel_self_play_with_predictor_impl, run_mini_self_play_with_predictor_impl,
    run_parallel_self_play_with_predictor_impl, run_self_play_with_predictor_impl,
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
    Ok(run_self_play_with_predictor_impl(predict_fn, cfg, num_games, worker_id))
}

/// 4x2 迷你暗棋版串行自对弈。
#[cfg(feature = "pyo3")]
#[pyfunction]
#[pyo3(signature = (predict_fn, config=None, num_games=1, worker_id=0))]
fn run_mini_self_play_with_predictor(
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
    Ok(run_mini_self_play_with_predictor_impl(predict_fn, cfg, num_games, worker_id))
}

/// 4x4 暗棋版串行自对弈。
#[cfg(feature = "pyo3")]
#[pyfunction]
#[pyo3(signature = (predict_fn, config=None, num_games=1, worker_id=0))]
fn run_game4x4_self_play_with_predictor(
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
    Ok(run_game4x4_self_play_with_predictor_impl(predict_fn, cfg, num_games, worker_id))
}

#[cfg(feature = "pyo3")]
#[pyfunction]
#[pyo3(signature = (predict_fn, config=None, num_games=1, concurrency=4, worker_id=0))]
fn run_batched_self_play_with_predictor(
    py: Python<'_>,
    predict_fn: PyObject,
    config: Option<PyRef<PySelfPlayConfig>>,
    num_games: usize,
    concurrency: usize,
    worker_id: usize,
) -> PyResult<Vec<PyGameEpisode>> {
    let cfg: SelfPlayConfig = match config {
        Some(c) => c.inner.clone(),
        None => SelfPlayConfig::default(),
    };
    Ok(run_batched_self_play_with_predictor_impl(
        py,
        predict_fn,
        cfg,
        num_games,
        concurrency,
        worker_id,
    ))
}

/// 4x2 迷你暗棋版批量自对弈。
#[cfg(feature = "pyo3")]
#[pyfunction]
#[pyo3(signature = (predict_fn, config=None, num_games=1, concurrency=4, worker_id=0))]
fn run_mini_batched_self_play_with_predictor(
    py: Python<'_>,
    predict_fn: PyObject,
    config: Option<PyRef<PySelfPlayConfig>>,
    num_games: usize,
    concurrency: usize,
    worker_id: usize,
) -> PyResult<Vec<PyGameEpisode>> {
    let cfg: SelfPlayConfig = match config {
        Some(c) => c.inner.clone(),
        None => SelfPlayConfig::default(),
    };
    Ok(run_mini_batched_self_play_with_predictor_impl(
        py,
        predict_fn,
        cfg,
        num_games,
        concurrency,
        worker_id,
    ))
}

/// 4x4 暗棋版批量自对弈。
#[cfg(feature = "pyo3")]
#[pyfunction]
#[pyo3(signature = (predict_fn, config=None, num_games=1, concurrency=4, worker_id=0))]
fn run_game4x4_batched_self_play_with_predictor(
    py: Python<'_>,
    predict_fn: PyObject,
    config: Option<PyRef<PySelfPlayConfig>>,
    num_games: usize,
    concurrency: usize,
    worker_id: usize,
) -> PyResult<Vec<PyGameEpisode>> {
    let cfg: SelfPlayConfig = match config {
        Some(c) => c.inner.clone(),
        None => SelfPlayConfig::default(),
    };
    Ok(run_game4x4_batched_self_play_with_predictor_impl(
        py,
        predict_fn,
        cfg,
        num_games,
        concurrency,
        worker_id,
    ))
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
    Ok(run_parallel_self_play_with_predictor_impl(
        predict_fn,
        cfg,
        num_workers,
        games_per_worker,
        worker_id,
    ))
}

/// 4x2 迷你暗棋版并行自对弈。
#[cfg(feature = "pyo3")]
#[pyfunction]
#[pyo3(signature = (predict_fn, config=None, num_workers=4, games_per_worker=1, worker_id=0))]
fn run_mini_parallel_self_play_with_predictor(
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
    Ok(run_mini_parallel_self_play_with_predictor_impl(
        predict_fn,
        cfg,
        num_workers,
        games_per_worker,
        worker_id,
    ))
}

/// 4x4 暗棋版并行自对弈。
#[cfg(feature = "pyo3")]
#[pyfunction]
#[pyo3(signature = (predict_fn, config=None, num_workers=4, games_per_worker=1, worker_id=0))]
fn run_game4x4_parallel_self_play_with_predictor(
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
    Ok(run_game4x4_parallel_self_play_with_predictor_impl(
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
    m.add_function(wrap_pyfunction!(run_batched_self_play_with_predictor, m)?)?;

    // --- 4x2 迷你暗棋自对弈绑定 ---
    m.add_function(wrap_pyfunction!(run_mini_self_play_with_predictor, m)?)?;
    m.add_function(wrap_pyfunction!(run_mini_parallel_self_play_with_predictor, m)?)?;
    m.add_function(wrap_pyfunction!(run_mini_batched_self_play_with_predictor, m)?)?;

    // --- 4x4 暗棋自对弈绑定 ---
    m.add_function(wrap_pyfunction!(run_game4x4_self_play_with_predictor, m)?)?;
    m.add_function(wrap_pyfunction!(run_game4x4_parallel_self_play_with_predictor, m)?)?;
    m.add_function(wrap_pyfunction!(run_game4x4_batched_self_play_with_predictor, m)?)?;

    m.add_function(wrap_pyfunction!(crate::py::describe_record, m)?)?;

    // --- 井字棋绑定（验证逻辑复用） ---
    m.add_class::<crate::py::ttt::PyTicTacToe>()?;
    m.add_function(wrap_pyfunction!(crate::py::ttt::ttt_mcts_search, m)?)?;
    m.add_function(wrap_pyfunction!(crate::py::ttt::run_ttt_self_play_with_predictor, m)?)?;

    // --- 暗棋环境绑定（视角反转验证） ---
    m.add_class::<crate::py::darkchess_env::PyDarkChess>()?;

    // --- 4x2 迷你暗棋环境绑定（训练验证） ---
    m.add_class::<crate::py::mini_darkchess_env::PyMiniDarkChess>()?;

    // --- 4x4 暗棋环境绑定（变体训练验证） ---
    m.add_class::<crate::py::game4x4_env::PyGame4x4>()?;

    m.add("BOARD_ROWS", BOARD_ROWS)?;
    m.add("BOARD_COLS", BOARD_COLS)?;
    m.add("BOARD_CHANNELS", BOARD_CHANNELS)?;
    m.add("SCALAR_FEATURE_COUNT", SCALAR_FEATURE_COUNT)?;
    m.add("ACTION_SPACE_SIZE", ACTION_SPACE_SIZE)?;

    m.add("MINI_BOARD_ROWS", MINI_BOARD_ROWS)?;
    m.add("MINI_BOARD_COLS", MINI_BOARD_COLS)?;
    m.add("MINI_BOARD_CHANNELS", MINI_BOARD_CHANNELS)?;
    m.add("MINI_SCALAR_FEATURE_COUNT", MINI_SCALAR_FEATURE_COUNT)?;
    m.add("MINI_ACTION_SPACE_SIZE", MINI_ACTION_SPACE_SIZE)?;

    m.add("GAME4X4_BOARD_ROWS", GAME4X4_BOARD_ROWS)?;
    m.add("GAME4X4_BOARD_COLS", GAME4X4_BOARD_COLS)?;
    m.add("GAME4X4_BOARD_CHANNELS", GAME4X4_BOARD_CHANNELS)?;
    m.add("GAME4X4_SCALAR_FEATURE_COUNT", GAME4X4_SCALAR_FEATURE_COUNT)?;
    m.add("GAME4X4_ACTION_SPACE_SIZE", GAME4X4_ACTION_SPACE_SIZE)?;

    m.add("TTT_ACTION_SPACE_SIZE", crate::TTT_ACTION_SPACE_SIZE)?;
    m.add("TTT_BOARD_ROWS", crate::TTT_BOARD_ROWS)?;
    m.add("TTT_BOARD_COLS", crate::TTT_BOARD_COLS)?;
    m.add("TTT_BOARD_CHANNELS", crate::TTT_BOARD_CHANNELS)?;
    m.add("TTT_SCALAR_FEATURE_COUNT", crate::TTT_SCALAR_FEATURE_COUNT)?;

    Ok(())
}
