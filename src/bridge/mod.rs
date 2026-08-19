//! 跨语言交互桥梁模块 (Language Interop Bridge)
//!
//! 包含 PyO3 Python C-Extension 绑定与 Python API 接口。
//!
//! `lib.rs` 仅声明顶层模块；本文件承载 PyO3 扩展模块（`banqi_4x8`）的
//! `#[pymodule]` 入口、对 Python 可见的 `#[pyfunction]` 转发包装，以及
//! 各类 pyclass / 常量的注册。

#[cfg(feature = "pyo3")]
pub mod python;

#[cfg(feature = "pyo3")]
use pyo3::prelude::*;

#[cfg(feature = "pyo3")]
use crate::bridge::python::{
    PyGameEpisode, PySelfPlayConfig, describe_record, decode_scalar_state,
    register_augment_functions, run_batched_self_play_with_predictor_impl,
    run_game4x4_batched_self_play_with_predictor_impl,
    run_game4x4_heuristic_self_play_impl, run_game4x4_minimax_self_play_impl,
    run_heuristic_self_play_impl, run_mini_batched_self_play_with_predictor_impl,
    run_mini_heuristic_self_play_impl, run_mini_minimax_self_play_impl,
    run_minimax_self_play_impl,
};
#[cfg(feature = "pyo3")]
use crate::core::env::{
    ACTION_SPACE_SIZE, BOARD_CHANNELS, BOARD_COLS, BOARD_ROWS, GAME4X4_ACTION_SPACE_SIZE,
    GAME4X4_BOARD_CHANNELS, GAME4X4_BOARD_COLS, GAME4X4_BOARD_ROWS, GAME4X4_SCALAR_FEATURE_COUNT,
    MINI_ACTION_SPACE_SIZE, MINI_BOARD_CHANNELS, MINI_BOARD_COLS, MINI_BOARD_ROWS,
    MINI_SCALAR_FEATURE_COUNT, SCALAR_FEATURE_COUNT, TTT_ACTION_SPACE_SIZE, TTT_BOARD_CHANNELS,
    TTT_BOARD_COLS, TTT_BOARD_ROWS, TTT_SCALAR_FEATURE_COUNT,
};
#[cfg(feature = "pyo3")]
use crate::pipeline::self_play::SelfPlayConfig;

// ---- 批量自对弈（流水线） ----

#[cfg(feature = "pyo3")]
#[pyfunction]
#[pyo3(signature = (predict_fn, config=None, num_games=1, concurrency=4, worker_id=0))]
fn run_batched_self_play_with_predictor(
    py: Python<'_>,
    predict_fn: Py<PyAny>,
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

#[cfg(feature = "pyo3")]
#[pyfunction]
#[pyo3(signature = (predict_fn, config=None, num_games=1, concurrency=4, worker_id=0))]
fn run_mini_batched_self_play_with_predictor(
    py: Python<'_>,
    predict_fn: Py<PyAny>,
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

#[cfg(feature = "pyo3")]
#[pyfunction]
#[pyo3(signature = (predict_fn, config=None, num_games=1, concurrency=4, worker_id=0))]
fn run_game4x4_batched_self_play_with_predictor(
    py: Python<'_>,
    predict_fn: Py<PyAny>,
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

// ---- 教师自对弈（4x8 启发式 / minimax） ----

#[cfg(feature = "pyo3")]
#[pyfunction]
#[pyo3(signature = (config=None, num_games=1, concurrency=8, worker_id=0))]
fn run_heuristic_self_play(
    py: Python<'_>,
    config: Option<PyRef<PySelfPlayConfig>>,
    num_games: usize,
    concurrency: usize,
    worker_id: usize,
) -> PyResult<Vec<PyGameEpisode>> {
    let cfg: SelfPlayConfig = match config {
        Some(c) => c.inner.clone(),
        None => SelfPlayConfig::default(),
    };
    Ok(run_heuristic_self_play_impl(
        py, &cfg, num_games, concurrency, worker_id,
    ))
}

#[cfg(feature = "pyo3")]
#[pyfunction]
#[pyo3(signature = (depth=2, num_games=1, concurrency=4, temperature=0.5))]
fn run_minimax_self_play(
    py: Python<'_>,
    depth: usize,
    num_games: usize,
    concurrency: usize,
    temperature: f32,
) -> PyResult<Vec<PyGameEpisode>> {
    Ok(run_minimax_self_play_impl(
        py, depth, num_games, concurrency, temperature,
    ))
}

// ---- 教师自对弈（4x2 迷你启发式 / minimax） ----

#[cfg(feature = "pyo3")]
#[pyfunction]
#[pyo3(signature = (config=None, num_games=1, concurrency=8, worker_id=0))]
fn run_mini_heuristic_self_play(
    py: Python<'_>,
    config: Option<PyRef<PySelfPlayConfig>>,
    num_games: usize,
    concurrency: usize,
    worker_id: usize,
) -> PyResult<Vec<PyGameEpisode>> {
    let cfg: SelfPlayConfig = match config {
        Some(c) => c.inner.clone(),
        None => SelfPlayConfig::default(),
    };
    Ok(run_mini_heuristic_self_play_impl(
        py, &cfg, num_games, concurrency, worker_id,
    ))
}

#[cfg(feature = "pyo3")]
#[pyfunction]
#[pyo3(signature = (depth=2, num_games=1, concurrency=4, temperature=0.5))]
fn run_mini_minimax_self_play(
    py: Python<'_>,
    depth: usize,
    num_games: usize,
    concurrency: usize,
    temperature: f32,
) -> PyResult<Vec<PyGameEpisode>> {
    Ok(run_mini_minimax_self_play_impl(
        py, depth, num_games, concurrency, temperature,
    ))
}

// ---- 教师自对弈（4x4 启发式 / minimax） ----

#[cfg(feature = "pyo3")]
#[pyfunction]
#[pyo3(signature = (config=None, num_games=1, concurrency=8, worker_id=0))]
fn run_game4x4_heuristic_self_play(
    py: Python<'_>,
    config: Option<PyRef<PySelfPlayConfig>>,
    num_games: usize,
    concurrency: usize,
    worker_id: usize,
) -> PyResult<Vec<PyGameEpisode>> {
    let cfg: SelfPlayConfig = match config {
        Some(c) => c.inner.clone(),
        None => SelfPlayConfig::default(),
    };
    Ok(run_game4x4_heuristic_self_play_impl(
        py, &cfg, num_games, concurrency, worker_id,
    ))
}

#[cfg(feature = "pyo3")]
#[pyfunction]
#[pyo3(signature = (depth=2, num_games=1, concurrency=4, temperature=0.5))]
fn run_game4x4_minimax_self_play(
    py: Python<'_>,
    depth: usize,
    num_games: usize,
    concurrency: usize,
    temperature: f32,
) -> PyResult<Vec<PyGameEpisode>> {
    Ok(run_game4x4_minimax_self_play_impl(
        py, depth, num_games, concurrency, temperature,
    ))
}

// ---- pymodule 入口 ----

#[cfg(feature = "pyo3")]
#[pymodule]
fn banqi_4x8(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyGameEpisode>()?;
    m.add_class::<PySelfPlayConfig>()?;
    m.add_function(wrap_pyfunction!(run_batched_self_play_with_predictor, m)?)?;

    // --- 4x2 迷你暗棋自对弈绑定 ---
    m.add_function(wrap_pyfunction!(run_mini_batched_self_play_with_predictor, m)?)?;

    // --- 4x4 暗棋自对弈绑定 ---
    m.add_function(wrap_pyfunction!(run_game4x4_batched_self_play_with_predictor, m)?)?;
    m.add_function(wrap_pyfunction!(run_game4x4_heuristic_self_play, m)?)?;
    m.add_function(wrap_pyfunction!(run_game4x4_minimax_self_play, m)?)?;

    // --- 教师自对弈（4x8 启发式 / minimax） ---
    m.add_function(wrap_pyfunction!(run_heuristic_self_play, m)?)?;
    m.add_function(wrap_pyfunction!(run_minimax_self_play, m)?)?;

    // --- 教师自对弈（4x2 迷你启发式 / minimax） ---
    m.add_function(wrap_pyfunction!(run_mini_heuristic_self_play, m)?)?;
    m.add_function(wrap_pyfunction!(run_mini_minimax_self_play, m)?)?;

    m.add_function(wrap_pyfunction!(describe_record, m)?)?;
    m.add_function(wrap_pyfunction!(decode_scalar_state, m)?)?;

    // --- 数据空间对称增强（Data Augmentation，动作置换表/board 重排下沉 Rust） ---
    register_augment_functions(m)?;

    // --- Rust 原生多线程评估接口 ---
    m.add_function(wrap_pyfunction!(crate::bridge::python::eval::run_eval_match, m)?)?;

    // --- 井字棋绑定（验证逻辑复用） ---
    m.add_class::<crate::bridge::python::ttt::PyTicTacToe>()?;
    m.add_function(wrap_pyfunction!(crate::bridge::python::ttt::ttt_mcts_search, m)?)?;
    m.add_function(wrap_pyfunction!(
        crate::bridge::python::ttt::run_ttt_self_play_with_predictor,
        m
    )?)?;

    // --- 暗棋环境绑定（统一入口：视角反转验证 / 训练验证 / 变体训练） ---
    m.add_class::<crate::bridge::python::chess_env::PyDarkChess>()?;
    m.add_class::<crate::bridge::python::chess_env::PyMiniDarkChess>()?;
    m.add_class::<crate::bridge::python::chess_env::PyGame4x4>()?;

    // --- Rust 持有模型的 Torch 数据收集器（需同时启用 torch + pyo3） ---
    // 模型加载进 Rust（LocalEvaluator），推理不经过 GIL，多线程/批量自对弈真正并行，
    // 且模型只加载一份，避免 spawn 多进程重复加载 libtorch 带来的内存开销。
    #[cfg(all(feature = "torch", feature = "pyo3"))]
    m.add_class::<crate::bridge::python::rust_collector::RustTorchCollector>()?;

    // --- Rust 持有模型的 ONNX 数据收集器（需同时启用 onnx + pyo3） ---
    // 与 RustTorchCollector 等价，但推理后端为 ONNX Runtime（不依赖 libtorch）。
    #[cfg(all(feature = "onnx", feature = "pyo3"))]
    m.add_class::<crate::bridge::python::onnx_collector::RustOnnxCollector>()?;

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

    m.add("TTT_ACTION_SPACE_SIZE", TTT_ACTION_SPACE_SIZE)?;
    m.add("TTT_BOARD_ROWS", TTT_BOARD_ROWS)?;
    m.add("TTT_BOARD_COLS", TTT_BOARD_COLS)?;
    m.add("TTT_BOARD_CHANNELS", TTT_BOARD_CHANNELS)?;
    m.add("TTT_SCALAR_FEATURE_COUNT", TTT_SCALAR_FEATURE_COUNT)?;

    Ok(())
}
