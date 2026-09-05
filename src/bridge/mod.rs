//! 跨语言交互桥梁模块 (Language Interop Bridge)
//!
//! 包含 PyO3 Python C-Extension 绑定与 Python API 接口。
//!
//! `lib.rs` 仅声明顶层模块；本文件承载 PyO3 扩展模块（`banqi_4x8`）的
//! `#[pymodule]` 入口、对 Python 可见的 `#[pyfunction]` 转发包装，以及
//! 各类 pyclass / 常量的注册。
//!
//! 自对弈 / 对战 / 评估对外仅保留两个唯一入口：
//! - `run_native_match`：Rust 侧持有 .pt/.onnx 模型或规则选手，rayon 多线程。
//! - `run_python_match`：Python 提供 `predict_fn` 推理服务，单线程。
//! 二者共同调用 `pipeline::self_play::run_match_core` 统一主干。其余旧入口
//! （`run_*_self_play*`、`run_eval_match`、`RustTorchCollector` 等）已彻底移除。

#[cfg(feature = "pyo3")]
pub mod python;

#[cfg(feature = "pyo3")]
use pyo3::prelude::*;

#[cfg(feature = "pyo3")]
use crate::bridge::python::{
    PyGameEpisode, PySelfPlayConfig, describe_record, decode_scalar_state, register_augment_functions,
    variant_dims,
};

// ---- pymodule 入口 ----

#[cfg(feature = "pyo3")]
#[pymodule]
fn banqi_4x8(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyGameEpisode>()?;
    m.add_class::<PySelfPlayConfig>()?;

    // --- 统一对局主干入口（唯一） ---
    m.add_function(wrap_pyfunction!(crate::bridge::python::eval::run_native_match, m)?)?;
    m.add_function(wrap_pyfunction!(crate::bridge::python::eval::run_python_match, m)?)?;

    // --- Expectimax + NNUE 自对弈入口（NNUE 训练回环） ---
    m.add_function(wrap_pyfunction!(
        crate::bridge::python::expectimax::run_expectimax_self_play,
        m
    )?)?;

    m.add_function(wrap_pyfunction!(describe_record, m)?)?;
    m.add_function(wrap_pyfunction!(decode_scalar_state, m)?)?;

    // --- 数据空间对称增强（Data Augmentation，动作置换表/board 重排下沉 Rust） ---
    register_augment_functions(m)?;

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

    // --- 统一变体维度查询（替代散落的模块级常量） ---
    m.add_function(wrap_pyfunction!(variant_dims, m)?)?;

    Ok(())
}
