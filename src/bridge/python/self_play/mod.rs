//! 自对弈 Python 绑定实现（串行 / 并行 / 批量 / 启发式 / minimax 教师）。
//!
//! 暴露给 `lib.rs` 的 `#[pyfunction]` 转发入口，统一以 `*_impl` 函数形式提供，
//! 由 `lib.rs` 的 `run_*_self_play_with_predictor` 包装后注册到 pymodule。
//!
//! 子模块划分：
//! - `config`：PySelfPlayConfig 类
//! - `batched`：批量（流水线）自对弈（唯一自对弈路径，concurrency=1 等价旧串行）
//! - `teacher`：4x4 启发式 / minimax 教师自对弈

#[cfg(feature = "pyo3")]
pub mod config;
#[cfg(feature = "pyo3")]
pub mod batched;
#[cfg(feature = "pyo3")]
pub mod teacher;

#[cfg(feature = "pyo3")]
pub use config::PySelfPlayConfig;
#[cfg(feature = "pyo3")]
pub use batched::{
    run_batched_self_play_with_predictor_impl, run_game4x4_batched_self_play_with_predictor_impl,
    run_mini_batched_self_play_with_predictor_impl,
};
#[cfg(feature = "pyo3")]
pub use teacher::{
    run_game4x4_heuristic_self_play_impl, run_game4x4_minimax_self_play_impl,
    run_heuristic_self_play_impl, run_mini_heuristic_self_play_impl, run_mini_minimax_self_play_impl,
    run_minimax_self_play_impl,
};
