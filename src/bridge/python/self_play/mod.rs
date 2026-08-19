//! 自对弈 Python 绑定配置。
//!
//! 仅保留 `PySelfPlayConfig` 配置类。旧的自对弈 / 教师 / 批量实现（`batched.rs`、
//! `teacher.rs`）已移除，统一由 `pipeline::self_play::match_core` 的 `run_match_core`
//! 主干承载，经 `bridge/python/eval.rs` 的 `run_native_match` / `run_python_match`
//! 两个唯一入口对外暴露。

#[cfg(feature = "pyo3")]
pub mod config;

#[cfg(feature = "pyo3")]
pub use config::PySelfPlayConfig;
