//! PyO3 Python 绑定模块。
//!
//! 按职责分层：
//! - `episode`:   `PyGameEpisode` 包装类 + episode 序列化为 PyDict
//! - `self_play`: `PySelfPlayConfig` + 串行/并行/批量/启发式/minimax 自对弈 impl
//! - `decode`:    `describe_record` / `decode_scalar_state` / `config_for_variant` 辅助函数
//! - `py_evaluator`: `PyEvaluator`（委托 Python predict_fn 的评估器）
//! - `chess_env`:   统一暗棋环境绑定（DarkChess / Game4x4 / MiniDarkChess）
//! - `ttt`:         井字棋环境绑定（验证复用）
//! - `rust_collector`: Rust 持有模型的 Torch 数据收集器（torch + pyo3）

#[cfg(feature = "pyo3")]
mod decode;
#[cfg(feature = "pyo3")]
mod episode;
#[cfg(feature = "pyo3")]
mod self_play;

#[cfg(feature = "pyo3")]
pub mod py_evaluator;

#[cfg(feature = "pyo3")]
pub mod chess_env;
#[cfg(feature = "pyo3")]
pub mod ttt;

#[cfg(all(feature = "torch", feature = "pyo3"))]
pub mod rust_collector;

// ---- 公共 re-export：保持 `crate::py::*` 与 `lib.rs` / py_data_collector.rs 的引用路径不变 ----

#[cfg(feature = "pyo3")]
pub use decode::{config_for_variant, decode_scalar_state, describe_record};
#[cfg(feature = "pyo3")]
pub use episode::{
    PyGameEpisode, episode_to_dict, episode_to_dict_darkchess,
};
#[cfg(feature = "pyo3")]
pub use py_evaluator::PyEvaluator;
#[cfg(feature = "pyo3")]
pub use self_play::PySelfPlayConfig;

#[cfg(feature = "pyo3")]
pub use self_play::{
    run_batched_self_play_with_predictor_impl, run_game4x4_batched_self_play_with_predictor_impl,
    run_game4x4_heuristic_self_play_impl, run_game4x4_minimax_self_play_impl,
    run_game4x4_parallel_self_play_with_predictor_impl,
    run_game4x4_self_play_with_predictor_impl, run_mini_batched_self_play_with_predictor_impl,
    run_mini_parallel_self_play_with_predictor_impl, run_mini_self_play_with_predictor_impl,
    run_parallel_self_play_with_predictor_impl, run_self_play_with_predictor_impl,
};
