//! PyO3 Python 绑定模块。
//!
//! 按职责分层：
//! - `episode`:   `PyGameEpisode` 包装类 + episode 序列化为 PyDict
//! - `self_play`: `PySelfPlayConfig` 配置类（旧自对弈 impl 已移除，统一走 match_core）
//! - `decode`:    `describe_record` / `decode_scalar_state` / `config_for_variant` 辅助函数
//! - `py_evaluator`: `PyEvaluator`（委托 Python predict_fn 的评估器）
//! - `eval`:      `run_native_match` / `run_python_match` 唯一对局入口
//! - `chess_env`:   统一暗棋环境绑定（DarkChess / Game4x4 / MiniDarkChess）
//! - `ttt`:         井字棋环境绑定（验证复用）

#[cfg(feature = "pyo3")]
mod augment;
#[cfg(feature = "pyo3")]
mod decode;
#[cfg(feature = "pyo3")]
pub mod variant;
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

#[cfg(feature = "pyo3")]
pub mod eval;

// ---- 公共 re-export：保持 `crate::bridge::python::*` 与既有调用方的引用路径不变 ----

#[cfg(feature = "pyo3")]
pub use decode::{config_for_variant, decode_scalar_state, describe_record};
#[cfg(feature = "pyo3")]
pub use variant::{SelfPlayVariant, variant_dims};
#[cfg(feature = "pyo3")]
pub use augment::register_augment_functions;
#[cfg(feature = "pyo3")]
pub use episode::{
    PyGameEpisode, episode_to_dict, episode_to_dict_darkchess,
};
#[cfg(feature = "pyo3")]
pub use py_evaluator::PyEvaluator;
#[cfg(feature = "pyo3")]
pub use self_play::PySelfPlayConfig;
