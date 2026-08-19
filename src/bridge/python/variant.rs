//! 变体枚举：统一 4x8 / 4x2 / 4x4 / 井字棋 的字符串解析与维度查询。
//!
//! 取代原先在 `bridge/mod.rs` 里散落注册的模块级常量
//! （`BOARD_ROWS` / `MINI_*` / `GAME4X4_*` / `TTT_*`），通过单一枚举 + 方法
//! 提供各变体的棋盘 / 特征维度；`variant_dims` 作为统一查询入口暴露给 Python。

#[cfg(feature = "pyo3")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "pyo3")]
use pyo3::prelude::*;
#[cfg(feature = "pyo3")]
use pyo3::types::PyDict;

#[cfg(feature = "pyo3")]
use crate::core::env::{
    ACTION_SPACE_SIZE, BOARD_CHANNELS, BOARD_COLS, BOARD_ROWS, GAME4X4_ACTION_SPACE_SIZE,
    GAME4X4_BOARD_CHANNELS, GAME4X4_BOARD_COLS, GAME4X4_BOARD_ROWS, GAME4X4_SCALAR_FEATURE_COUNT,
    MINI_ACTION_SPACE_SIZE, MINI_BOARD_CHANNELS, MINI_BOARD_COLS, MINI_BOARD_ROWS,
    MINI_SCALAR_FEATURE_COUNT, SCALAR_FEATURE_COUNT, TTT_ACTION_SPACE_SIZE, TTT_BOARD_CHANNELS,
    TTT_BOARD_COLS, TTT_BOARD_ROWS, TTT_SCALAR_FEATURE_COUNT,
};

/// 自对弈 / 对战 / 评估统一变体枚举。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SelfPlayVariant {
    /// 4x8 暗棋（无前缀变体）。
    Dark4x8,
    /// 4x2 迷你暗棋。
    Mini4x2,
    /// 4x4 暗棋。
    Game4x4,
    /// 井字棋（验证用，仅维度查询，不进入对局入口）。
    Ttt,
}

impl SelfPlayVariant {
    /// 解析变体字符串：支持 "4x8"/"dark"、"4x2"/"mini"、"4x4"、"ttt"。
    pub fn parse(s: &str) -> PyResult<Self> {
        match s.to_ascii_lowercase().as_str() {
            "4x8" | "dark" => Ok(Self::Dark4x8),
            "4x2" | "mini" => Ok(Self::Mini4x2),
            "4x4" | "game4x4" => Ok(Self::Game4x4),
            "ttt" | "tic_tac_toe" => Ok(Self::Ttt),
            other => Err(PyValueError::new_err(format!(
                "未知变体: {other:?}（应为 4x8 | 4x2 | 4x4 | ttt）"
            ))),
        }
    }

    /// 对局入口使用的 episode 变体代码：0=4x8、1=4x2、2=4x4（见 `PyGameEpisode.variant`）。
    /// ttt 不进入对局入口，借用 4x8 形状。
    pub fn episode_code(self) -> u8 {
        match self {
            Self::Dark4x8 => 0,
            Self::Mini4x2 => 1,
            Self::Game4x4 => 2,
            Self::Ttt => 0,
        }
    }

    /// 棋盘行数。
    pub fn board_rows(self) -> usize {
        match self {
            Self::Dark4x8 => BOARD_ROWS,
            Self::Mini4x2 => MINI_BOARD_ROWS,
            Self::Game4x4 => GAME4X4_BOARD_ROWS,
            Self::Ttt => TTT_BOARD_ROWS,
        }
    }

    /// 棋盘列数。
    pub fn board_cols(self) -> usize {
        match self {
            Self::Dark4x8 => BOARD_COLS,
            Self::Mini4x2 => MINI_BOARD_COLS,
            Self::Game4x4 => GAME4X4_BOARD_COLS,
            Self::Ttt => TTT_BOARD_COLS,
        }
    }

    /// 特征通道数。
    pub fn board_channels(self) -> usize {
        match self {
            Self::Dark4x8 => BOARD_CHANNELS,
            Self::Mini4x2 => MINI_BOARD_CHANNELS,
            Self::Game4x4 => GAME4X4_BOARD_CHANNELS,
            Self::Ttt => TTT_BOARD_CHANNELS,
        }
    }

    /// 标量特征数。
    pub fn scalar_feature_count(self) -> usize {
        match self {
            Self::Dark4x8 => SCALAR_FEATURE_COUNT,
            Self::Mini4x2 => MINI_SCALAR_FEATURE_COUNT,
            Self::Game4x4 => GAME4X4_SCALAR_FEATURE_COUNT,
            Self::Ttt => TTT_SCALAR_FEATURE_COUNT,
        }
    }

    /// 动作空间大小。
    pub fn action_space_size(self) -> usize {
        match self {
            Self::Dark4x8 => ACTION_SPACE_SIZE,
            Self::Mini4x2 => MINI_ACTION_SPACE_SIZE,
            Self::Game4x4 => GAME4X4_ACTION_SPACE_SIZE,
            Self::Ttt => TTT_ACTION_SPACE_SIZE,
        }
    }
}

/// 统一维度查询入口（替代散落的模块级常量）。
#[pyfunction]
#[pyo3(signature = (variant = "4x8"))]
pub fn variant_dims<'py>(py: Python<'py>, variant: &str) -> PyResult<Bound<'py, PyDict>> {
    let v = SelfPlayVariant::parse(variant)?;
    let dict = PyDict::new(py);
    dict.set_item("variant", variant)?;
    dict.set_item("board_rows", v.board_rows())?;
    dict.set_item("board_cols", v.board_cols())?;
    dict.set_item("board_channels", v.board_channels())?;
    dict.set_item("scalar_feature_count", v.scalar_feature_count())?;
    dict.set_item("action_space_size", v.action_space_size())?;
    Ok(dict)
}
