//! Episode 包装类与序列化。
//!
//! 将 Rust 侧的 `GameEpisode` 包装为 `PyGameEpisode`（暴露给 Python），
//! 并提供 `episode_to_dict*` 系列函数将样本序列化为 PyDict。
//! `episode_to_dict` / `episode_to_dict_darkchess` 被 `py_data_collector.rs` 复用。

#[cfg(feature = "pyo3")]
use pyo3::prelude::*;
#[cfg(feature = "pyo3")]
use pyo3::types::PyDict;

#[cfg(feature = "pyo3")]
use crate::core::env::{
    ACTION_SPACE_SIZE, BOARD_CHANNELS, BOARD_COLS, BOARD_ROWS, SCALAR_FEATURE_COUNT,
};
#[cfg(feature = "pyo3")]
use crate::pipeline::self_play::GameEpisode;

#[cfg(feature = "pyo3")]
#[pyclass(name = "GameEpisode", skip_from_py_object)]
#[derive(Clone)]
pub struct PyGameEpisode {
    pub inner: GameEpisode,
    /// 变体标识：0=4x8 暗棋，1=4x2 迷你，2=4x4。
    /// 决定 episode dict 中的 shape 字段。
    pub variant: u8,
}

#[cfg(feature = "pyo3")]
#[pymethods]
impl PyGameEpisode {
    #[getter]
    fn game_length(slf: PyRef<'_, Self>) -> usize {
        slf.inner.game_length
    }

    #[getter]
    fn winner(slf: PyRef<'_, Self>) -> Option<i32> {
        slf.inner.winner
    }

    #[getter]
    fn num_samples(slf: PyRef<'_, Self>) -> usize {
        slf.inner.samples.len()
    }

    #[allow(clippy::type_complexity)]
    fn get_samples(slf: PyRef<'_, Self>) -> (Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<f32>, Vec<f32>, Vec<u32>, Vec<f32>, Vec<Vec<i32>>, Vec<usize>, Vec<f32>, Vec<bool>) {
        let n = slf.inner.samples.len();
        let mut boards: Vec<Vec<f32>> = Vec::with_capacity(n);
        let mut scalars: Vec<Vec<f32>> = Vec::with_capacity(n);
        let mut policies: Vec<Vec<f32>> = Vec::with_capacity(n);
        let mut mcts_values: Vec<f32> = Vec::with_capacity(n);
        let mut completed_qs: Vec<f32> = Vec::with_capacity(n);
        let mut root_visits: Vec<u32> = Vec::with_capacity(n);
        let mut game_results: Vec<f32> = Vec::with_capacity(n);
        let mut action_masks: Vec<Vec<i32>> = Vec::with_capacity(n);
        let mut actions: Vec<usize> = Vec::with_capacity(n);
        let mut health_diffs: Vec<f32> = Vec::with_capacity(n);
        let mut is_full_searches: Vec<bool> = Vec::with_capacity(n);

        for (obs, policy, mcts_val, completed_q, root_visit, game_result, mask, action, health_diff, is_full_search) in &slf.inner.samples {
            boards.push(obs.board.as_slice().unwrap().to_vec());
            scalars.push(obs.scalars.as_slice().unwrap().to_vec());
            policies.push(policy.clone());
            mcts_values.push(*mcts_val);
            completed_qs.push(*completed_q);
            root_visits.push(*root_visit);
            game_results.push(*game_result);
            action_masks.push(mask.clone());
            actions.push(*action);
            health_diffs.push(*health_diff);
            is_full_searches.push(*is_full_search);
        }

        (
            boards,
            scalars,
            policies,
            mcts_values,
            completed_qs,
            root_visits,
            game_results,
            action_masks,
            actions,
            health_diffs,
            is_full_searches,
        )
    }

    fn to_dict<'py>(slf: PyRef<'py, Self>, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        episode_to_dict_with_shapes(py, &slf.inner, slf.variant)
    }
}

/// 将 GameEpisode 序列化为 PyDict（供 `PyGameEpisode::to_dict` 和
/// `py_data_collector.rs` 共用，消除重复逻辑）。
/// `variant`：0=4x8 暗棋，1=4x2 迷你，2=4x4。
#[cfg(feature = "pyo3")]
pub fn episode_to_dict<'py>(
    py: Python<'py>,
    episode: &GameEpisode,
    mini: bool,
) -> PyResult<Bound<'py, PyDict>> {
    episode_to_dict_with_shapes(py, episode, if mini { 1 } else { 0 })
}

/// 4x8 暗棋变体的 episode dict（供 py_data_collector.rs 兼容调用）。
#[cfg(feature = "pyo3")]
pub fn episode_to_dict_darkchess<'py>(
    py: Python<'py>,
    episode: &GameEpisode,
) -> PyResult<Bound<'py, PyDict>> {
    episode_to_dict_with_shapes(py, episode, 0)
}

#[cfg(feature = "pyo3")]
fn episode_to_dict_with_shapes<'py>(
    py: Python<'py>,
    episode: &GameEpisode,
    variant: u8,
) -> PyResult<Bound<'py, PyDict>> {
    let (bc, br, bcol, sc, ac): (usize, usize, usize, usize, usize) = match variant {
        1 => (
            crate::core::env::MINI_BOARD_CHANNELS,
            crate::core::env::MINI_BOARD_ROWS,
            crate::core::env::MINI_BOARD_COLS,
            crate::core::env::MINI_SCALAR_FEATURE_COUNT,
            crate::core::env::MINI_ACTION_SPACE_SIZE,
        ),
        2 => (
            crate::core::env::GAME4X4_BOARD_CHANNELS,
            crate::core::env::GAME4X4_BOARD_ROWS,
            crate::core::env::GAME4X4_BOARD_COLS,
            crate::core::env::GAME4X4_SCALAR_FEATURE_COUNT,
            crate::core::env::GAME4X4_ACTION_SPACE_SIZE,
        ),
        _ => (BOARD_CHANNELS, BOARD_ROWS, BOARD_COLS, SCALAR_FEATURE_COUNT, ACTION_SPACE_SIZE),
    };
    let n = episode.samples.len();
    let mut boards: Vec<Vec<f32>> = Vec::with_capacity(n);
    let mut scalars: Vec<Vec<f32>> = Vec::with_capacity(n);
    let mut policies: Vec<Vec<f32>> = Vec::with_capacity(n);
    let mut mcts_values: Vec<f32> = Vec::with_capacity(n);
    let mut completed_qs: Vec<f32> = Vec::with_capacity(n);
    let mut root_visits: Vec<u32> = Vec::with_capacity(n);
    let mut game_results: Vec<f32> = Vec::with_capacity(n);
    let mut action_masks: Vec<Vec<i32>> = Vec::with_capacity(n);
    let mut actions: Vec<usize> = Vec::with_capacity(n);
    let mut health_diffs: Vec<f32> = Vec::with_capacity(n);
    let mut is_full_searches: Vec<bool> = Vec::with_capacity(n);

    for (obs, policy, mcts_val, completed_q, root_visit, game_result, mask, action, health_diff, is_full_search) in &episode.samples {
        boards.push(obs.board.as_slice().unwrap().to_vec());
        scalars.push(obs.scalars.as_slice().unwrap().to_vec());
        policies.push(policy.clone());
        mcts_values.push(*mcts_val);
        completed_qs.push(*completed_q);
        root_visits.push(*root_visit);
        game_results.push(*game_result);
        action_masks.push(mask.clone());
        actions.push(*action);
        health_diffs.push(*health_diff);
        is_full_searches.push(*is_full_search);
    }

    let dict = PyDict::new(py);
    dict.set_item("game_length", episode.game_length)?;
    dict.set_item("winner", episode.winner)?;
    dict.set_item("num_samples", n)?;
    dict.set_item("boards", boards)?;
    dict.set_item("scalars", scalars)?;
    dict.set_item("policies", policies)?;
    dict.set_item("mcts_values", mcts_values)?;
    dict.set_item("completed_qs", completed_qs)?;
    dict.set_item("root_visits", root_visits)?;
    dict.set_item("game_results", game_results)?;
    dict.set_item("health_diffs", health_diffs)?;
    dict.set_item("action_masks", action_masks)?;
    dict.set_item("actions", actions)?;
    dict.set_item("is_full_search", is_full_searches)?;
    dict.set_item("health_diff_red", episode.health_diff_red)?;
    dict.set_item("board_shape", vec![bc, br, bcol])?;
    dict.set_item("scalar_shape", vec![sc])?;
    dict.set_item("action_space", ac)?;

    Ok(dict)
}
