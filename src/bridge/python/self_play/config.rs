// src/py/self_play/config.rs
// Python 绑定层：PySelfPlayConfig 类（自对弈配置）。

#[cfg(feature = "pyo3")]
use pyo3::prelude::*;

#[cfg(feature = "pyo3")]
use crate::pipeline::self_play::{ScenarioType, SelfPlayConfig};

#[cfg(feature = "pyo3")]
#[pyclass(name = "SelfPlayConfig", skip_from_py_object)]
#[derive(Clone)]
pub struct PySelfPlayConfig {
    pub inner: SelfPlayConfig,
}

#[cfg(feature = "pyo3")]
#[pymethods]
impl PySelfPlayConfig {
    #[new]
    #[pyo3(signature = (
        mcts_sims = 64,
        max_considered_actions = 16,
        c_scale = 1.0,
        gumbel_scale = 1.0,
        playout_cap_random_enabled = true,
        fast_mcts_sims = 16,
        full_search_prob = 0.25,
    ))]
    fn new(
        mcts_sims: usize,
        max_considered_actions: usize,
        c_scale: f32,
        gumbel_scale: f32,
        playout_cap_random_enabled: bool,
        fast_mcts_sims: usize,
        full_search_prob: f32,
    ) -> Self {
        Self {
            inner: SelfPlayConfig {
                mcts_sims,
                max_considered_actions,
                scenario: ScenarioType::Standard,
                c_scale,
                gumbel_scale,
                playout_cap_random_enabled,
                fast_mcts_sims,
                full_search_prob,
            },
        }
    }

    #[getter]
    fn mcts_sims(slf: PyRef<'_, Self>) -> usize {
        slf.inner.mcts_sims
    }

    #[getter]
    fn max_considered_actions(slf: PyRef<'_, Self>) -> usize {
        slf.inner.max_considered_actions
    }
}
