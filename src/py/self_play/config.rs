// src/py/self_play/config.rs
// Python 绑定层：PySelfPlayConfig 类（自对弈配置）。

#[cfg(feature = "pyo3")]
use pyo3::prelude::*;

#[cfg(feature = "pyo3")]
use crate::self_play::{ScenarioType, SelfPlayConfig};

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
        temperature_steps = 12,
        c_scale = 1.0,
        gumbel_scale = 1.0,
    ))]
    fn new(
        mcts_sims: usize,
        max_considered_actions: usize,
        temperature_steps: usize,
        c_scale: f32,
        gumbel_scale: f32,
    ) -> Self {
        Self {
            inner: SelfPlayConfig {
                mcts_sims,
                max_considered_actions,
                // 注意：Dirichlet 噪声注入已移除（Gumbel AlphaZero 探索由
                // Gumbel 噪声 + Sequential Halving 提供），不再暴露对应参数。
                temperature_steps,
                scenario: ScenarioType::Standard,
                c_scale,
                gumbel_scale,
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
