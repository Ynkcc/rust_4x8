//! src/bridge/python/expectimax.rs — Expectimax + NNUE 自对弈 PyO3 入口。
//!
//! 一键拉起 Rust 侧 Expectimax 自对弈（局间多 worker 并发 + 局内可选 Lazy SMP），
//! 流式写出 NNUE 训练 JSONL，供 `python/banqi/nnue/train.py` 直接消费，
//! 形成「自对弈 → 训练 → 导出 .nnue → 再自对弈」的独立训练回环。

use std::path::PathBuf;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::pipeline::self_play::{
    ExpectimaxSelfPlayConfig, run_expectimax_self_play as run_expectimax_self_play_inner,
};

/// Expectimax + NNUE 自对弈（Rust 原生，局间多线程并发）。
///
/// 返回统计 dict：{games, a_wins, b_wins, draws, steps}。
/// `out_jsonl` 每局完成即追加一行 NNUE episode JSON（契约与
/// `NnueSampleDataset` 一致，`value_source="completed_q"` 直接可用）。
#[pyfunction]
#[pyo3(signature = (nnue_path, n_games=16, num_workers=4, node_budget=500_000, max_depth=8, threads_per_search=1, seed=None, out_jsonl=None))]
pub fn run_expectimax_self_play<'py>(
    py: Python<'py>,
    nnue_path: &str,
    n_games: usize,
    num_workers: usize,
    node_budget: u64,
    max_depth: i32,
    threads_per_search: usize,
    seed: Option<u64>,
    out_jsonl: Option<PathBuf>,
) -> PyResult<pyo3::Bound<'py, pyo3::types::PyDict>> {
    if n_games == 0 {
        return Err(PyValueError::new_err("n_games 必须大于 0"));
    }
    let config = ExpectimaxSelfPlayConfig {
        node_budget,
        max_depth,
        threads_per_search: threads_per_search.max(1),
    };
    let out_ref = out_jsonl.as_deref();

    let stats = py
        .detach(|| {
            run_expectimax_self_play_inner(
                nnue_path,
                &config,
                n_games,
                num_workers,
                seed,
                out_ref,
            )
        })
        .map_err(PyValueError::new_err)?;

    let dict = pyo3::types::PyDict::new(py);
    dict.set_item("games", stats.games)?;
    dict.set_item("a_wins", stats.a_wins)?;
    dict.set_item("b_wins", stats.b_wins)?;
    dict.set_item("draws", stats.draws)?;
    dict.set_item("steps", stats.steps)?;
    Ok(dict)
}
