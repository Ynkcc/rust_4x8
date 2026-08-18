// src/py/self_play/batched.rs
// Python 绑定层：批量（流水线）自对弈。

#[cfg(feature = "pyo3")]
use pyo3::prelude::*;

#[cfg(feature = "pyo3")]
use crate::core::env::{DarkChessEnv, Game4x4Env, GameEnv, MiniDarkChessEnv};
#[cfg(feature = "pyo3")]
use crate::pipeline::self_play::SelfPlayConfig;
#[cfg(feature = "pyo3")]
use crate::bridge::python::episode::PyGameEpisode;
#[cfg(feature = "pyo3")]
use crate::bridge::python::py_evaluator::PyEvaluator;

/// 批量版：同时驱动 `concurrency` 局自对弈，并把多棵树的 MCTS 叶子评估合并成一个大 batch。
#[cfg(feature = "pyo3")]
pub fn run_batched_self_play_with_predictor_impl<'py>(
    py: Python<'py>,
    predict_fn: Py<PyAny>,
    cfg: SelfPlayConfig,
    num_games: usize,
    concurrency: usize,
    worker_id: usize,
) -> Vec<PyGameEpisode> {
    let evaluator = PyEvaluator::new(predict_fn);
    run_batched_core(
        py, &evaluator, &cfg, num_games, concurrency, worker_id, 0, DarkChessEnv::new,
    )
}

/// 4x2 迷你暗棋版批量自对弈。
#[cfg(feature = "pyo3")]
pub fn run_mini_batched_self_play_with_predictor_impl<'py>(
    py: Python<'py>,
    predict_fn: Py<PyAny>,
    cfg: SelfPlayConfig,
    num_games: usize,
    concurrency: usize,
    worker_id: usize,
) -> Vec<PyGameEpisode> {
    let evaluator = PyEvaluator::new(predict_fn);
    run_batched_core(
        py,
        &evaluator,
        &cfg,
        num_games,
        concurrency,
        worker_id,
        1,
        MiniDarkChessEnv::new,
    )
}

/// 4x4 暗棋版批量自对弈。
#[cfg(feature = "pyo3")]
pub fn run_game4x4_batched_self_play_with_predictor_impl<'py>(
    py: Python<'py>,
    predict_fn: Py<PyAny>,
    cfg: SelfPlayConfig,
    num_games: usize,
    concurrency: usize,
    worker_id: usize,
) -> Vec<PyGameEpisode> {
    let evaluator = PyEvaluator::new(predict_fn);
    run_batched_core(
        py,
        &evaluator,
        &cfg,
        num_games,
        concurrency,
        worker_id,
        2,
        Game4x4Env::new,
    )
}

#[cfg(feature = "pyo3")]
fn run_batched_core<G: GameEnv + Sync>(
    py: Python<'_>,
    evaluator: &PyEvaluator<G>,
    cfg: &SelfPlayConfig,
    num_games: usize,
    concurrency: usize,
    worker_id: usize,
    variant: u8,
    make_env: fn() -> G,
) -> Vec<PyGameEpisode> {
    let mut episodes: Vec<PyGameEpisode> = Vec::with_capacity(num_games);
    let mut game_count = 0;

    while game_count < num_games {
        let batch: Vec<_> =
            py.detach(|| crate::pipeline::self_play::run_batched_self_play(
                evaluator, cfg, num_games - game_count, concurrency, make_env,
            ));
        for ep in batch {
            if ep.samples.is_empty() {
                eprintln!("[Worker-{}] ⚠️ 生成了空游戏数据，跳过", worker_id);
                continue;
            }
            episodes.push(PyGameEpisode { inner: ep, variant });
            game_count += 1;
            if game_count >= num_games {
                break;
            }
        }
    }
    episodes
}
