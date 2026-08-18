// src/py/self_play/parallel.rs
// Python 绑定层：并行（rayon）自对弈。

#[cfg(feature = "pyo3")]
use pyo3::prelude::*;

#[cfg(feature = "pyo3")]
use crate::game_env::{DarkChessEnv, Game4x4Env, GameEnv, MiniDarkChessEnv};
#[cfg(feature = "pyo3")]
use crate::self_play::SelfPlayConfig;
#[cfg(feature = "pyo3")]
use crate::py::episode::PyGameEpisode;
#[cfg(feature = "pyo3")]
use crate::py::py_evaluator::PyEvaluator;

/// 并行版：使用 rayon 线程池运行自对弈。
#[cfg(feature = "pyo3")]
pub fn run_parallel_self_play_with_predictor_impl(
    predict_fn: Py<PyAny>,
    cfg: SelfPlayConfig,
    num_workers: usize,
    games_per_worker: usize,
    worker_id: usize,
) -> Vec<PyGameEpisode> {
    run_parallel_core(
        predict_fn, cfg, num_workers, games_per_worker, worker_id, 0, DarkChessEnv::new,
    )
}

/// 4x2 迷你暗棋版并行自对弈。
#[cfg(feature = "pyo3")]
pub fn run_mini_parallel_self_play_with_predictor_impl(
    predict_fn: Py<PyAny>,
    cfg: SelfPlayConfig,
    num_workers: usize,
    games_per_worker: usize,
    worker_id: usize,
) -> Vec<PyGameEpisode> {
    run_parallel_core(
        predict_fn, cfg, num_workers, games_per_worker, worker_id, 1, MiniDarkChessEnv::new,
    )
}

/// 4x4 暗棋版并行自对弈。
#[cfg(feature = "pyo3")]
pub fn run_game4x4_parallel_self_play_with_predictor_impl(
    predict_fn: Py<PyAny>,
    cfg: SelfPlayConfig,
    num_workers: usize,
    games_per_worker: usize,
    worker_id: usize,
) -> Vec<PyGameEpisode> {
    run_parallel_core(
        predict_fn, cfg, num_workers, games_per_worker, worker_id, 2, Game4x4Env::new,
    )
}

#[cfg(feature = "pyo3")]
fn run_parallel_core<G: GameEnv>(
    predict_fn: Py<PyAny>,
    cfg: SelfPlayConfig,
    num_workers: usize,
    games_per_worker: usize,
    worker_id: usize,
    variant: u8,
    _make_env: fn() -> G,
) -> Vec<PyGameEpisode> {
    use rayon::prelude::*;

    let _ = worker_id;
    let total_games = num_workers.max(1) * games_per_worker;

    // 在持有 GIL 的情况下，为每个 worker 克隆一份 predict_fn 引用
    let predict_fn_per_worker: Vec<Py<PyAny>> = Python::attach(|py| {
        (0..num_workers.max(1))
            .map(|_| predict_fn.clone_ref(py))
            .collect()
    });

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(num_workers.max(1))
        .build()
        .expect("failed to build rayon thread pool for parallel self_play");

    // 关键：用 allow_threads 释放 GIL 后再进入 rayon 并行区。
    let episodes_by_worker: Vec<Vec<PyGameEpisode>> = Python::attach(|py| {
        py.detach(|| {
            pool.install(|| {
                predict_fn_per_worker
                    .into_par_iter()
                    .enumerate()
                    .map(|(wid, pf)| {
                        let evaluator = PyEvaluator::new(pf);
                        let mut local = Vec::with_capacity(games_per_worker);
                        for g in 0..games_per_worker {
                            let start = std::time::Instant::now();
                            let episode = crate::self_play::run_self_play(&evaluator, &cfg, _make_env);
                            if episode.samples.is_empty() {
                                eprintln!(
                                    "[ParallelWorker-{}/game{}] ⚠️ 空游戏数据，跳过",
                                    wid, g
                                );
                                continue;
                            }
                            let dur = start.elapsed().as_secs_f64();
                            let winner_str = match episode.winner {
                                Some(1) => "红胜",
                                Some(-1) => "黑胜",
                                _ => "平局",
                            };
                            println!(
                                "[PW-{}] #{}/{} steps={} {} {:.2}s ({:.0} steps/s)",
                                wid,
                                g + 1,
                                games_per_worker,
                                episode.game_length,
                                winner_str,
                                dur,
                                episode.game_length as f64 / dur.max(1e-9)
                            );
                            local.push(PyGameEpisode {
                                inner: episode,
                                variant,
                            });
                        }
                        local
                    })
                    .collect()
            })
        })
    });

    episodes_by_worker
        .into_iter()
        .flatten()
        .take(total_games)
        .collect()
}
