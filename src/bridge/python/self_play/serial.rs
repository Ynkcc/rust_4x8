// src/py/self_play/serial.rs
// Python 绑定层：串行自对弈（暗棋 / 4x2 迷你 / 4x4）。

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

/// 串行版：连续生成直到累计 `num_games` 个**非空** episode。
#[cfg(feature = "pyo3")]
pub fn run_self_play_with_predictor_impl(
    predict_fn: Py<PyAny>,
    cfg: SelfPlayConfig,
    num_games: usize,
    worker_id: usize,
) -> Vec<PyGameEpisode> {
    let evaluator = PyEvaluator::new(predict_fn);
    run_self_play_serial_core(&evaluator, &cfg, num_games, worker_id, 0, DarkChessEnv::new)
}

/// 4x2 迷你暗棋版串行自对弈。
#[cfg(feature = "pyo3")]
pub fn run_mini_self_play_with_predictor_impl(
    predict_fn: Py<PyAny>,
    cfg: SelfPlayConfig,
    num_games: usize,
    worker_id: usize,
) -> Vec<PyGameEpisode> {
    let evaluator = PyEvaluator::new(predict_fn);
    run_self_play_serial_core(
        &evaluator,
        &cfg,
        num_games,
        worker_id,
        1,
        MiniDarkChessEnv::new,
    )
}

/// 4x4 暗棋版串行自对弈。
#[cfg(feature = "pyo3")]
pub fn run_game4x4_self_play_with_predictor_impl(
    predict_fn: Py<PyAny>,
    cfg: SelfPlayConfig,
    num_games: usize,
    worker_id: usize,
) -> Vec<PyGameEpisode> {
    let evaluator = PyEvaluator::new(predict_fn);
    run_self_play_serial_core(
        &evaluator,
        &cfg,
        num_games,
        worker_id,
        2,
        Game4x4Env::new,
    )
}

#[cfg(feature = "pyo3")]
fn run_self_play_serial_core<G: GameEnv>(
    evaluator: &PyEvaluator<G>,
    cfg: &SelfPlayConfig,
    num_games: usize,
    worker_id: usize,
    variant: u8,
    make_env: fn() -> G,
) -> Vec<PyGameEpisode> {
    let mut episodes = Vec::with_capacity(num_games);
    let mut game_count = 0;

    loop {
        let start_time = std::time::Instant::now();
        let episode = crate::pipeline::self_play::run_self_play(evaluator, cfg, make_env);
        let duration = start_time.elapsed();

        if episode.samples.is_empty() {
            eprintln!("[Worker-{}] ⚠️ 生成了空游戏数据，跳过", worker_id);
            continue;
        }

        let winner_str = match episode.winner {
            Some(1) => "红胜",
            Some(-1) => "黑胜",
            _ => "平局",
        };
        println!(
            "[Worker-{}] Game #{}: 步数={}, 结果={}, 耗时={:.1}s ({:.1} steps/s)",
            worker_id,
            game_count + 1,
            episode.game_length,
            winner_str,
            duration.as_secs_f64(),
            episode.game_length as f64 / duration.as_secs_f64()
        );

        episodes.push(PyGameEpisode {
            inner: episode,
            variant,
        });

        game_count += 1;
        if game_count >= num_games {
            break;
        }
    }

    episodes
}
