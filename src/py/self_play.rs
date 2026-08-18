//! 自对弈 Python 绑定实现（串行 / 并行 / 批量 / 启发式 / minimax 教师）。
//!
//! 暴露给 `lib.rs` 的 `#[pyfunction]` 转发入口，统一以 `*_impl` 函数形式提供，
//! 由 `lib.rs` 的 `run_*_self_play_with_predictor` 包装后注册到 pymodule。

#[cfg(feature = "pyo3")]
use pyo3::prelude::*;

#[cfg(feature = "pyo3")]
use rayon::prelude::*;

#[cfg(feature = "pyo3")]
use crate::game_env::{
    DarkChessEnv, Game4x4Env, GameEnv, MiniDarkChessEnv,
};
#[cfg(feature = "pyo3")]
use crate::self_play::{self, GameEpisode, ScenarioType, SelfPlayConfig, finalize_episode};
#[cfg(feature = "pyo3")]
use crate::mcts::{Evaluator, GumbelConfig};

#[cfg(feature = "pyo3")]
use super::py_evaluator::PyEvaluator;
#[cfg(feature = "pyo3")]
use super::episode::PyGameEpisode;

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
        let episode = self_play::run_self_play(evaluator, cfg, make_env);
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
                            let episode = self_play::run_self_play(&evaluator, &cfg, _make_env);
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

/// 4x4 启发式教师自对弈：用纯计算启发式评估器驱动 Gumbel MCTS 自对弈。
#[cfg(feature = "pyo3")]
pub fn run_game4x4_heuristic_self_play_impl<'py>(
    py: Python<'py>,
    cfg: &SelfPlayConfig,
    num_games: usize,
    concurrency: usize,
    worker_id: usize,
) -> Vec<PyGameEpisode> {
    use crate::ai::eval::EvalParams;
    use crate::ai::heuristic_mcts::prior_logit;
    use crate::ai::movegen::generate_moves;
    use crate::mcts::Evaluator;

    /// 适配 Game4x4Env 的启发式评估器（内部委托给 HeuristicEvaluator 的规则）。
    struct HeuristicEval4x4 {
        params: EvalParams,
        prior_scale: f32,
    }

    impl Evaluator<Game4x4Env> for HeuristicEval4x4 {
        fn evaluate(&self, envs: &[Game4x4Env]) -> (Vec<Vec<f32>>, Vec<f32>) {
            let params = &self.params;
            let prior_scale = self.prior_scale;
            let n = envs.len();
            let results: Vec<_> = envs
                .par_iter()
                .map(|env| {
                    let inner = &env.inner;
                    let mut lg = vec![0.0f32; inner.config.action_space_size];
                    for m in generate_moves(inner, inner.get_current_player()) {
                        lg[m.action] = prior_logit(inner, &m, params, prior_scale);
                    }
                    let val = crate::ai::eval::evaluate(inner, params);
                    (lg, val)
                })
                .collect();
            let mut logits = Vec::with_capacity(n);
            let mut values = Vec::with_capacity(n);
            for (lg, val) in results {
                logits.push(lg);
                values.push(val);
            }
            (logits, values)
        }
    }

    let evaluator = HeuristicEval4x4 {
        params: EvalParams::default(),
        prior_scale: 0.5,
    };
    let mut episodes: Vec<PyGameEpisode> = Vec::with_capacity(num_games);
    let mut game_count = 0;
    let _ = worker_id;
    while game_count < num_games {
        let batch: Vec<GameEpisode> = py.detach(|| {
            self_play::run_batched_self_play::<Game4x4Env, HeuristicEval4x4>(
                &evaluator,
                cfg,
                num_games - game_count,
                concurrency,
                Game4x4Env::new,
            )
        });
        for ep in batch {
            if ep.samples.is_empty() {
                continue;
            }
            episodes.push(PyGameEpisode {
                inner: ep,
                variant: 2,
            });
            game_count += 1;
            if game_count >= num_games {
                break;
            }
        }
    }
    episodes
}

/// 4x4 Minimax 教师自对弈：用 expectiminimax + alpha-beta（深度 `depth`）驱动对局。
#[cfg(feature = "pyo3")]
struct MinimaxEval4x4 {
    depth: usize,
    lambda: f32,
}

#[cfg(feature = "pyo3")]
impl Evaluator<Game4x4Env> for MinimaxEval4x4 {
    fn evaluate(&self, envs: &[Game4x4Env]) -> (Vec<Vec<f32>>, Vec<f32>) {
        let mut logits = Vec::with_capacity(envs.len());
        let mut values = Vec::with_capacity(envs.len());
        for env in envs {
            let inner = &env.inner;
            let mut lg = vec![0.0f32; inner.config.action_space_size];
            let best = crate::ai::minimax::minimax_best_action(inner, self.depth);
            let best_val = best.map(|r| r.value).unwrap_or(0.0);
            if let Some(b) = best {
                lg[b.action] = 6.0 * self.lambda; // 给最优动作高先验
            }
            logits.push(lg);
            values.push(best_val);
        }
        (logits, values)
    }
}

#[cfg(feature = "pyo3")]
fn minimax_self_play_one(
    evaluator: &MinimaxEval4x4,
    cfg: &GumbelConfig,
    temperature: f32,
) -> GameEpisode {
    use crate::mcts::GumbelMCTS;
    let mut env = Game4x4Env::new();
    let mut episode_data = Vec::new();
    let mut mcts = GumbelMCTS::new(&env, evaluator, cfg.clone());
    let mut step = 0;
    loop {
        let search_result = match mcts.run() {
            Some(r) => r,
            None => {
                let (_, _, winner) = env.check_game_over_conditions();
                return finalize_episode(episode_data, winner, env.terminal_health_diff_red());
            }
        };
        let q_policy = mcts.get_root_completed_q_policy(temperature);
        let action = GumbelMCTS::<Game4x4Env, MinimaxEval4x4>::sample_action_from_policy(
            &q_policy, &search_result.action_mask);
        let completed_q = mcts.get_root_completed_q(action);
        episode_data.push((
            search_result.state,
            search_result.improved_policy,
            search_result.mcts_value,
            completed_q,
            search_result.root_visit_count,
            search_result.player,
            search_result.action_mask,
            action,
        ));
        match env.step(action) {
            Ok((_, _, terminated, truncated, winner)) => {
                mcts.step_next(&env, action);
                if terminated || truncated {
                    return finalize_episode(episode_data, winner, env.terminal_health_diff_red());
                }
            }
            Err(e) => {
                eprintln!("⚠️ minimax 教师自对弈 step 错误: {}", e);
                return GameEpisode {
                    samples: Vec::new(),
                    game_length: step,
                    winner: None,
                    health_diff_red: env.terminal_health_diff_red(),
                };
            }
        }
        step += 1;
        if step >= Game4x4Env::max_steps() {
            return finalize_episode(episode_data, Some(0), env.terminal_health_diff_red());
        }
    }
}

#[cfg(feature = "pyo3")]
pub fn run_game4x4_minimax_self_play_impl(
    py: Python<'_>,
    depth: usize,
    num_games: usize,
    concurrency: usize,
    temperature: f32,
) -> Vec<PyGameEpisode> {
    let _ = concurrency;
    let evaluator = MinimaxEval4x4 { depth, lambda: 1.0 };
    let gumbel_cfg = GumbelConfig {
        num_simulations: 16, // 教师无需太多 Gumbel 模拟：价值已由 minimax 提供
        max_considered_actions: 8,
        c_scale: 1.0,
        gumbel_scale: 1.0,
    };
    let mut episodes: Vec<PyGameEpisode> = Vec::with_capacity(num_games);
    let mut game_count = 0;
    while game_count < num_games {
        let batch: Vec<GameEpisode> = py.detach(|| {
            let mut eps = Vec::with_capacity(num_games - game_count);
            for _ in 0..(num_games - game_count) {
                let ep = minimax_self_play_one(&evaluator, &gumbel_cfg, temperature);
                if !ep.samples.is_empty() {
                    eps.push(ep);
                }
            }
            eps
        });
        for ep in batch {
            episodes.push(PyGameEpisode { inner: ep, variant: 2 });
            game_count += 1;
            if game_count >= num_games {
                break;
            }
        }
    }
    episodes
}

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
        let batch: Vec<GameEpisode> =
            py.detach(|| self_play::run_batched_self_play(
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
