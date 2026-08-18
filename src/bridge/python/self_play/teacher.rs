// src/py/self_play/teacher.rs
// Python 绑定层：教师自对弈（启发式 / minimax），泛型化支持多变体。
//
// 通过 `PyChessEnvCore::as_darkchess()` 把任意暗棋变体（4x8 `DarkChessEnv`、
// 4x2 `MiniDarkChessEnv`、4x4 `Game4x4Env`）统一映射到底层 `DarkChessEnv`，
// 从而复用引擎侧全部纯规则评估 / 走子生成 / minimax 搜索（均以 DarkChessEnv 为输入）。

#[cfg(feature = "pyo3")]
use pyo3::prelude::*;

#[cfg(feature = "pyo3")]
use rayon::prelude::*;

#[cfg(feature = "pyo3")]
use crate::bridge::python::chess_env::PyChessEnvCore;
#[cfg(feature = "pyo3")]
use crate::core::env::{DarkChessEnv, Game4x4Env, GameEnv, MiniDarkChessEnv};
#[cfg(feature = "pyo3")]
use crate::pipeline::self_play::finalize_episode;
#[cfg(feature = "pyo3")]
use crate::pipeline::self_play::{GameEpisode, SelfPlayConfig};
#[cfg(feature = "pyo3")]
use crate::core::mcts::{Evaluator, GumbelConfig, GumbelMCTS};
#[cfg(feature = "pyo3")]
use crate::bridge::python::episode::PyGameEpisode;

// ---------------------------------------------------------------------------
// 启发式教师：用纯计算启发式评估器驱动 Gumbel MCTS 自对弈
// ---------------------------------------------------------------------------

/// 适配任意暗棋变体（`G: GameEnv + PyChessEnvCore`）的启发式评估器。
#[cfg(feature = "pyo3")]
struct HeuristicEval<G: GameEnv + PyChessEnvCore> {
    params: crate::engine::evaluation::EvalParams,
    prior_scale: f32,
    _marker: std::marker::PhantomData<G>,
}

#[cfg(feature = "pyo3")]
impl<G: GameEnv + PyChessEnvCore + Sync> Evaluator<G> for HeuristicEval<G> {
    fn evaluate(&self, envs: &[G]) -> (Vec<Vec<f32>>, Vec<f32>) {
        use crate::engine::evaluation::evaluate;
        use crate::engine::mcts_heuristic::prior_logit;
        use crate::engine::movegen::generate_moves;
        let params = &self.params;
        let prior_scale = self.prior_scale;
        let n = envs.len();
        let results: Vec<_> = envs
            .par_iter()
            .map(|env| {
                let inner = env.as_darkchess();
                let mut lg = vec![0.0f32; inner.config.action_space_size];
                for m in generate_moves(inner, inner.get_current_player()) {
                    lg[m.action] = prior_logit(inner, &m, params, prior_scale);
                }
                let val = evaluate(inner, params);
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

/// 泛型启发式教师自对弈核心：驱动 `run_batched_self_play` 生成 `num_games` 局。
#[cfg(feature = "pyo3")]
fn run_heuristic_self_play_core<G: GameEnv + PyChessEnvCore + Sync>(
    py: Python<'_>,
    cfg: &SelfPlayConfig,
    num_games: usize,
    concurrency: usize,
    worker_id: usize,
    variant: u8,
    make_env: fn() -> G,
) -> Vec<PyGameEpisode> {
    let evaluator = HeuristicEval::<G> {
        params: crate::engine::evaluation::EvalParams::default(),
        prior_scale: 0.5,
        _marker: std::marker::PhantomData,
    };
    let mut episodes: Vec<PyGameEpisode> = Vec::with_capacity(num_games);
    let mut game_count = 0;
    let _ = worker_id;
    while game_count < num_games {
        let batch: Vec<GameEpisode> = py.detach(|| {
            crate::pipeline::self_play::run_batched_self_play::<G, HeuristicEval<G>>(
                &evaluator,
                cfg,
                num_games - game_count,
                concurrency,
                make_env,
            )
        });
        for ep in batch {
            if ep.samples.is_empty() {
                continue;
            }
            episodes.push(PyGameEpisode {
                inner: ep,
                variant,
            });
            game_count += 1;
            if game_count >= num_games {
                break;
            }
        }
    }
    episodes
}

// ---------------------------------------------------------------------------
// Minimax 教师：用 expectiminimax + alpha-beta 驱动对局
// ---------------------------------------------------------------------------

/// 适配任意暗棋变体（`G: GameEnv + PyChessEnvCore`）的 minimax 评估器。
#[cfg(feature = "pyo3")]
struct MinimaxEval<G: GameEnv + PyChessEnvCore> {
    depth: usize,
    lambda: f32,
    _marker: std::marker::PhantomData<G>,
}

#[cfg(feature = "pyo3")]
impl<G: GameEnv + PyChessEnvCore> Evaluator<G> for MinimaxEval<G> {
    fn evaluate(&self, envs: &[G]) -> (Vec<Vec<f32>>, Vec<f32>) {
        let mut logits = Vec::with_capacity(envs.len());
        let mut values = Vec::with_capacity(envs.len());
        for env in envs {
            let inner = env.as_darkchess();
            let mut lg = vec![0.0f32; inner.config.action_space_size];
            let best = crate::engine::minimax::minimax_best_action(inner, self.depth);
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

/// 泛型 minimax 教师单局自对弈。
#[cfg(feature = "pyo3")]
fn minimax_self_play_one<G: GameEnv + PyChessEnvCore + Sync>(
    evaluator: &MinimaxEval<G>,
    cfg: &GumbelConfig,
    temperature: f32,
    make_env: fn() -> G,
) -> GameEpisode {
    let mut env = make_env();
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
        let action = GumbelMCTS::<G, MinimaxEval<G>>::sample_action_from_policy(
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
        if step >= G::max_steps() {
            return finalize_episode(episode_data, Some(0), env.terminal_health_diff_red());
        }
    }
}

/// 泛型 minimax 教师自对弈核心：生成 `num_games` 局。
#[cfg(feature = "pyo3")]
fn run_minimax_self_play_core<G: GameEnv + PyChessEnvCore + Sync>(
    py: Python<'_>,
    depth: usize,
    num_games: usize,
    concurrency: usize,
    temperature: f32,
    variant: u8,
    make_env: fn() -> G,
) -> Vec<PyGameEpisode> {
    let _ = concurrency;
    let evaluator = MinimaxEval::<G> {
        depth,
        lambda: 1.0,
        _marker: std::marker::PhantomData,
    };
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
                let ep = minimax_self_play_one(&evaluator, &gumbel_cfg, temperature, make_env);
                if !ep.samples.is_empty() {
                    eps.push(ep);
                }
            }
            eps
        });
        for ep in batch {
            episodes.push(PyGameEpisode { inner: ep, variant });
            game_count += 1;
            if game_count >= num_games {
                break;
            }
        }
    }
    episodes
}

// ---------------------------------------------------------------------------
// 各变体导出入口（4x8=DarkChessEnv / 4x2=MiniDarkChessEnv / 4x4=Game4x4Env）
// ---------------------------------------------------------------------------

// --- 4x8 暗棋 ---

#[cfg(feature = "pyo3")]
pub fn run_heuristic_self_play_impl<'py>(
    py: Python<'py>,
    cfg: &SelfPlayConfig,
    num_games: usize,
    concurrency: usize,
    worker_id: usize,
) -> Vec<PyGameEpisode> {
    run_heuristic_self_play_core::<DarkChessEnv>(
        py, cfg, num_games, concurrency, worker_id, 0, DarkChessEnv::new,
    )
}

#[cfg(feature = "pyo3")]
pub fn run_minimax_self_play_impl(
    py: Python<'_>,
    depth: usize,
    num_games: usize,
    concurrency: usize,
    temperature: f32,
) -> Vec<PyGameEpisode> {
    run_minimax_self_play_core::<DarkChessEnv>(
        py, depth, num_games, concurrency, temperature, 0, DarkChessEnv::new,
    )
}

// --- 4x2 迷你暗棋 ---

#[cfg(feature = "pyo3")]
pub fn run_mini_heuristic_self_play_impl<'py>(
    py: Python<'py>,
    cfg: &SelfPlayConfig,
    num_games: usize,
    concurrency: usize,
    worker_id: usize,
) -> Vec<PyGameEpisode> {
    run_heuristic_self_play_core::<MiniDarkChessEnv>(
        py, cfg, num_games, concurrency, worker_id, 1, MiniDarkChessEnv::new,
    )
}

#[cfg(feature = "pyo3")]
pub fn run_mini_minimax_self_play_impl(
    py: Python<'_>,
    depth: usize,
    num_games: usize,
    concurrency: usize,
    temperature: f32,
) -> Vec<PyGameEpisode> {
    run_minimax_self_play_core::<MiniDarkChessEnv>(
        py, depth, num_games, concurrency, temperature, 1, MiniDarkChessEnv::new,
    )
}

// --- 4x4 暗棋（保持既有导出名，内部改用泛型实现） ---

#[cfg(feature = "pyo3")]
pub fn run_game4x4_heuristic_self_play_impl<'py>(
    py: Python<'py>,
    cfg: &SelfPlayConfig,
    num_games: usize,
    concurrency: usize,
    worker_id: usize,
) -> Vec<PyGameEpisode> {
    run_heuristic_self_play_core::<Game4x4Env>(
        py, cfg, num_games, concurrency, worker_id, 2, Game4x4Env::new,
    )
}

#[cfg(feature = "pyo3")]
pub fn run_game4x4_minimax_self_play_impl(
    py: Python<'_>,
    depth: usize,
    num_games: usize,
    concurrency: usize,
    temperature: f32,
) -> Vec<PyGameEpisode> {
    run_minimax_self_play_core::<Game4x4Env>(
        py, depth, num_games, concurrency, temperature, 2, Game4x4Env::new,
    )
}
