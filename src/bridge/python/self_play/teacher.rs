// src/py/self_play/teacher.rs
// Python 绑定层：教师自对弈（4x4 启发式 / 4x4 minimax）。

#[cfg(feature = "pyo3")]
use pyo3::prelude::*;

#[cfg(feature = "pyo3")]
use rayon::prelude::*;

#[cfg(feature = "pyo3")]
use crate::core::env::{Game4x4Env, GameEnv};
#[cfg(feature = "pyo3")]
use crate::pipeline::self_play::finalize_episode;
#[cfg(feature = "pyo3")]
use crate::pipeline::self_play::{GameEpisode, SelfPlayConfig};
#[cfg(feature = "pyo3")]
use crate::core::mcts::{Evaluator, GumbelConfig};
#[cfg(feature = "pyo3")]
use crate::bridge::python::episode::PyGameEpisode;

/// 4x4 启发式教师自对弈：用纯计算启发式评估器驱动 Gumbel MCTS 自对弈。
#[cfg(feature = "pyo3")]
pub fn run_game4x4_heuristic_self_play_impl<'py>(
    py: Python<'py>,
    cfg: &SelfPlayConfig,
    num_games: usize,
    concurrency: usize,
    worker_id: usize,
) -> Vec<PyGameEpisode> {
    use crate::engine::evaluation::EvalParams;
    use crate::engine::mcts_heuristic::prior_logit;
    use crate::engine::movegen::generate_moves;
    use crate::core::mcts::Evaluator;

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
                    let val = crate::engine::evaluation::evaluate(inner, params);
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
            crate::pipeline::self_play::run_batched_self_play::<Game4x4Env, HeuristicEval4x4>(
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

#[cfg(feature = "pyo3")]
fn minimax_self_play_one(
    evaluator: &MinimaxEval4x4,
    cfg: &GumbelConfig,
    temperature: f32,
) -> GameEpisode {
    use crate::core::mcts::GumbelMCTS;
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
