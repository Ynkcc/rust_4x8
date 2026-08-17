#[cfg(feature = "pyo3")]
use pyo3::prelude::*;
#[cfg(feature = "pyo3")]
use pyo3::types::PyDict;

#[cfg(feature = "pyo3")]
use rayon::prelude::*;

#[cfg(feature = "pyo3")]
mod py_evaluator;

#[cfg(feature = "pyo3")]
pub use py_evaluator::PyEvaluator;

#[cfg(feature = "pyo3")]
pub mod ttt;

#[cfg(feature = "pyo3")]
pub mod darkchess_env;
#[cfg(feature = "pyo3")]
pub mod game4x4_env;
#[cfg(feature = "pyo3")]
pub mod mini_darkchess_env;

#[cfg(feature = "pyo3")]
use crate::game_env::{
    ACTION_SPACE_SIZE, BOARD_CHANNELS, BOARD_COLS, BOARD_ROWS, DarkChessEnv, Game4x4Env,
    GameEnv, MiniDarkChessEnv, SCALAR_FEATURE_COUNT,
};
#[cfg(feature = "pyo3")]
use crate::game_env::config::{GameConfig, darkchess_config, game_4x4_config, mini_config};
#[cfg(feature = "pyo3")]
use crate::self_play::{self, GameEpisode, ScenarioType, SelfPlayConfig, finalize_episode};

#[cfg(feature = "pyo3")]
use crate::mcts::{Evaluator, GumbelConfig};

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
    fn get_samples(slf: PyRef<'_, Self>) -> (Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<f32>, Vec<f32>, Vec<u32>, Vec<f32>, Vec<Vec<i32>>, Vec<usize>) {
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

        for (obs, policy, mcts_val, completed_q, root_visit, game_result, mask, action) in &slf.inner.samples {
            boards.push(obs.board.as_slice().unwrap().to_vec());
            scalars.push(obs.scalars.as_slice().unwrap().to_vec());
            policies.push(policy.clone());
            mcts_values.push(*mcts_val);
            completed_qs.push(*completed_q);
            root_visits.push(*root_visit);
            game_results.push(*game_result);
            action_masks.push(mask.clone());
            actions.push(*action);
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
            crate::MINI_BOARD_CHANNELS,
            crate::MINI_BOARD_ROWS,
            crate::MINI_BOARD_COLS,
            crate::MINI_SCALAR_FEATURE_COUNT,
            crate::MINI_ACTION_SPACE_SIZE,
        ),
        2 => (
            crate::GAME4X4_BOARD_CHANNELS,
            crate::GAME4X4_BOARD_ROWS,
            crate::GAME4X4_BOARD_COLS,
            crate::GAME4X4_SCALAR_FEATURE_COUNT,
            crate::GAME4X4_ACTION_SPACE_SIZE,
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

    for (obs, policy, mcts_val, completed_q, root_visit, game_result, mask, action) in &episode.samples {
        boards.push(obs.board.as_slice().unwrap().to_vec());
        scalars.push(obs.scalars.as_slice().unwrap().to_vec());
        policies.push(policy.clone());
        mcts_values.push(*mcts_val);
        completed_qs.push(*completed_q);
        root_visits.push(*root_visit);
        game_results.push(*game_result);
        action_masks.push(mask.clone());
        actions.push(*action);
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
    dict.set_item("action_masks", action_masks)?;
    dict.set_item("actions", actions)?;
    dict.set_item("board_shape", vec![bc, br, bcol])?;
    dict.set_item("scalar_shape", vec![sc])?;
    dict.set_item("action_space", ac)?;

    Ok(dict)
}

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
///
/// 空局（`samples` 为空）不计入目标局数，打印告警后跳过并继续生成，
/// 保证返回值长度恰好为 `num_games`。
///
/// 注意：并行版（`run_parallel_self_play_with_predictor_impl`）是"每 worker 固定运行
/// `games_per_worker` 轮、空局跳过"，二者在"空局不占配额、返回非空局数量"上语义一致，
/// 但并行版每 worker 返回数 ≤ `games_per_worker`，总量以 `take(total_games)` 兜底。
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
            // 空局不计入目标局数：继续生成，直到累计 num_games 个非空 episode。
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
///
/// - 每个 worker 线程拥有自己的 PyEvaluator（通过 clone_ref 增加 Python 引用计数）
/// - 每个 PyEvaluator.evaluate 内部用 Python::with_gil 获取 GIL；
///   若 predictor 内部有 time.sleep / IO 等待，sleep 会释放 GIL，从而让多个 worker 的等待
///   可以真正并发叠加，吞吐随 worker 数近似线性扩展。
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
    // (本质只是增加 Python 对象的引用计数，不做深拷贝)
    let predict_fn_per_worker: Vec<Py<PyAny>> = Python::attach(|py| {
        (0..num_workers.max(1))
            .map(|_| predict_fn.clone_ref(py))
            .collect()
    });

    // 创建固定大小的 rayon 线程池
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(num_workers.max(1))
        .build()
        .expect("failed to build rayon thread pool for parallel self_play");

    // 关键：用 allow_threads 释放 GIL 后再进入 rayon 并行区。
    // pool.install 会阻塞主线程直到全部并行任务完成；若此时仍持有 GIL，
    // worker 线程内的 Python::with_gil 将永远等不到 GIL，形成互等死锁。
    // allow_threads 在等待期间释放 GIL，worker 按需获取；predictor 内部
    // sleep/IO 会再次释放 GIL，实现多 worker 的等待真正并发叠加。
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

/// 4x4 启发式教师自对弈：用纯计算启发式评估器（规则先验 + 多特征价值）驱动
/// Gumbel MCTS 自对弈，生成高质量训练目标（improved_policy / mcts_value），
/// 供网络做「模仿学习」预热——网络学习强教师的走子与评估，避免从随机自对弈
/// 冷启动时目标噪声过大导致的训练停滞。
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
            // 用 rayon 并行计算启发式评估，充分利用多核。
            // 背景：run_batched_self_play 只有单个 eval_worker 线程串行处理所有
            // 树的叶子评估，导致并发高但 CPU 利用率极低（实测多核仅 ~12%）。
            // 对纯计算启发式（无 Python 推理），并行 evaluate 是吞吐关键。
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

/// 4x4 Minimax 教师自对弈：用 expectiminimax + alpha-beta（深度 `depth`）驱动对局，
/// 每步记录 minimax 的价值与走子分布作为训练目标。
///
/// 它比启发式 Gumbel MCTS 更强（minimax2 当前能碾压启发式 MCTS64），作为"更优教师"
/// 让网络在模仿阶段学到更深的战术，从而有潜力超越启发式教师的上限。
///
/// 注意：minimax 返回单一最优动作，无完整策略分布；因此 policy 目标采用
/// softmax(λ·minimax_value) 的"价值加权先验"——把价值最高动作概率设为接近 1，
/// 其余动作按价值 softmax。value 目标直接用 minimax 搜索值（[-1,1]）。
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
                return finalize_episode(episode_data, winner);
            }
        };
        // 温度采样：τ=1 时按价值加权探索，τ→0 时 argmax
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
                    return finalize_episode(episode_data, winner);
                }
            }
            Err(e) => {
                eprintln!("⚠️ minimax 教师自对弈 step 错误: {}", e);
                return GameEpisode { samples: Vec::new(), game_length: step, winner: None };
            }
        }
        step += 1;
        if step >= Game4x4Env::max_steps() {
            return finalize_episode(episode_data, Some(0));
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
    use crate::mcts::GumbelConfig;
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

/// 批量版：同时驱动 `concurrency` 局自对弈，并把多棵树的 MCTS 叶子评估合并成
/// 一个大 batch 送给 predictor，显著提升推理吞吐。
///
/// - 空局（samples 为空）不计入目标局数，跳过后继续生成，保证返回长度 = `num_games`。
/// - 内部使用 `self_play::run_batched_self_play`，每波并发 `concurrency` 局。
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

    // 循环生成，直到累计 num_games 个非空 episode。
    // 关键：`run_batched_self_play` 内部起了一个后台评估线程，评估时会
    // `Python::with_gil`；此处必须 `py.allow_threads` 释放 GIL，否则后台线程
    // 拿不到 GIL、主线程又等它返回，会形成互等死锁。
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

/// 从对局记录 dict（`GameEpisode::to_dict()` 的输出）解析人类可读的中文棋谱描述。
///
/// 内部使用 boards/scalars 逐手还原棋盘 → 重建环境 → 重新生成 action_masks 并与记录
/// 断言一致，同时断言 actions[i] 一定在合法掩码内；阵营由手数奇偶决定
/// （i%2==0 → 红方、i%2==1 → 黑方），无需手动传入颜色。
///
/// `variant`：0=4x8 暗棋（默认）、1=4x2 迷你、2=4x4。
#[cfg(feature = "pyo3")]
#[pyfunction]
#[pyo3(signature = (record, variant = 0))]
pub fn describe_record(record: &Bound<'_, PyDict>, variant: u8) -> PyResult<String> {
    let boards: Vec<Vec<f32>> = record
        .get_item("boards")?
        .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("记录缺少 boards"))?
        .extract()?;
    let scalars: Vec<Vec<f32>> = record
        .get_item("scalars")?
        .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("记录缺少 scalars"))?
        .extract()?;
    let action_masks: Vec<Vec<i32>> = record
        .get_item("action_masks")?
        .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("记录缺少 action_masks"))?
        .extract()?;
    let actions: Vec<usize> = record
        .get_item("actions")?
        .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("记录缺少 actions"))?
        .extract()?;

    let cfg = config_for_variant(variant)?;
    Ok(crate::replay::describe_record_with_config(
        &boards,
        &scalars,
        &action_masks,
        &actions,
        &cfg,
    ))
}

/// 按 variant 返回游戏配置：0=4x8 暗棋、1=4x2 迷你、2=4x4。
#[cfg(feature = "pyo3")]
pub fn config_for_variant(variant: u8) -> PyResult<GameConfig> {
    match variant {
        0 => Ok(darkchess_config()),
        1 => Ok(mini_config()),
        2 => Ok(game_4x4_config()),
        _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "未知 variant: {}（应为 0=4x8、1=4x2、2=4x4）",
            variant
        ))),
    }
}

/// 解码单手标量特征（MongoDB sample.scalar_state）为结构化/人类可读信息，由 Rust 侧解析。
///
/// scalars 布局（见 replay.rs）：`[0]` 连续无吃子步数/判和步数、`[1]` 当前方 HP/上限、
/// `[2]` 对方 HP/上限、之后为双方存活 one-hot。
///
/// 返回 dict：
///   - `move_counter`: 连续无吃子步数（原始值）
///   - `my_hp` / `opp_hp`: 当前方 / 对方 HP（原始值）
///   - `my_survival` / `opp_survival`: 按 active_types 顺序的存活数列表
///   - `my_dead` / `opp_dead`: 已阵亡棋子中文名列表
///   - `text`: 人类可读摘要
///
/// `variant`：0=4x8、1=4x2、2=4x4（默认 2）。`current_player`：1=红、-1=黑（仅影响描述文案）。
#[cfg(feature = "pyo3")]
#[pyfunction]
#[pyo3(signature = (scalars, variant = 2, current_player = 1))]
pub fn decode_scalar_state(
    py: Python<'_>,
    scalars: Vec<f32>,
    variant: u8,
    current_player: i32,
) -> PyResult<Bound<'_, PyDict>> {
    use crate::game_env::types::{PieceType, Player};
    use crate::replay::{decode_scalar_state as decode_rs, survival_to_dead_vec};

    let cfg = config_for_variant(variant)?;
    let r = decode_rs(&scalars, &cfg);
    let cur = if current_player == -1 { Player::Black } else { Player::Red };
    let opp = cur.opposite();

    let piece_name = |pt: PieceType, player: Player| -> String {
        let name = match pt {
            PieceType::General => match player {
                Player::Red => "帅",
                Player::Black => "将",
            },
            PieceType::Cannon => "炮",
            PieceType::Horse => "马",
            PieceType::Chariot => "车",
            PieceType::Elephant => "象",
            PieceType::Advisor => "士",
            PieceType::Soldier => "兵",
        };
        format!("{}{}", if player == Player::Red { "红" } else { "黑" }, name)
    };

    let my_survival: Vec<i32> = r.my_survival.iter().map(|&v| v as i32).collect();
    let opp_survival: Vec<i32> = r.opp_survival.iter().map(|&v| v as i32).collect();

    let my_dead: Vec<String> = survival_to_dead_vec(&r.my_survival, &cfg)
        .into_iter()
        .map(|pt| piece_name(pt, cur))
        .collect();
    let opp_dead: Vec<String> = survival_to_dead_vec(&r.opp_survival, &cfg)
        .into_iter()
        .map(|pt| piece_name(pt, opp))
        .collect();

    // 存活摘要：仅显示存活数 > 0 的棋子
    let mut my_alive = Vec::new();
    let mut opp_alive = Vec::new();
    for (ci, &pt) in cfg.active_types.iter().enumerate().take(cfg.num_active) {
        let pt = crate::game_env::types::PieceType::from_index(pt);
        if r.my_survival[ci] > 0 {
            my_alive.push(format!("{}x{}", piece_name(pt, cur), r.my_survival[ci]));
        }
        if r.opp_survival[ci] > 0 {
            opp_alive.push(format!("{}x{}", piece_name(pt, opp), r.opp_survival[ci]));
        }
    }

    let text = format!(
        "{}方回合 连续无吃子步数={} HP {}={} vs {}={} | {}存活: [{}] | {}存活: [{}] | {}阵亡: [{}] | {}阵亡: [{}]",
        if cur == Player::Red { "红" } else { "黑" },
        r.move_counter,
        if cur == Player::Red { "红" } else { "黑" },
        r.my_hp,
        if opp == Player::Red { "红" } else { "黑" },
        r.opp_hp,
        if cur == Player::Red { "红" } else { "黑" },
        my_alive.join(" "),
        if opp == Player::Red { "红" } else { "黑" },
        opp_alive.join(" "),
        if cur == Player::Red { "红" } else { "黑" },
        if my_dead.is_empty() { "无".to_string() } else { my_dead.join("、") },
        if opp == Player::Red { "红" } else { "黑" },
        if opp_dead.is_empty() { "无".to_string() } else { opp_dead.join("、") },
    );

    let dict = PyDict::new(py);
    dict.set_item("move_counter", r.move_counter)?;
    dict.set_item("my_hp", r.my_hp)?;
    dict.set_item("opp_hp", r.opp_hp)?;
    dict.set_item("my_survival", my_survival)?;
    dict.set_item("opp_survival", opp_survival)?;
    dict.set_item("my_dead", my_dead)?;
    dict.set_item("opp_dead", opp_dead)?;
    dict.set_item("text", text)?;
    dict.set_item("variant", variant)?;
    Ok(dict)
}
