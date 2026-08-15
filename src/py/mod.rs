#[cfg(feature = "pyo3")]
use pyo3::prelude::*;
#[cfg(feature = "pyo3")]
use pyo3::types::PyDict;

#[cfg(feature = "pyo3")]
mod py_evaluator;

#[cfg(feature = "pyo3")]
pub use py_evaluator::PyEvaluator;

#[cfg(feature = "pyo3")]
pub mod ttt;

#[cfg(feature = "pyo3")]
pub mod darkchess_env;
#[cfg(feature = "pyo3")]
pub mod mini_darkchess_env;

#[cfg(feature = "pyo3")]
use crate::game_env::{
    ACTION_SPACE_SIZE, BOARD_CHANNELS, BOARD_COLS, BOARD_ROWS, DarkChessEnv, GameEnv,
    MiniDarkChessEnv, SCALAR_FEATURE_COUNT,
};
#[cfg(feature = "pyo3")]
use crate::self_play::{self, GameEpisode, ScenarioType, SelfPlayConfig};

#[cfg(feature = "pyo3")]
#[pyclass(name = "GameEpisode")]
#[derive(Clone)]
pub struct PyGameEpisode {
    pub inner: GameEpisode,
    /// 是否为 4x2 迷你变体（决定 episode dict 中的 shape 字段）。
    pub mini: bool,
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
        episode_to_dict(py, &slf.inner, slf.mini)
    }
}

/// 将 GameEpisode 序列化为 PyDict（供 `PyGameEpisode::to_dict` 和
/// `py_data_collector.rs` 共用，消除重复逻辑）。
/// `mini` 为 true 时输出 4x2 迷你变体的 shape 字段，否则输出 4x8 暗棋 shape。
#[cfg(feature = "pyo3")]
pub fn episode_to_dict<'py>(
    py: Python<'py>,
    episode: &GameEpisode,
    mini: bool,
) -> PyResult<Bound<'py, PyDict>> {
    episode_to_dict_with_shapes(py, episode, mini)
}

/// 4x8 暗棋变体的 episode dict（供 py_data_collector.rs 兼容调用）。
#[cfg(feature = "pyo3")]
pub fn episode_to_dict_darkchess<'py>(
    py: Python<'py>,
    episode: &GameEpisode,
) -> PyResult<Bound<'py, PyDict>> {
    episode_to_dict_with_shapes(py, episode, false)
}

#[cfg(feature = "pyo3")]
fn episode_to_dict_with_shapes<'py>(
    py: Python<'py>,
    episode: &GameEpisode,
    mini: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let (bc, br, bcol, sc, ac): (usize, usize, usize, usize, usize) = if mini {
        (
            crate::MINI_BOARD_CHANNELS,
            crate::MINI_BOARD_ROWS,
            crate::MINI_BOARD_COLS,
            crate::MINI_SCALAR_FEATURE_COUNT,
            crate::MINI_ACTION_SPACE_SIZE,
        )
    } else {
        (BOARD_CHANNELS, BOARD_ROWS, BOARD_COLS, SCALAR_FEATURE_COUNT, ACTION_SPACE_SIZE)
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

    let dict = PyDict::new_bound(py);
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
#[pyclass(name = "SelfPlayConfig")]
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
        c_visit = 50.0,
    ))]
    fn new(
        mcts_sims: usize,
        max_considered_actions: usize,
        temperature_steps: usize,
        c_visit: f32,
    ) -> Self {
        Self {
            inner: SelfPlayConfig {
                mcts_sims,
                max_considered_actions,
                // 注意：Dirichlet 噪声注入已移除（Gumbel AlphaZero 探索由
                // Gumbel 噪声 + Sequential Halving 提供），不再暴露对应参数。
                temperature_steps,
                scenario: ScenarioType::Standard,
                c_visit,
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
    predict_fn: PyObject,
    cfg: SelfPlayConfig,
    num_games: usize,
    worker_id: usize,
) -> Vec<PyGameEpisode> {
    let evaluator = PyEvaluator::new(predict_fn);
    run_self_play_serial_core(&evaluator, &cfg, num_games, worker_id, false, DarkChessEnv::new)
}

/// 4x2 迷你暗棋版串行自对弈。
#[cfg(feature = "pyo3")]
pub fn run_mini_self_play_with_predictor_impl(
    predict_fn: PyObject,
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
        true,
        MiniDarkChessEnv::new,
    )
}

#[cfg(feature = "pyo3")]
fn run_self_play_serial_core<G: GameEnv>(
    evaluator: &PyEvaluator<G>,
    cfg: &SelfPlayConfig,
    num_games: usize,
    worker_id: usize,
    mini: bool,
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
            mini,
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
    predict_fn: PyObject,
    cfg: SelfPlayConfig,
    num_workers: usize,
    games_per_worker: usize,
    worker_id: usize,
) -> Vec<PyGameEpisode> {
    run_parallel_core(
        predict_fn, cfg, num_workers, games_per_worker, worker_id, false, DarkChessEnv::new,
    )
}

/// 4x2 迷你暗棋版并行自对弈。
#[cfg(feature = "pyo3")]
pub fn run_mini_parallel_self_play_with_predictor_impl(
    predict_fn: PyObject,
    cfg: SelfPlayConfig,
    num_workers: usize,
    games_per_worker: usize,
    worker_id: usize,
) -> Vec<PyGameEpisode> {
    run_parallel_core(
        predict_fn, cfg, num_workers, games_per_worker, worker_id, true, MiniDarkChessEnv::new,
    )
}

#[cfg(feature = "pyo3")]
fn run_parallel_core<G: GameEnv>(
    predict_fn: PyObject,
    cfg: SelfPlayConfig,
    num_workers: usize,
    games_per_worker: usize,
    worker_id: usize,
    mini: bool,
    _make_env: fn() -> G,
) -> Vec<PyGameEpisode> {
    use rayon::prelude::*;

    let _ = worker_id;
    let total_games = num_workers.max(1) * games_per_worker;

    // 在持有 GIL 的情况下，为每个 worker 克隆一份 predict_fn 引用
    // (本质只是增加 Python 对象的引用计数，不做深拷贝)
    let predict_fn_per_worker: Vec<PyObject> = Python::with_gil(|py| {
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
    let episodes_by_worker: Vec<Vec<PyGameEpisode>> = Python::with_gil(|py| {
        py.allow_threads(|| {
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
                                mini,
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

/// 批量版：同时驱动 `concurrency` 局自对弈，并把多棵树的 MCTS 叶子评估合并成
/// 一个大 batch 送给 predictor，显著提升推理吞吐。
///
/// - 空局（samples 为空）不计入目标局数，跳过后继续生成，保证返回长度 = `num_games`。
/// - 内部使用 `self_play::run_batched_self_play`，每波并发 `concurrency` 局。
#[cfg(feature = "pyo3")]
pub fn run_batched_self_play_with_predictor_impl<'py>(
    py: Python<'py>,
    predict_fn: PyObject,
    cfg: SelfPlayConfig,
    num_games: usize,
    concurrency: usize,
    worker_id: usize,
) -> Vec<PyGameEpisode> {
    let evaluator = PyEvaluator::new(predict_fn);
    run_batched_core(
        py, &evaluator, &cfg, num_games, concurrency, worker_id, false, DarkChessEnv::new,
    )
}

/// 4x2 迷你暗棋版批量自对弈。
#[cfg(feature = "pyo3")]
pub fn run_mini_batched_self_play_with_predictor_impl<'py>(
    py: Python<'py>,
    predict_fn: PyObject,
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
        true,
        MiniDarkChessEnv::new,
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
    mini: bool,
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
            py.allow_threads(|| self_play::run_batched_self_play(
                evaluator, cfg, num_games - game_count, concurrency, make_env,
            ));
        for ep in batch {
            if ep.samples.is_empty() {
                eprintln!("[Worker-{}] ⚠️ 生成了空游戏数据，跳过", worker_id);
                continue;
            }
            episodes.push(PyGameEpisode { inner: ep, mini });
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
#[cfg(feature = "pyo3")]
#[pyfunction]
pub fn describe_record(record: &Bound<'_, PyDict>) -> PyResult<String> {
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

    Ok(crate::replay::describe_record(
        &boards,
        &scalars,
        &action_masks,
        &actions,
    ))
}
