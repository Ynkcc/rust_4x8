#[cfg(feature = "pyo3")]
use pyo3::prelude::*;
#[cfg(feature = "pyo3")]
use pyo3::types::PyDict;

#[cfg(feature = "pyo3")]
mod py_evaluator;

#[cfg(feature = "pyo3")]
pub use py_evaluator::PyEvaluator;

#[cfg(feature = "pyo3")]
use crate::game_env::{
    ACTION_SPACE_SIZE, BOARD_CHANNELS, BOARD_COLS, BOARD_ROWS, SCALAR_FEATURE_COUNT,
};
#[cfg(feature = "pyo3")]
use crate::self_play::{self, GameEpisode, ScenarioType, SelfPlayConfig};

#[cfg(feature = "pyo3")]
#[pyclass(name = "GameEpisode")]
#[derive(Clone)]
pub struct PyGameEpisode {
    pub inner: GameEpisode,
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

    fn get_samples(slf: PyRef<'_, Self>) -> (Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<f32>, Vec<f32>, Vec<u32>, Vec<f32>, Vec<Vec<i32>>) {
        let n = slf.inner.samples.len();
        let mut boards: Vec<Vec<f32>> = Vec::with_capacity(n);
        let mut scalars: Vec<Vec<f32>> = Vec::with_capacity(n);
        let mut policies: Vec<Vec<f32>> = Vec::with_capacity(n);
        let mut mcts_values: Vec<f32> = Vec::with_capacity(n);
        let mut completed_qs: Vec<f32> = Vec::with_capacity(n);
        let mut root_visits: Vec<u32> = Vec::with_capacity(n);
        let mut game_results: Vec<f32> = Vec::with_capacity(n);
        let mut action_masks: Vec<Vec<i32>> = Vec::with_capacity(n);

        for (obs, policy, mcts_val, completed_q, root_visit, game_result, mask) in &slf.inner.samples {
            boards.push(obs.board.as_slice().unwrap().to_vec());
            scalars.push(obs.scalars.as_slice().unwrap().to_vec());
            policies.push(policy.clone());
            mcts_values.push(*mcts_val);
            completed_qs.push(*completed_q);
            root_visits.push(*root_visit);
            game_results.push(*game_result);
            action_masks.push(mask.clone());
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
        )
    }

    fn to_dict<'py>(slf: PyRef<'py, Self>, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let game_length = slf.inner.game_length;
        let winner = slf.inner.winner;
        let num_samples = slf.inner.samples.len();
        let (boards, scalars, policies, mcts_values, completed_qs, root_visits, game_results, action_masks) =
            Self::get_samples(slf);

        let dict = PyDict::new_bound(py);
        dict.set_item("game_length", game_length)?;
        dict.set_item("winner", winner)?;
        dict.set_item("num_samples", num_samples)?;
        dict.set_item("boards", boards)?;
        dict.set_item("scalars", scalars)?;
        dict.set_item("policies", policies)?;
        dict.set_item("mcts_values", mcts_values)?;
        dict.set_item("completed_qs", completed_qs)?;
        dict.set_item("root_visits", root_visits)?;
        dict.set_item("game_results", game_results)?;
        dict.set_item("action_masks", action_masks)?;
        dict.set_item("board_shape", vec![BOARD_CHANNELS, BOARD_ROWS, BOARD_COLS])?;
        dict.set_item("scalar_shape", vec![SCALAR_FEATURE_COUNT])?;
        dict.set_item("action_space", ACTION_SPACE_SIZE)?;

        Ok(dict)
    }
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
    ))]
    fn new(
        mcts_sims: usize,
        max_considered_actions: usize,
        temperature_steps: usize,
    ) -> Self {
        Self {
            inner: SelfPlayConfig {
                mcts_sims,
                max_considered_actions,
                // 注意：Dirichlet 噪声注入已移除（Gumbel AlphaZero 探索由
                // Gumbel 噪声 + Sequential Halving 提供），不再暴露对应参数。
                temperature_steps,
                scenario: ScenarioType::Standard,
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
/// 注意：并行版（`_run_parallel_self_play_with_predictor`）是"每 worker 固定运行
/// `games_per_worker` 轮、空局跳过"，二者在"空局不占配额、返回非空局数量"上语义一致，
/// 但并行版每 worker 返回数 ≤ `games_per_worker`，总量以 `take(total_games)` 兜底。
#[cfg(feature = "pyo3")]
pub fn _run_self_play_with_predictor(
    predict_fn: PyObject,
    cfg: SelfPlayConfig,
    num_games: usize,
    worker_id: usize,
) -> Vec<PyGameEpisode> {
    let evaluator = PyEvaluator::new(predict_fn);

    let mut episodes = Vec::with_capacity(num_games);
    let mut game_count = 0;

    loop {
        let start_time = std::time::Instant::now();
        let episode = self_play::run_self_play(&evaluator, &cfg);
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

        episodes.push(PyGameEpisode { inner: episode });

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
pub fn _run_parallel_self_play_with_predictor(
    predict_fn: PyObject,
    cfg: SelfPlayConfig,
    num_workers: usize,
    games_per_worker: usize,
    worker_id: usize,
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

    let episodes_by_worker: Vec<Vec<PyGameEpisode>> = pool.install(|| {
        predict_fn_per_worker
            .into_par_iter()
            .enumerate()
            .map(|(wid, pf)| {
                let evaluator = PyEvaluator::new(pf);
                let mut local = Vec::with_capacity(games_per_worker);
                for g in 0..games_per_worker {
                    let start = std::time::Instant::now();
                    let episode = self_play::run_self_play(&evaluator, &cfg);
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
                    local.push(PyGameEpisode { inner: episode });
                }
                local
            })
            .collect()
    });

    episodes_by_worker
        .into_iter()
        .flatten()
        .take(total_games)
        .collect()
}
