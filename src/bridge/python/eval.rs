//! src/bridge/python/eval.rs — Rust 原生对战 / 自对弈唯一入口（PyO3 绑定）。
//!
//! 在 Rust 侧直接加载 .pt (TorchScript) 或 .onnx 模型，结合 rayon 多线程并发
//! 模拟对战对局（彻底释放 GIL 锁），支持固定随机种子（Seed）与两端玩家显式组合
//! （模型 / 启发式 MCTS / Minimax / 随机）。对局推进全部下沉到统一的
//! `pipeline::self_play::run_match_core` 主干，不再持有独立的对局循环。
//!
//! 这是 `run_native_match` 唯一入口：替代旧的 `run_eval_match`，并吸收旧
//! `RustTorchCollector` / `RustOnnxCollector` 的「Rust 持模型批量自对弈」能力
//! （`record_episodes=true` 时返回 `PyGameEpisode` 列表）。

use std::sync::Arc;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::core::env::{DarkChessEnv, Game4x4Env, GameEnv, MiniDarkChessEnv};
use crate::core::mcts::Evaluator;
use crate::pipeline::self_play::{
    AsDarkChessRef, MatchParams, PlayerSpec, SeedableEnv, SelfPlayConfig, run_match_core,
};

use crate::bridge::python::{PyEvaluator, PyGameEpisode, PySelfPlayConfig, SelfPlayVariant};

#[cfg(feature = "torch")]
use crate::inference::torchscript::LocalEvaluator;

#[cfg(feature = "onnx")]
use crate::inference::onnx::{OnnxEvaluator, OnnxModel};

// ============================================================================
// 模型评估器加载（.pt / .onnx → Arc<dyn Evaluator<G> + Send + Sync>）
// ============================================================================

fn load_model_eval<G: GameEnv>(
    path: &str,
) -> Result<Arc<dyn Evaluator<G> + Send + Sync>, String> {
    let path_obj = std::path::Path::new(path);
    let ext = path_obj.extension().and_then(|s| s.to_str()).unwrap_or("");
    if ext == "onnx" {
        #[cfg(feature = "onnx")]
        {
            let model = OnnxModel::new(path, "cpu")?;
            let eval = OnnxEvaluator::<G>::new(Arc::new(model));
            Ok(Arc::new(eval))
        }
        #[cfg(not(feature = "onnx"))]
        {
            Err(format!("需要启用 onnx 特性才能加载 ONNX 模型 ({path})"))
        }
    } else {
        #[cfg(feature = "torch")]
        {
            let eval = LocalEvaluator::<G>::new(path, tch::Device::Cpu)
                .map_err(|e| format!("加载 TorchScript 失败: {e}"))?;
            Ok(Arc::new(eval))
        }
        #[cfg(not(feature = "torch"))]
        {
            Err(format!("需要启用 torch 特性才能加载 TorchScript 模型 ({path})"))
        }
    }
}

// ============================================================================
// 选手规范解析（模型 / 启发式 / Minimax / 随机）
// ============================================================================

fn parse_player_spec<G: GameEnv>(
    spec_str: &str,
    heuristic_sims: Option<usize>,
) -> Result<PlayerSpec<G>, String> {
    if spec_str.starts_with("minimax") {
        let depth = spec_str
            .strip_prefix("minimax")
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(3);
        Ok(PlayerSpec::Minimax { depth })
    } else if spec_str.starts_with("heuristic") || spec_str.starts_with("mcts") {
        let default_sims = if let Some(rest) = spec_str.strip_prefix("heuristic") {
            rest.parse::<usize>().ok().unwrap_or(128)
        } else if let Some(rest) = spec_str.strip_prefix("mcts") {
            rest.parse::<usize>().ok().unwrap_or(128)
        } else {
            128
        };
        let sims = heuristic_sims.unwrap_or(default_sims);
        Ok(PlayerSpec::Heuristic { sims })
    } else if spec_str.starts_with("random") {
        Ok(PlayerSpec::Random)
    } else {
        let eval = load_model_eval::<G>(spec_str)?;
        Ok(PlayerSpec::ModelEval(eval))
    }
}

// ============================================================================
// 变体泛型对战 / 自对弈驱动
// ============================================================================

#[allow(clippy::too_many_arguments)]
fn run_native_for_variant<G>(
    py: Python<'_>,
    player_a_str: &str,
    player_b_str: &str,
    n: usize,
    model_sims: usize,
    heuristic_sims: Option<usize>,
    seed: Option<u64>,
    num_threads: usize,
    config: &SelfPlayConfig,
    record_episodes: bool,
    variant: u8,
) -> PyResult<(usize, usize, usize, Vec<f32>, f32, Vec<PyGameEpisode>)>
where
    G: GameEnv + AsDarkChessRef + SeedableEnv + Default + Send + Sync + 'static,
{
    let player_a = parse_player_spec::<G>(player_a_str, heuristic_sims)
        .map_err(|e| PyValueError::new_err(format!("加载选手 A 失败 ({player_a_str}): {e}")))?;
    let player_b = parse_player_spec::<G>(player_b_str, heuristic_sims)
        .map_err(|e| PyValueError::new_err(format!("加载选手 B 失败 ({player_b_str}): {e}")))?;

    let num_threads = num_threads.max(1);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build()
        .map_err(|e| PyValueError::new_err(format!("创建线程池失败: {e}")))?;

    let result = py.detach(|| {
        run_match_core::<G>(MatchParams {
            player_a: &player_a,
            player_b: &player_b,
            n_games: n,
            config,
            seed,
            record_episodes,
            model_sims,
            thread_pool: Some(&pool),
            make_env: G::default,
        })
    });

    let episodes = result
        .episodes
        .into_iter()
        .map(|inner| PyGameEpisode { inner, variant })
        .collect();

    Ok((
        result.wins,
        result.draws,
        result.losses,
        result.block_wr,
        result.avg_moves,
        episodes,
    ))
}

// ============================================================================
// PyO3 导出接口（唯一 Rust 原生入口）
// ============================================================================

#[pyfunction]
#[pyo3(signature = (player_a, player_b, n=100, variant_id="4x4", model_sims=64, heuristic_sims=None, seed=None, num_threads=4, config=None, record_episodes=false))]
pub fn run_native_match(
    py: Python<'_>,
    player_a: &str,
    player_b: &str,
    n: usize,
    variant_id: &str,
    model_sims: usize,
    heuristic_sims: Option<usize>,
    seed: Option<u64>,
    num_threads: usize,
    config: Option<PyRef<PySelfPlayConfig>>,
    record_episodes: bool,
) -> PyResult<(usize, usize, usize, Vec<f32>, f32, Vec<PyGameEpisode>)> {
    let cfg = match config {
        Some(c) => c.inner.clone(),
        None => SelfPlayConfig::default(),
    };
    match SelfPlayVariant::parse(variant_id)? {
        SelfPlayVariant::Dark4x8 => run_native_for_variant::<DarkChessEnv>(
            py, player_a, player_b, n, model_sims, heuristic_sims, seed, num_threads, &cfg,
            record_episodes, SelfPlayVariant::Dark4x8.episode_code(),
        ),
        SelfPlayVariant::Mini4x2 => run_native_for_variant::<MiniDarkChessEnv>(
            py, player_a, player_b, n, model_sims, heuristic_sims, seed, num_threads, &cfg,
            record_episodes, SelfPlayVariant::Mini4x2.episode_code(),
        ),
        SelfPlayVariant::Game4x4 => run_native_for_variant::<Game4x4Env>(
            py, player_a, player_b, n, model_sims, heuristic_sims, seed, num_threads, &cfg,
            record_episodes, SelfPlayVariant::Game4x4.episode_code(),
        ),
        SelfPlayVariant::Ttt => Err(PyValueError::new_err("run_native_match 不支持 ttt 变体")),
    }
}

// ============================================================================
// PyO3 导出接口（Python 推理单线程入口）
// ============================================================================

#[allow(clippy::too_many_arguments)]
fn run_python_for_variant<G>(
    py: Python<'_>,
    predict_fn: Py<PyAny>,
    cfg: &SelfPlayConfig,
    num_games: usize,
    variant: u8,
) -> PyResult<Vec<PyGameEpisode>>
where
    G: GameEnv + AsDarkChessRef + SeedableEnv + Default + Send + Sync + 'static,
{
    let evaluator = Arc::new(PyEvaluator::<G>::new(predict_fn));
    let player = PlayerSpec::<G>::PyPredictor(evaluator);

    let result = py.detach(|| {
        run_match_core::<G>(MatchParams {
            player_a: &player,
            player_b: &player,
            n_games: num_games,
            config: cfg,
            seed: None,
            record_episodes: true,
            model_sims: cfg.mcts_sims,
            thread_pool: None,
            make_env: G::default,
        })
    });

    Ok(result
        .episodes
        .into_iter()
        .map(|inner| PyGameEpisode { inner, variant })
        .collect())
}

#[pyfunction]
#[pyo3(signature = (predict_fn, config=None, num_games=1, concurrency=1, worker_id=0, variant_id="4x8"))]
pub fn run_python_match(
    py: Python<'_>,
    predict_fn: Py<PyAny>,
    config: Option<PyRef<PySelfPlayConfig>>,
    num_games: usize,
    concurrency: usize,
    worker_id: usize,
    variant_id: &str,
) -> PyResult<Vec<PyGameEpisode>> {
    // Python 推理路径统一单线程推进；concurrency 参数保留以兼容既有调用方。
    let _ = (concurrency, worker_id);
    let cfg = match config {
        Some(c) => c.inner.clone(),
        None => SelfPlayConfig::default(),
    };
    match SelfPlayVariant::parse(variant_id)? {
        SelfPlayVariant::Dark4x8 => {
            run_python_for_variant::<DarkChessEnv>(
                py,
                predict_fn,
                &cfg,
                num_games,
                SelfPlayVariant::Dark4x8.episode_code(),
            )
        }
        SelfPlayVariant::Mini4x2 => {
            run_python_for_variant::<MiniDarkChessEnv>(
                py,
                predict_fn,
                &cfg,
                num_games,
                SelfPlayVariant::Mini4x2.episode_code(),
            )
        }
        SelfPlayVariant::Game4x4 => {
            run_python_for_variant::<Game4x4Env>(
                py,
                predict_fn,
                &cfg,
                num_games,
                SelfPlayVariant::Game4x4.episode_code(),
            )
        }
        SelfPlayVariant::Ttt => Err(PyValueError::new_err("run_python_match 不支持 ttt 变体")),
    }
}
