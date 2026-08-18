//! src/bridge/python/eval.rs — Rust 原生高性能评估引擎（PyO3 绑定）。
//!
//! 在 Rust 侧直接加载 .pt (TorchScript) 或 .onnx 模型，结合 rayon 多线程并发
//! 模拟对战对局（彻底释放 GIL 锁），提供秒级完成 100 局的大样本对战评估能力。
//! 支持固定随机种子（Seed）和两端玩家显式组合设定（模型/规则/MCTS/Minimax/随机）。

use std::sync::Arc;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::core::env::{DarkChessEnv, Game4x4Env, GameEnv, MiniDarkChessEnv};
use crate::core::mcts::{Evaluator, GumbelConfig, GumbelMCTS};
use crate::engine::minimax::minimax_best_action;
use crate::engine::HeuristicMctsPolicy;

#[cfg(feature = "torch")]
use crate::inference::torchscript::LocalEvaluator;

#[cfg(feature = "onnx")]
use crate::inference::onnx::{OnnxEvaluator, OnnxModel};

// ============================================================================
// 辅助 Trait：提取 &DarkChessEnv 与种子设置
// ============================================================================

pub trait AsDarkChessRef {
    fn as_darkchess_ref(&self) -> &DarkChessEnv;
}

impl AsDarkChessRef for DarkChessEnv {
    fn as_darkchess_ref(&self) -> &DarkChessEnv {
        self
    }
}

impl AsDarkChessRef for MiniDarkChessEnv {
    fn as_darkchess_ref(&self) -> &DarkChessEnv {
        &self.inner
    }
}

impl AsDarkChessRef for Game4x4Env {
    fn as_darkchess_ref(&self) -> &DarkChessEnv {
        &self.inner
    }
}

pub trait SeedableEnv {
    fn set_seed(&mut self, seed: u64);
}

impl SeedableEnv for DarkChessEnv {
    fn set_seed(&mut self, seed: u64) {
        self.seed = Some(seed);
        self.reset_internal_state();
        self.initialize_board();
    }
}

impl SeedableEnv for MiniDarkChessEnv {
    fn set_seed(&mut self, seed: u64) {
        self.inner.seed = Some(seed);
        self.inner.reset_internal_state();
        self.inner.initialize_board();
    }
}

impl SeedableEnv for Game4x4Env {
    fn set_seed(&mut self, seed: u64) {
        self.inner.seed = Some(seed);
        self.inner.reset_internal_state();
        self.inner.initialize_board();
    }
}

// ============================================================================
// 统一模型评估器包装 (TorchScript / ONNX)
// ============================================================================

enum ModelEval<G: GameEnv> {
    #[cfg(feature = "torch")]
    Torch(LocalEvaluator<G>),
    #[cfg(feature = "onnx")]
    Onnx(OnnxEvaluator<G>),
    #[allow(dead_code)]
    Phantom(std::marker::PhantomData<G>),
}

impl<G: GameEnv> Evaluator<G> for ModelEval<G> {
    fn evaluate(&self, envs: &[G]) -> (Vec<Vec<f32>>, Vec<f32>) {
        match self {
            #[cfg(feature = "torch")]
            ModelEval::Torch(eval) => eval.evaluate(envs),
            #[cfg(feature = "onnx")]
            ModelEval::Onnx(eval) => eval.evaluate(envs),
            _ => (Vec::new(), Vec::new()),
        }
    }
}

fn load_model_eval<G: GameEnv>(path: &str) -> Result<ModelEval<G>, String> {
    let path_obj = std::path::Path::new(path);
    let ext = path_obj.extension().and_then(|s| s.to_str()).unwrap_or("");
    if ext == "onnx" {
        #[cfg(feature = "onnx")]
        {
            let model = OnnxModel::new(path, "cpu")?;
            let arc_model = Arc::new(model);
            let eval = OnnxEvaluator::<G>::new(arc_model);
            Ok(ModelEval::Onnx(eval))
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
            Ok(ModelEval::Torch(eval))
        }
        #[cfg(not(feature = "torch"))]
        {
            Err(format!("需要启用 torch 特性才能加载 TorchScript 模型 ({path})"))
        }
    }
}

// ============================================================================
// 选手规范 (Minimax / 启发式 MCTS / 神经网络模型)
// ============================================================================

enum PlayerSpec<G: GameEnv> {
    Minimax(usize),
    Heuristic(usize),
    Model(ModelEval<G>),
}

fn parse_player_spec<G: GameEnv>(
    spec_str: &str,
    heuristic_sims: Option<usize>,
) -> Result<PlayerSpec<G>, String> {
    if spec_str.starts_with("minimax") {
        let depth = spec_str
            .strip_prefix("minimax")
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(3);
        Ok(PlayerSpec::Minimax(depth))
    } else if spec_str.starts_with("heuristic") || spec_str.starts_with("mcts") {
        let default_sims = if let Some(rest) = spec_str.strip_prefix("heuristic") {
            rest.parse::<usize>().ok().unwrap_or(128)
        } else if let Some(rest) = spec_str.strip_prefix("mcts") {
            rest.parse::<usize>().ok().unwrap_or(128)
        } else {
            128
        };
        let sims = heuristic_sims.unwrap_or(default_sims);
        Ok(PlayerSpec::Heuristic(sims))
    } else {
        let eval = load_model_eval::<G>(spec_str)?;
        Ok(PlayerSpec::Model(eval))
    }
}

fn model_mcts_action<G: GameEnv>(
    env: &G,
    evaluator: &ModelEval<G>,
    sims: usize,
) -> Option<usize> {
    let config = GumbelConfig {
        num_simulations: sims,
        max_considered_actions: 16,
        c_scale: 0.25,
        gumbel_scale: 1.0,
    };
    let mut mcts = GumbelMCTS::new(env, evaluator, config);
    mcts.run().map(|r| r.action)
}

fn get_player_action<G>(
    env: &G,
    spec: &PlayerSpec<G>,
    model_sims: usize,
) -> Option<usize>
where
    G: GameEnv + AsDarkChessRef,
{
    match spec {
        PlayerSpec::Minimax(depth) => {
            let dark_ref = env.as_darkchess_ref();
            minimax_best_action(dark_ref, *depth).map(|r| r.action)
        }
        PlayerSpec::Heuristic(sims) => {
            let dark_ref = env.as_darkchess_ref();
            let policy = HeuristicMctsPolicy::new(*sims);
            policy.choose_action(dark_ref)
        }
        PlayerSpec::Model(evaluator) => {
            model_mcts_action(env, evaluator, model_sims)
        }
    }
}

fn play_one_game<G>(
    player_a_spec: &PlayerSpec<G>,
    player_b_spec: &PlayerSpec<G>,
    player_a_is_red: bool,
    model_sims: usize,
    game_seed: Option<u64>,
    max_moves: usize,
) -> (i32, usize)
where
    G: GameEnv + AsDarkChessRef + SeedableEnv + Default,
{
    let mut env = G::default();
    if let Some(s) = game_seed {
        env.set_seed(s);
    }
    let mut moves = 0;

    while !env.check_game_over_conditions().0 {
        if env.check_game_over_conditions().2.is_some() {
            break;
        }
        let cur = env.get_current_player().val();
        let is_a_turn = (cur == 1) == player_a_is_red;

        let action = if is_a_turn {
            get_player_action(&env, player_a_spec, model_sims)
        } else {
            get_player_action(&env, player_b_spec, model_sims)
        };

        let Some(a) = action else {
            break;
        };

        if GameEnv::step(&mut env, a).is_err() {
            break;
        }

        moves += 1;
        if moves >= max_moves {
            break;
        }
    }

    let winner = env.check_game_over_conditions().2;
    let r = match winner {
        Some(1) => {
            if player_a_is_red { 1 } else { -1 }
        }
        Some(-1) => {
            if player_a_is_red { -1 } else { 1 }
        }
        _ => 0,
    };
    (r, moves)
}

fn run_match_for_variant<G>(
    py: Python<'_>,
    player_a_str: &str,
    player_b_str: &str,
    n: usize,
    model_sims: usize,
    heuristic_sims: Option<usize>,
    seed: Option<u64>,
    num_threads: usize,
) -> PyResult<(usize, usize, usize, Vec<f32>, f32)>
where
    G: GameEnv + AsDarkChessRef + SeedableEnv + Default + Send + Sync + 'static,
{
    let player_a_spec = parse_player_spec::<G>(player_a_str, heuristic_sims)
        .map_err(|e| PyValueError::new_err(format!("加载选手 A 失败 ({player_a_str}): {e}")))?;
    let player_b_spec = parse_player_spec::<G>(player_b_str, heuristic_sims)
        .map_err(|e| PyValueError::new_err(format!("加载选手 B 失败 ({player_b_str}): {e}")))?;

    let arc_player_a = Arc::new(player_a_spec);
    let arc_player_b = Arc::new(player_b_spec);

    let num_threads = num_threads.max(1);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build()
        .map_err(|e| PyValueError::new_err(format!("创建线程池失败: {e}")))?;

    let indices: Vec<usize> = (0..n).collect();

    let results: Vec<(i32, usize)> = py.detach(|| {
        pool.install(|| {
            indices
                .into_par_iter()
                .map(|i| {
                    let player_a_is_red = (i % 2) == 0;
                    let game_seed = seed.map(|s| s.wrapping_add(i as u64));
                    play_one_game::<G>(
                        arc_player_a.as_ref(),
                        arc_player_b.as_ref(),
                        player_a_is_red,
                        model_sims,
                        game_seed,
                        400,
                    )
                })
                .collect()
        })
    });

    let mut wins = 0;
    let mut draws = 0;
    let mut losses = 0;
    let mut total_moves = 0;
    let block_size = 20;
    let mut block_wr = Vec::new();

    let mut blk_w = 0;
    let mut blk_tot = 0;

    for (i, &(r, moves)) in results.iter().enumerate() {
        total_moves += moves;
        if r == 1 {
            wins += 1;
            blk_w += 1;
        } else if r == -1 {
            losses += 1;
        } else {
            draws += 1;
        }
        blk_tot += 1;

        if (i + 1) % block_size == 0 {
            block_wr.push(100.0 * (blk_w as f32) / (blk_tot as f32));
            blk_w = 0;
            blk_tot = 0;
        }
    }
    if blk_tot > 0 && n % block_size != 0 {
        block_wr.push(100.0 * (blk_w as f32) / (blk_tot as f32));
    }

    let avg_moves = (total_moves as f32) / (n.max(1) as f32);
    Ok((wins, draws, losses, block_wr, avg_moves))
}

// ============================================================================
// PyO3 导出接口
// ============================================================================

#[pyfunction]
#[pyo3(signature = (player_a, player_b, n=100, variant_id="4x4", model_sims=64, heuristic_sims=None, seed=None, num_threads=4))]
pub fn run_eval_match(
    py: Python<'_>,
    player_a: &str,
    player_b: &str,
    n: usize,
    variant_id: &str,
    model_sims: usize,
    heuristic_sims: Option<usize>,
    seed: Option<u64>,
    num_threads: usize,
) -> PyResult<(usize, usize, usize, Vec<f32>, f32)> {
    match variant_id {
        "4x2" | "mini" => run_match_for_variant::<MiniDarkChessEnv>(
            py,
            player_a,
            player_b,
            n,
            model_sims,
            heuristic_sims,
            seed,
            num_threads,
        ),
        "4x4" => run_match_for_variant::<Game4x4Env>(
            py,
            player_a,
            player_b,
            n,
            model_sims,
            heuristic_sims,
            seed,
            num_threads,
        ),
        "4x8" | "dark" => run_match_for_variant::<DarkChessEnv>(
            py,
            player_a,
            player_b,
            n,
            model_sims,
            heuristic_sims,
            seed,
            num_threads,
        ),
        _ => Err(PyValueError::new_err(format!("未知的变体 ID: {variant_id}"))),
    }
}
