// src/pipeline/self_play/match_core.rs
// 统一对局主干引擎（Unified Match Core Engine）。
//
// 承载 A vs B 双选手对战 / 自对弈的核心驱动逻辑，供 `run_native_match`（Rust 持有
// 模型 + rayon 多线程）与 `run_python_match`（Python predict_fn 单线程）两个唯一
// 上层入口共同调用。两端必须经此主干推进对局，杜绝重复的对局调度 / 动作选择 /
// 胜败逻辑。
//
// 设计要点：
// - `PlayerSpec`：统一选手抽象（模型 / 启发式 MCTS / Minimax / Python 推理 / 随机）。
// - `run_match_core`：统一主干，支持固定 Seed、记录 Episode 或仅收集胜负统计。
// - `AsDarkChessRef` / `SeedableEnv` 从旧 `bridge/python/eval.rs` 迁入，供规则选手
//   取底层棋盘与设置种子。

use std::marker::PhantomData;
use std::sync::Arc;

use rayon::prelude::*;

use crate::core::env::{DarkChessEnv, Game4x4Env, GameEnv, MiniDarkChessEnv};
use crate::core::mcts::{Evaluator, EvaluatorOutput, GumbelConfig, GumbelMCTS};
use crate::engine::evaluation::{evaluate, EvalParams};
use crate::engine::mcts_heuristic::prior_logit;
use crate::engine::minimax::minimax_best_action;
use crate::engine::movegen::generate_moves;
use crate::engine::HeuristicMctsPolicy;

#[cfg(feature = "pyo3")]
use crate::bridge::python::py_evaluator::PyEvaluator;

use super::{finalize_episode, GameEpisode, SelfPlayConfig};

// ============================================================================
// 辅助 Trait：提取 &DarkChessEnv 与种子设置（自 eval.rs 迁入）
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
// 统一选手抽象 PlayerSpec
// ============================================================================

/// 统一选手抽象：支持模型 / 启发式 MCTS / Minimax / Python 推理 / 随机任意组合。
pub enum PlayerSpec<G: GameEnv> {
    Minimax { depth: usize },
    Heuristic { sims: usize },
    /// Rust 侧持有 .pt / .onnx 模型的评估器（推理不经过 GIL）。
    ModelEval(Arc<dyn Evaluator<G> + Send + Sync>),
    /// Python 侧 predict_fn 推理服务（单线程，GIL 边界内调用）。
    #[cfg(feature = "pyo3")]
    PyPredictor(Arc<PyEvaluator<G>>),
    Random,
}

// ============================================================================
// 规则 / 随机评估器（供记录 Episode 的 MCTS 自对弈路径使用）
// ============================================================================

/// 用纯计算启发式评估器驱动 Gumbel MCTS。
struct HeuristicEval<G: GameEnv + AsDarkChessRef> {
    params: EvalParams,
    prior_scale: f32,
    _marker: PhantomData<G>,
}

impl<G: GameEnv + AsDarkChessRef + Sync> Evaluator<G> for HeuristicEval<G> {
    fn evaluate(&self, envs: &[G]) -> EvaluatorOutput {
        let params = &self.params;
        let prior_scale = self.prior_scale;
        let n = envs.len();
        let results: Vec<_> = envs
            .par_iter()
            .map(|env| {
                let inner = env.as_darkchess_ref();
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
        EvaluatorOutput { logits, values, health: None }
    }
}

/// 用 expectiminimax + alpha-beta 评估器驱动 Gumbel MCTS。
struct MinimaxEval<G: GameEnv + AsDarkChessRef> {
    depth: usize,
    lambda: f32,
    _marker: PhantomData<G>,
}

impl<G: GameEnv + AsDarkChessRef> Evaluator<G> for MinimaxEval<G> {
    fn evaluate(&self, envs: &[G]) -> EvaluatorOutput {
        let mut logits = Vec::with_capacity(envs.len());
        let mut values = Vec::with_capacity(envs.len());
        for env in envs {
            let inner = env.as_darkchess_ref();
            let mut lg = vec![0.0f32; inner.config.action_space_size];
            let best = minimax_best_action(inner, self.depth);
            let best_val = best.map(|r| r.value).unwrap_or(0.0);
            if let Some(b) = best {
                lg[b.action] = 6.0 * self.lambda;
            }
            logits.push(lg);
            values.push(best_val);
        }
        EvaluatorOutput { logits, values, health: None }
    }
}

/// 随机评估器：对所有合法动作给均匀先验。
struct RandomEval;

impl<G: GameEnv> Evaluator<G> for RandomEval {
    fn evaluate(&self, envs: &[G]) -> EvaluatorOutput {
        let mut logits = Vec::with_capacity(envs.len());
        let mut values = Vec::with_capacity(envs.len());
        for env in envs {
            let mut lg = vec![0.0f32; G::action_space_size()];
            let mut masks = vec![0i32; G::action_space_size()];
            env.action_masks_into(&mut masks);
            for (i, &m) in masks.iter().enumerate() {
                if m == 1 {
                    lg[i] = 1.0;
                }
            }
            logits.push(lg);
            values.push(0.0);
        }
        EvaluatorOutput { logits, values, health: None }
    }
}

/// 具体（Sized）评估器包装：把 `PlayerSpec` 统一为可驱动 Gumbel MCTS 的评估器。
enum PlayerEval<G: GameEnv + AsDarkChessRef> {
    Model(Arc<dyn Evaluator<G> + Send + Sync>),
    #[cfg(feature = "pyo3")]
    Py(Arc<PyEvaluator<G>>),
    Heuristic(HeuristicEval<G>),
    Minimax(MinimaxEval<G>),
    Random(RandomEval),
}

impl<G: GameEnv + AsDarkChessRef + Sync> Evaluator<G> for PlayerEval<G> {
    fn evaluate(&self, envs: &[G]) -> EvaluatorOutput {
        match self {
            PlayerEval::Model(e) => e.evaluate(envs),
            #[cfg(feature = "pyo3")]
            PlayerEval::Py(e) => e.evaluate(envs),
            PlayerEval::Heuristic(e) => e.evaluate(envs),
            PlayerEval::Minimax(e) => e.evaluate(envs),
            PlayerEval::Random(e) => e.evaluate(envs),
        }
    }
}

/// 把 `PlayerSpec` 转成具体评估器（供记录 Episode 的 MCTS 路径）。
fn make_evaluator<G>(spec: &PlayerSpec<G>) -> PlayerEval<G>
where
    G: GameEnv + AsDarkChessRef + Sync,
{
    match spec {
        PlayerSpec::ModelEval(e) => PlayerEval::Model(e.clone()),
        #[cfg(feature = "pyo3")]
        PlayerSpec::PyPredictor(p) => PlayerEval::Py(p.clone()),
        PlayerSpec::Heuristic { .. } => PlayerEval::Heuristic(HeuristicEval {
            params: EvalParams::default(),
            prior_scale: 0.5,
            _marker: PhantomData,
        }),
        PlayerSpec::Minimax { depth } => PlayerEval::Minimax(MinimaxEval {
            depth: *depth,
            lambda: 1.0,
            _marker: PhantomData,
        }),
        PlayerSpec::Random => PlayerEval::Random(RandomEval),
    }
}

// ============================================================================
// 动作选择（非记录 / 评估路径）
// ============================================================================

/// 用 Gumbel MCTS 驱动的模型动作选择。
fn model_mcts_action<G>(env: &G, evaluator: &PlayerEval<G>, sims: usize) -> Option<usize>
where
    G: GameEnv + AsDarkChessRef + Sync,
{
    let config = GumbelConfig {
        num_simulations: sims,
        max_considered_actions: 16,
        c_scale: 0.25,
        gumbel_scale: 1.0,
        ..Default::default()
    };
    let mut mcts = GumbelMCTS::new(env, evaluator, config);
    mcts.run().map(|r| r.action)
}

fn random_action<G: GameEnv>(env: &G) -> Option<usize> {
    let mut masks = vec![0i32; G::action_space_size()];
    env.action_masks_into(&mut masks);
    let legal: Vec<usize> = masks
        .iter()
        .enumerate()
        .filter(|(_, m)| **m == 1)
        .map(|(i, _)| i)
        .collect();
    if legal.is_empty() {
        return None;
    }
    use rand::Rng;
    Some(legal[rand::thread_rng().gen_range(0..legal.len())])
}

/// 按选手类型选择一步动作（评估路径，不记录 Episode）。
fn get_player_action<G>(env: &G, spec: &PlayerSpec<G>, model_sims: usize) -> Option<usize>
where
    G: GameEnv + AsDarkChessRef + Sync,
{
    match spec {
        PlayerSpec::Minimax { depth } => {
            minimax_best_action(env.as_darkchess_ref(), *depth).map(|r| r.action)
        }
        PlayerSpec::Heuristic { sims } => {
            let dark_ref = env.as_darkchess_ref();
            let policy = HeuristicMctsPolicy::new(*sims);
            policy.choose_action(dark_ref)
        }
        PlayerSpec::ModelEval(e) => model_mcts_action(env, &PlayerEval::Model(e.clone()), model_sims),
        #[cfg(feature = "pyo3")]
        PlayerSpec::PyPredictor(p) => {
            model_mcts_action(env, &PlayerEval::Py(p.clone()), model_sims)
        }
        PlayerSpec::Random => random_action(env),
    }
}

// ============================================================================
// 单局驱动
// ============================================================================

/// 单局结果（从选手 A 视角）。`episode` 仅在记录模式下 Some。
struct GameOutcome {
    result: i32,
    moves: usize,
    episode: Option<GameEpisode>,
}

/// 非记录（评估）路径：A vs B 直接推进，不生成 Episode。
#[allow(clippy::too_many_arguments)]
fn play_one_game<G>(
    player_a_spec: &PlayerSpec<G>,
    player_b_spec: &PlayerSpec<G>,
    player_a_is_red: bool,
    model_sims: usize,
    game_seed: Option<u64>,
    make_env: fn() -> G,
) -> GameOutcome
where
    G: GameEnv + AsDarkChessRef + SeedableEnv + Default,
{
    let mut env = make_env();
    if let Some(s) = game_seed {
        env.set_seed(s);
    }
    let mut moves = 0;
    let max_moves = G::max_steps();

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
    GameOutcome {
        result: r,
        moves,
        episode: None,
    }
}

/// 记录（自对弈）路径：每步用当前选手的评估器驱动 Gumbel MCTS，收集完整 Episode。
fn play_one_game_recorded<G>(
    player_a_spec: &PlayerSpec<G>,
    player_b_spec: &PlayerSpec<G>,
    player_a_is_red: bool,
    config: &SelfPlayConfig,
    game_seed: Option<u64>,
    make_env: fn() -> G,
) -> GameOutcome
where
    G: GameEnv + AsDarkChessRef + SeedableEnv + Default,
{
    let mut env = make_env();
    if let Some(s) = game_seed {
        env.set_seed(s);
    }

    let gumbel_cfg = GumbelConfig {
        num_simulations: config.mcts_sims,
        max_considered_actions: config.max_considered_actions,
        c_scale: config.c_scale,
        gumbel_scale: config.gumbel_scale,
        health_enabled: config.health_enabled,
        health_weight: config.health_weight,
        health_confidence_exp: config.health_confidence_exp,
    };

    let mut episode_data = Vec::new();
    let mut step = 0;

    let eval_a = make_evaluator(player_a_spec);
    let eval_b = make_evaluator(player_b_spec);

    loop {
        let cur = env.get_current_player().val();
        let is_a_turn = (cur == 1) == player_a_is_red;
        let evaluator = if is_a_turn { &eval_a } else { &eval_b };

        let is_full_search = if config.playout_cap_random_enabled {
            if step == 0 {
                true
            } else {
                rand::random::<f32>() < config.full_search_prob
            }
        } else {
            true
        };
        let step_sims = if is_full_search {
            config.mcts_sims
        } else {
            config.fast_mcts_sims
        };

        let mut step_gumbel_cfg = gumbel_cfg.clone();
        step_gumbel_cfg.num_simulations = step_sims;

        let mut mcts = GumbelMCTS::new(&env, evaluator, step_gumbel_cfg);
        let search_result = match mcts.run() {
            Some(r) => r,
            None => {
                let (_, _, winner) = env.check_game_over_conditions();
                let ep = finalize_episode(episode_data, winner, env.terminal_health_diff_red());
                return outcome_from_episode(ep, player_a_is_red);
            }
        };

        // 落子：直接采用 Gumbel 搜索选出的动作。探索由每次搜索重新抽的 Gumbel
        // 噪声提供（sample_gumbel_top_k），无需根温度采样，详见 SelfPlayConfig。
        let action = search_result.action;
        let completed_q = search_result.completed_q;

        // 全记录：无论 Full/Fast 都收集样本，并标记 is_full_search，交由 Python 侧
        // 选择性使用（losses.py 仅让 Full Search 样本参与训练）。
        episode_data.push((
            search_result.state,
            search_result.improved_policy,
            search_result.mcts_value,
            completed_q,
            search_result.root_visit_count,
            search_result.player,
            search_result.action_mask,
            action,
            is_full_search,
        ));

        match env.step(action) {
            Ok((_, terminated, truncated, winner)) => {
                if terminated || truncated {
                    let ep = finalize_episode(episode_data, winner, env.terminal_health_diff_red());
                    return outcome_from_episode(ep, player_a_is_red);
                }
            }
            Err(e) => {
                eprintln!("⚠️ 记录对局 step 错误: {e}");
                let ep = GameEpisode {
                    samples: Vec::new(),
                    game_length: step,
                    winner: None,
                    health_diff_red: env.terminal_health_diff_red(),
                };
                return outcome_from_episode(ep, player_a_is_red);
            }
        }

        step += 1;
        if step >= G::max_steps() {
            let ep = finalize_episode(episode_data, Some(0), env.terminal_health_diff_red());
            return outcome_from_episode(ep, player_a_is_red);
        }
    }
}

/// 从 Episode 推导选手 A 视角结果与步数。
fn outcome_from_episode(ep: GameEpisode, player_a_is_red: bool) -> GameOutcome {
    let moves = ep.game_length;
    let r = match ep.winner {
        Some(1) => {
            if player_a_is_red { 1 } else { -1 }
        }
        Some(-1) => {
            if player_a_is_red { -1 } else { 1 }
        }
        _ => 0,
    };
    GameOutcome {
        result: r,
        moves,
        episode: Some(ep),
    }
}

// ============================================================================
// 统一主干 run_match_core
// ============================================================================

/// 多局比赛参数。
pub struct MatchParams<'a, G: GameEnv> {
    pub player_a: &'a PlayerSpec<G>,
    pub player_b: &'a PlayerSpec<G>,
    pub n_games: usize,
    pub config: &'a SelfPlayConfig,
    pub seed: Option<u64>,
    /// 是否记录完整 Episode（自对弈数据）；false 时仅收集胜负统计（评估）。
    pub record_episodes: bool,
    /// 模型选手的 MCTS 模拟数（评估路径）。
    pub model_sims: usize,
    /// 线程池：Some = 原生多线程（Rust 持模型 / 规则），None = 单线程（Python 推理）。
    pub thread_pool: Option<&'a rayon::ThreadPool>,
    pub make_env: fn() -> G,
}

/// 多局比赛结果。
pub struct MatchResult {
    pub episodes: Vec<GameEpisode>,
    pub wins: usize,
    pub draws: usize,
    pub losses: usize,
    pub block_wr: Vec<f32>,
    pub avg_moves: f32,
}

/// 统一对局主干：驱动 `n_games` 局 A vs B。
pub fn run_match_core<G>(params: MatchParams<'_, G>) -> MatchResult
where
    G: GameEnv + AsDarkChessRef + SeedableEnv + Send + Sync + Default + 'static,
{
    let n = params.n_games;
    let indices: Vec<usize> = (0..n).collect();

    let play = |i: usize| -> GameOutcome {
        let player_a_is_red = (i % 2) == 0;
        let game_seed = params.seed.map(|s| s.wrapping_add(i as u64));
        if params.record_episodes {
            play_one_game_recorded(
                params.player_a,
                params.player_b,
                player_a_is_red,
                params.config,
                game_seed,
                params.make_env,
            )
        } else {
            play_one_game(
                params.player_a,
                params.player_b,
                player_a_is_red,
                params.model_sims,
                game_seed,
                params.make_env,
            )
        }
    };

    let games: Vec<GameOutcome> = match params.thread_pool {
        Some(pool) => pool.install(|| indices.into_par_iter().map(play).collect()),
        None => indices.into_iter().map(play).collect(),
    };

    let mut wins = 0;
    let mut draws = 0;
    let mut losses = 0;
    let mut total_moves = 0;
    let mut episodes = Vec::new();
    let block_size = 20;
    let mut block_wr = Vec::new();
    let mut blk_w = 0;
    let mut blk_tot = 0;

    for (i, g) in games.iter().enumerate() {
        total_moves += g.moves;
        if g.result == 1 {
            wins += 1;
            blk_w += 1;
        } else if g.result == -1 {
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
        if let Some(ep) = &g.episode {
            if !ep.samples.is_empty() {
                episodes.push(ep.clone());
            }
        }
    }
    if blk_tot > 0 && n % block_size != 0 {
        block_wr.push(100.0 * (blk_w as f32) / (blk_tot as f32));
    }

    let avg_moves = (total_moves as f32) / (n.max(1) as f32);
    MatchResult {
        episodes,
        wins,
        draws,
        losses,
        block_wr,
        avg_moves,
    }
}
