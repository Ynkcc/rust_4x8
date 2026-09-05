//! Expecti-Alpha-Beta 主搜索
//!
//! 包含 Star1 期望值概率节点剪枝、置换表 (TT)、静态搜索 (Quiescence)、
//! 晚走子减深 (LMR)、重复局面检测与迭代加深。

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use crate::core::env::DarkChessEnv;
use crate::core::env::symmetry::{Symmetry, search_group};
use crate::engine::movegen::generate_moves;
use crate::inference::nnue::{DualAccumulator, NnueEvaluator};

use super::ordering;
use super::smp::SharedTT;
use super::zobrist;
use super::{Ctx, FEAT_LMR, FEAT_ORDERING, FEAT_REP, FEAT_TT};

pub use zobrist::{INF, TT_EMPTY, TtEntry, VMAX, VMIN};

/// 搜索引擎配置
#[derive(Clone, Debug)]
pub struct SearchConfig {
    /// 节点预算（总节点数上限；超出即中止当前迭代）
    pub node_budget: u64,
    /// 时间预算（毫秒；0 = 仅按节点预算）
    pub time_limit_ms: u64,
    /// 迭代加深最大深度
    pub max_depth: i32,
    /// 和棋偏差（contempt；正数 = 领先方避和、落后方求和）
    pub contempt: f32,
    /// 是否启用静态搜索
    pub quiesce: bool,
    /// 静态搜索最大深度
    pub quiesce_max: i32,
    /// 特性位掩码（FEAT_*）
    pub features: u32,
    /// 置换表大小（2^tt_bits 项）
    pub tt_bits: u32,
    /// Lazy SMP 并发线程数（1 = 单线程，>1 时共享置换表协同搜索）
    pub threads: usize,
    /// 量化开关：TT 原始键 miss 后用对称视角键二次探测并统计（只计数，不参与存储/截断）
    pub tt_sym_probe: bool,
    /// 可选 NNUE 求值网络引擎
    pub nnue_evaluator: Option<Arc<NnueEvaluator>>,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            node_budget: 500_000,
            time_limit_ms: 0,
            max_depth: 24,
            contempt: 0.1,
            quiesce: true,
            quiesce_max: 8,
            features: FEAT_ORDERING | FEAT_TT | FEAT_LMR | FEAT_REP,
            tt_bits: 18,
            threads: 1,
            tt_sym_probe: false,
            nnue_evaluator: None,
        }
    }
}

impl SearchConfig {
    #[inline]
    pub(super) fn feat(&self, bit: u32) -> bool {
        self.features & bit != 0
    }
}

/// 搜索引擎评估与搜索结果
#[derive(Debug, Clone, Copy)]
pub struct SearchResult {
    pub action: usize,
    /// 根走子方视角的评估值（最深完成迭代）
    pub value: f32,
    /// 完成的最深迭代层数
    pub depth: i32,
    /// 消耗的总节点数
    pub nodes: u64,
}

/// 叶节点局面评估入口（NNUE 为唯一评估来源）
#[inline]
pub fn eval_state(env: &DarkChessEnv, cfg: &SearchConfig) -> f32 {
    match cfg.nnue_evaluator.as_ref() {
        Some(nnue) => nnue.evaluate(env),
        None => 0.0,
    }
}

/// 基于双累加器的 O(1) 叶节点评估（当前行棋方视角）。
#[inline]
fn eval_acc(env: &DarkChessEnv, acc: &DualAccumulator, cfg: &SearchConfig) -> f32 {
    match cfg.nnue_evaluator.as_ref() {
        Some(nnue) => nnue.forward_accumulator(acc.get(env.get_current_player())),
        None => eval_state(env, cfg),
    }
}

/// 在父局面上执行一步动作，生成子局面并增量更新双累加器。
#[inline]
fn step_with_acc(
    env: &DarkChessEnv,
    acc: &DualAccumulator,
    action: usize,
    child: &mut DarkChessEnv,
    nnue: Option<&NnueEvaluator>,
) -> DualAccumulator {
    let before = *env;
    let _ = child.step(action, None);
    match nnue {
        Some(nnue) => {
            let mut child_acc = *acc;
            let (diff_red, diff_black) =
                crate::inference::nnue::compute_step_diff(&before, child, action);
            child_acc.apply_diffs(&diff_red, &diff_black, nnue);
            child_acc
        }
        None => *acc,
    }
}

/// 机会节点结果局面的双累加器（结果环境由 chance_outcomes 产生）。
#[inline]
fn outcome_acc(
    env: &DarkChessEnv,
    acc: &DualAccumulator,
    action: usize,
    next_env: &DarkChessEnv,
    nnue: Option<&NnueEvaluator>,
) -> DualAccumulator {
    match nnue {
        Some(nnue) => {
            let mut child_acc = *acc;
            let (diff_red, diff_black) =
                crate::inference::nnue::compute_step_diff(env, next_env, action);
            child_acc.apply_diffs(&diff_red, &diff_black, nnue);
            child_acc
        }
        None => *acc,
    }
}

/// 静态搜索：仅延展吃明子走法。
fn quiesce(
    env: &DarkChessEnv,
    acc: &DualAccumulator,
    mut alpha: f32,
    beta: f32,
    cfg: &SearchConfig,
    ctx: &mut Ctx,
    qdepth: i32,
) -> Result<f32, ()> {
    ctx.tick()?;
    let moves = generate_moves(env, env.get_current_player());
    if let Some(winner) = ordering::terminal_info(env, &moves) {
        return Ok(ordering::terminal_value(env, Some(winner), cfg, ctx));
    }
    let stand = eval_acc(env, acc, cfg);
    if stand >= beta || qdepth <= 0 {
        return Ok(stand);
    }
    if stand > alpha {
        alpha = stand;
    }
    let mut caps: Vec<(i32, crate::engine::movegen::Move)> = moves
        .iter()
        .filter(|m| m.is_capture)
        .map(|&m| (ordering::victim_value(env, &m), m))
        .collect();
    caps.sort_by(|a, b| b.0.cmp(&a.0));
    let nnue = cfg.nnue_evaluator.as_deref();
    let mut best = stand;
    for (_, m) in caps {
        let mut child = *env;
        let child_acc = step_with_acc(env, acc, m.action, &mut child, nnue);
        let v = -quiesce(&child, &child_acc, -beta, -alpha, cfg, ctx, qdepth - 1)?;
        if v > best {
            best = v;
        }
        if best > alpha {
            alpha = best;
        }
        if alpha >= beta {
            break;
        }
    }
    Ok(best)
}

/// Star1 机会节点：按概率加权期望值，用区间边界做剪枝。
fn flip_value(
    env: &DarkChessEnv,
    acc: &DualAccumulator,
    action: usize,
    depth: i32,
    alpha: f32,
    beta: f32,
    cfg: &SearchConfig,
    ctx: &mut Ctx,
) -> Result<f32, ()> {
    let outcomes = env.chance_outcomes(action);
    if outcomes.is_empty() {
        return Ok(0.0);
    }
    let nnue = cfg.nnue_evaluator.as_deref();
    let (l, u) = (VMIN, VMAX);
    let mut vsum = 0.0f32;
    let mut rem = 1.0f32;
    for (_, p, next_env) in outcomes {
        rem -= p;
        if rem < 0.0 {
            rem = 0.0;
        }
        let ai = (alpha - vsum - rem * u) / p;
        let bi = (beta - vsum - rem * l) / p;
        if ai >= u {
            return Ok(alpha);
        }
        if bi <= l {
            return Ok(beta);
        }
        let cl = if ai > l { ai } else { l };
        let cu = if bi < u { bi } else { u };
        let next_acc = outcome_acc(env, acc, action, &next_env, nnue);
        let v = -negamax(&next_env, &next_acc, depth - 1, -cu, -cl, cfg, ctx)?;
        if v <= ai {
            return Ok(alpha);
        }
        if v >= bi {
            return Ok(beta);
        }
        vsum += p * v;
    }
    Ok(vsum)
}

/// 单条走子的值。
fn move_value(
    env: &DarkChessEnv,
    acc: &DualAccumulator,
    action: usize,
    depth: i32,
    alpha: f32,
    beta: f32,
    cfg: &SearchConfig,
    ctx: &mut Ctx,
) -> Result<f32, ()> {
    if env.is_chance_action(action) {
        return flip_value(env, acc, action, depth, alpha, beta, cfg, ctx);
    }
    let nnue = cfg.nnue_evaluator.as_deref();
    let mut child = *env;
    let child_acc = step_with_acc(env, acc, action, &mut child, nnue);
    Ok(-negamax(&child, &child_acc, depth - 1, -beta, -alpha, cfg, ctx)?)
}

fn negamax(
    env: &DarkChessEnv,
    acc: &DualAccumulator,
    depth: i32,
    mut alpha: f32,
    beta: f32,
    cfg: &SearchConfig,
    ctx: &mut Ctx,
) -> Result<f32, ()> {
    ctx.tick()?;
    let moves = generate_moves(env, env.get_current_player());
    if let Some(winner) = ordering::terminal_info(env, &moves) {
        return Ok(ordering::terminal_value(env, Some(winner), cfg, ctx));
    }
    if depth <= 0 {
        if cfg.quiesce {
            return quiesce(env, acc, alpha, beta, cfg, ctx, cfg.quiesce_max);
        }
        return Ok(eval_acc(env, acc, cfg));
    }
    let alpha_orig = alpha;
    let key = if cfg.feat(FEAT_TT) || cfg.feat(FEAT_REP) {
        zobrist::zkey(env)
    } else {
        0
    };

    // 重复检测：静走循环会产生相同 zkey（吃子/翻棋改变袋或棋盘 → 键不同）。
    if cfg.feat(FEAT_REP) && ctx.path.iter().any(|&k| k == key) {
        return Ok(ordering::terminal_value(env, Some(0), cfg, ctx));
    }

    // 置换表探测（决策节点；SharedTT 原子读，跨线程安全）
    let mut tt_move: Option<usize> = None;
    if cfg.feat(FEAT_TT) {
        match ctx.tt.probe(key) {
            Some((value, e_depth, flag, best_hint)) => {
                ctx.tt.bump(1);
                if e_depth >= depth {
                    match flag {
                        1 => return Ok(value), // exact
                        2 => {
                            if value >= beta {
                                return Ok(value);
                            }
                        }
                        3 => {
                            if value <= alpha {
                                return Ok(value);
                            }
                        }
                        _ => {}
                    }
                }
                tt_move = Some(best_hint as usize);
            }
            None => {
                // 对称合并量化：原始键 miss 后用对称视角键二次探测（只统计，不参与截断）。
                if cfg.tt_sym_probe {
                    ctx.tt.bump(0);
                    for &sym in search_group(env.config.rows, env.config.cols) {
                        if sym == Symmetry::Identity {
                            continue;
                        }
                        if let Some((_, e_depth, _, _)) =
                            ctx.tt.probe(zobrist::sym_zkey(env, sym))
                        {
                            ctx.tt.bump(2);
                            if e_depth >= depth {
                                ctx.tt.bump(3);
                            }
                            break;
                        }
                    }
                }
            }
        }
    }

    let mut ordered = moves;
    ordering::order_moves(env, &mut ordered, depth, cfg, ctx);
    if let Some(tm) = tt_move {
        if let Some(pos) = ordered.iter().position(|m| m.action == tm) {
            let m = ordered.remove(pos);
            ordered.insert(0, m);
        }
    }
    if cfg.feat(FEAT_REP) {
        ctx.path.push(key);
    }

    let mut best = -INF;
    let mut best_m = ordered[0].action;
    let nnue = cfg.nnue_evaluator.as_deref();
    for (i, &m) in ordered.iter().enumerate() {
        let quiet = !m.is_chance && !m.is_capture;
        let v = if cfg.feat(FEAT_LMR) && quiet && i >= 3 && depth >= 3 {
            let mut child = *env;
            let child_acc = step_with_acc(env, acc, m.action, &mut child, nnue);
            let probe = -negamax(&child, &child_acc, depth - 2, -alpha - 1e-6, -alpha, cfg, ctx)?;
            if probe > alpha {
                -negamax(&child, &child_acc, depth - 1, -beta, -alpha, cfg, ctx)?
            } else {
                probe
            }
        } else {
            move_value(env, acc, m.action, depth, alpha, beta, cfg, ctx)?
        };
        if v > best {
            best = v;
            best_m = m.action;
        }
        if best > alpha {
            alpha = best;
        }
        if alpha >= beta {
            if cfg.feat(FEAT_ORDERING) && quiet {
                ctx.record_cutoff(&m, depth, env.config.total_positions);
            }
            break;
        }
    }
    if cfg.feat(FEAT_REP) {
        ctx.path.pop();
    }

    // 置换表存储（深度优先替换；last-write-wins，跨线程安全）
    if cfg.feat(FEAT_TT) {
        let flag = if best <= alpha_orig {
            3 // fail-low → 上界
        } else if best >= beta {
            2 // fail-high → 下界
        } else {
            1 // exact
        };
        ctx.tt.store_cond(
            key,
            &TtEntry {
                key,
                value: best,
                depth: depth as i16,
                flag,
                best: best_m,
            },
        );
    }
    Ok(best)
}

/// 单层根搜索：返回 (最优动作, 根走子方视角值)。
fn best_at_depth(
    env: &DarkChessEnv,
    acc: &DualAccumulator,
    depth: i32,
    cfg: &SearchConfig,
    ctx: &mut Ctx,
    hint: Option<usize>,
) -> Result<Option<(usize, f32)>, ()> {
    let mut moves = generate_moves(env, env.get_current_player());
    if moves.is_empty() {
        return Ok(None);
    }
    ordering::order_moves(env, &mut moves, depth, cfg, ctx);
    if let Some(h) = hint {
        if let Some(pos) = moves.iter().position(|m| m.action == h) {
            let m = moves.remove(pos);
            moves.insert(0, m);
        }
    }
    let mut best_val = -INF;
    let mut best = None;
    let mut alpha = VMIN;
    for &m in &moves {
        let v = move_value(env, acc, m.action, depth, alpha, VMAX, cfg, ctx)?;
        if v > best_val {
            best_val = v;
            best = Some(m.action);
            if v > alpha {
                alpha = v;
            }
        }
    }
    Ok(best.map(|a| (a, best_val)))
}

/// 节点/时间预算驱动的迭代加深 Expectimax 搜索。返回 `None` 表示无合法动作（终局）。
pub fn search(env: &DarkChessEnv, cfg: &SearchConfig) -> Option<SearchResult> {
    let shared = Arc::new(SharedTT::new(cfg.tt_bits));
    search_with_tt(env, cfg, shared)
}

/// Lazy SMP 并发搜索：`threads <= 1` 时等价 `search`；>1 时主线程迭代加深，
/// 助线程以独立上下文共享置换表协同搜索，主线程结果为准。
pub fn search_par(env: &DarkChessEnv, cfg: &SearchConfig) -> Option<SearchResult> {
    let threads = cfg.threads.max(1);
    if threads == 1 {
        return search(env, cfg);
    }
    if let Some(nnue) = &cfg.nnue_evaluator {
        if let Err(msg) = nnue.validate_feature_dim(env.config.nnue_feature_dim()) {
            eprintln!("❌ {msg}");
            return None;
        }
    }
    let moves = generate_moves(env, env.get_current_player());
    if moves.is_empty() {
        return None;
    }
    let shared = Arc::new(SharedTT::new(cfg.tt_bits));
    let stop = Arc::new(AtomicBool::new(false));

    std::thread::scope(|scope| {
        for _ in 1..threads {
            let shared = Arc::clone(&shared);
            let stop = Arc::clone(&stop);
            let helper_env = *env;
            let helper_cfg = cfg.clone();
            scope.spawn(move || {
                // 助线程独立迭代加深，仅通过共享 TT 贡献信息，结果被丢弃。
                let budget = (helper_cfg.node_budget / threads as u64).max(1024);
                let mut helper_cfg = helper_cfg;
                helper_cfg.node_budget = budget;
                let root_acc = match helper_cfg.nnue_evaluator.as_ref() {
                    Some(nnue) => DualAccumulator::init_from_env(&helper_env, nnue),
                    None => DualAccumulator::default(),
                };
                let mut ctx = Ctx::with_tt(&helper_cfg, &helper_env, shared);
                if helper_cfg.feat(FEAT_REP) {
                    ctx.path.push(zobrist::zkey(&helper_env));
                }
                let mut hint: Option<usize> = None;
                for depth in 1..=helper_cfg.max_depth {
                    if stop.load(Ordering::Relaxed) {
                        break;
                    }
                    match best_at_depth(&helper_env, &root_acc, depth, &helper_cfg, &mut ctx, hint) {
                        Ok(Some((a, _))) => hint = Some(a),
                        _ => break,
                    }
                }
            });
        }
        let result = search_with_tt(env, cfg, Arc::clone(&shared));
        stop.store(true, Ordering::Relaxed);
        result
    })
}

/// 以指定共享置换表执行迭代加深主搜索。
fn search_with_tt(
    env: &DarkChessEnv,
    cfg: &SearchConfig,
    shared: Arc<SharedTT>,
) -> Option<SearchResult> {
    if let Some(nnue) = &cfg.nnue_evaluator {
        if let Err(msg) = nnue.validate_feature_dim(env.config.nnue_feature_dim()) {
            eprintln!("❌ {msg}");
            return None;
        }
    }
    let moves = generate_moves(env, env.get_current_player());
    if moves.is_empty() {
        return None;
    }
    let root_acc = match cfg.nnue_evaluator.as_ref() {
        Some(nnue) => DualAccumulator::init_from_env(env, nnue),
        None => DualAccumulator::default(),
    };
    let mut ctx = Ctx::with_tt(cfg, env, shared);
    if cfg.feat(FEAT_REP) {
        ctx.path.push(zobrist::zkey(env));
    }
    let mut best = moves[0].action;
    let mut best_score = 0.0f32;
    let mut hint: Option<usize> = None;
    let mut depth_reached = 0;
    for depth in 1..=cfg.max_depth {
        match best_at_depth(env, &root_acc, depth, cfg, &mut ctx, hint) {
            Ok(Some((a, v))) => {
                best = a;
                best_score = v;
                hint = Some(a);
                depth_reached = depth;
            }
            _ => break,
        }
    }
    if cfg.tt_sym_probe && cfg.feat(FEAT_TT) {
        let [misses, raw_hits, sym_hits, sym_deep] = ctx.tt.tt_stats();
        let rate = if misses > 0 { sym_hits as f64 / misses as f64 } else { 0.0 };
        eprintln!(
            "[tt-sym-quant] 探测节点={} 原始命中={raw_hits} miss={misses} 对称命中={sym_hits} 对称命中(深度足够)={sym_deep} miss挽回率={rate:.1}%",
            misses + raw_hits
        );
    }
    Some(SearchResult {
        action: best,
        value: best_score,
        depth: depth_reached,
        nodes: ctx.nodes,
    })
}
