//! 核心 Expecti-Alpha-Beta 搜索引擎
//!
//! 包含 Star1 期望值概率节点剪枝、置换表 (TT)、静态搜索 (Quiescence) 与走子排序。

use std::sync::Arc;
use std::time::Instant;

use crate::core::env::DarkChessEnv;
use crate::inference::nnue::NnueEvaluator;

use super::ordering;
use super::zobrist;
use super::zobrist::TtFlag;
use crate::core::zobrist::zkey;

pub use zobrist::{TT_EMPTY, TtEntry, VMAX, VMIN, INF};


/// 搜索引擎配置
#[derive(Clone, Debug)]
pub struct SearchConfig {
    pub node_budget: u64,
    pub time_limit_ms: u64,
    pub max_depth: i32,
    pub contempt: f32,
    pub quiesce: bool,
    pub quiesce_max: i32,
    pub tt_bits: u32,
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
            tt_bits: 18,
            nnue_evaluator: None,
        }
    }
}

/// 搜索引擎评估与搜索结果
#[derive(Debug, Clone, Copy)]
pub struct SearchResult {
    pub action: usize,
    pub value: f32,
    pub depth: i32,
    pub nodes: u64,
}

struct SearchCtx {
    nodes: u64,
    budget: u64,
    start: Instant,
    time_limit_ms: u64,
    tt: Vec<TtEntry>,
    tt_mask: usize,
}

impl SearchCtx {
    fn new(cfg: &SearchConfig) -> Self {
        let tt_size = 1usize << cfg.tt_bits;
        Self {
            nodes: 0,
            budget: cfg.node_budget.max(1),
            start: Instant::now(),
            time_limit_ms: cfg.time_limit_ms,
            tt: vec![TT_EMPTY; tt_size],
            tt_mask: tt_size.wrapping_sub(1),
        }
    }

    /// 探测置换表：键匹配且非空时返回表项。
    #[inline]
    fn probe(&self, key: u64) -> Option<TtEntry> {
        let e = self.tt[key as usize & self.tt_mask];
        if e.depth >= 0 && e.key == key {
            Some(e)
        } else {
            None
        }
    }

    /// 写入置换表（depth 优先替换：空槽 / 同键 / 更深搜索覆盖）。
    #[inline]
    fn store(&mut self, entry: TtEntry) {
        let idx = entry.key as usize & self.tt_mask;
        let cur = self.tt[idx];
        if cur.depth < 0 || cur.key == entry.key || entry.depth >= cur.depth {
            self.tt[idx] = entry;
        }
    }

    #[inline]
    fn tick(&mut self) -> Result<(), ()> {
        self.nodes += 1;
        if self.nodes > self.budget {
            return Err(());
        }
        if self.time_limit_ms > 0
            && (self.nodes & 1023) == 0
            && self.start.elapsed().as_millis() as u64 >= self.time_limit_ms
        {
            return Err(());
        }
        Ok(())
    }
}

#[inline]
fn eval_state(env: &DarkChessEnv, cfg: &SearchConfig) -> f32 {
    if let Some(ref nnue) = cfg.nnue_evaluator {
        nnue.evaluate(env)
    } else {
        0.0
    }
}

/// 静态搜索
fn quiesce(
    env: &DarkChessEnv,
    mut alpha: f32,
    beta: f32,
    cfg: &SearchConfig,
    ctx: &mut SearchCtx,
    qdepth: i32,
) -> Result<f32, ()> {
    ctx.tick()?;
    let stand = eval_state(env, cfg);
    if stand >= beta || qdepth <= 0 {
        return Ok(stand);
    }
    if stand > alpha {
        alpha = stand;
    }
    Ok(alpha)
}

/// Star1 机会节点概率加权剪枝
fn flip_value(
    env: &DarkChessEnv,
    action: usize,
    depth: i32,
    alpha: f32,
    beta: f32,
    cfg: &SearchConfig,
    ctx: &mut SearchCtx,
) -> Result<f32, ()> {
    let outcomes = env.chance_outcomes(action);
    if outcomes.is_empty() {
        return Ok(0.0);
    }
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
        let v = -negamax(&next_env, depth - 1, -cu, -cl, cfg, ctx)?;
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

fn move_value(
    env: &DarkChessEnv,
    action: usize,
    depth: i32,
    alpha: f32,
    beta: f32,
    cfg: &SearchConfig,
    ctx: &mut SearchCtx,
) -> Result<f32, ()> {
    if env.is_chance_action(action) {
        return flip_value(env, action, depth, alpha, beta, cfg, ctx);
    }
    let mut child = *env;
    let _ = child.step(action, None);
    Ok(-negamax(&child, depth - 1, -beta, -alpha, cfg, ctx)?)
}

fn negamax(
    env: &DarkChessEnv,
    depth: i32,
    mut alpha: f32,
    beta: f32,
    cfg: &SearchConfig,
    ctx: &mut SearchCtx,
) -> Result<f32, ()> {
    ctx.tick()?;
    let legal = env.legal_action_indices();
    if legal.is_empty() {
        return Ok(ordering::terminal_value(env, Some(0), cfg.contempt));
    }
    if depth <= 0 {
        if cfg.quiesce {
            return quiesce(env, alpha, beta, cfg, ctx, cfg.quiesce_max);
        }
        return Ok(eval_state(env, cfg));
    }

    let mut best = -INF;
    let mut best_act = legal[0];

    // --- 置换表探测（仅决策节点；机会节点期望值经 Star1 截断后仍为可靠界，
    //     Exact/Lower/Upper 语义在全树保持成立，故此处可安全剪枝） ---
    let key = zkey(env);
    let tt_hit = ctx.probe(key);
    if let Some(e) = tt_hit {
        if i32::from(e.depth) >= depth {
            match TtFlag::from_u8(e.flag) {
                Some(TtFlag::Exact) => return Ok(e.value),
                Some(TtFlag::LowerBound) if e.value >= beta => return Ok(e.value),
                Some(TtFlag::UpperBound) if e.value <= alpha => return Ok(e.value),
                _ => {}
            }
        }
    }

    // 走子排序：TT 最佳着法置于首位
    let mut ordered = legal;
    if let Some(e) = tt_hit {
        if let Some(pos) = ordered.iter().position(|&a| a as u16 == e.best_action) {
            ordered.swap(0, pos);
        }
    }

    let alpha_orig = alpha;
    for &act in &ordered {
        let v = move_value(env, act, depth, alpha, beta, cfg, ctx)?;
        if v > best {
            best = v;
            best_act = act;
        }
        if best > alpha {
            alpha = best;
        }
        if alpha >= beta {
            break;
        }
    }

    // --- 置换表写入：fail-soft 值按界方向归类 ---
    let flag = if best <= alpha_orig {
        TtFlag::UpperBound
    } else if best >= beta {
        TtFlag::LowerBound
    } else {
        TtFlag::Exact
    };
    ctx.store(TtEntry {
        key,
        depth: depth as i8,
        flag: flag as u8,
        value: best,
        best_action: best_act as u16,
    });

    Ok(best)
}

/// 执行 Expectimax 搜索并产出最佳走子
pub fn search(env: &DarkChessEnv, cfg: &SearchConfig) -> Option<SearchResult> {
    let legal = env.legal_action_indices();
    if legal.is_empty() {
        return None;
    }
    let mut ctx = SearchCtx::new(cfg);
    let mut best_action = legal[0];
    let mut best_val = -INF;

    for depth in 1..=cfg.max_depth {
        let mut alpha = VMIN;
        let mut depth_best_act = legal[0];
        let mut depth_best_val = -INF;

        for &act in &legal {
            if let Ok(v) = move_value(env, act, depth, alpha, VMAX, cfg, &mut ctx) {
                if v > depth_best_val {
                    depth_best_val = v;
                    depth_best_act = act;
                    if v > alpha {
                        alpha = v;
                    }
                }
            } else {
                break;
            }
        }
        if depth_best_val > -INF {
            best_action = depth_best_act;
            best_val = depth_best_val;
        }
    }

    Some(SearchResult {
        action: best_action,
        value: best_val,
        depth: cfg.max_depth,
        nodes: ctx.nodes,
    })
}
