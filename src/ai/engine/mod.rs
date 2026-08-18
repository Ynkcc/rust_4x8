// src/ai/engine/mod.rs
// 纯计算强引擎：αβ + Star1（机会节点区间剪枝）+ 置换表 + 静态搜索 + 走子排序
// + 迭代加深（节点预算 / 时间预算）。
//
// 移植自 misty-banqi 引擎的搜索骨架，适配本项目 DarkChessEnv：
//   - 机会节点（翻棋 / 吃暗子）用 Star1 期望值区间剪枝替代全量展开；
//   - 置换表只在“决策节点”探测/存储（含机会子树的期望值下界/上界标志）；
//   - 走子排序：MVV-LVA（吃子）+ 杀手走 + 历史启发 + 翻棋垫底；
//   - 静态搜索：深度耗尽后仅延展吃明子的走法；
//   - 迭代加深 + 节点预算/时间预算控制，根层 hint 传递上一深度最优动作。
//
// 子模块分层：
//   - zobrist:  Zobrist 哈希 + 置换表 TtEntry + 暗子袋计数
//   - ordering: 走子排序 + 终局检测/价值 + 根层送子检测
//   - tests:    单元测试
//
// 值约定：所有搜索值均为“当前节点走子方视角”，范围约 [-1, 1]。

use std::time::Instant;

use crate::DarkChessEnv;

use super::eval::{EvalParams, evaluate_for};
use super::movegen::generate_moves;

pub use zobrist::{TtEntry, TT_EMPTY, VMIN, VMAX, INF};

mod ordering;
mod tests;
pub mod zobrist;

// 搜索特性标志位
pub const FEAT_ORDERING: u32 = 1 << 0; // 走子排序（MVV-LVA + 杀手 + 历史）
pub const FEAT_TT: u32 = 1 << 1; // 置换表（决策节点）
pub const FEAT_LMR: u32 = 1 << 2; // 晚走子减深（late move reductions）
pub const FEAT_REP: u32 = 1 << 3; // 重复局面检测（路径 zkey）
pub const FEAT_NO_DRAW_SAC: u32 = 1 << 4; // 根层禁止“为躲和棋送子”

/// 正交方向（用于送子检测）。
pub(crate) const ORTHO: [(i32, i32); 4] = [(1, 0), (-1, 0), (0, 1), (0, -1)];

/// 引擎配置。
#[derive(Clone, Copy, Debug)]
pub struct EngineConfig {
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
    /// 评估参数
    pub params: EvalParams,
    /// 特性位掩码（FEAT_*）
    pub features: u32,
    /// 置换表大小（2^tt_bits 项）
    pub tt_bits: u32,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            node_budget: 500_000,
            time_limit_ms: 0,
            max_depth: 24,
            contempt: 0.1,
            quiesce: true,
            quiesce_max: 8,
            params: EvalParams::default(),
            features: FEAT_ORDERING | FEAT_TT | FEAT_LMR | FEAT_REP | FEAT_NO_DRAW_SAC,
            tt_bits: 18,
        }
    }
}

/// 单次搜索的结果。
#[derive(Debug, Clone, Copy)]
pub struct EngineResult {
    pub action: usize,
    /// 根走子方视角的评估值（最深完成迭代）
    pub value: f32,
    /// 完成的最深迭代层数
    pub depth: i32,
    /// 消耗的总节点数
    pub nodes: u64,
}

impl EngineConfig {
    fn feat(&self, bit: u32) -> bool {
        self.features & bit != 0
    }
}

/// 搜索上下文。
pub(crate) struct Ctx {
    nodes: u64,
    budget: u64,
    start: Instant,
    time_limit_ms: u64,
    killers: Vec<[usize; 2]>, // 每个剩余深度的两个杀手动作
    history: Vec<i32>,        // [total_positions*total_positions] 静走子截断历史
    tt: Vec<TtEntry>,
    tt_mask: usize,
    path: Vec<u64>, // 当前搜索路径上的祖先 zkey（重复检测）
    root: usize,    // 根走子方 idx（contempt 方向）
}

impl Ctx {
    fn new(cfg: &EngineConfig, env: &DarkChessEnv) -> Self {
        let total = env.config.total_positions;
        let kd = (cfg.max_depth.max(1) + 2) as usize;
        let tt_size = if cfg.feat(FEAT_TT) {
            1usize << cfg.tt_bits
        } else {
            0
        };
        Self {
            nodes: 0,
            budget: cfg.node_budget.max(1),
            start: Instant::now(),
            time_limit_ms: cfg.time_limit_ms,
            killers: vec![[0; 2]; kd],
            history: vec![0; total * total],
            tt: vec![TT_EMPTY; tt_size],
            tt_mask: tt_size.wrapping_sub(1),
            path: Vec::with_capacity(64),
            root: env.get_current_player().idx(),
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

    /// 记录静走子截断：提升为杀手走并累加历史分（深度²）。
    #[inline]
    fn record_cutoff(&mut self, m: &crate::ai::movegen::Move, depth: i32, total: usize) {
        let d = depth as usize;
        if d < self.killers.len() && self.killers[d][0] != m.action {
            self.killers[d][1] = self.killers[d][0];
            self.killers[d][0] = m.action;
        }
        let key = m.from * total + m.to;
        if key < self.history.len() {
            self.history[key] += depth * depth;
        }
    }
}

/// 静态搜索：仅延展吃明子走法。
fn quiesce(
    env: &DarkChessEnv,
    mut alpha: f32,
    beta: f32,
    cfg: &EngineConfig,
    ctx: &mut Ctx,
    qdepth: i32,
) -> Result<f32, ()> {
    ctx.tick()?;
    let moves = generate_moves(env, env.get_current_player());
    if let Some(winner) = ordering::terminal_info(env, &moves) {
        return Ok(ordering::terminal_value(env, Some(winner), cfg, ctx));
    }
    let stand = evaluate_for(env, env.get_current_player(), &cfg.params);
    if stand >= beta || qdepth <= 0 {
        return Ok(stand);
    }
    if stand > alpha {
        alpha = stand;
    }
    let mut caps: Vec<(i32, crate::ai::movegen::Move)> = moves
        .iter()
        .filter(|m| m.is_capture)
        .map(|&m| (ordering::victim_value(env, &m, cfg) as i32, m))
        .collect();
    caps.sort_by(|a, b| b.0.cmp(&a.0));
    let mut best = stand;
    for (_, m) in caps {
        let mut child = *env;
        let _ = child.step(m.action, None);
        let v = -quiesce(&child, -beta, -alpha, cfg, ctx, qdepth - 1)?;
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
    action: usize,
    depth: i32,
    alpha: f32,
    beta: f32,
    cfg: &EngineConfig,
    ctx: &mut Ctx,
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

/// 单条走子的值。
fn move_value(
    env: &DarkChessEnv,
    action: usize,
    depth: i32,
    alpha: f32,
    beta: f32,
    cfg: &EngineConfig,
    ctx: &mut Ctx,
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
    cfg: &EngineConfig,
    ctx: &mut Ctx,
) -> Result<f32, ()> {
    ctx.tick()?;
    let moves = generate_moves(env, env.get_current_player());
    if let Some(winner) = ordering::terminal_info(env, &moves) {
        return Ok(ordering::terminal_value(env, Some(winner), cfg, ctx));
    }
    if depth <= 0 {
        if cfg.quiesce {
            return quiesce(env, alpha, beta, cfg, ctx, cfg.quiesce_max);
        }
        return Ok(evaluate_for(env, env.get_current_player(), &cfg.params));
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

    // 置换表探测（决策节点）
    let mut tt_move: Option<usize> = None;
    if cfg.feat(FEAT_TT) {
        let e = ctx.tt[(key as usize) & ctx.tt_mask];
        if e.flag != 0 && e.key == key {
            if e.depth as i32 >= depth {
                match e.flag {
                    1 => return Ok(e.value), // exact
                    2 => {
                        if e.value >= beta {
                            return Ok(e.value);
                        }
                    }
                    3 => {
                        if e.value <= alpha {
                            return Ok(e.value);
                        }
                    }
                    _ => {}
                }
            }
            tt_move = Some(e.best);
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
    for (i, &m) in ordered.iter().enumerate() {
        let quiet = !m.is_chance && !m.is_capture;
        let v = if cfg.feat(FEAT_LMR) && quiet && i >= 3 && depth >= 3 {
            let mut child = *env;
            let _ = child.step(m.action, None);
            let probe = -negamax(&child, depth - 2, -alpha - 1e-6, -alpha, cfg, ctx)?;
            if probe > alpha {
                -negamax(&child, depth - 1, -beta, -alpha, cfg, ctx)?
            } else {
                probe
            }
        } else {
            move_value(env, m.action, depth, alpha, beta, cfg, ctx)?
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

    // 置换表存储（深度优先替换）
    if cfg.feat(FEAT_TT) {
        let flag = if best <= alpha_orig {
            3 // fail-low → 上界
        } else if best >= beta {
            2 // fail-high → 下界
        } else {
            1 // exact
        };
        let idx = (key as usize) & ctx.tt_mask;
        let cur = ctx.tt[idx];
        if cur.flag == 0 || cur.key != key || (cur.depth as i32) <= depth {
            ctx.tt[idx] = TtEntry {
                key,
                value: best,
                depth: depth as i16,
                flag,
                best: best_m,
            };
        }
    }
    Ok(best)
}

/// 单层根搜索：返回 (最优动作, 根走子方视角值)。
fn best_at_depth(
    env: &DarkChessEnv,
    depth: i32,
    cfg: &EngineConfig,
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
    let draw_v = -cfg.contempt;
    for &m in &moves {
        let mut v = move_value(env, m.action, depth, alpha, VMAX, cfg, ctx)?;
        // 根层防“送子躲和”：明显亏本吃子且不构成赢棋（value < 0.3）时压到和棋值之下。
        if cfg.feat(FEAT_NO_DRAW_SAC) && v < 0.3 && ordering::losing_capture(env, &m, cfg) {
            v = v.min(draw_v - 1e-3);
        }
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

/// 节点/时间预算驱动的迭代加深搜索。返回 `None` 表示无合法动作（终局）。
pub fn best_move(env: &DarkChessEnv, cfg: &EngineConfig) -> Option<EngineResult> {
    let moves = generate_moves(env, env.get_current_player());
    if moves.is_empty() {
        return None;
    }
    let mut ctx = Ctx::new(cfg, env);
    if cfg.feat(FEAT_REP) {
        ctx.path.push(zobrist::zkey(env));
    }
    let mut best = moves[0].action;
    let mut best_score = 0.0f32;
    let mut hint: Option<usize> = None;
    let mut depth_reached = 0;
    for depth in 1..=cfg.max_depth {
        match best_at_depth(env, depth, cfg, &mut ctx, hint) {
            Ok(Some((a, v))) => {
                best = a;
                best_score = v;
                hint = Some(a);
                depth_reached = depth;
            }
            _ => break,
        }
    }
    Some(EngineResult {
        action: best,
        value: best_score,
        depth: depth_reached,
        nodes: ctx.nodes,
    })
}
