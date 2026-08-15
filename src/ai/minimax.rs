// src/ai/minimax.rs
// Expectiminimax + Alpha-Beta 剪枝（迷你暗棋 / 暗棋通用），已升级：
//
//   - 多特征启发式评估（src/ai/eval.rs，校正价值表 + 覆盖物质 + 将帅情境 +
//     支配价值 + 机动性 + 将帅危险度）替代单一 HP 差；
//   - 置换表（决策节点，含机会子树期望值的精确/边界标志，剪枝安全）；
//   - 走子排序（MVV-LVA + 杀手走 + 历史启发 + 翻棋垫底）；
//   - 静态搜索（深度耗尽后仅延展吃明子走法）。
//
// 暗棋是部分可观察游戏，严格来说应为 `expectiminimax`：
//   - 确定性动作（普通移动 / 吃明子）：negamax + alpha-beta 剪枝；
//   - 机会动作（翻棋 / 吃暗子）：枚举 `chance_outcomes` 的所有可能结果，
//     按概率加权取期望值（期望节点不能剪枝，其子搜索使用全开边界保证正确性）。
//
// 值的约定：所有函数返回「从传入环境当前玩家视角」的效用，范围 [-1, 1]：
//   +1 = 当前玩家必胜，-1 = 当前玩家必败，0 = 平局；
//   深度耗尽时用启发式评估截断。

use crate::game_env::types::Player;
use crate::DarkChessEnv;

use super::eval::{EvalParams, evaluate_for};
use super::movegen::{Move, generate_moves};

/// 机会节点（翻棋/吃暗子）消耗的搜索深度。
///
/// 机会动作每个要枚举 2*num_active 种翻棋结果，若不惩罚会指数爆炸
/// （alpha-beta 对期望节点无效，无法剪枝）。设 2：机会层等价于两倍普通层代价。
const CHANCE_DEPTH_PENALTY: usize = 2;

const VMIN: f32 = -1.0;
const VMAX: f32 = 1.0;
const INF: f32 = f32::INFINITY;

/// 单次搜索的结果。
#[derive(Debug, Clone, Copy)]
pub struct MinimaxResult {
    /// 从当前玩家视角选择的最优动作。
    pub action: usize,
    /// 该动作的期望效用（当前玩家视角，[-1, 1]）。
    pub value: f32,
    /// 搜索展开的节点数（不含根）。
    pub nodes: u64,
}

/// Minimax 搜索配置。
#[derive(Clone, Copy, Debug)]
pub struct MinimaxConfig {
    /// 是否使用多特征启发式评估（关闭则退回 HP 差评估）
    pub rich_eval: bool,
    /// 是否启用置换表
    pub use_tt: bool,
    /// 是否启用走子排序
    pub use_ordering: bool,
    /// 是否启用静态搜索
    pub use_quiescence: bool,
    /// 置换表大小（2^tt_bits 项）
    pub tt_bits: u32,
    /// 评估参数（rich_eval 时生效）
    pub params: EvalParams,
}

impl Default for MinimaxConfig {
    fn default() -> Self {
        Self {
            rich_eval: true,
            use_tt: true,
            use_ordering: true,
            use_quiescence: true,
            tt_bits: 16,
            params: EvalParams::default(),
        }
    }
}

/// 启发式静态评估：多特征评估（默认），或当前玩家 HP 差 / 初始 HP。
pub fn heuristic_value(env: &DarkChessEnv) -> f32 {
    let cfg = MinimaxConfig::default();
    eval_leaf(env, &cfg)
}

fn eval_leaf(env: &DarkChessEnv, cfg: &MinimaxConfig) -> f32 {
    if cfg.rich_eval {
        evaluate_for(env, env.get_current_player(), &cfg.params)
    } else {
        let my = env.get_hp(env.get_current_player());
        let opp = env.get_hp(env.get_current_player().opposite());
        (my - opp) as f32 / env.config.initial_health as f32
    }
}

/// 终局值：从当前玩家视角的 ±1 / 0。
fn terminal_value(env: &DarkChessEnv, winner: Option<i32>) -> f32 {
    match winner {
        Some(w) if w == env.get_current_player().val() => 1.0,
        Some(w) if w == 0 => 0.0,
        _ => -1.0,
    }
}

// --- 置换表 ---

#[derive(Clone, Copy)]
struct TtEntry {
    key: u64,
    value: f32,
    depth: i16,
    flag: u8, // 0 空, 1 exact, 2 下界, 3 上界
    best: usize,
}

const TT_EMPTY: TtEntry = TtEntry {
    key: 0,
    value: 0.0,
    depth: 0,
    flag: 0,
    best: 0,
};

struct SearchState {
    nodes: u64,
    tt: Vec<TtEntry>,
    tt_mask: usize,
    killers: Vec<[usize; 2]>,
    history: Vec<i32>,
}

impl SearchState {
    fn new(cfg: &MinimaxConfig, env: &DarkChessEnv, max_depth: usize) -> Self {
        let total = env.config.total_positions;
        let tt_size = if cfg.use_tt { 1usize << cfg.tt_bits } else { 0 };
        Self {
            nodes: 0,
            tt: vec![TT_EMPTY; tt_size],
            tt_mask: tt_size.wrapping_sub(1),
            killers: vec![[0; 2]; max_depth + 2],
            history: vec![0; total * total],
        }
    }

    fn record_cutoff(&mut self, m: &Move, depth: usize, total: usize) {
        let d = depth as usize;
        if d < self.killers.len() && self.killers[d][0] != m.action {
            self.killers[d][1] = self.killers[d][0];
            self.killers[d][0] = m.action;
        }
        let key = m.from * total + m.to;
        if key < self.history.len() {
            self.history[key] += (depth * depth) as i32;
        }
    }
}

/// 局面 zkey（棋盘槽位 + 暗子袋 + 走子方），与引擎共用同一哈希。
fn zkey(env: &DarkChessEnv) -> u64 {
    crate::ai::engine::zkey(env)
}

/// 轻量终局检测（复用走子列表）。
fn terminal_info(env: &DarkChessEnv, moves: &[Move]) -> Option<i32> {
    if env.get_score(Player::Red) <= 0 {
        return Some(Player::Black.val());
    }
    if env.get_score(Player::Black) <= 0 {
        return Some(Player::Red.val());
    }
    if env.get_dead_pieces(Player::Red).len() == env.config.total_pieces_per_player {
        return Some(Player::Black.val());
    }
    if env.get_dead_pieces(Player::Black).len() == env.config.total_pieces_per_player {
        return Some(Player::Red.val());
    }
    if moves.is_empty() {
        return Some(env.get_current_player().opposite().val());
    }
    if env.get_move_counter() >= env.config.max_consecutive_moves_for_draw {
        return Some(0);
    }
    if env.get_total_steps() >= env.config.max_steps_per_episode {
        return Some(0);
    }
    None
}

#[inline]
fn victim_value(env: &DarkChessEnv, m: &Move, cfg: &MinimaxConfig) -> f32 {
    if let crate::Slot::Revealed(p) = &env.get_board_slots()[m.to] {
        cfg.params.values[p.piece_type as usize]
    } else {
        0.0
    }
}

#[inline]
fn order_key(env: &DarkChessEnv, m: &Move, depth: usize, cfg: &MinimaxConfig, ss: &SearchState) -> i32 {
    if m.is_flip {
        return -1_000_000;
    }
    if m.is_chance {
        return -900_000;
    }
    if m.is_capture {
        let victim = victim_value(env, m, cfg) as i32;
        let attacker = if let crate::Slot::Revealed(p) = &env.get_board_slots()[m.from] {
            cfg.params.values[p.piece_type as usize] as i32
        } else {
            0
        };
        return 1_000_000 + victim * 8 - attacker;
    }
    let d = depth as usize;
    if d < ss.killers.len() {
        if ss.killers[d][0] == m.action {
            return 900_000;
        }
        if ss.killers[d][1] == m.action {
            return 800_000;
        }
    }
    let key = m.from * env.config.total_positions + m.to;
    if key < ss.history.len() {
        ss.history[key]
    } else {
        0
    }
}

fn order_moves(env: &DarkChessEnv, mv: &mut [Move], depth: usize, cfg: &MinimaxConfig, ss: &SearchState) {
    if cfg.use_ordering {
        mv.sort_by(|a, b| order_key(env, b, depth, cfg, ss).cmp(&order_key(env, a, depth, cfg, ss)));
    }
}

/// 静态搜索：深度耗尽后延展吃明子走法。
fn quiesce(
    env: &DarkChessEnv,
    mut alpha: f32,
    beta: f32,
    cfg: &MinimaxConfig,
    ss: &mut SearchState,
    qdepth: i32,
) -> f32 {
    ss.nodes += 1;
    let moves = generate_moves(env, env.get_current_player());
    if let Some(winner) = terminal_info(env, &moves) {
        return terminal_value(env, Some(winner));
    }
    let stand = eval_leaf(env, cfg);
    if stand >= beta || qdepth <= 0 {
        return stand;
    }
    if stand > alpha {
        alpha = stand;
    }
    let mut caps: Vec<(i32, Move)> = moves
        .iter()
        .filter(|m| m.is_capture)
        .map(|&m| (victim_value(env, &m, cfg) as i32, m))
        .collect();
    caps.sort_by(|a, b| b.0.cmp(&a.0));
    let mut best = stand;
    for (_, m) in caps {
        let mut child = *env;
        let _ = child.step(m.action, None);
        let v = -quiesce(&child, -beta, -alpha, cfg, ss, qdepth - 1);
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
    best
}

/// 机会节点：按概率加权期望（精确期望）。
///
/// 子搜索使用全开边界 (VMIN, VMAX)，保证每个结果值都是精确值，
/// 期望值严格正确；代价是机会子树内部无法剪枝（由 CHANCE_DEPTH_PENALTY 补偿）。
fn expected_value(
    env: &DarkChessEnv,
    action: usize,
    depth: usize,
    cfg: &MinimaxConfig,
    ss: &mut SearchState,
) -> f32 {
    let outcomes = env.chance_outcomes(action);
    let child_depth = depth.saturating_sub(CHANCE_DEPTH_PENALTY);
    let mut expected = 0.0;
    for (_, prob, next_env) in outcomes {
        let v = -search(&next_env, child_depth, VMIN, VMAX, cfg, ss);
        expected += prob * v;
    }
    expected
}

/// 确定性走子：执行并进入子搜索。
fn deterministic_value(
    env: &DarkChessEnv,
    m: &Move,
    depth: usize,
    alpha: f32,
    beta: f32,
    cfg: &MinimaxConfig,
    ss: &mut SearchState,
) -> f32 {
    let mut next_env = *env;
    let _ = next_env.step(m.action, None);
    -search(&next_env, depth - 1, -beta, -alpha, cfg, ss)
}

/// 搜索当前节点（当前玩家视角）。
fn search(
    env: &DarkChessEnv,
    depth: usize,
    mut alpha: f32,
    beta: f32,
    cfg: &MinimaxConfig,
    ss: &mut SearchState,
) -> f32 {
    ss.nodes += 1;
    let moves = generate_moves(env, env.get_current_player());
    if let Some(winner) = terminal_info(env, &moves) {
        return terminal_value(env, Some(winner));
    }
    if depth == 0 {
        if cfg.use_quiescence {
            return quiesce(env, alpha, beta, cfg, ss, 4);
        }
        return eval_leaf(env, cfg);
    }

    let alpha_orig = alpha;
    let key = if cfg.use_tt { zkey(env) } else { 0 };

    let mut tt_move: Option<usize> = None;
    if cfg.use_tt {
        let e = ss.tt[(key as usize) & ss.tt_mask];
        if e.flag != 0 && e.key == key {
            if e.depth as usize >= depth {
                match e.flag {
                    1 => return e.value,
                    2 => {
                        if e.value >= beta {
                            return e.value;
                        }
                    }
                    3 => {
                        if e.value <= alpha {
                            return e.value;
                        }
                    }
                    _ => {}
                }
            }
            tt_move = Some(e.best);
        }
    }

    let mut ordered = moves;
    order_moves(env, &mut ordered, depth, cfg, ss);
    if let Some(tm) = tt_move {
        if let Some(pos) = ordered.iter().position(|m| m.action == tm) {
            let m = ordered.remove(pos);
            ordered.insert(0, m);
        }
    }

    let mut best = -INF;
    let mut best_m = ordered[0].action;
    let mut a = alpha;
    for (i, &m) in ordered.iter().enumerate() {
        let quiet = !m.is_chance && !m.is_capture;
        // 晚到静走子减深（LMR）：null-window 探测后再决定是否完整重搜。
        let v = if cfg.use_ordering && quiet && i >= 3 && depth >= 3 {
            let probe = deterministic_value(env, &m, depth - 1, a, a + 1e-6, cfg, ss);
            if probe > a {
                deterministic_value(env, &m, depth, a, beta, cfg, ss)
            } else {
                probe
            }
        } else if env.is_chance_action(m.action) {
            expected_value(env, m.action, depth, cfg, ss)
        } else {
            deterministic_value(env, &m, depth, a, beta, cfg, ss)
        };
        if v > best {
            best = v;
            best_m = m.action;
        }
        if best > alpha {
            alpha = best;
        }
        a = a.max(best);
        if a >= beta {
            if cfg.use_ordering && quiet {
                ss.record_cutoff(&m, depth, env.config.total_positions);
            }
            break;
        }
    }

    if cfg.use_tt {
        let flag = if best <= alpha_orig {
            3
        } else if best >= beta {
            2
        } else {
            1
        };
        let idx = (key as usize) & ss.tt_mask;
        let cur = ss.tt[idx];
        if cur.flag == 0 || cur.key != key || (cur.depth as usize) <= depth {
            ss.tt[idx] = TtEntry {
                key,
                value: best,
                depth: depth as i16,
                flag,
                best: best_m,
            };
        }
    }
    best
}

/// 求解当前局面的最优动作（expectiminimax + alpha-beta，带配置）。
pub fn minimax_best_action_with_config(
    env: &DarkChessEnv,
    max_depth: usize,
    cfg: &MinimaxConfig,
) -> Option<MinimaxResult> {
    let moves = generate_moves(env, env.get_current_player());
    if moves.is_empty() {
        return None;
    }
    let mut ss = SearchState::new(cfg, env, max_depth);

    let mut best_action = moves[0].action;
    let mut best_value = -INF;
    for &m in &moves {
        let v = if env.is_chance_action(m.action) {
            let outcomes = env.chance_outcomes(m.action);
            let child_depth = max_depth.saturating_sub(CHANCE_DEPTH_PENALTY);
            let mut expected = 0.0;
            for (_, prob, next_env) in outcomes {
                let child_v = -search(&next_env, child_depth, VMIN, VMAX, cfg, &mut ss);
                expected += prob * child_v;
            }
            expected
        } else {
            let mut next_env = *env;
            let _ = next_env.step(m.action, None);
            // 后续根动作用当前最优值作 beta 边界（保守，避免跨动作误剪）。
            let beta = if best_value > -INF { -best_value } else { VMAX };
            -search(&next_env, max_depth - 1, VMIN, beta, cfg, &mut ss)
        };
        if v > best_value {
            best_value = v;
            best_action = m.action;
        }
    }

    Some(MinimaxResult {
        action: best_action,
        value: best_value,
        nodes: ss.nodes,
    })
}

/// 求解当前局面的最优动作（expectiminimax + alpha-beta，升级默认配置）。
///
/// `max_depth` 为搜索深度（不含根；机会节点每层消耗 CHANCE_DEPTH_PENALTY 深度）。
/// 返回 `None` 表示无合法动作（终局）。
pub fn minimax_best_action(env: &DarkChessEnv, max_depth: usize) -> Option<MinimaxResult> {
    minimax_best_action_with_config(env, max_depth, &MinimaxConfig::default())
}

/// 以固定深度搜索并返回动作；等价于 `minimax_best_action(env, depth).map(|r| r.action)`。
pub fn minimax_choose_action(env: &DarkChessEnv, max_depth: usize) -> Option<usize> {
    minimax_best_action(env, max_depth).map(|r| r.action)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn minimax_returns_legal_action() {
        let mut env = DarkChessEnv::new();
        env.seed = Some(21);
        env.reset();
        let res = minimax_best_action(&env, 2).expect("应返回动作");
        let mut masks = vec![0i32; env.config.action_space_size];
        env.action_masks_into(&mut masks);
        assert_eq!(masks[res.action], 1, "minimax 返回非法动作 {}", res.action);
    }

    #[test]
    fn minimax_survives_random_games() {
        let mut env = DarkChessEnv::new();
        env.seed = Some(33);
        env.reset();
        let mut steps = 0;
        loop {
            let Some(res) = minimax_best_action(&env, 2) else { break };
            let mut next = env;
            assert!(next.step(res.action, None).is_ok());
            env = next;
            let (term, _, _) = env.check_game_over_conditions();
            steps += 1;
            if term || steps > 40 {
                break;
            }
        }
    }
}
