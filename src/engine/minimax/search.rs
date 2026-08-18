// src/ai/minimax/search.rs - expectiminimax + alpha-beta 核心搜索

use crate::core::env::DarkChessEnv;

use super::eval::{eval_leaf, terminal_value};
use super::ordering::{order_moves, victim_value};
use super::types::{
    CHANCE_DEPTH_PENALTY, INF, MinimaxConfig, MinimaxResult, SearchState, TtEntry, VMAX, VMIN,
};
use crate::engine::movegen::{Move, generate_moves};
use crate::core::env::types::Player;

/// 局面 zkey（棋盘槽位 + 暗子袋 + 走子方），与引擎共用同一哈希。
fn zkey(env: &DarkChessEnv) -> u64 {
    crate::engine::alpha_beta::zobrist::zkey(env)
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
