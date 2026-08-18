// src/ai/minimax/ordering.rs - 走子排序（MVV-LVA + 杀手走 + 历史启发 + 翻棋垫底）

use crate::DarkChessEnv;

use super::types::{MinimaxConfig, SearchState};
use crate::ai::movegen::Move;

#[inline]
pub(super) fn victim_value(env: &DarkChessEnv, m: &Move, cfg: &MinimaxConfig) -> f32 {
    if let crate::Slot::Revealed(p) = &env.get_board_slots()[m.to] {
        cfg.params.values[p.piece_type as usize]
    } else {
        0.0
    }
}

#[inline]
pub(super) fn order_key(env: &DarkChessEnv, m: &Move, depth: usize, cfg: &MinimaxConfig, ss: &SearchState) -> i32 {
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

pub(super) fn order_moves(env: &DarkChessEnv, mv: &mut [Move], depth: usize, cfg: &MinimaxConfig, ss: &SearchState) {
    if cfg.use_ordering {
        mv.sort_by(|a, b| order_key(env, b, depth, cfg, ss).cmp(&order_key(env, a, depth, cfg, ss)));
    }
}
