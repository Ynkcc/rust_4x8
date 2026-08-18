//! 走子排序（MVV-LVA + 杀手 + 历史）与终局评估辅助。

use crate::ai::engine::{Ctx, EngineConfig};
use crate::ai::movegen::{Move, can_capture};
use crate::game_env::types::Player;
use crate::DarkChessEnv;

use super::{FEAT_ORDERING, ORTHO};

/// 终局值（从当前走子方视角；winner 为全局 1/-1/0）。
pub(super) fn terminal_value(
    env: &DarkChessEnv,
    winner: Option<i32>,
    cfg: &EngineConfig,
    ctx: &Ctx,
) -> f32 {
    let my = env.get_current_player();
    match winner {
        Some(w) if w == my.val() => 1.0,
        Some(w) if w == 0 => {
            // 和棋：按 contempt 偏向（根走子方视角）
            if cfg.contempt != 0.0 {
                if ctx.root == my.idx() {
                    -cfg.contempt
                } else {
                    cfg.contempt
                }
            } else {
                0.0
            }
        }
        Some(_) => -1.0,
        None => 0.0,
    }
}

/// 轻量终局检测（复用已生成的走子列表，避免重复计算动作掩码）。
pub(super) fn terminal_info(env: &DarkChessEnv, moves: &[Move]) -> Option<i32> {
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

/// 价值表（评估用）——来自 EvalParams。
#[inline]
pub(super) fn values(cfg: &EngineConfig) -> &[f32; 7] {
    &cfg.params.values
}

/// 吃子目标的棋子价值（m 为目标明子的吃子/炮击）。
#[inline]
pub(super) fn victim_value(env: &DarkChessEnv, m: &Move, cfg: &EngineConfig) -> f32 {
    if let SlotKind::Revealed(p) = slot_kind(&env.get_board_slots()[m.to]) {
        values(cfg)[p.piece_type as usize]
    } else {
        0.0
    }
}

/// 走子排序键（降序 = 优先搜索）：
/// 吃子（MVV-LVA）> 杀手走 > 历史静走 > 炮吃暗子（机会） > 翻棋垫底。
#[inline]
pub(super) fn order_key(env: &DarkChessEnv, m: &Move, depth: i32, cfg: &EngineConfig, ctx: &Ctx) -> i32 {
    if m.is_flip {
        return -1_000_000; // 翻棋最后（最贵且信息量最低）
    }
    if m.is_chance {
        return -900_000; // 炮吃暗子：机会动作
    }
    if m.is_capture {
        let victim = victim_value(env, m, cfg) as i32;
        let attacker = if let SlotKind::Revealed(p) = slot_kind(&env.get_board_slots()[m.from]) {
            values(cfg)[p.piece_type as usize] as i32
        } else {
            0
        };
        return 1_000_000 + victim * 8 - attacker; // MVV-LVA
    }
    // 静走子：杀手 + 历史
    let d = depth as usize;
    if d < ctx.killers.len() {
        if ctx.killers[d][0] == m.action {
            return 900_000;
        }
        if ctx.killers[d][1] == m.action {
            return 800_000;
        }
    }
    let key = m.from * env.config.total_positions + m.to;
    if key < ctx.history.len() {
        ctx.history[key]
    } else {
        0
    }
}

pub(super) fn order_moves(
    env: &DarkChessEnv,
    mv: &mut [Move],
    depth: i32,
    cfg: &EngineConfig,
    ctx: &Ctx,
) {
    if cfg.feat(FEAT_ORDERING) {
        mv.sort_by(|a, b| order_key(env, b, depth, cfg, ctx).cmp(&order_key(env, a, depth, cfg, ctx)));
    }
}

/// 根层检查：`m` 是否为明显亏本吃子（攻击方价值高于被吃方 + 落点被敌方防守）。
/// 保守实现（忽略炮架回吃与深层交换链），只用于根层“不送子躲和棋”约束。
pub(super) fn losing_capture(env: &DarkChessEnv, m: &Move, cfg: &EngineConfig) -> bool {
    if !m.is_capture {
        return false;
    }
    let slots = env.get_board_slots();
    let attacker = match slot_kind(&slots[m.from]) {
        SlotKind::Revealed(p) => *p,
        _ => return false,
    };
    let victim = match slot_kind(&slots[m.to]) {
        SlotKind::Revealed(p) => *p,
        _ => return false,
    };
    let vals = values(cfg);
    if vals[attacker.piece_type as usize] <= vals[victim.piece_type as usize] {
        return false; // 等值或占优的吃子不是亏本交换
    }
    let opp = attacker.player.opposite();
    let cols = env.config.cols as i32;
    let rows = env.config.rows as i32;
    let in_bounds = |r: i32, c: i32| r >= 0 && r < rows && c >= 0 && c < cols;
    let df = (m.to / env.config.cols) as i32;
    let dc = (m.to % env.config.cols) as i32;
    for (dr, dcc) in ORTHO {
        let (r, c) = (df + dr, dc + dcc);
        if !in_bounds(r, c) {
            continue;
        }
        let s = (r * cols + c) as usize;
        if s == m.from {
            continue; // 攻击方腾出的源格
        }
        if let SlotKind::Revealed(d) = slot_kind(&slots[s]) {
            if d.player == opp && can_capture(d.piece_type, attacker.piece_type) {
                return true; // 敌方相邻子能回吃攻击方 → 亏本交换
            }
        }
    }
    false
}

// --- 局部 Slot 枚举（避免直接依赖 game_env::Slot 模式匹配，保持模块内聚） ---
enum SlotKind<'a> {
    Empty,
    Hidden,
    Revealed(&'a crate::game_env::types::Piece),
}

fn slot_kind(slot: &crate::game_env::types::Slot) -> SlotKind<'_> {
    match slot {
        crate::game_env::types::Slot::Empty => SlotKind::Empty,
        crate::game_env::types::Slot::Hidden => SlotKind::Hidden,
        crate::game_env::types::Slot::Revealed(p) => SlotKind::Revealed(p),
    }
}
