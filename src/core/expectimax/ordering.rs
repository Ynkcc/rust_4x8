//! 走子排序（MVV-LVA + 杀手 + 历史）与终局评估辅助。

use crate::core::env::DarkChessEnv;
use crate::core::env::types::Player;
use crate::engine::movegen::Move;

use super::search::SearchConfig;
use super::{Ctx, FEAT_ORDERING};

/// 终局值（从当前走子方视角；winner 为全局 1/-1/0）。
pub(super) fn terminal_value(
    env: &DarkChessEnv,
    winner: Option<i32>,
    cfg: &SearchConfig,
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

/// 轻量终局检测：委托 env 核心，复用已生成的走子列表，避免重复计算动作掩码。
pub(super) fn terminal_info(env: &DarkChessEnv, moves: &[Move]) -> Option<i32> {
    env.check_game_over_with_moves(moves.is_empty()).2
}

/// 吃子目标的棋子价值（m 为目标明子的吃子/炮击）。
#[inline]
pub(super) fn victim_value(env: &DarkChessEnv, m: &Move) -> i32 {
    if let SlotKind::Revealed(p) = slot_kind(&env.get_board_slots()[m.to]) {
        p.piece_type.value()
    } else {
        0
    }
}

/// 走子排序键（降序 = 优先搜索）：
/// 吃子（MVV-LVA）> 杀手走 > 历史静走 > 炮吃暗子（机会） > 翻棋垫底。
#[inline]
pub(super) fn order_key(env: &DarkChessEnv, m: &Move, depth: i32, ctx: &Ctx) -> i32 {
    if m.is_flip {
        return -1_000_000; // 翻棋最后（最贵且信息量最低）
    }
    if m.is_chance {
        return -900_000; // 炮吃暗子：机会动作
    }
    if m.is_capture {
        let victim = victim_value(env, m);
        let attacker = if let SlotKind::Revealed(p) = slot_kind(&env.get_board_slots()[m.from]) {
            p.piece_type.value()
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
    cfg: &SearchConfig,
    ctx: &Ctx,
) {
    if cfg.feat(FEAT_ORDERING) {
        mv.sort_by(|a, b| order_key(env, b, depth, ctx).cmp(&order_key(env, a, depth, ctx)));
    }
}

// --- 局部 Slot 枚举（避免直接依赖 game_env::Slot 模式匹配，保持模块内聚） ---
enum SlotKind<'a> {
    Empty,
    Hidden,
    Revealed(&'a crate::core::env::types::Piece),
}

fn slot_kind(slot: &crate::core::env::types::Slot) -> SlotKind<'_> {
    match slot {
        crate::core::env::types::Slot::Empty => SlotKind::Empty,
        crate::core::env::types::Slot::Hidden => SlotKind::Hidden,
        crate::core::env::types::Slot::Revealed(p) => SlotKind::Revealed(p),
    }
}
