// src/ai/minimax/eval.rs - 叶子评估（启发式 / HP 差）

use crate::DarkChessEnv;

use super::types::MinimaxConfig;
use crate::ai::eval::evaluate_for;

/// 启发式静态评估：多特征评估（默认），或当前玩家 HP 差 / 初始 HP。
pub fn heuristic_value(env: &DarkChessEnv) -> f32 {
    let cfg = MinimaxConfig::default();
    eval_leaf(env, &cfg)
}

pub(super) fn eval_leaf(env: &DarkChessEnv, cfg: &MinimaxConfig) -> f32 {
    if cfg.rich_eval {
        evaluate_for(env, env.get_current_player(), &cfg.params)
    } else {
        let my = env.get_hp(env.get_current_player());
        let opp = env.get_hp(env.get_current_player().opposite());
        (my - opp) as f32 / env.config.initial_health as f32
    }
}

/// 终局值：从当前玩家视角的 ±1 / 0。
pub(super) fn terminal_value(env: &DarkChessEnv, winner: Option<i32>) -> f32 {
    match winner {
        Some(w) if w == env.get_current_player().val() => 1.0,
        Some(w) if w == 0 => 0.0,
        _ => -1.0,
    }
}
