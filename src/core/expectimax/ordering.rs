//! Expectimax 搜索引擎走子排序与终端估值辅助

use crate::core::env::{DarkChessEnv, Player, Slot};

/// 判断吃子双方的 MVV-LVA 价值得分
pub fn victim_value(env: &DarkChessEnv, from: usize, to: usize) -> i32 {
    let attacker_val = match env.board[from] {
        Slot::Revealed(p) => p.piece_type.value(),
        _ => 1,
    };
    let defender_val = match env.board[to] {
        Slot::Revealed(p) => p.piece_type.value(),
        _ => 1,
    };
    defender_val * 10 - attacker_val
}

/// 简单终局检查：受胜负规则判定
pub fn terminal_value(env: &DarkChessEnv, winner: Option<i32>, contempt: f32) -> f32 {
    match winner {
        Some(1) => {
            if env.get_current_player() == Player::Red {
                1.0
            } else {
                -1.0
            }
        }
        Some(-1) => {
            if env.get_current_player() == Player::Black {
                1.0
            } else {
                -1.0
            }
        }
        Some(0) => -contempt,
        _ => 0.0,
    }
}
