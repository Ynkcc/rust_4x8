//! 暗棋 NNUE 特征提取与累加器模块
//!
//! 将 DarkChessEnv 状态映射为固定 562 维稀疏特征输入，并维护增量累加向量。

use crate::core::env::{DarkChessEnv, MAX_POSITIONS, NUM_PIECE_TYPES_MAX, Player, Slot};

/// NNUE 总输入特征维度 (32格*16状态 + 7种暗子*6数量 + 8标量 = 562)
pub const FEATURE_DIM: usize = 562;
/// 特征累加器/第一层隐层维度
pub const TRANSFORMER_OUT_DIM: usize = 256;

/// NNUE 累加器（Feature Transformer 输出）
#[derive(Clone, Debug)]
pub struct Accumulator {
    pub vals: [f32; TRANSFORMER_OUT_DIM],
}

impl Default for Accumulator {
    fn default() -> Self {
        Self {
            vals: [0.0; TRANSFORMER_OUT_DIM],
        }
    }
}

impl Accumulator {
    /// O(1) 增量更新：从累加器中移除 old_feature，增加 new_feature
    pub fn update_feature(
        &mut self,
        evaluator: &super::network::NnueEvaluator,
        old_feature: usize,
        new_feature: usize,
    ) {
        if old_feature == new_feature {
            return;
        }
        let old_offset = old_feature * TRANSFORMER_OUT_DIM;
        let new_offset = new_feature * TRANSFORMER_OUT_DIM;

        for j in 0..TRANSFORMER_OUT_DIM {
            let mut val = self.vals[j];
            if old_feature < FEATURE_DIM {
                val -= evaluator.feature_weights[old_offset + j];
            }
            if new_feature < FEATURE_DIM {
                val += evaluator.feature_weights[new_offset + j];
            }
            self.vals[j] = val;
        }
    }
}

/// 计算单格槽位对应的全局输入特征索引 (0..512)
#[inline]
pub fn sq_state_feature(sq: usize, slot: Slot) -> usize {
    let state_offset = match slot {
        Slot::Empty => 0,
        Slot::Hidden => 1,
        Slot::Revealed(piece) => {
            let p_offset = match piece.player {
                Player::Red => 0,
                Player::Black => 7,
            };
            2 + p_offset + piece.piece_type as usize
        }
    };
    sq * 16 + state_offset
}

/// 从当前 DarkChessEnv 中提取活性的输入特征索引列表 (完全兼容 4x8, 4x4, 4x2 变体)
pub fn extract_active_features(env: &DarkChessEnv) -> Vec<usize> {
    let mut active = Vec::with_capacity(48);

    for sq in 0..MAX_POSITIONS {
        let slot = if sq < env.config.total_positions {
            env.board[sq]
        } else {
            Slot::Empty
        };
        active.push(sq_state_feature(sq, slot));
    }

    let bag_base = 512;
    let mut unrevealed_counts = [0usize; NUM_PIECE_TYPES_MAX];
    for i in 0..env.hidden_pieces_count {
        let pt = env.hidden_pieces_pool[i].piece_type as usize;
        if pt < NUM_PIECE_TYPES_MAX {
            unrevealed_counts[pt] += 1;
        }
    }
    for pt_idx in 0..NUM_PIECE_TYPES_MAX {
        let count = unrevealed_counts[pt_idx].min(5);
        active.push(bag_base + pt_idx * 6 + count);
    }

    let scalar_base = 554;
    if env.get_current_player() == Player::Black {
        active.push(scalar_base);
    }
    let step_bucket = (env.total_step_counter / 5).min(5);
    active.push(scalar_base + 1 + step_bucket);
    let no_eat_bucket = env.move_counter.min(8);
    if no_eat_bucket > 0 {
        active.push(scalar_base + 6);
    }

    active
}
