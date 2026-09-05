//! 暗棋 NNUE 累加器与增量差分模块
//!
//! 支持双视角（Red/Black）累加器缓存及 O(1) 走子特征差分更新。

use crate::core::env::types::{Player, Slot};
use crate::core::env::DarkChessEnv;
use super::network::NnueEvaluator;

/// 特征累加器/第一层隐层维度
pub const TRANSFORMER_OUT_DIM: usize = 256;

/// 单个视角的特征增量变化
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FeatureDiff {
    pub removed: [u16; 8],
    pub added: [u16; 8],
    pub removed_count: u8,
    pub added_count: u8,
}

impl FeatureDiff {
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline]
    pub fn remove(&mut self, feat: usize) {
        if (self.removed_count as usize) < self.removed.len() {
            self.removed[self.removed_count as usize] = feat as u16;
            self.removed_count += 1;
        }
    }

    #[inline]
    pub fn add(&mut self, feat: usize) {
        if (self.added_count as usize) < self.added.len() {
            self.added[self.added_count as usize] = feat as u16;
            self.added_count += 1;
        }
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.removed_count == 0 && self.added_count == 0
    }
}

/// NNUE 累加器（Feature Transformer 输出）
///
/// 对齐到 64 字节（缓存行），消除多线程 Lazy SMP 场景中的 false sharing，
/// 并保证 AVX2/AVX-512 对齐加载的前提。
#[derive(Clone, Copy, Debug)]
#[repr(align(64))]
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
        evaluator: &NnueEvaluator,
        old_feature: usize,
        new_feature: usize,
    ) {
        if old_feature == new_feature {
            return;
        }
        let vals: &mut [f32; TRANSFORMER_OUT_DIM] = &mut self.vals;
        let weights = &evaluator.feature_weights;

        if old_feature < evaluator.feature_dim {
            let w = &weights[old_feature * TRANSFORMER_OUT_DIM..][..TRANSFORMER_OUT_DIM];
            for j in 0..TRANSFORMER_OUT_DIM {
                vals[j] -= w[j];
            }
        }
        if new_feature < evaluator.feature_dim {
            let w = &weights[new_feature * TRANSFORMER_OUT_DIM..][..TRANSFORMER_OUT_DIM];
            for j in 0..TRANSFORMER_OUT_DIM {
                vals[j] += w[j];
            }
        }
    }

    /// 应用单视角特征差分
    ///
    /// 内层循环固定长度 TRANSFORMER_OUT_DIM=256，编译器可展开并生成 AVX2/AVX-512 向量指令。
    #[inline]
    pub fn apply_diff(&mut self, diff: &FeatureDiff, evaluator: &NnueEvaluator) {
        let vals: &mut [f32; TRANSFORMER_OUT_DIM] = &mut self.vals;
        let weights = &evaluator.feature_weights;

        for i in 0..diff.removed_count as usize {
            let feat = diff.removed[i] as usize;
            if feat < evaluator.feature_dim {
                let w = &weights[feat * TRANSFORMER_OUT_DIM..][..TRANSFORMER_OUT_DIM];
                for j in 0..TRANSFORMER_OUT_DIM {
                    vals[j] -= w[j];
                }
            }
        }
        for i in 0..diff.added_count as usize {
            let feat = diff.added[i] as usize;
            if feat < evaluator.feature_dim {
                let w = &weights[feat * TRANSFORMER_OUT_DIM..][..TRANSFORMER_OUT_DIM];
                for j in 0..TRANSFORMER_OUT_DIM {
                    vals[j] += w[j];
                }
            }
        }
    }
}

/// 红黑双视角累加器
#[derive(Clone, Copy, Debug)]
pub struct DualAccumulator {
    /// 0: Red 视角, 1: Black 视角
    pub accumulators: [Accumulator; 2],
}

impl Default for DualAccumulator {
    fn default() -> Self {
        Self {
            accumulators: [Accumulator::default(), Accumulator::default()],
        }
    }
}

impl DualAccumulator {
    /// 从环境完整初始化红黑双视角累加器
    pub fn init_from_env(env: &DarkChessEnv, evaluator: &NnueEvaluator) -> Self {
        let mut red_features = Vec::with_capacity(env.config.total_positions + 16);
        let mut black_features = Vec::with_capacity(env.config.total_positions + 16);
        env.nnue_active_features_for_player_into(Player::Red, &mut red_features);
        env.nnue_active_features_for_player_into(Player::Black, &mut black_features);

        let acc_red = evaluator.compute_accumulator(&red_features);
        let acc_black = evaluator.compute_accumulator(&black_features);

        Self {
            accumulators: [acc_red, acc_black],
        }
    }

    /// 应用红黑两个视角的 FeatureDiff
    #[inline]
    pub fn apply_diffs(
        &mut self,
        diff_red: &FeatureDiff,
        diff_black: &FeatureDiff,
        evaluator: &NnueEvaluator,
    ) {
        self.accumulators[0].apply_diff(diff_red, evaluator);
        self.accumulators[1].apply_diff(diff_black, evaluator);
    }

    /// 获取特定玩家视角的累加器
    #[inline]
    pub fn get(&self, player: Player) -> &Accumulator {
        &self.accumulators[player.idx()]
    }
}

/// 计算环境从 before_env 到 after_env 发生的一步动作对红黑双方视角产生的 FeatureDiff。
pub fn compute_step_diff(
    before_env: &DarkChessEnv,
    after_env: &DarkChessEnv,
    action: usize,
) -> (FeatureDiff, FeatureDiff) {
    let cfg = &before_env.config;
    let lookup = crate::core::env::action_lookup_tables(cfg);
    let mut diff_red = FeatureDiff::new();
    let mut diff_black = FeatureDiff::new();

    let affected_squares: &[usize] = if action < cfg.reveal_actions_count {
        &lookup.action_to_coords[action][0..1]
    } else {
        &lookup.action_to_coords[action][0..2]
    };

    // 1. 格位段增量变化
    for &sq in affected_squares {
        let old_slot = before_env.get_board_slots()[sq];
        let new_slot = after_env.get_board_slots()[sq];
        if old_slot != new_slot {
            for (player, diff) in [(Player::Red, &mut diff_red), (Player::Black, &mut diff_black)] {
                let old_feat = DarkChessEnv::nnue_slot_feature_index(cfg, player, sq, old_slot);
                let new_feat = DarkChessEnv::nnue_slot_feature_index(cfg, player, sq, new_slot);
                if old_feat != new_feat {
                    diff.remove(old_feat);
                    diff.add(new_feat);
                }
            }
        }
    }

    // 2. 暗子包段变化（如果有暗子被翻开）
    let mut revealed_pt = None;
    for &sq in affected_squares {
        if matches!(before_env.get_board_slots()[sq], Slot::Hidden) {
            if let Some(p) = after_env.last_revealed_piece {
                revealed_pt = Some(p.piece_type);
                break;
            }
        }
    }

    if let Some(pt) = revealed_pt {
        let compact = cfg.compact_index(pt as usize);
        let stride = cfg.nnue_bag_stride();
        let states = cfg.nnue_states_per_square();
        let bag_base = cfg.total_positions * states;

        let mut old_count = 0usize;
        for piece in before_env.get_hidden_pieces_raw() {
            if piece.piece_type == pt {
                old_count += 1;
            }
        }
        let new_count = old_count.saturating_sub(1);

        let old_feat = bag_base + compact * stride + old_count.min(stride - 1);
        let new_feat = bag_base + compact * stride + new_count.min(stride - 1);
        if old_feat != new_feat {
            diff_red.remove(old_feat);
            diff_red.add(new_feat);
            diff_black.remove(old_feat);
            diff_black.add(new_feat);
        }
    }

    // 3. 标量段变化：无吃子标记
    let states = cfg.nnue_states_per_square();
    let bag_base = cfg.total_positions * states;
    let stride = cfg.nnue_bag_stride();
    let scalar_base = bag_base + cfg.num_active * stride;

    let old_flag = before_env.get_move_counter().min(8) > 0;
    let new_flag = after_env.get_move_counter().min(8) > 0;
    if old_flag != new_flag {
        let flag_feat = scalar_base;
        if old_flag {
            diff_red.remove(flag_feat);
            diff_black.remove(flag_feat);
        } else {
            diff_red.add(flag_feat);
            diff_black.add(flag_feat);
        }
    }

    (diff_red, diff_black)
}
