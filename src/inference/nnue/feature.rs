//! 暗棋 NNUE 累加器模块（纯变换，不含 env 依赖）
//!
//! 特征提取在 `core::env`（`DarkChessEnv::nnue_active_features_into`），
//! 本模块只负责把稀疏特征索引增量累加为隐层向量。

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
            if old_feature < evaluator.feature_dim {
                val -= evaluator.feature_weights[old_offset + j];
            }
            if new_feature < evaluator.feature_dim {
                val += evaluator.feature_weights[new_offset + j];
            }
            self.vals[j] = val;
        }
    }
}
