//! 暗棋 NNUE 推理网络模块

pub mod feature;
pub mod network;

pub use feature::{Accumulator, FEATURE_DIM, TRANSFORMER_OUT_DIM, extract_active_features, sq_state_feature};
pub use network::NnueEvaluator;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::env::DarkChessEnv;

    #[test]
    fn test_nnue_evaluator_basic() {
        let env = DarkChessEnv::default();
        let eval = NnueEvaluator::new_dummy();

        let val = eval.evaluate(&env);
        assert!(val >= -1.0 && val <= 1.0);

        let active = extract_active_features(&env);
        assert!(active.len() >= 40);
    }
}
