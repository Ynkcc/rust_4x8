//! 暗棋 NNUE 推理网络模块

pub mod feature;
pub mod network;

pub use feature::{compute_step_diff, Accumulator, DualAccumulator, FeatureDiff, TRANSFORMER_OUT_DIM};
pub use network::{NnueBoard, NnueEvaluator};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::env::types::Player;
    use crate::core::env::DarkChessEnv;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    #[test]
    fn test_nnue_evaluator_basic() {
        let env = DarkChessEnv::default();
        let eval = NnueEvaluator::new_dummy(env.config.nnue_feature_dim());

        let val = eval.evaluate(&env);
        assert!(val >= -1.0 && val <= 1.0);

        let active = env.nnue_active_features();
        assert!(active.len() >= env.config.total_positions + env.config.num_active);
    }

    #[test]
    fn test_nnue_incremental_update_consistency() {
        let eval = NnueEvaluator::new_dummy(DarkChessEnv::default().config.nnue_feature_dim());

        for seed in [42u64, 100u64, 2024u64] {
            let mut env = DarkChessEnv::default();
            env.seed = Some(seed);
            env.reset();

            let mut board = NnueBoard::new(env, &eval);
            let mut rng = StdRng::seed_from_u64(seed);

            for step_idx in 0..40 {
                let mut masks = vec![0i32; board.env.config.action_space_size];
                board.env.action_masks_into(&mut masks);
                let legal_actions: Vec<usize> = masks
                    .iter()
                    .enumerate()
                    .filter_map(|(act, &m)| if m == 1 { Some(act) } else { None })
                    .collect();

                if legal_actions.is_empty() {
                    break;
                }

                let act = legal_actions[rng.gen_range(0..legal_actions.len())];
                let (_, done, _, _) = board.step(act, None, &eval).expect("Step should succeed");

                // 1. 验证根据当前行棋方求值的数值严格一致
                let incremental_val = board.evaluate(&eval);
                let full_val = eval.evaluate(&board.env);
                let diff = (incremental_val - full_val).abs();
                assert!(
                    diff < 1e-4,
                    "Seed {}, Step {}: 评估值不一致! 增量值: {}, 全量值: {}, 差异: {}",
                    seed, step_idx, incremental_val, full_val, diff
                );

                // 2. 验证红黑双视角累加器内部数值与全量从头计算严格一致
                for &p in &[Player::Red, Player::Black] {
                    let mut active_feats = Vec::new();
                    board.env.nnue_active_features_for_player_into(p, &mut active_feats);
                    let full_acc = eval.compute_accumulator(&active_feats);
                    let inc_acc = board.accumulators.get(p);

                    for j in 0..TRANSFORMER_OUT_DIM {
                        let acc_diff = (inc_acc.vals[j] - full_acc.vals[j]).abs();
                        assert!(
                            acc_diff < 1e-4,
                            "Seed {}, Step {}, Player {:?}, dim {}: 累加器不一致! 增量: {}, 全量: {}",
                            seed, step_idx, p, j, inc_acc.vals[j], full_acc.vals[j]
                        );
                    }
                }

                if done {
                    break;
                }
            }
        }
    }
}
