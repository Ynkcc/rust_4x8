use crate::{DarkChessEnv, REVEAL_ACTIONS_COUNT};
use rand::seq::SliceRandom;

use super::Policy;

/// 优先翻棋策略：
/// - 若存在“翻棋”类有效动作（索引 < REVEAL_ACTIONS_COUNT），随机选其一
/// - 否则在剩余所有有效动作中随机选择
pub struct RevealFirstPolicy;

impl Policy for RevealFirstPolicy {
    fn choose_action(env: &DarkChessEnv) -> Option<usize> {
        let masks = env.action_masks();
        let mut reveal_actions: Vec<usize> = Vec::new();
        let mut fallback_actions: Vec<usize> = Vec::new();

        for (idx, &m) in masks.iter().enumerate() {
            if m == 1 {
                if idx < REVEAL_ACTIONS_COUNT {
                    reveal_actions.push(idx);
                } else {
                    fallback_actions.push(idx);
                }
            }
        }

        let mut rng = rand::thread_rng();
        if !reveal_actions.is_empty() {
            return reveal_actions.choose(&mut rng).copied();
        }
        fallback_actions.choose(&mut rng).copied()
    }
}
