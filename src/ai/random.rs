use crate::DarkChessEnv;
use rand::seq::SliceRandom;

use super::Policy;

/// 随机策略：在所有有效动作中等概率选择
pub struct RandomPolicy;

impl Policy for RandomPolicy {
    fn choose_action(env: &DarkChessEnv) -> Option<usize> {
        let masks = env.action_masks();
        let valid_actions: Vec<usize> = masks
            .iter()
            .enumerate()
            .filter_map(|(idx, &val)| if val == 1 { Some(idx) } else { None })
            .collect();

        if valid_actions.is_empty() {
            return None;
        }

        let mut rng = rand::thread_rng();
        valid_actions.choose(&mut rng).copied()
    }
}
