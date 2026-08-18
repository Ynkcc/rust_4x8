// src/ai/minimax.rs
// Expectiminimax + Alpha-Beta 剪枝（迷你暗棋 / 暗棋通用），已升级：
//
//   - 多特征启发式评估（src/ai/eval.rs，校正价值表 + 覆盖物质 + 将帅情境 +
//     支配价值 + 机动性 + 将帅危险度）替代单一 HP 差；
//   - 置换表（决策节点，含机会子树期望值的精确/边界标志，剪枝安全）；
//   - 走子排序（MVV-LVA + 杀手走 + 历史启发 + 翻棋垫底）；
//   - 静态搜索（深度耗尽后仅延展吃明子走法）。
//
// 暗棋是部分可观察游戏，严格来说应为 `expectiminimax`：
//   - 确定性动作（普通移动 / 吃明子）：negamax + alpha-beta 剪枝；
//   - 机会动作（翻棋 / 吃暗子）：枚举 `chance_outcomes` 的所有可能结果，
//     按概率加权取期望值（期望节点不能剪枝，其子搜索使用全开边界保证正确性）。
//
// 值的约定：所有函数返回「从传入环境当前玩家视角」的效用，范围 [-1, 1]：
//   +1 = 当前玩家必胜，-1 = 当前玩家必败，0 = 平局；
//   深度耗尽时用启发式评估截断。

mod eval;
mod ordering;
mod search;
mod types;

use crate::core::env::DarkChessEnv;

pub use eval::heuristic_value;
pub use search::minimax_best_action_with_config;
pub use types::{MinimaxConfig, MinimaxResult};

/// 求解当前局面的最优动作（expectiminimax + alpha-beta，升级默认配置）。
///
/// `max_depth` 为搜索深度（不含根；机会节点每层消耗 CHANCE_DEPTH_PENALTY 深度）。
/// 返回 `None` 表示无合法动作（终局）。
pub fn minimax_best_action(env: &DarkChessEnv, max_depth: usize) -> Option<MinimaxResult> {
    minimax_best_action_with_config(env, max_depth, &MinimaxConfig::default())
}

/// 以固定深度搜索并返回动作；等价于 `minimax_best_action(env, depth).map(|r| r.action)`。
pub fn minimax_choose_action(env: &DarkChessEnv, max_depth: usize) -> Option<usize> {
    minimax_best_action(env, max_depth).map(|r| r.action)
}

#[cfg(test)]
mod tests {
    use super::minimax_best_action;
    use crate::core::env::DarkChessEnv;

    #[test]
    fn minimax_returns_legal_action() {
        let mut env = DarkChessEnv::new();
        env.seed = Some(21);
        env.reset();
        let res = minimax_best_action(&env, 2).expect("应返回动作");
        let mut masks = vec![0i32; env.config.action_space_size];
        env.action_masks_into(&mut masks);
        assert_eq!(masks[res.action], 1, "minimax 返回非法动作 {}", res.action);
    }

    #[test]
    fn minimax_survives_random_games() {
        let mut env = DarkChessEnv::new();
        env.seed = Some(33);
        env.reset();
        let mut steps = 0;
        loop {
            let Some(res) = minimax_best_action(&env, 2) else { break };
            let mut next = env;
            assert!(next.step(res.action, None).is_ok());
            env = next;
            let (term, _, _) = env.check_game_over_conditions();
            steps += 1;
            if term || steps > 40 {
                break;
            }
        }
    }
}
