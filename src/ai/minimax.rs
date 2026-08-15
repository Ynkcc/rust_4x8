// src/ai/minimax.rs
// Expectiminimax + Alpha-Beta 剪枝（迷你暗棋 / 暗棋通用）。
//
// 暗棋是部分可观察游戏（翻棋 / 吃暗子有随机性），因此严格来说应为
// `expectiminimax`：
//   - 确定性动作（普通移动 / 吃明子）：negamax + alpha-beta 剪枝；
//   - 机会动作（翻棋 / 吃暗子）：枚举 `chance_outcomes` 的所有可能结果，
//     按概率加权取期望值（期望节点不能剪枝，其子搜索使用全开边界保证正确性）。
//
// 值的约定：所有函数返回「从传入环境当前玩家视角」的效用，范围 [-1, 1]：
//   +1 = 当前玩家必胜，-1 = 当前玩家必败，0 = 平局；
//   深度耗尽时用启发式（当前玩家 HP 差归一化到 [-1, 1]）截断。

use crate::game_env::board::DarkChessEnv;

/// 机会节点（翻棋/吃暗子）消耗的搜索深度。
///
/// 机会动作每个要枚举 2*num_active 种翻棋结果，若不惩罚会指数爆炸
/// （alpha-beta 对期望节点无效，无法剪枝）。设 2：机会层等价于两倍普通层代价。
const CHANCE_DEPTH_PENALTY: usize = 2;

/// 单次搜索的结果。
#[derive(Debug, Clone, Copy)]
pub struct MinimaxResult {
    /// 从当前玩家视角选择的最优动作。
    pub action: usize,
    /// 该动作的期望效用（当前玩家视角，[-1, 1]）。
    pub value: f32,
    /// 搜索展开的节点数（不含根）。
    pub nodes: u64,
}

/// 启发式静态评估：当前玩家 HP 差 / 初始 HP，归一化到 [-1, 1]。
///
/// 深度耗尽时作为叶子估值，保证与终局值（±1）同量纲。
pub fn heuristic_value(env: &DarkChessEnv) -> f32 {
    let my = env.get_hp(env.get_current_player());
    let opp = env.get_hp(env.get_current_player().opposite());
    (my - opp) as f32 / env.config.initial_health as f32
}

/// 终局值：从当前玩家视角的 ±1 / 0。
fn terminal_value(env: &DarkChessEnv, winner: Option<i32>) -> f32 {
    match winner {
        Some(w) if w == env.get_current_player().val() => 1.0,
        Some(w) if w == 0 => 0.0,
        _ => -1.0,
    }
}

/// 搜索当前节点（当前玩家视角）。`stats` 累计节点数。
fn search(env: &DarkChessEnv, depth: usize, alpha: f32, beta: f32, stats: &mut u64) -> f32 {
    let (terminated, _truncated, winner) = env.check_game_over_conditions();
    if terminated {
        return terminal_value(env, winner);
    }
    if depth == 0 {
        return heuristic_value(env);
    }

    let action_space = env.config.action_space_size;
    let mut masks = vec![0i32; action_space];
    env.action_masks_into(&mut masks);

    let mut best = f32::NEG_INFINITY;
    let mut a = alpha;
    for action in 0..action_space {
        if masks[action] != 1 {
            continue;
        }
        *stats += 1;

        if env.is_chance_action(action) {
            // --- 机会节点：期望值（期望本身不剪枝，但子搜索传递边界让确定性子树可剪枝） ---
            // 机会层"昂贵"：每个机会动作枚举 2*num_active 种翻棋结果，继续展开会指数爆炸，
            // 因此机会节点的子搜索深度惩罚 `CHANCE_DEPTH_PENALTY`（消耗更多深度，抑制爆炸）。
            let outcomes = env.chance_outcomes(action);
            let mut expected = 0.0;
            let child_depth = depth.saturating_sub(CHANCE_DEPTH_PENALTY);
            for (_, prob, next_env) in outcomes {
                // step 后玩家已翻转，search 返回 next_env 当前玩家视角，取负换回本视角
                let mut child_stats = 0u64;
                let v = -search(&next_env, child_depth, -beta, -alpha, &mut child_stats);
                *stats += child_stats;
                expected += prob * v;
            }
            best = best.max(expected);
        } else {
            // --- 确定性节点：negamax + alpha-beta 剪枝 ---
            let mut next_env = *env;
            let _ = next_env.step(action, None);
            let v = -search(&next_env, depth - 1, -beta, -a, stats);
            best = best.max(v);
        }

        a = a.max(best);
        if a >= beta {
            break; // alpha-beta 剪枝
        }
    }
    best
}

/// 求解当前局面的最优动作（expectiminimax + alpha-beta）。
///
/// `max_depth` 为搜索深度（不含根；机会节点每层消耗 1 深度）。
/// 返回 `None` 表示无合法动作（终局）。
pub fn minimax_best_action(env: &DarkChessEnv, max_depth: usize) -> Option<MinimaxResult> {
    let action_space = env.config.action_space_size;
    let mut masks = vec![0i32; action_space];
    env.action_masks_into(&mut masks);

    let legal: Vec<usize> = (0..action_space).filter(|&i| masks[i] == 1).collect();
    if legal.is_empty() {
        return None;
    }

    let mut best_action = legal[0];
    let mut best_value = f32::NEG_INFINITY;
    let mut total_nodes = 0u64;

    // 根层：对每个合法动作求值（当前玩家视角），取最大。
    for action in legal {
        let v = if env.is_chance_action(action) {
            let outcomes = env.chance_outcomes(action);
            let mut expected = 0.0;
            let child_depth = max_depth.saturating_sub(CHANCE_DEPTH_PENALTY);
            for (_, prob, next_env) in outcomes {
                let mut child_stats = 0u64;
                let child_v = -search(&next_env, child_depth, f32::NEG_INFINITY, f32::INFINITY, &mut child_stats);
                expected += prob * child_v;
                total_nodes += child_stats;
            }
            expected
        } else {
            let mut next_env = *env;
            let _ = next_env.step(action, None);
            let mut child_stats = 0u64;
            // 根层逐动作独立剪枝：用当前 best_value 作为 alpha 边界（保守，避免跨动作误剪）
            let v = -search(&next_env, max_depth - 1, f32::NEG_INFINITY, -best_value, &mut child_stats);
            total_nodes += child_stats;
            v
        };

        if v > best_value {
            best_value = v;
            best_action = action;
        }
    }

    Some(MinimaxResult {
        action: best_action,
        value: best_value,
        nodes: total_nodes,
    })
}

/// 以固定深度搜索并返回动作；等价于 `minimax_best_action(env, depth).map(|r| r.action)`。
pub fn minimax_choose_action(env: &DarkChessEnv, max_depth: usize) -> Option<usize> {
    minimax_best_action(env, max_depth).map(|r| r.action)
}
