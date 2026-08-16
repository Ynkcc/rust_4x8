// src/ai/heuristic_mcts.rs
// 纯计算启发式 Gumbel MCTS 对手（无需 torch）：
//   - `HeuristicEvaluator`：实现泛型 `Evaluator<DarkChessEnv>`，
//     输出基于规则的走子先验 Logits 与多特征启发式 Value；
//   - `HeuristicMctsPolicy`：用该评估器驱动现有 Gumbel MCTS 搜索选择动作。

use crate::mcts::{Evaluator, GumbelConfig, GumbelMCTS};
use crate::DarkChessEnv;

use super::eval::{EvalParams, evaluate};
use super::movegen::Move;

/// 基于规则的走子先验 Logits。
///
/// 直觉：吃子（按被吃子价值排序）> 吃暗子（机会） > 翻棋 > 静走子，
/// 使 Gumbel Top-K 采样优先考虑吃子与翻棋这类“信息量/收益高”的动作，
/// 静走子仅在没有更好选择时进入候选。
fn prior_logit(env: &DarkChessEnv, m: &Move, params: &EvalParams, scale: f32) -> f32 {
    if m.is_capture {
        let victim = match &env.get_board_slots()[m.to] {
            crate::Slot::Revealed(p) => params.values[p.piece_type as usize],
            _ => 0.0,
        };
        return 3.0 + victim * scale;
    }
    if m.is_chance {
        // 吃暗子：结果不可控，比明子吃子略低
        return 1.5;
    }
    if m.is_flip {
        return 1.0;
    }
    0.5 // 静走子
}

/// 纯计算启发式评估器（实现 `Evaluator<DarkChessEnv>`）。
pub struct HeuristicEvaluator {
    pub params: EvalParams,
    /// 被吃子价值对先验 Logits 的缩放（吃子优先级强度）
    pub prior_scale: f32,
}

impl HeuristicEvaluator {
    pub fn new() -> Self {
        Self {
            params: EvalParams::default(),
            prior_scale: 0.5,
        }
    }
}

impl Default for HeuristicEvaluator {
    fn default() -> Self {
        Self::new()
    }
}

impl Evaluator<DarkChessEnv> for HeuristicEvaluator {
    fn evaluate(&self, envs: &[DarkChessEnv]) -> (Vec<Vec<f32>>, Vec<f32>) {
        let mut logits = Vec::with_capacity(envs.len());
        let mut values = Vec::with_capacity(envs.len());
        for env in envs {
            // config 驱动：动作空间大小随变体变化（4x8=352 / 4x2=40 / 4x4=112）
            let mut lg = vec![0.0f32; env.config.action_space_size];
            for m in super::movegen::generate_moves(env, env.get_current_player()) {
                lg[m.action] = prior_logit(env, &m, &self.params, self.prior_scale);
            }
            logits.push(lg);
            values.push(evaluate(env, &self.params));
        }
        (logits, values)
    }
}

/// 纯计算启发式 MCTS 策略（每次调用创建新的 GumbelMCTS，简单可靠）。
pub struct HeuristicMctsPolicy {
    /// 模拟次数
    pub sims: usize,
    /// Gumbel Top-K 候选动作数
    pub max_considered_actions: usize,
    /// 评估参数
    pub params: EvalParams,
}

impl HeuristicMctsPolicy {
    pub fn new(sims: usize) -> Self {
        Self {
            sims: sims.max(1),
            max_considered_actions: 16,
            params: EvalParams::default(),
        }
    }

    pub fn set_iterations(&mut self, sims: usize) {
        self.sims = sims.max(1);
    }

    pub fn set_max_considered_actions(&mut self, k: usize) {
        self.max_considered_actions = k.max(1);
    }

    pub fn choose_action(&self, env: &DarkChessEnv) -> Option<usize> {
        let evaluator = HeuristicEvaluator {
            params: self.params,
            prior_scale: 0.5,
        };
        let config = GumbelConfig {
            num_simulations: self.sims,
            max_considered_actions: self.max_considered_actions,
            c_scale: 1.0,
            gumbel_scale: 1.0,
        };
        let mut mcts = GumbelMCTS::new(env, &evaluator, config);
        mcts.run().map(|r| r.action)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn heuristic_policy_returns_legal_action() {
        let mut env = DarkChessEnv::new();
        env.seed = Some(9);
        env.reset();
        let policy = HeuristicMctsPolicy::new(24);
        let action = policy.choose_action(&env).expect("应返回动作");
        let mut masks = vec![0i32; env.config.action_space_size];
        env.action_masks_into(&mut masks);
        assert_eq!(masks[action], 1, "启发式 MCTS 返回非法动作 {}", action);
    }
}
