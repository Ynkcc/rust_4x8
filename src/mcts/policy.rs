// src/mcts/policy.rs
// 策略计算层：概率分布、根策略、改进策略（Gumbel AlphaZero 训练目标）
//
// 分层说明：
// - 本模块只负责「由 MCTS 树状态产出概率 / 策略」的只读计算；
// - 树结构的构建与回溯见 tree.rs，搜索主循环见 search.rs。

use super::evaluator::Evaluator;
use super::search::GumbelMCTS;
use crate::game_env::GameEnv;
use rand::prelude::*;

impl<'a, G: GameEnv, E: Evaluator<G>> GumbelMCTS<'a, G, E> {
    /// 根据 Logits 和动作掩码计算概率分布
    pub(crate) fn compute_probs_from_logits(&self, logits: &[f32], masks: &[i32]) -> Vec<f32> {
        let mut probs = vec![0.0; logits.len()];
        let mut max_logit = f32::NEG_INFINITY;

        // 第一遍：找到最大 logit（数值稳定性）
        for (i, &logit) in logits.iter().enumerate() {
            if masks[i] == 1 && logit > max_logit {
                max_logit = logit;
            }
        }

        if !max_logit.is_finite() {
            return probs;
        }

        // 第二遍：计算指数并求和
        let mut sum = 0.0;
        for (i, &logit) in logits.iter().enumerate() {
            if masks[i] == 1 {
                let value = (logit - max_logit).exp();
                probs[i] = value;
                sum += value;
            }
        }

        // 第三遍：归一化
        if sum > 0.0 {
            for p in &mut probs {
                *p /= sum;
            }
        }

        probs
    }

    // ========================================================================
    // 注意：根节点 Dirichlet 噪声注入已被移除，请勿重新添加！
    //
    // 原因：本项目使用 Gumbel AlphaZero（Gumbel 论文探索方案），探索由以下
    // 机制提供，根节点先验 prior 不参与任何搜索决策：
    //   1. Gumbel Top-K 采样（sample_gumbel_top_k）使用子节点的 logit；
    //   2. Sequential Halving 淘汰基于 completed_q（Q 值）；
    //   3. 根节点第一跳由候选动作直接指定，不经过根节点 PUCT；
    //   4. 训练目标 get_improved_policy 使用 logit + σ·Q。
    // 因此修改根节点子节点的 prior（Dirichlet 注入）在搜索中是无效的空转，
    // 既不能提供探索，也不影响训练目标。
    // ========================================================================

    /// 获取根节点的访问概率分布
    ///
    /// 返回基于访问次数归一化的概率分布，可用于训练策略网络。已弃用，建议使用 `get_improved_policy` 获取 Gumbel AlphaZero 的改进策略。
    pub fn get_root_probabilities(&self) -> Vec<f32> {
        let mut probs = vec![0.0; G::action_space_size()];
        let root = self.arena.get(self.root_idx);
        let total = root.visit_count as f32;
        if total == 0.0 {
            return probs;
        }
        for (action, child_idx) in &root.children {
            let child = self.arena.get(*child_idx);
            if *action < probs.len() {
                probs[*action] = child.visit_count as f32 / total;
            }
        }
        probs
    }

    /// 基于根节点 completed Q 的温度策略（Gumbel AlphaZero 论文标准动作选择）
    ///
    /// π(a) ∝ exp(Q_comp(a) / τ)
    /// - τ = 1: 对 completed Q 做 softmax，鼓励探索
    /// - τ → 0: 趋向 argmax，确定性选择
    /// 仅对合法动作计算，非法动作保持 0。
    ///
    /// 注意：此处刻意使用 completed Q 而非访问计数 N^(1/τ)（经典 AlphaZero 做法）。
    /// Sequential Halving 结束后 surviving 候选的访问次数基本均分，基于 N 的策略
    /// 会退化为近似均匀采样、丢失动作质量信息；而 completed Q 保留了质量排序，
    /// 符合 Gumbel AlphaZero 论文（Policy improvement by planning with Gumbel）
    /// 的动作选择方式。请勿替换回基于 visit_count 的实现。
    pub fn get_root_completed_q_policy(&self, temperature: f32) -> Vec<f32> {
        let mut policy = vec![0.0; G::action_space_size()];

        let env = match self.arena.get(self.root_idx).env.as_ref() {
            Some(env) => env,
            None => return policy,
        };
        let mut masks = vec![0; G::action_space_size()];
        env.action_masks_into(&mut masks);

        let tau = temperature.max(1e-4);
        let inv_tau = 1.0 / tau;

        // 数值稳定性：先减去最大 completed Q 再做 exp，避免溢出
        let mut max_q = f32::NEG_INFINITY;
        for action in 0..G::action_space_size() {
            if masks[action] == 1 {
                max_q = max_q.max(self.completed_q(action));
            }
        }
        if !max_q.is_finite() {
            return policy;
        }

        let mut sum = 0.0;
        for action in 0..G::action_space_size() {
            if masks[action] == 1 {
                let value = ((self.completed_q(action) - max_q) * inv_tau).exp();
                policy[action] = value;
                sum += value;
            }
        }

        if sum > 0.0 {
            for p in policy.iter_mut() {
                *p /= sum;
            }
        }
        policy
    }

    /// 从离散概率分布中采样一个动作（仅合法动作）
    pub fn sample_action_from_policy(probs: &[f32], masks: &[i32]) -> usize {
        let mut rng = thread_rng();
        let mut sum = 0.0;
        for i in 0..probs.len() {
            if masks[i] == 1 {
                sum += probs[i];
            }
        }
        if sum <= 0.0 {
            for i in 0..masks.len() {
                if masks[i] == 1 {
                    return i;
                }
            }
            return 0;
        }
        let mut r: f32 = rng.gen_range(0.0..sum);
        for i in 0..probs.len() {
            if masks[i] == 1 {
                r -= probs[i];
                if r <= 0.0 {
                    return i;
                }
            }
        }
        for i in (0..probs.len()).rev() {
            if masks[i] == 1 {
                return i;
            }
        }
        0
    }

    /// 获取 Gumbel AlphaZero 的改进策略 (pi_target)
    ///
    /// 使用 root 的先验 logit 与 completed_Q 直接组合，计算 softmax 概率。
    pub fn get_improved_policy(&self) -> Vec<f32> {
        let mut policy = vec![0.0; G::action_space_size()];
        let env = match self.arena.get(self.root_idx).env.as_ref() {
            Some(env) => env,
            None => return policy,
        };

        let mut masks = vec![0; G::action_space_size()];
        env.action_masks_into(&mut masks);

        // 1. 计算打分: logit + sigma * completed_q
        // sigma = c_scale * ln(1 + N_root)  —— 按 Gumbel AlphaZero 论文
        let root = self.arena.get(self.root_idx);
        let root_visit_count = root.visit_count as f32;
        let sigma_scale = self.config.c_scale * (1.0 + root_visit_count).ln();

        let mut scores = vec![f32::NEG_INFINITY; G::action_space_size()];
        let mut max_score = f32::NEG_INFINITY;

        for action in 0..G::action_space_size() {
            if masks[action] != 1 {
                continue;
            }
            let child_idx = match root
                .children
                .iter()
                .find(|(act, _)| *act == action)
                .map(|(_, idx)| *idx)
            {
                Some(idx) => idx,
                None => continue,
            };
            let child = self.arena.get(child_idx);
            let q = self.completed_q(action);
            let score = child.logit + sigma_scale * q;
            scores[action] = score;
            if score > max_score {
                max_score = score;
            }
        }

        // 3. 计算 Softmax（带数值稳定性）
        // 若所有合法动作的 score 均非有限（如网络输出 NaN/Inf logit），
        // 直接回退到均匀分布，而不是返回全 0 policy：
        //   - 全 0 policy 进入训练会让 policy_loss 变为 0（梯度消失，策略头退化）；
        //   - 若与 -inf 的 log_softmax 相乘还会产生 NaN。
        // 均匀回退至少保留一个合法归一化分布，避免训练目标被污染。
        if !max_score.is_finite() {
            let count = masks.iter().sum::<i32>() as f32;
            if count > 0.0 {
                for i in 0..G::action_space_size() {
                    if masks[i] == 1 {
                        policy[i] = 1.0 / count;
                    }
                }
            }
            return policy;
        }

        let mut sum = 0.0;
        for action in 0..G::action_space_size() {
            let score = scores[action];
            if score.is_finite() {
                let value = (score - max_score).exp();
                policy[action] = value;
                sum += value;
            }
        }

        // 4. 归一化概率，异常时回退到均匀分布
        if sum > 0.0 {
            for p in policy.iter_mut() {
                *p /= sum;
            }
        } else {
            let count = masks.iter().sum::<i32>() as f32;
            if count > 0.0 {
                for i in 0..G::action_space_size() {
                    if masks[i] == 1 {
                        policy[i] = 1.0 / count;
                    }
                }
            }
        }

        policy
    }
}
