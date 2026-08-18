// src/mcts/sampling.rs
// 采样层：Gumbel Top-K 采样与机会结果采样（泛型化：G = 游戏环境）
//
// 分层说明：
// - 本模块只负责「从 Logits / 概率分布中采样」的算法，不触碰树结构；
// - 树结构的构建与回溯见 tree.rs，搜索主循环见 search.rs。

use super::evaluator::Evaluator;
use super::search::GumbelMCTS;
use crate::game_env::GameEnv;
use rand::prelude::*;
use rand_distr::Gumbel;

impl<'a, G: GameEnv, E: Evaluator<G>> GumbelMCTS<'a, G, E> {
    /// 执行 Gumbel-Top-K 采样
    ///
    /// 从 Logits 中添加 Gumbel 噪声并选择前 K 个动作。
    /// 这是 Gumbel AlphaZero 的核心机制，用于在不进行完全树搜索的情况下选择候选动作。
    /// 使用内部 scratch_gumbel 缓存以避免重复堆分配。
    pub(crate) fn sample_gumbel_top_k(
        &mut self,
        logits: &[f32],
        masks: &[i32],
        k: usize,
    ) -> Vec<usize> {
        let gumbel_dist = Gumbel::new(0.0, self.config.gumbel_scale as f64).unwrap();

        // 清空并复用 scratch_gumbel
        self.scratch_gumbel.clear();
        for (i, &logit) in logits.iter().enumerate() {
            if masks[i] == 1 {
                let noise: f64 = gumbel_dist.sample(&mut self.rng);
                self.scratch_gumbel.push((i, logit + noise as f32));
            }
        }

        // 按加噪后的 Logits 降序排序
        self.scratch_gumbel
            .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let actual_k = k.min(self.scratch_gumbel.len());
        self.scratch_gumbel
            .iter()
            .take(actual_k)
            .map(|(i, _)| *i)
            .collect()
    }

    /// 从机会节点的可能结果中采样
    ///
    /// 根据各种结果的概率分布，随机采样一个结果 ID。
    /// 主要用于模拟阶段，决定在机会节点走向哪个分支。
    pub(crate) fn sample_outcome_id(
        outcomes: &[(usize, f32, usize)],
        rng: &mut impl Rng,
    ) -> Option<usize> {
        if outcomes.is_empty() {
            return None;
        }
        let total: f32 = outcomes.iter().map(|(_, p, _)| p).sum();
        if total <= 0.0 {
            return outcomes.first().map(|(id, _, _)| *id);
        }
        let mut pick = rng.gen_range(0.0..1.0) * total;
        for (outcome_id, prob, _) in outcomes {
            pick -= *prob;
            if pick <= 0.0 {
                return Some(*outcome_id);
            }
        }
        outcomes.first().map(|(id, _, _)| *id)
    }
}
