// src/mcts/evaluator.rs
// 神经网络评估接口定义

use crate::DarkChessEnv;

/// 评估器特征 (Trait)
///
/// 定义了评估游戏状态的接口。
/// 实现该特征的结构体 (如神经网络模型) 需要提供状态评估功能。
pub trait Evaluator {
    /// 评估给定的游戏环境批次
    ///
    /// # 参数
    ///
    /// * `envs` - 需要评估的 `DarkChessEnv` 列表
    ///
    /// # 返回
    ///
    /// 返回一个元组 `(logits, values)`:
    /// * `logits`: 每个环境的动作原始 Logits（未 mask/softmax）
    /// * `values`: 每个环境的状态价值
    fn evaluate(&self, envs: &[DarkChessEnv]) -> (Vec<Vec<f32>>, Vec<f32>);

    /// 评估并返回 Logits 和 Value
    ///
    /// 默认实现直接返回 `evaluate` 的结果。
    /// Logits 用于 Gumbel 分布的采样。
    ///
    /// # 参数
    ///
    /// * `envs` - 需要评估的 `DarkChessEnv` 列表
    ///
    /// # 返回
    ///
    /// * `logits`: 每个环境的动作对数概率
    /// * `values`: 每个环境的状态价值
    fn evaluate_logits(&self, envs: &[DarkChessEnv]) -> (Vec<Vec<f32>>, Vec<f32>) {
        self.evaluate(envs)
    }
}
