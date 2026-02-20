// src/mcts/config.rs
// MCTS 搜索配置与结果定义

use crate::{Observation, Player};

/// Gumbel MCTS 配置参数
///
/// 用于控制 MCTS 搜索过程的超参数。
#[derive(Clone)]
pub struct GumbelConfig {
    /// 模拟总次数 (N_sim)
    pub num_simulations: usize,
    /// 初始考虑的最大动作数 (m)
    /// 在 Gumbel Top-K 采样中，最多选择多少个动作进行评估。
    pub max_considered_actions: usize,
    /// 访问次数缩放因子 (c_visit)
    /// 预留用于策略或 completed_Q 的尺度调节（当前不直接参与核心计算）
    pub c_visit: f32,
    /// Gumbel 噪声缩放因子 (c_scale)
    pub c_scale: f32,
    /// 是否处于训练模式
    /// 如果为 true，可能会影响某些行为 (如噪声注入)，目前主要作为标记。
    pub train: bool,
}

impl Default for GumbelConfig {
    /// 默认配置
    ///
    /// * simulations: 64
    /// * max_considered_actions: 16
    /// * c_visit: 50.0
    /// * c_scale: 1.0
    /// * train: false
    fn default() -> Self {
        Self {
            num_simulations: 64,
            max_considered_actions: 16,
            c_visit: 50.0,
            c_scale: 1.0,
            train: false,
        }
    }
}

/// MCTS 搜索结果
///
/// 包含 MCTS 搜索后的所有关键数据，避免在 self-play 中重复计算
#[derive(Debug, Clone)]
pub struct MctsSearchResult {
    /// 选择的动作索引
    pub action: usize,
    /// 当前状态的观测
    pub state: Observation,
    /// 改进的策略概率分布
    pub improved_policy: Vec<f32>,
    /// MCTS 根节点价值
    pub mcts_value: f32,
    /// 选择动作的 completed_Q 值
    pub completed_q: f32,
    /// 根节点访问次数
    pub root_visit_count: u32,
    /// 当前玩家
    pub player: Player,
    /// 动作掩码
    pub action_mask: Vec<i32>,
}
