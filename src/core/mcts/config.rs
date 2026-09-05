// src/mcts/config.rs
// MCTS 搜索配置与结果定义

use crate::core::env::{ResNetObservation, Player};

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
    /// PUCT 探索系数（即经典 c_puct）：
    /// - 非根节点选择阶段：u_score = c_scale * prior * sqrt(N_parent) / (1 + N_child)
    /// - 训练目标 improved_policy：sigma = c_scale * ln(1 + N_root)
    pub(crate) c_scale: f32,
    /// Gumbel 噪声尺度（Gumbel(0, gumbel_scale)）。
    /// Gumbel AlphaZero 根探索主力：Top-K 采样为每个候选动作 logit 加该尺度噪声。
    /// 越大探索越强，越小越接近纯 logit 排序；1.0 为标准 Gumbel。
    pub(crate) gumbel_scale: f32,
    // 注意：Gumbel AlphaZero 的根节点探索由 Gumbel 噪声（Top-K 采样）与
    // Sequential Halving 提供；根节点子节点的 prior 不参与任何搜索决策
    // （Top-K 用 logit、根选择不经 PUCT、训练目标用 logit + σ·Q）。
    // 曾存在 Dirichlet 噪声注入及配套的 train 标记字段，因无效已移除，
    // 请勿重新添加。
    // --------------------------------------------------------------------
    // 血量差异头（可选）：启用后把血量期望并入动作选择效用
    //   U = Q_win + λ(|Q_win|) · Q_hp
    // health_enabled=false（默认）时复合效用退化为纯胜率 Q_win，行为与旧版逐位等价。
    /// 是否启用血量差异头参与搜索（复合效用）
    pub health_enabled: bool,
    /// 复合效用中血量期望权重 λ（0 = 纯胜率）
    pub health_weight: f32,
    /// λ 随 |v_win| 的自适应幂指数；0 = 常量 λ
    pub health_confidence_exp: f32,
}

impl Default for GumbelConfig {
    /// 默认配置
    ///
    /// * simulations: 64
    /// * max_considered_actions: 16
    /// * c_scale: 1.0
    /// * gumbel_scale: 1.0
    fn default() -> Self {
        Self {
            num_simulations: 64,
            max_considered_actions: 16,
            c_scale: 1.0,
            gumbel_scale: 1.0,
            health_enabled: false,
            health_weight: 0.0,
            health_confidence_exp: 0.0,
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
    pub state: ResNetObservation,
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
