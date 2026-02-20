// src/self_play.rs - 自对弈与数据生成模块 (同步版)
//
// 本模块实现了自对弈（Self-Play）逻辑，用于生成强化学习所需的训练数据。
// 重构说明：
// - 移除异步依赖，改为同步执行
// - 直接持有模型引用，无需 Channel 通信
// - 使用 Gumbel AlphaZero MCTS

use crate::game_env::{DarkChessEnv, Observation};
use crate::mcts::{Evaluator, GumbelConfig, GumbelMCTS};
use std::time::Instant;

// ================ 数据结构定义 ================

/// 游戏简要统计信息
#[derive(Debug, Clone)]
pub struct GameStats {
    /// 游戏总步数
    pub steps: usize,
    /// 获胜方: Some(1)=红胜, Some(-1)=黑胜, None/Some(0)=平局
    pub winner: Option<i32>,
}

/// 单局游戏的完整数据记录
///
/// 包含该局游戏中每一步的观测状态、MCTS 搜索产生的策略概率、
/// MCTS 估算的根节点价值以及最终的游戏结果。
#[derive(Debug, Clone)]
pub struct GameEpisode {
    /// 训练样本列表: (观测状态, 策略概率分布, MCTS根节点价值, completed_Q, 根节点访问次数, 最终回报, 动作掩码)
    pub samples: Vec<(Observation, Vec<f32>, f32, f32, u32, f32, Vec<i32>)>,
    /// 游戏总步数
    pub game_length: usize,
    /// 获胜方
    pub winner: Option<i32>,
}

// ================ 场景定义 ================

/// 训练场景类型枚举
#[derive(Debug, Clone, Copy)]
pub enum ScenarioType {
    /// 场景1: 双士残局 (R_A vs B_A)
    TwoAdvisors,
    /// 场景2: 隐藏威胁 (Hidden Threat)
    HiddenThreats,
    /// 标准开局 - 正常的完整游戏
    Standard,
}

impl ScenarioType {
    /// 根据枚举值创建对应的游戏环境
    pub fn create_env(&self) -> DarkChessEnv {
        let env = DarkChessEnv::new();
        match self {
            ScenarioType::TwoAdvisors => {
                // env.setup_two_advisors(Player::Black); // Removed in refactor
            }
            ScenarioType::HiddenThreats => {
                // env.setup_hidden_threats(); // Removed in refactor
            }
            ScenarioType::Standard => {}
        }
        env
    }

    /// 获取场景的描述名称
    pub fn name(&self) -> &'static str {
        match self {
            ScenarioType::TwoAdvisors => "TwoAdvisors (R_A vs B_A)",
            ScenarioType::HiddenThreats => "HiddenThreats",
            ScenarioType::Standard => "Standard",
        }
    }

    /// 获取该场景下的期望最优动作索引 (用于验证/调试)
    pub fn expected_action(&self) -> usize {
        match self {
            ScenarioType::TwoAdvisors => 38,
            ScenarioType::HiddenThreats => 3,
            ScenarioType::Standard => 0,
        }
    }
}

// ================ 自对弈配置 ================

/// 自对弈配置
#[derive(Clone)]
pub struct SelfPlayConfig {
    /// 每次决策执行的 MCTS 模拟次数
    pub mcts_sims: usize,
    /// Gumbel Top-K 候选动作数
    pub max_considered_actions: usize,
    /// MCTS 根节点 Dirichlet 噪声的 Alpha 参数
    pub dirichlet_alpha: f32,
    /// MCTS 根节点 Dirichlet 噪声的权重 (Epsilon)
    pub dirichlet_epsilon: f32,
    /// 温度采样的步数阈值
    pub temperature_steps: usize,
    /// 训练场景
    pub scenario: ScenarioType,
}

impl Default for SelfPlayConfig {
    fn default() -> Self {
        Self {
            mcts_sims: 64,
            max_considered_actions: 16,
            dirichlet_alpha: 0.3,
            dirichlet_epsilon: 0.25,
            temperature_steps: 10,
            scenario: ScenarioType::Standard,
        }
    }
}

// ================ 自对弈运行器 (同步) ================

/// 自对弈运行器
///
/// 直接持有评估器引用，同步执行
pub struct SelfPlayRunner<'a, E: Evaluator> {
    evaluator: &'a E,
    config: SelfPlayConfig,
}

impl<'a, E: Evaluator> SelfPlayRunner<'a, E> {
    /// 创建新的自对弈运行器
    pub fn new(evaluator: &'a E, config: SelfPlayConfig) -> Self {
        Self { evaluator, config }
    }
    
    /// 使用默认配置创建
    pub fn with_defaults(evaluator: &'a E, mcts_sims: usize) -> Self {
        let config = SelfPlayConfig {
            mcts_sims,
            ..Default::default()
        };
        Self { evaluator, config }
    }

    /// 执行一局完整的自对弈 (同步)
    pub fn play_episode(&self, _episode_num: usize) -> GameEpisode {
        let _start_time = Instant::now();

        // 1. 初始化环境
        let mut env = self.config.scenario.create_env();

        // 2. 配置 MCTS
        let mcts_config = GumbelConfig {
            num_simulations: self.config.mcts_sims,
            max_considered_actions: self.config.max_considered_actions,
            c_visit: 50.0,
            c_scale: 1.0,
            train: true,
        };
        let mut mcts = GumbelMCTS::new(&env, self.evaluator, mcts_config.clone());

        let mut episode_data = Vec::new();
        let mut step = 0;

        // 3. 游戏主循环
        loop {
            // --- MCTS 搜索 (同步) ---
            let search_result = match mcts.run() {
                Some(result) => result,
                None => {
                    // 无有效动作，游戏结束
                    let mut samples = Vec::new();
                    for (obs, p, mcts_val, completed_q, root_visit_count, _, mask) in episode_data {
                        samples.push((obs, p, mcts_val, completed_q, root_visit_count, 0.0, mask));
                    }
                    return GameEpisode {
                        samples,
                        game_length: step,
                        winner: None,
                    };
                }
            };
            
            let action = search_result.action;
            let completed_q = search_result.completed_q;

            // --- 收集样本数据 ---
            episode_data.push((
                search_result.state,
                search_result.improved_policy,
                search_result.mcts_value,
                completed_q,
                search_result.root_visit_count,
                search_result.player,
                search_result.action_mask,
            ));

            // --- 执行动作 ---
            match env.step(action, None) {
                Ok((_, _, terminated, truncated, winner)) => {
                    // 推进 MCTS 树
                    mcts.step_next(&env, action);

                    if terminated || truncated {
                        // --- 游戏结束处理 ---
                        let reward_red: f32 = match winner {
                            Some(1) => 1.0,
                            Some(-1) => -1.0,
                            _ => 0.0,
                        };

                        // --- 回填价值 ---
                        let mut samples = Vec::new();
                        for (obs, p, mcts_val, completed_q, root_visit_count, player, mask) in episode_data {
                            let game_result_val: f32 = if player.val() == 1 {
                                reward_red
                            } else {
                                -reward_red
                            };
                            samples.push((obs, p, mcts_val, completed_q, root_visit_count, game_result_val, mask));
                        }

                        return GameEpisode {
                            samples,
                            game_length: step,
                            winner,
                        };
                    }
                }
                Err(e) => {
                    eprintln!("  ⚠️ 游戏错误 (step={}, action={}): {}", step, action, e);
                    return GameEpisode {
                        samples: Vec::new(),
                        game_length: step,
                        winner: None,
                    };
                }
            }

            // --- 步数限制检查 ---
            step += 1;
            if step > 200 {
                let mut samples = Vec::new();
                for (obs, p, mcts_val, completed_q, root_visit_count, _, mask) in episode_data {
                    samples.push((obs, p, mcts_val, completed_q, root_visit_count, 0.0, mask));
                }
                return GameEpisode {
                    samples,
                    game_length: step,
                    winner: None,
                };
            }
        }
    }
}

// ================ 高级 API ================

/// 运行单局自对弈
pub fn run_self_play<E: Evaluator>(
    evaluator: &E,
    config: &SelfPlayConfig,
) -> GameEpisode {
    let runner = SelfPlayRunner::new(evaluator, config.clone());
    runner.play_episode(0)
}

/// 批量运行多局自对弈
pub fn run_batch_self_play<E: Evaluator>(
    evaluator: &E,
    config: &SelfPlayConfig,
    num_games: usize,
) -> Vec<GameEpisode> {
    (0..num_games)
        .map(|i| {
            let runner = SelfPlayRunner::new(evaluator, config.clone());
            runner.play_episode(i)
        })
        .collect()
}

// ================ 辅助函数 ================

/// 选择 completed_Q 最大的动作（确定性）
pub fn select_completed_q_action<E: Evaluator>(
    mcts: &GumbelMCTS<E>,
    masks: &[i32],
) -> (usize, f32) {
    let mut best_action: Option<usize> = None;
    let mut best_completed_q = f32::NEG_INFINITY;

    for (action, &mask) in masks.iter().enumerate() {
        if mask != 1 {
            continue;
        }
        let completed_q = mcts.get_root_completed_q(action);
        if completed_q > best_completed_q {
            best_completed_q = completed_q;
            best_action = Some(action);
        }
    }

    let action = best_action.expect("无有效动作");
    (action, best_completed_q)
}

/// 获取 Top-K 动作 (用于调试)
pub fn get_top_k_actions(probs: &[f32], k: usize) -> Vec<(usize, f32)> {
    let mut indexed: Vec<(usize, f32)> = probs.iter().enumerate().map(|(i, &p)| (i, p)).collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    indexed.into_iter().take(k).collect()
}