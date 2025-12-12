// src/self_play.rs - 自对弈与数据生成模块
//
// 本模块实现了并行的自对弈（Self-Play）逻辑，用于生成强化学习所需的训练数据。
//
// 更新说明：
// - SelfPlayWorker 改为泛型 Struct，支持任意实现了 Evaluator trait 的评估器。
// - play_episode 改为 async 方法，以支持异步 MCTS。

use crate::game_env::{DarkChessEnv, Observation, Player};
use crate::mcts::{Evaluator, MCTSConfig, MCTS};
use rand::distributions::WeightedIndex;
use rand::prelude::*;
use std::sync::Arc;
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
    /// 训练样本列表: (观测状态, 策略概率分布, MCTS根节点价值, 最终回报, 动作掩码)
    pub samples: Vec<(Observation, Vec<f32>, f32, f32, Vec<i32>)>,
    /// 游戏总步数
    pub game_length: usize,
    /// 获胜方
    pub winner: Option<i32>,
}

// ================ 场景定义 ================

/// 训练场景类型枚举
///
/// 用于在自对弈开始时设置特定的棋盘局面，以便模型能针对性地学习某些特定战术或残局。
#[derive(Debug, Clone, Copy)]
pub enum ScenarioType {
    /// 场景1: 双士残局 (R_A vs B_A) - 测试基本的移动和吃子逻辑
    TwoAdvisors,
    /// 场景2: 隐藏威胁 (Hidden Threat) - 测试翻棋与炮击等复杂逻辑
    HiddenThreats,
    /// 标准开局 - 正常的完整游戏
    Standard,
}

impl ScenarioType {
    /// 根据枚举值创建对应的游戏环境
    pub fn create_env(&self) -> DarkChessEnv {
        let mut env = DarkChessEnv::new();
        match self {
            ScenarioType::TwoAdvisors => env.setup_two_advisors(Player::Black),
            ScenarioType::HiddenThreats => env.setup_hidden_threats(),
            ScenarioType::Standard => {} // 默认为标准开局
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
            ScenarioType::Standard => 0, // 标准开局无特定单一最优解，此处仅为占位
        }
    }
}

// ================ 自对弈工作器 (泛型 & 异步) ================

/// 自对弈工作器
///
/// 泛型参数 `E`: 必须实现 `Evaluator` trait (如 ChannelEvaluator 或 GrpcEvaluator)
pub struct SelfPlayWorker<E: Evaluator> {
    /// 工作器 ID，用于日志区分
    pub worker_id: usize,
    /// 神经网络评估器，通过通道与推理服务器通信
    pub evaluator: Arc<E>,
    /// 每次决策执行的 MCTS 模拟次数
    pub mcts_sims: usize,
    /// 指定的训练场景 (None 表示随机/标准)
    pub scenario: Option<ScenarioType>,
    /// MCTS 根节点 Dirichlet 噪声的 Alpha 参数 (控制噪声分布的集中程度)
    pub dirichlet_alpha: f32,
    /// MCTS 根节点 Dirichlet 噪声的权重 (Epsilon)
    pub dirichlet_epsilon: f32,
    /// 温度采样的步数阈值。前 N 步使用温度=1进行探索，之后使用温度=0进行贪婪选择。
    pub temperature_steps: usize,
}

impl<E: Evaluator + 'static> SelfPlayWorker<E> {
    /// 创建一个新的自对弈工作器 (默认参数)
    pub fn new(worker_id: usize, evaluator: Arc<E>, mcts_sims: usize) -> Self {
        Self {
            worker_id,
            evaluator,
            mcts_sims,
            scenario: None,
            dirichlet_alpha: 0.3,
            dirichlet_epsilon: 0.25,
            temperature_steps: 10,
        }
    }

    /// 创建指定场景的工作器
    pub fn with_scenario(
        worker_id: usize,
        evaluator: Arc<E>,
        mcts_sims: usize,
        scenario: ScenarioType,
    ) -> Self {
        Self {
            worker_id,
            evaluator,
            mcts_sims,
            scenario: Some(scenario),
            dirichlet_alpha: 0.3,
            dirichlet_epsilon: 0.25,
            temperature_steps: 10,
        }
    }
    
    /// 创建指定场景并自定义 Dirichlet 噪声参数的工作器
    pub fn with_scenario_and_dirichlet(
        worker_id: usize,
        evaluator: Arc<E>,
        mcts_sims: usize,
        scenario: ScenarioType,
        dirichlet_alpha: f32,
        dirichlet_epsilon: f32,
    ) -> Self {
        Self {
            worker_id,
            evaluator,
            mcts_sims,
            scenario: Some(scenario),
            dirichlet_alpha,
            dirichlet_epsilon,
            temperature_steps: 10,
        }
    }
    
    /// 全配置构造函数：场景、噪声参数、温度采样步数
    pub fn with_scenario_dirichlet_and_temperature(
        worker_id: usize,
        evaluator: Arc<E>,
        mcts_sims: usize,
        scenario: ScenarioType,
        dirichlet_alpha: f32,
        dirichlet_epsilon: f32,
        temperature_steps: usize,
    ) -> Self {
        Self {
            worker_id,
            evaluator,
            mcts_sims,
            scenario: Some(scenario),
            dirichlet_alpha,
            dirichlet_epsilon,
            temperature_steps,
        }
    }

    /// 执行一局完整的自对弈 (异步)
    ///
    /// # 参数
    /// - `episode_num`: 当前局数的索引 (主要用于日志)
    ///
    /// # 返回
    /// - `GameEpisode`: 包含本局游戏的所有训练样本和结果
    pub async fn play_episode(&self, _episode_num: usize) -> GameEpisode {
        let _scenario_name = self.scenario.map(|s| s.name()).unwrap_or("Random");
        let start_time = Instant::now();

        // 1. 初始化环境
        let mut env = match self.scenario {
            Some(scenario) => scenario.create_env(),
            None => DarkChessEnv::new(),
        };

        // 2. 配置 MCTS
        let config = MCTSConfig {
            num_simulations: self.mcts_sims,
            cpuct: 1.0,
            virtual_loss: 1.0,
            num_mcts_workers: 8,
            dirichlet_alpha: self.dirichlet_alpha,
            dirichlet_epsilon: self.dirichlet_epsilon,
            train: true,
            max_pending_inference: 32,
        };
        let mut mcts = MCTS::new(&env, self.evaluator.clone(), config);

        let mut episode_data = Vec::new();
        let mut step = 0;

        // 预分配掩码缓冲区
        let mut masks = vec![0; crate::game_env::ACTION_SPACE_SIZE];

        // 3. 游戏主循环
        loop {
            // --- MCTS 搜索 (异步等待) ---
            mcts.run().await;
            
            let probs = mcts.get_root_probabilities().await;
            env.action_masks_into(&mut masks);
            let mcts_value = mcts.root.read().await.q_value();

            // --- 收集样本数据 ---
            // 存储当前状态、MCTS计算出的策略概率、MCTS价值、当前玩家、动作掩码
            episode_data.push((
                env.get_state(),
                probs.clone(),
                mcts_value,
                env.get_current_player(),
                masks.clone(),
            ));

            // --- 动作选择 ---
            // 根据当前步数决定使用探索性采样 (Temperature=1) 还是贪婪采样 (Temperature=0)
            let current_step = env.get_total_steps();
            let temperature = if current_step < self.temperature_steps {
                1.0
            } else {
                0.0
            };
            let action = sample_action(&probs, &env, temperature);

            // --- 执行动作 ---
            match env.step(action, None) {
                Ok((_, _, terminated, truncated, winner)) => {
                    // 推进 MCTS 树 (异步)
                    mcts.step_next(&env, action).await;

                    if terminated || truncated {
                        // --- 游戏结束处理 ---
                        
                        // 计算红方视角的最终奖励
                        let reward_red = match winner {
                            Some(1) => 1.0,
                            Some(-1) => -1.0,
                            _ => 0.0,
                        };

                        let _elapsed = start_time.elapsed();

                        // --- 回填价值 (Value Backfilling) ---
                        // 将最终的游戏结果 (Win/Loss/Draw) 作为真实的 Value Target 回填给每一步
                        // 注意：价值是相对于当前行动玩家的，所以需要根据玩家身份翻转符号
                        let mut samples = Vec::new();
                        for (obs, p, mcts_val, player, mask) in episode_data {
                            let game_result_val = if player.val() == 1 {
                                reward_red
                            } else {
                                -reward_red
                            };
                            samples.push((obs, p, mcts_val, game_result_val, mask));
                        }

                        return GameEpisode {
                            samples,
                            game_length: step,
                            winner,
                        };
                    }
                }
                Err(e) => {
                    eprintln!("  ⚠️ [Worker-{}] 游戏错误 (step={}, action={}): {}", 
                        self.worker_id, step, action, e);
                    // 发生错误，丢弃本局数据
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
                // 超过最大步数强制平局
                let mut samples = Vec::new();
                for (obs, p, mcts_val, _, mask) in episode_data {
                    samples.push((obs, p, mcts_val, 0.0, mask));
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

// ================ 辅助函数 ================

/// 动作采样函数
///
/// 根据概率分布 `probs` 和温度参数 `temperature` 选择一个动作。
/// - `temperature = 0`: 贪婪选择，直接选概率最大的动作。
/// - `temperature = 1`: 按照概率分布进行随机采样。
/// - `temperature > 1`: 使分布更平滑，增加探索。
/// - `temperature < 1`: 使分布更尖锐，减少探索。
///
/// 始终会应用动作掩码 `env.action_masks()` 以确保采样的合法性。
pub fn sample_action(probs: &[f32], env: &DarkChessEnv, temperature: f32) -> usize {
    // 1. 获取合法动作掩码
    let masks = env.action_masks();
    
    // 2. 应用掩码过滤概率 (Probs * Mask)
    // 确保不会采样到非法动作
    let masked_probs: Vec<f32> = probs
        .iter()
        .zip(masks.iter())
        .map(|(&p, &m)| if m == 1 { p } else { 0.0 })
        .collect();

    let non_zero_sum: f32 = masked_probs.iter().sum();

    if non_zero_sum == 0.0 {
        // 防御性编程：如果所有合法动作概率均为0 (MCTS异常)，则均匀随机选择一个合法动作
        eprint!("⚠️ 警告: 所有合法动作概率均为0，执行均匀随机选择动作。");
        let valid_actions: Vec<usize> = masks
            .iter()
            .enumerate()
            .filter_map(|(i, &m)| if m == 1 { Some(i) } else { None })
            .collect();

        let mut rng = thread_rng();
        *valid_actions.choose(&mut rng).expect("无有效动作")
    } else if temperature == 0.0 {
        // τ=0: 贪心选择最大概率动作
        masked_probs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .expect("无有效动作")
    } else {
        // 应用温度参数调整分布
        let adjusted_probs: Vec<f32> = if temperature != 1.0 {
            // p^(1/T)
            let sum: f32 = masked_probs.iter().map(|&p| p.powf(1.0 / temperature)).sum();
            masked_probs
                .iter()
                .map(|&p| p.powf(1.0 / temperature) / sum)
                .collect()
        } else {
            // 仅做归一化
            masked_probs.iter().map(|&p| p / non_zero_sum).collect()
        };

        let dist = WeightedIndex::new(&adjusted_probs).unwrap();
        let mut rng = thread_rng();
        dist.sample(&mut rng)
    }
}

/// 获取 Top-K 动作 (用于调试)
pub fn get_top_k_actions(probs: &[f32], k: usize) -> Vec<(usize, f32)> {
    let mut indexed: Vec<(usize, f32)> = probs.iter().enumerate().map(|(i, &p)| (i, p)).collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    indexed.into_iter().take(k).collect()
}