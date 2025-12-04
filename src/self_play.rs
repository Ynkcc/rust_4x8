// self_play.rs - 自对弈工作器模块
//
// 提供自对弈游戏的执行逻辑，包括工作器管理、动作采样、场景类型等

use crate::game_env::{DarkChessEnv, Observation, Player};
use crate::inference::ChannelEvaluator;
use crate::mcts::{MCTSConfig, MCTS};
use rand::distributions::WeightedIndex;
use rand::prelude::*;
use std::sync::Arc;
use std::time::Instant;

// ================ 游戏统计信息 ================

/// 游戏统计信息
#[derive(Debug, Clone)]
pub struct GameStats {
    pub steps: usize,
    pub winner: Option<i32>, // Some(1)=红胜, Some(-1)=黑胜, None/Some(0)=平局
}

/// 单局游戏的完整数据（包含样本和元数据）
#[derive(Debug, Clone)]
pub struct GameEpisode {
    pub samples: Vec<(Observation, Vec<f32>, f32, f32, Vec<i32>)>, // (观察, 策略概率, MCTS价值, 游戏结果价值, 动作掩码)
    pub game_length: usize,
    pub winner: Option<i32>,
}

// ================ 场景环境枚举 ================

/// 场景类型枚举，用于指定自对弈使用的场景
#[derive(Debug, Clone, Copy)]
pub enum ScenarioType {
    /// 场景1: R_A vs B_A (红仕对黑仕)
    TwoAdvisors,
    /// 场景2: Hidden Threat (隐藏威胁)
    HiddenThreats,
    /// 标准开局
    Standard,
}

impl ScenarioType {
    /// 创建对应场景的环境
    pub fn create_env(&self) -> DarkChessEnv {
        let mut env = DarkChessEnv::new();
        match self {
            ScenarioType::TwoAdvisors => env.setup_two_advisors(Player::Black),
            ScenarioType::HiddenThreats => env.setup_hidden_threats(),
            ScenarioType::Standard => {}
        }
        env
    }

    /// 获取场景名称
    pub fn name(&self) -> &'static str {
        match self {
            ScenarioType::TwoAdvisors => "TwoAdvisors (R_A vs B_A)",
            ScenarioType::HiddenThreats => "HiddenThreats",
            ScenarioType::Standard => "Standard",
        }
    }

    /// 获取该场景的期望最优动作
    pub fn expected_action(&self) -> usize {
        match self {
            ScenarioType::TwoAdvisors => 38,
            ScenarioType::HiddenThreats => 3,
            ScenarioType::Standard => 0,
        }
    }
}

// ================ 并行自对弈工作器 ================

/// 自对弈工作器
pub struct SelfPlayWorker {
    pub worker_id: usize,
    pub evaluator: Arc<ChannelEvaluator>,
    pub mcts_sims: usize,
    pub scenario: Option<ScenarioType>, // 指定场景类型，None 表示使用随机初始化
    pub dirichlet_alpha: f32,            // Dirichlet 噪声 alpha 参数
    pub dirichlet_epsilon: f32,          // Dirichlet 噪声权重
}

impl SelfPlayWorker {
    pub fn new(worker_id: usize, evaluator: Arc<ChannelEvaluator>, mcts_sims: usize) -> Self {
        Self {
            worker_id,
            evaluator,
            mcts_sims,
            scenario: None,
            dirichlet_alpha: 0.3,
            dirichlet_epsilon: 0.25,
        }
    }

    /// 创建使用指定场景的工作器
    pub fn with_scenario(
        worker_id: usize,
        evaluator: Arc<ChannelEvaluator>,
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
        }
    }
    
    /// 创建使用指定场景和 Dirichlet 参数的工作器
    pub fn with_scenario_and_dirichlet(
        worker_id: usize,
        evaluator: Arc<ChannelEvaluator>,
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
        }
    }

    /// 运行一局自对弈游戏，返回GameEpisode
    pub fn play_episode(&self, episode_num: usize) -> GameEpisode {
        let _scenario_name = self.scenario.map(|s| s.name()).unwrap_or("Random");
        // println!("  [Worker-{}] 开始第 {} 局游戏 (场景: {})", self.worker_id, episode_num + 1, _scenario_name);
        let start_time = Instant::now();

        // 根据场景类型创建环境
        let mut env = match self.scenario {
            Some(scenario) => scenario.create_env(),
            None => DarkChessEnv::new(),
        };
        let config = MCTSConfig {
            num_simulations: self.mcts_sims,
            cpuct: 1.0,
            virtual_loss: 1.0,
            max_concurrent_inferences: 8,
            dirichlet_alpha: self.dirichlet_alpha,
            dirichlet_epsilon: self.dirichlet_epsilon,
            train: true, // 自对弈训练时开启 Dirichlet 噪声
        };
        let mut mcts = MCTS::new(&env, self.evaluator.clone(), config);

        let mut episode_data = Vec::new();
        let mut step = 0;

        // 🐛 DEBUG: 记录首步MCTS详情
        let debug_first_step = episode_num < 2; // 只调试前2局

        loop {
            // 运行MCTS
            mcts.run();
            let probs = mcts.get_root_probabilities();
            let masks = env.action_masks();
            
            // 获取MCTS根节点的价值（从当前玩家视角）
            let mcts_value = mcts.root.q_value();

            // 🐛 DEBUG: 打印MCTS根节点详情
            if debug_first_step && step < 3 {
                // println!("    [Worker-{}] Step {}: MCTS根节点详情", self.worker_id, step);
                let _top_actions = get_top_k_actions(&probs, 5);
                // for (_action, _prob) in _top_actions {
                //     println!("      action={}, prob={:.3}", _action, _prob);
                // }
            }

            // 保存数据（包含MCTS价值）
            episode_data.push((
                env.get_state(),
                probs.clone(),
                mcts_value,
                env.get_current_player(),
                masks,
            ));

            // 选择动作（使用访问计数比例，不再使用温度采样）
            // Dirichlet 噪声已经在 MCTS 根节点扩展时添加
            let action = sample_action(&probs, &env, 1.0);

            // 🐛 DEBUG: 记录动作选择
            if debug_first_step && step < 3 {
                // println!("      选择: action={}", action);
            }

            // 执行动作
            match env.step(action, None) {
                Ok((_, _, terminated, truncated, winner)) => {
                    mcts.step_next(&env, action);

                    if terminated || truncated {
                        // 分配奖励
                        let reward_red = match winner {
                            Some(1) => 1.0,
                            Some(-1) => -1.0,
                            _ => 0.0,
                        };

                        let _elapsed = start_time.elapsed();
                        // println!("  [Worker-{}] 第 {} 局结束: {} 步, 胜者={:?}, 耗时 {:.1}s",
                        //     self.worker_id, episode_num + 1, step, winner, _elapsed.as_secs_f64());

                        // 🐛 DEBUG: 检查价值标签分布
                        if debug_first_step {
                            let mut red_values = Vec::new();
                            let mut black_values = Vec::new();
                            for (_, _, _, player, _) in &episode_data {
                                let val = if player.val() == 1 {
                                    reward_red
                                } else {
                                    -reward_red
                                };
                                if player.val() == 1 {
                                    red_values.push(val);
                                } else {
                                    black_values.push(val);
                                }
                            }
                            // println!("    [Worker-{}] 价值标签统计: 红方样本数={}, 黑方样本数={}",
                            //     self.worker_id, red_values.len(), black_values.len());
                            if !red_values.is_empty() {
                                // println!("      红方价值标签: {:.2} (winner={:?})", red_values[0], winner);
                            }
                            if !black_values.is_empty() {
                                // println!("      黑方价值标签: {:.2} (winner={:?})", black_values[0], winner);
                            }
                        }

                        // 回填价值
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
                Err(_e) => {
                    // eprintln!("[Worker-{}] 游戏错误: {}", self.worker_id, _e);
                    return GameEpisode {
                        samples: Vec::new(),
                        game_length: step,
                        winner: None,
                    };
                }
            }

            step += 1;
            if step > 200 {
                // 超过最大步数，游戏平局
                // println!("  [Worker-{}] 第 {} 局超时: {} 步", self.worker_id, episode_num + 1, step);
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

/// 动作采样（带温度参数）
pub fn sample_action(probs: &[f32], env: &DarkChessEnv, temperature: f32) -> usize {
    let non_zero_sum: f32 = probs.iter().sum();

    if non_zero_sum == 0.0 {
        // 回退：从有效动作中均匀选择
        let masks = env.action_masks();
        let valid_actions: Vec<usize> = masks
            .iter()
            .enumerate()
            .filter_map(|(i, &m)| if m == 1 { Some(i) } else { None })
            .collect();

        let mut rng = thread_rng();
        *valid_actions.choose(&mut rng).expect("无有效动作")
    } else {
        // 应用温度参数
        let adjusted_probs: Vec<f32> = if temperature != 1.0 {
            let sum: f32 = probs.iter().map(|&p| p.powf(1.0 / temperature)).sum();
            probs
                .iter()
                .map(|&p| p.powf(1.0 / temperature) / sum)
                .collect()
        } else {
            probs.to_vec()
        };

        let dist = WeightedIndex::new(&adjusted_probs).unwrap();
        let mut rng = thread_rng();
        dist.sample(&mut rng)
    }
}

/// 🐛 DEBUG: 获取top-k动作
pub fn get_top_k_actions(probs: &[f32], k: usize) -> Vec<(usize, f32)> {
    let mut indexed: Vec<(usize, f32)> = probs.iter().enumerate().map(|(i, &p)| (i, p)).collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    indexed.into_iter().take(k).collect()
}
