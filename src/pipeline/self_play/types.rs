// src/self_play/types.rs - 自对弈数据结构与同步运行器
//
// 本模块实现自对弈（Self-Play）逻辑，用于生成强化学习所需的训练数据。
// 重构说明：
// - 移除异步依赖，改为同步执行
// - 直接持有模型引用，无需 Channel 通信
// - 使用 Gumbel AlphaZero MCTS
// - 泛型化：`G: GameEnv` 可为暗棋（DarkChessEnv）或井字棋（TicTacToeEnv），
//   环境由调用方以 `fn() -> G` 工厂注入。

use crate::core::env::{GameEnv, Observation};
use crate::core::mcts::{Evaluator, GumbelConfig, GumbelMCTS};
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
/// MCTS 估算的根节点价值、实际选择的动作以及最终的游戏结果。
///
/// 样本中的观测统一使用 `Observation`（各游戏按自身通道/尺寸编码），
/// 因此 `GameEpisode` 本身不携带游戏泛型参数。
#[derive(Debug, Clone)]
pub struct GameEpisode {
    /// 训练样本列表: (观测状态, 策略概率分布, MCTS根节点价值, completed_Q, 根节点访问次数, 最终回报, 动作掩码, 实际动作, 终局归一化血量差)
    /// 最后一项 health_diff 为终局血量差按该样本玩家视角取号（红方视角为正）。
    pub samples: Vec<(Observation, Vec<f32>, f32, f32, u32, f32, Vec<i32>, usize, f32)>,
    /// 游戏总步数
    pub game_length: usize,
    /// 获胜方
    pub winner: Option<i32>,
    /// 终局归一化血量差（红方视角为正）：(红HP-黑HP)/(初始总HP+最大子力分值)，大致落在 [-1,1]。
    pub health_diff_red: Option<f32>,
}

// ================ 场景定义 ================

/// 训练场景类型枚举（暗棋专用）
///
/// 预留扩展点：未来可实现特定残局/开局场景 (如 TwoAdvisors, HiddenThreats)。
/// 目前所有场景均退化为标准开局。
#[derive(Debug, Clone, Copy)]
pub enum ScenarioType {
    /// 场景1: 双士残局 (R_A vs B_A) — 未实现，回退为 Standard
    TwoAdvisors,
    /// 场景2: 隐藏威胁 (Hidden Threat) — 未实现，回退为 Standard
    HiddenThreats,
    /// 标准开局 - 正常的完整游戏
    Standard,
}

impl ScenarioType {
    /// 根据枚举值创建对应的游戏环境（当前所有场景均创建标准环境）
    pub fn create_env(&self) -> crate::core::env::DarkChessEnv {
        crate::core::env::DarkChessEnv::new()
    }

    /// 获取场景的描述名称
    pub fn name(&self) -> &'static str {
        match self {
            ScenarioType::TwoAdvisors => "TwoAdvisors (R_A vs B_A) [unimplemented=Standard]",
            ScenarioType::HiddenThreats => "HiddenThreats [unimplemented=Standard]",
            ScenarioType::Standard => "Standard",
        }
    }

    /// 获取该场景下的期望最优动作索引 (预留验证接口，未实现场景默认返回 0)
    pub fn expected_action(&self) -> usize {
        match self {
            ScenarioType::TwoAdvisors | ScenarioType::HiddenThreats | ScenarioType::Standard => 0,
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
    // 注意：根节点 Dirichlet 噪声注入已移除。Gumbel AlphaZero 的探索由
    // Gumbel 噪声（Top-K 采样）与 Sequential Halving 提供，根节点子节点
    // prior 不参与任何搜索决策（Top-K 用 logit、根选择不经 PUCT），
    // 注入 Dirichlet 无效，请勿重新添加 dirichlet_alpha / dirichlet_epsilon 字段。
    /// 温度采样的步数阈值
    pub temperature_steps: usize,
    /// 训练场景
    pub scenario: ScenarioType,
    /// PUCT 探索系数（c_puct）与训练目标 σ 的缩放因子。默认 1.0。
    pub c_scale: f32,
    /// Gumbel 噪声尺度（根节点 Top-K 采样探索强度）。默认 1.0（标准 Gumbel）。
    pub gumbel_scale: f32,
}

impl Default for SelfPlayConfig {
    fn default() -> Self {
        Self {
            mcts_sims: 64,
            max_considered_actions: 16,
            temperature_steps: 10,
            scenario: ScenarioType::Standard,
            c_scale: 1.0,
            gumbel_scale: 1.0,
        }
    }
}

// ================ 自对弈运行器 (同步) ================

/// 自对弈运行器
///
/// 直接持有评估器引用，同步执行。
/// `G` 为游戏环境类型，环境由 `make_env` 工厂创建。
pub struct SelfPlayRunner<'a, G: GameEnv, E: Evaluator<G>> {
    evaluator: &'a E,
    config: SelfPlayConfig,
    make_env: fn() -> G,
}

impl<'a, G: GameEnv, E: Evaluator<G>> SelfPlayRunner<'a, G, E> {
    /// 创建新的自对弈运行器
    pub fn new(evaluator: &'a E, config: SelfPlayConfig, make_env: fn() -> G) -> Self {
        Self {
            evaluator,
            config,
            make_env,
        }
    }

    /// 使用默认配置创建
    pub fn with_defaults(evaluator: &'a E, mcts_sims: usize, make_env: fn() -> G) -> Self {
        let config = SelfPlayConfig {
            mcts_sims,
            ..Default::default()
        };
        Self {
            evaluator,
            config,
            make_env,
        }
    }

    /// 执行一局完整的自对弈 (同步)
    pub fn play_episode(&self, _episode_num: usize) -> GameEpisode {
        let _start_time = Instant::now();

        // 1. 初始化环境
        let mut env = (self.make_env)();

        // 2. 配置 MCTS
        let mcts_config = GumbelConfig {
            num_simulations: self.config.mcts_sims,
            max_considered_actions: self.config.max_considered_actions,
            c_scale: self.config.c_scale,
            gumbel_scale: self.config.gumbel_scale,
        };
        let mut mcts = GumbelMCTS::new(&env, self.evaluator, mcts_config.clone());

        let mut episode_data = Vec::new();
        let mut step = 0;

        // 3. 游戏主循环
        loop {
            // 注意：这里不再注入根节点 Dirichlet 噪声 —— Gumbel AlphaZero 的
            // 探索由 Gumbel 噪声 + Sequential Halving 提供，根节点 prior 不参与
            // 搜索决策，注入无效（详见 src/mcts/search.rs 中的说明）。请勿加回。

            // --- MCTS 搜索 (同步) ---
            let search_result = match mcts.run() {
                Some(result) => result,
                None => {
                    // mcts.run() 返回 None = 当前玩家无合法走法 → 该玩家判负。
                    // 调用环境终止条件获取真实 winner，回填正确的 ±1 胜负。
                    let (_, _, winner) = env.check_game_over_conditions();
                    return crate::pipeline::self_play::finalize_episode(
                        episode_data,
                        winner,
                        env.terminal_health_diff_red(),
                    );
                }
            };

            // --- 温度采样：前 temperature_steps 用 τ=1（探索），之后用 argmax（利用）---
            let temperature: f32 = if step < self.config.temperature_steps {
                1.0
            } else {
                1e-3
            };
            let sampled_action = {
                // Gumbel AlphaZero 标准动作选择：基于 completed Q 的温度 softmax（π ∝ exp(Q/τ)）
                let q_policy = mcts.get_root_completed_q_policy(temperature);
                GumbelMCTS::<G, E>::sample_action_from_policy(
                    &q_policy,
                    &search_result.action_mask,
                )
            };
            let action = sampled_action;
            let completed_q = mcts.get_root_completed_q(action);

            // --- 收集样本数据 ---
            // 注意: improved_policy 仍使用 Gumbel AlphaZero 的 σ(Q) + logit 公式作为训练目标
            // 实际动作 action 一并记录，用于对局回放 / 文字棋谱还原与交叉校验
            episode_data.push((
                search_result.state,
                search_result.improved_policy,
                search_result.mcts_value,
                completed_q,
                search_result.root_visit_count,
                search_result.player,
                search_result.action_mask,
                action,
            ));

            // --- 执行动作 ---
            match env.step(action) {
                Ok((_, _, terminated, truncated, winner)) => {
                    // 推进 MCTS 树
                    mcts.step_next(&env, action);

                    if terminated || truncated {
                        // --- 游戏结束处理：统一回填 ---
                        return crate::pipeline::self_play::finalize_episode(
                            episode_data,
                            winner,
                            env.terminal_health_diff_red(),
                        );
                    }
                }
                Err(e) => {
                    eprintln!("  ⚠️ 游戏错误 (step={}, action={}): {}", step, action, e);
                    return GameEpisode {
                        samples: Vec::new(),
                        game_length: step,
                        winner: None,
                        health_diff_red: env.terminal_health_diff_red(),
                    };
                }
            }

            // --- 步数限制检查：使用环境给定的步数上限 ---
            step += 1;
            if step >= G::max_steps() {
                // 步数上限截断：环境视其为 truncated 平局 (winner=Some(0))，
                // 与终局分支语义对齐，game_result 回填 0.0。
                return crate::pipeline::self_play::finalize_episode(
                    episode_data,
                    Some(0),
                    env.terminal_health_diff_red(),
                );
            }
        }
    }
}

// ================ 高级 API ================

/// 运行单局自对弈
///
/// `make_env` 为环境工厂（如 `DarkChessEnv::new` 或 `TicTacToeEnv::new`）。
pub fn run_self_play<G: GameEnv, E: Evaluator<G>>(
    evaluator: &E,
    config: &SelfPlayConfig,
    make_env: fn() -> G,
) -> GameEpisode {
    let runner = SelfPlayRunner::new(evaluator, config.clone(), make_env);
    runner.play_episode(0)
}

/// 批量运行多局自对弈
pub fn run_batch_self_play<G: GameEnv, E: Evaluator<G>>(
    evaluator: &E,
    config: &SelfPlayConfig,
    num_games: usize,
    make_env: fn() -> G,
) -> Vec<GameEpisode> {
    (0..num_games)
        .map(|i| {
            let runner = SelfPlayRunner::new(evaluator, config.clone(), make_env);
            runner.play_episode(i)
        })
        .collect()
}
