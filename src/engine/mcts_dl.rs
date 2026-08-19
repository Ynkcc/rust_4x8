// src/ai/mcts_dl.rs
//! MCTS + 深度学习策略（支持搜索树复用）- 同步版本，基于 `GameConfig` 泛化。
//!
//! 通过 `GameEnv` trait 的关联常量（棋盘通道/尺寸/标量数）与 `action_space_size()`
//! 适配任意变体（4x8 / 4x4 / 4x2），使同一份 TorchScript 推理代码服务所有棋盘。
//!
//! 提供：
//! - `ModelWrapper`：加载 TorchScript `.pt` 模型（`CModule`）
//! - `TchEvaluator<G>`：实现 `Evaluator<G>`，批量前向 `(board, scalars) -> (logits, value)`
//! - `MctsDlPolicy<G>`：基于 Gumbel MCTS 的落子策略
//!
//! 使用流程：
//! 1. 加载模型 -> `ModelWrapper::load_from_file`
//! 2. 创建策略 -> `MctsDlPolicy::<G>::new(model, &env, sims)`
//! 3. 需要选择动作时调用 `choose_action(&env)`

use crate::core::env::GameEnv;
use crate::core::mcts::{Evaluator, EvaluatorOutput, GumbelConfig, GumbelMCTS};
use std::marker::PhantomData;
use std::sync::{Arc, Mutex};
use tch::{CModule, Device, Tensor};

// ---------------- Model 封装 ----------------

pub struct ModelWrapper {
    model: CModule,
    device: Device,
    gate: Mutex<()>, // 串行化前向以保线程安全
}

impl ModelWrapper {
    pub fn load_from_file(path: &str) -> Result<Self, String> {
        let device = Device::Cpu;
        let model = CModule::load(path).map_err(|e| format!("模型加载失败: {}", e))?;
        Ok(Self {
            model,
            device,
            gate: Mutex::new(()),
        })
    }

    pub fn get_device(&self) -> Device {
        self.device
    }
}

// 由于内部有互斥锁保护，允许跨线程共享
unsafe impl Send for ModelWrapper {}
unsafe impl Sync for ModelWrapper {}

// ---------------- Evaluator (泛型，基于 GameEnv) ----------------

/// 基于 `GameEnv` 关联常量适配任意变体的批量评估器。
pub struct TchEvaluator<G: GameEnv> {
    pub model: Arc<ModelWrapper>,
    pub _marker: PhantomData<G>,
}

impl<G: GameEnv> TchEvaluator<G> {
    pub fn new(model: Arc<ModelWrapper>) -> Self {
        Self {
            model,
            _marker: PhantomData,
        }
    }
}

impl<G: GameEnv> Evaluator<G> for TchEvaluator<G> {
    fn evaluate(&self, envs: &[G]) -> EvaluatorOutput {
        if envs.is_empty() {
            return EvaluatorOutput {
                logits: Vec::new(),
                values: Vec::new(),
                health: None,
            };
        }

        // 动作空间由环境类型决定（4x8/4x4/4x2 各自的 GameEnv 关联常量）。
        let action_space = G::action_space_size();

        let _guard = self.model.gate.lock().unwrap();
        tch::no_grad(|| {
            let batch_size = envs.len();

            // 从首个环境运行时观测推导特征维度（由 config 驱动，适配任意变体）。
            let ref_obs = envs[0].get_state();
            let board_channels = ref_obs.board.shape()[0];
            let board_rows = ref_obs.board.shape()[1];
            let board_cols = ref_obs.board.shape()[2];
            let scalar_count = ref_obs.scalars.len();

            let mut board_flat: Vec<f32> =
                Vec::with_capacity(batch_size * board_channels * board_rows * board_cols);
            let mut scalars_flat: Vec<f32> = Vec::with_capacity(batch_size * scalar_count);

            for env in envs {
                let obs = env.get_state();
                board_flat.extend(obs.board.iter().cloned());
                scalars_flat.extend(obs.scalars.iter().cloned());
            }

            let board_t = Tensor::from_slice(&board_flat)
                .to_device(self.model.device)
                .view([
                    batch_size as i64,
                    board_channels as i64,
                    board_rows as i64,
                    board_cols as i64,
                ]);

            let scalars_t = Tensor::from_slice(&scalars_flat)
                .to_device(self.model.device)
                .view([batch_size as i64, scalar_count as i64]);

            let board_ivalue = tch::IValue::Tensor(board_t);
            let scalars_ivalue = tch::IValue::Tensor(scalars_t);
            let outputs = self
                .model
                .model
                .forward_is(&[board_ivalue, scalars_ivalue])
                .expect("TorchScript forward failed");

            // 兼容 2 输出（旧模型）与 3 输出（带血量差异头）。
            let (policy_logits, value_t, health_t) = match outputs {
                tch::IValue::Tuple(mut tensors) if tensors.len() == 2 => {
                    let value_t = match tensors.pop().unwrap() {
                        tch::IValue::Tensor(t) => t,
                        _ => panic!("Expected Tensor for value"),
                    };
                    let policy_logits = match tensors.pop().unwrap() {
                        tch::IValue::Tensor(t) => t,
                        _ => panic!("Expected Tensor for policy"),
                    };
                    (policy_logits, value_t, None)
                }
                tch::IValue::Tuple(mut tensors) if tensors.len() == 3 => {
                    let health = match tensors.pop().unwrap() {
                        tch::IValue::Tensor(t) => t,
                        _ => panic!("Expected Tensor for health"),
                    };
                    let value_t = match tensors.pop().unwrap() {
                        tch::IValue::Tensor(t) => t,
                        _ => panic!("Expected Tensor for value"),
                    };
                    let policy_logits = match tensors.pop().unwrap() {
                        tch::IValue::Tensor(t) => t,
                        _ => panic!("Expected Tensor for policy"),
                    };
                    (policy_logits, value_t, Some(health))
                }
                _ => panic!("Expected tuple of 2 or 3 tensors from model"),
            };

            // 模型输出的策略 logits 长度即该模型的动作空间；若小于环境动作空间
            // （例如 4x4 模型 112 vs DarkChessEnv 关联常量 192），不足部分补 -inf，
            // 它们在合法动作掩码下无效，不影响搜索。
            let model_action = policy_logits.size()[1] as usize;
            let mut raw_flat = vec![0.0f32; batch_size * model_action];
            let raw_len = raw_flat.len();
            policy_logits
                .to_device(Device::Cpu)
                .copy_data(&mut raw_flat, raw_len);

            let mut logits_flat = vec![f32::NEG_INFINITY; batch_size * action_space];
            let copy_n = model_action.min(action_space);
            for b in 0..batch_size {
                let dst = &mut logits_flat[b * action_space..b * action_space + copy_n];
                dst.copy_from_slice(&raw_flat[b * model_action..b * model_action + copy_n]);
            }
            let logits_vec: Vec<Vec<f32>> = logits_flat
                .chunks(action_space)
                .map(|chunk| chunk.to_vec())
                .collect();

            let mut values = vec![0.0f32; batch_size];
            let values_len = values.len();
            value_t
                .to_device(Device::Cpu)
                .view([batch_size as i64])
                .copy_data(&mut values, values_len);

            // 血量差异头：[B, K] 分桶 logits；旧模型为 None。
            let health = health_t.map(|h| {
                let k = h.size()[1] as usize;
                let mut health_flat = vec![0.0f32; batch_size * k];
                let n = health_flat.len();
                h.to_device(Device::Cpu).copy_data(&mut health_flat, n);
                health_flat.chunks(k).map(|c| c.to_vec()).collect()
            });

            EvaluatorOutput {
                logits: logits_vec,
                values,
                health,
            }
        })
    }

    fn evaluate_logits(&self, envs: &[G]) -> EvaluatorOutput {
        self.evaluate(envs)
    }
}

// ---------------- 策略对象（泛型，每次创建新 MCTS）----------------

/// MCTS + 深度学习策略
///
/// 为了避免生命周期问题，每次调用 choose_action 时创建新的 MCTS 实例
/// 虽然失去了搜索树复用的优势，但实现更简单可靠
pub struct MctsDlPolicy<G: GameEnv> {
    model: Arc<ModelWrapper>,
    num_simulations: usize,
    _marker: PhantomData<G>,
}

impl<G: GameEnv> MctsDlPolicy<G> {
    pub fn new(model: Arc<ModelWrapper>, _env: &G, num_simulations: usize) -> Self {
        Self {
            model,
            num_simulations,
            _marker: PhantomData,
        }
    }

    pub fn set_iterations(&mut self, sims: usize) {
        self.num_simulations = sims.max(1);
    }

    /// 选择动作（每次创建新 MCTS）
    pub fn choose_action(&self, env: &G) -> Option<usize> {
        choose_action_once(&self.model, env, self.num_simulations)
    }
}

// ---------------- 简化的一次性策略 ----------------

/// 为给定环境选择最佳动作（每次创建新 MCTS）
pub fn choose_action_once<G: GameEnv>(
    model: &Arc<ModelWrapper>,
    env: &G,
    num_simulations: usize,
) -> Option<usize> {
    let evaluator = TchEvaluator::<G>::new(model.clone());
    let config = GumbelConfig {
        num_simulations,
        max_considered_actions: 16,
        c_scale: 1.0,
        gumbel_scale: 1.0,
        ..Default::default()
    };

    let mut mcts = GumbelMCTS::new(env, &evaluator, config);
    // 只返回动作索引，忽略完整搜索结果
    mcts.run().map(|result| result.action)
}
