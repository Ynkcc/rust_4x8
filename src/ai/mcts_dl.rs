// src/ai/mcts_dl.rs
//! MCTS + 深度学习策略（支持搜索树复用）- 同步版本
//!
//! 提供 `MctsDlPolicy`：
//! - 维持一个持久的 `GumbelMCTS` 实例
//! - 可动态调整模拟次数
//! - 基于已加载的 `BanqiNet` 模型做策略+价值评估
//! - 在每个外部动作后调用 `advance(action, env)` 来复用搜索树
//!
//! 使用流程：
//! 1. 加载模型 -> `ModelWrapper::load_from_file`
//! 2. 创建策略 -> `MctsDlPolicy::new(model, &env, sims)`
//! 3. 每当玩家或 AI 执行动作后调用 `advance`
//! 4. 需要选择动作时调用 `choose_action(&env)`

use crate::mcts::{Evaluator, GumbelConfig, GumbelMCTS};
use crate::game_env::{DarkChessEnv, ACTION_SPACE_SIZE, BOARD_CHANNELS, BOARD_COLS, BOARD_ROWS, SCALAR_FEATURE_COUNT};
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
        let model = CModule::load(path)
            .map_err(|e| format!("模型加载失败: {}", e))?;
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

// ---------------- Evaluator (同步版本) ----------------

pub struct TchEvaluator {
    pub model: Arc<ModelWrapper>,
}

impl TchEvaluator {
    pub fn new(model: Arc<ModelWrapper>) -> Self {
        Self { model }
    }
}

impl Evaluator for TchEvaluator {
    fn evaluate(&self, envs: &[DarkChessEnv]) -> (Vec<Vec<f32>>, Vec<f32>) {
        if envs.is_empty() {
            return (Vec::new(), Vec::new());
        }

        let _guard = self.model.gate.lock().unwrap();
        tch::no_grad(|| {
            let batch_size = envs.len();
            let mut board_flat: Vec<f32> = Vec::with_capacity(batch_size * BOARD_CHANNELS * BOARD_ROWS * BOARD_COLS);
            let mut scalars_flat: Vec<f32> = Vec::with_capacity(batch_size * SCALAR_FEATURE_COUNT);

            for env in envs {
                let obs = env.get_state();
                board_flat.extend(obs.board.iter().cloned());
                scalars_flat.extend(obs.scalars.iter().cloned());
            }

            let board_t = Tensor::from_slice(&board_flat)
                .to_device(self.model.device)
                .view([
                    batch_size as i64,
                    BOARD_CHANNELS as i64,
                    BOARD_ROWS as i64,
                    BOARD_COLS as i64,
                ]);

            let scalars_t = Tensor::from_slice(&scalars_flat)
                .to_device(self.model.device)
                .view([batch_size as i64, SCALAR_FEATURE_COUNT as i64]);

            let board_ivalue = tch::IValue::Tensor(board_t);
            let scalars_ivalue = tch::IValue::Tensor(scalars_t);
            let outputs = self.model.model.forward_is(&[board_ivalue, scalars_ivalue])
                .expect("TorchScript forward failed");

            let (policy_logits, value_t) = match outputs {
                tch::IValue::Tuple(mut tensors) if tensors.len() == 2 => {
                    let value_t = match tensors.pop().unwrap() {
                        tch::IValue::Tensor(t) => t,
                        _ => panic!("Expected Tensor for value"),
                    };
                    let policy_logits = match tensors.pop().unwrap() {
                        tch::IValue::Tensor(t) => t,
                        _ => panic!("Expected Tensor for policy"),
                    };
                    (policy_logits, value_t)
                }
                _ => panic!("Expected tuple of 2 tensors from model"),
            };

            let mut logits_flat = vec![0.0f32; batch_size * ACTION_SPACE_SIZE];
            let logits_len = logits_flat.len();
            policy_logits
                .to_device(Device::Cpu)
                .copy_data(&mut logits_flat, logits_len);
            let logits_vec: Vec<Vec<f32>> = logits_flat
                .chunks(ACTION_SPACE_SIZE)
                .map(|chunk| chunk.to_vec())
                .collect();

            let mut values = vec![0.0f32; batch_size];
            let values_len = values.len();
            value_t
                .to_device(Device::Cpu)
                .view([batch_size as i64])
                .copy_data(&mut values, values_len);

            (logits_vec, values)
        })
    }

    fn evaluate_logits(&self, envs: &[DarkChessEnv]) -> (Vec<Vec<f32>>, Vec<f32>) {
        self.evaluate(envs)
    }
}

// ---------------- 策略对象（简化版：每次创建新 MCTS）----------------

/// MCTS + 深度学习策略
/// 
/// 为了避免生命周期问题，每次调用 choose_action 时创建新的 MCTS 实例
/// 虽然失去了搜索树复用的优势，但实现更简单可靠
pub struct MctsDlPolicy {
    model: Arc<ModelWrapper>,
    num_simulations: usize,
}

impl MctsDlPolicy {
    pub fn new(model: Arc<ModelWrapper>, _env: &DarkChessEnv, num_simulations: usize) -> Self {
        Self {
            model,
            num_simulations,
        }
    }

    pub fn set_iterations(&mut self, sims: usize) {
        self.num_simulations = sims.max(1);
    }

    /// 选择动作（每次创建新 MCTS）
    pub fn choose_action(&self, env: &DarkChessEnv) -> Option<usize> {
        choose_action_once(&self.model, env, self.num_simulations)
    }
}

// ---------------- 简化的一次性策略 ----------------

/// 为给定环境选择最佳动作（每次创建新 MCTS）
pub fn choose_action_once(
    model: &Arc<ModelWrapper>,
    env: &DarkChessEnv,
    num_simulations: usize,
) -> Option<usize> {
    let evaluator = TchEvaluator::new(model.clone());
    let config = GumbelConfig {
        num_simulations,
        max_considered_actions: 16,
        c_visit: 50.0,
        c_scale: 1.0,
        train: false,
    };
    
    let mut mcts = GumbelMCTS::new(env, &evaluator, config);
    // 只返回动作索引，忽略完整搜索结果
    mcts.run().map(|result| result.action)
}
