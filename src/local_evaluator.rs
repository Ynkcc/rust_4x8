// src/local_evaluator.rs
//
// 本地模型评估器 - 独立模块

use anyhow::Result;
use banqi_4x8::game_env::{
    ACTION_SPACE_SIZE, BOARD_CHANNELS, BOARD_COLS, BOARD_ROWS, DarkChessEnv, SCALAR_FEATURE_COUNT,
};
use banqi_4x8::mcts::Evaluator;
use tch::{CModule, Device, Kind, Tensor};

// ============================================================================
// 本地模型评估器
// ============================================================================

/// 直接使用 tch-rs CModule 加载 TorchScript 模型的评估器
pub struct LocalEvaluator {
    model: CModule,
    device: Device,
}

impl LocalEvaluator {
    pub fn new(model_path: &str, device: Device) -> Result<Self> {
        let model = CModule::load(model_path)?;
        Ok(Self { model, device })
    }
}

impl Evaluator for LocalEvaluator {
    fn evaluate(&self, envs: &[DarkChessEnv]) -> (Vec<Vec<f32>>, Vec<f32>) {
        if envs.is_empty() {
            return (Vec::new(), Vec::new());
        }

        tch::no_grad(|| {
            let batch_size = envs.len();
            let mut board_data: Vec<f32> =
                Vec::with_capacity(batch_size * BOARD_CHANNELS * BOARD_ROWS * BOARD_COLS);
            let mut scalar_data: Vec<f32> = Vec::with_capacity(batch_size * SCALAR_FEATURE_COUNT);

            for env in envs {
                let obs = env.get_state();
                board_data.extend_from_slice(obs.board.as_slice().unwrap());
                scalar_data.extend_from_slice(obs.scalars.as_slice().unwrap());
            }

            let board_tensor = Tensor::from_slice(&board_data)
                .view([
                    batch_size as i64,
                    BOARD_CHANNELS as i64,
                    BOARD_ROWS as i64,
                    BOARD_COLS as i64,
                ])
                .to_device(self.device)
                .to_kind(Kind::Float);

            let scalar_tensor = Tensor::from_slice(&scalar_data)
                .view([batch_size as i64, SCALAR_FEATURE_COUNT as i64])
                .to_device(self.device)
                .to_kind(Kind::Float);

            let board_ivalue = tch::IValue::Tensor(board_tensor);
            let scalar_ivalue = tch::IValue::Tensor(scalar_tensor);
            let outputs = self
                .model
                .forward_is(&[board_ivalue, scalar_ivalue])
                .expect("TorchScript forward failed");

            let (policy_logits, value) = match outputs {
                tch::IValue::Tuple(mut tensors) if tensors.len() == 2 => {
                    let value = match tensors.pop().unwrap() {
                        tch::IValue::Tensor(t) => t,
                        _ => panic!("Expected Tensor for value"),
                    };
                    let policy_logits = match tensors.pop().unwrap() {
                        tch::IValue::Tensor(t) => t,
                        _ => panic!("Expected Tensor for policy"),
                    };
                    (policy_logits, value)
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
            value
                .to_device(Device::Cpu)
                .view([batch_size as i64])
                .copy_data(&mut values, values_len);

            (logits_vec, values)
        })
    }
}
