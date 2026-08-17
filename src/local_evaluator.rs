// src/local_evaluator.rs
//
// 本地模型评估器 - 独立模块（泛型化 G = 游戏环境）。
//
// 该评估器直接在 Rust 侧用 tch-rs 加载 TorchScript 模型（CModule），
// 推理不经过 Python / GIL，因此可以在多线程 / 批量自对弈中被安全共享：
//   - 模型只在构造时加载一次（单份内存，避免 spawn 多进程重复加载 libtorch + 权重）；
//   - `evaluate` 内部在 `no_grad` 下只读推理，`LocalEvaluator` 标记为 Send + Sync，
//     可跨线程并行评估，彻底规避「Python 侧 predict_fn 被 GIL 串行化」的瓶颈。
//
// 泛型 `G: GameEnv` 可为暗棋（DarkChessEnv）、4x2 迷你（MiniDarkChessEnv）或
// 4x4（Game4x4Env）；模型前向契约统一为：
//   forward(board[B, C, H, W], scalars[B, S]) -> (policy_logits[B, A], value[B])

use anyhow::Result;
use crate::game_env::GameEnv;
use crate::mcts::Evaluator;
use tch::{CModule, Device, Kind, Tensor};
use std::marker::PhantomData;

// ============================================================================
// 本地模型评估器
// ============================================================================

/// 直接使用 tch-rs CModule 加载 TorchScript 模型的评估器。
///
/// 仅依赖 `CModule`（libtorch 的 TorchScript 模块），本身是纯 Rust 结构，
/// 推理不触碰 Python 解释器，因此天然不受 GIL 影响。
pub struct LocalEvaluator<G: GameEnv> {
    model: CModule,
    device: Device,
    /// 游戏环境类型标记
    _marker: PhantomData<G>,
}

// CModule 已在 tch 中实现 Send + Sync（libtorch 推理线程安全）；
// Device 与 PhantomData 亦为 Send + Sync，因此无需手写 unsafe impl。
impl<G: GameEnv> LocalEvaluator<G> {
    pub fn new(model_path: &str, device: Device) -> Result<Self> {
        let model = CModule::load(model_path)?;
        Ok(Self {
            model,
            device,
            _marker: PhantomData,
        })
    }
}

impl<G: GameEnv> Evaluator<G> for LocalEvaluator<G> {
    fn evaluate(&self, envs: &[G]) -> (Vec<Vec<f32>>, Vec<f32>) {
        if envs.is_empty() {
            return (Vec::new(), Vec::new());
        }

        tch::no_grad(|| {
            let batch_size = envs.len();
            let mut board_data: Vec<f32> =
                Vec::with_capacity(batch_size * G::BOARD_CHANNELS * G::BOARD_ROWS * G::BOARD_COLS);
            let mut scalar_data: Vec<f32> =
                Vec::with_capacity(batch_size * G::SCALAR_FEATURE_COUNT);

            // 复用临时缓冲，避免每个 env 新建堆分配（与 PyEvaluator 一致）。
            let mut board_buf = Vec::new();
            let mut scalar_buf = Vec::new();
            for env in envs {
                env.encode_features_flat_into(&mut board_buf, &mut scalar_buf);
                board_data.extend_from_slice(&board_buf);
                scalar_data.extend_from_slice(&scalar_buf);
            }

            let board_tensor = Tensor::from_slice(&board_data)
                .view([
                    batch_size as i64,
                    G::BOARD_CHANNELS as i64,
                    G::BOARD_ROWS as i64,
                    G::BOARD_COLS as i64,
                ])
                .to_device(self.device)
                .to_kind(Kind::Float);

            let scalar_tensor = Tensor::from_slice(&scalar_data)
                .view([batch_size as i64, G::SCALAR_FEATURE_COUNT as i64])
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

            let action_space = G::action_space_size();
            let mut logits_flat = vec![0.0f32; batch_size * action_space];
            let logits_len = logits_flat.len();
            policy_logits
                .to_device(Device::Cpu)
                .copy_data(&mut logits_flat, logits_len);
            let logits_vec: Vec<Vec<f32>> = logits_flat
                .chunks(action_space)
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
