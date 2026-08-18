// src/onnx/mod.rs
//
// ONNX Runtime 推理服务（feature = "onnx"）。
//
// 用途：
//   - 自对弈收集器：`RustOnnxCollector`（见 src/py/onnx_collector.rs）在 Rust 侧
//     持有 ONNX 模型，推理不经过 Python / GIL，供 run_batched / run_parallel 使用。
//   - banqi-tauri：`OnnxMctsPolicy` 作为「MCTS + ONNX」对手，无需 libtorch。
//
// 模型前向契约与 TorchScript 一致：
//   forward(board[B, C, H, W], scalars[B, S]) -> (policy_logits[B, A], value[B, 1])
// 输入名固定为 "board" / "scalars"，输出名固定为 "policy_logits" / "value"
// （由 Python 侧 banqi/checkpoint.py 的 export_onnx 导出时指定）。

use std::marker::PhantomData;
use std::sync::{Arc, Mutex};

use ort::session::Session;
use ort::value::Tensor;

use crate::core::env::GameEnv;
use crate::core::mcts::{Evaluator, GumbelConfig, GumbelMCTS};

// ============================================================================
// ONNX 模型封装
// ============================================================================

/// 加载并持有 ONNX 模型的推理服务。
///
/// - `Mutex<Session>`：onnxruntime 的 `Session::run` 需要 `&mut self`（内部 EP 非
///   线程安全），用互斥锁串行化推理；批量自对弈通过「合并大 batch」获得并行度。
/// - 结构为 `Send + Sync`，可跨线程共享（`Arc<OnnxModel>`）。
pub struct OnnxModel {
    session: Mutex<Session>,
    model_path: String,
}

impl OnnxModel {
    /// 加载 ONNX 模型。
    ///
    /// `device`: "cpu" 强制 CPU；"cuda" / "auto" 在启用 `onnx-cuda` feature 时
    /// 尝试 CUDA EP（失败自动回退 CPU），否则直接使用 CPU。
    pub fn new(model_path: &str, device: &str) -> Result<Self, String> {
        let prefer_gpu = matches!(device, "cuda" | "auto");
        let session = build_session(model_path, prefer_gpu)?;
        Ok(Self {
            session: Mutex::new(session),
            model_path: model_path.to_string(),
        })
    }

    pub fn model_path(&self) -> &str {
        &self.model_path
    }

    /// 批量前向推理。
    ///
    /// 参数为扁平特征（由 `GameEnv::encode_features_flat_into` 填充）：
    ///   - board_data:   batch * channels * rows * cols
    ///   - scalars_data: batch * scalar_count
    ///
    /// 返回 `(policy_logits[B, A_model], values[B])`（A_model 为模型输出动作维度，
    /// 可能小于环境动作空间，由 `OnnxEvaluator` 负责补齐）。
    pub fn run(
        &self,
        board_data: &[f32],
        scalars_data: &[f32],
        batch_size: usize,
        board_channels: usize,
        board_rows: usize,
        board_cols: usize,
        scalar_count: usize,
    ) -> Result<(Vec<Vec<f32>>, Vec<f32>), String> {
        let board_tensor = Tensor::from_array((
            [batch_size, board_channels, board_rows, board_cols],
            board_data.to_vec().into_boxed_slice(),
        ))
        .map_err(|e| format!("构建 board 张量失败: {e}"))?;

        let scalars_tensor = Tensor::from_array((
            [batch_size, scalar_count],
            scalars_data.to_vec().into_boxed_slice(),
        ))
        .map_err(|e| format!("构建 scalars 张量失败: {e}"))?;

        // SessionOutputs 借用自 Session，需让互斥锁守卫存活到提取完输出为止。
        let mut session = self
            .session
            .lock()
            .map_err(|e| format!("ONNX 会话锁中毒: {e}"))?;
        let outputs = session
            .run(ort::inputs![
                "board" => board_tensor,
                "scalars" => scalars_tensor,
            ])
            .map_err(|e| format!("ONNX 推理失败: {e}"))?;

        // 输出顺序与 export_onnx 的 output_names 一致：policy_logits, value
        let model_action = extract_dim(&outputs[0], 1, "policy_logits")?;
        let mut logits_flat = vec![0.0f32; batch_size * model_action];
        copy_tensor(&outputs[0], &mut logits_flat)?;
        let logits: Vec<Vec<f32>> = logits_flat
            .chunks(model_action)
            .map(|c| c.to_vec())
            .collect();

        let mut values = vec![0.0f32; batch_size];
        copy_tensor(&outputs[1], &mut values)?;

        Ok((logits, values))
    }
}

// ============================================================================
// 会话构建（CUDA EP 为可选项，失败自动回退 CPU）
// ============================================================================

#[cfg(feature = "onnx-cuda")]
fn build_cuda_session(model_path: &str) -> Result<Session, String> {
    use ort::execution_providers::CUDAExecutionProvider;
    let provider = CUDAExecutionProvider::default()
        .build()
        .map_err(|e| format!("CUDA EP 构建失败: {e}"))?;
    Session::builder()
        .and_then(|mut b| b.with_execution_providers([provider]))
        .and_then(|mut b| b.commit_from_file(model_path))
        .map_err(|e| format!("加载 ONNX 模型（CUDA EP）失败 ({model_path}): {e}"))
}

fn build_session(model_path: &str, prefer_gpu: bool) -> Result<Session, String> {
    #[cfg(feature = "onnx-cuda")]
    if prefer_gpu {
        match build_cuda_session(model_path) {
            Ok(s) => {
                println!("[onnx] 已使用 CUDA EP: {model_path}");
                return Ok(s);
            }
            Err(e) => eprintln!("[onnx] CUDA EP 不可用，回退 CPU: {e}"),
        }
    }
    #[cfg(not(feature = "onnx-cuda"))]
    let _ = prefer_gpu;
    Session::builder()
        .and_then(|mut b| b.commit_from_file(model_path))
        .map_err(|e| format!("加载 ONNX 模型失败 ({model_path}): {e}"))
}

// ============================================================================
// 输出张量辅助函数
// ============================================================================

fn extract_dim(value: &ort::value::Value, dim: usize, name: &str) -> Result<usize, String> {
    let (shape, _data) = value
        .try_extract_tensor::<f32>()
        .map_err(|e| format!("提取输出 {name} 失败: {e}"))?;
    shape
        .get(dim)
        .copied()
        .map(|v| v as usize)
        .ok_or_else(|| format!("输出 {name} 维度不足: {:?}", &shape[..]))
}

fn copy_tensor(value: &ort::value::Value, out: &mut [f32]) -> Result<(), String> {
    let (_shape, data) = value
        .try_extract_tensor::<f32>()
        .map_err(|e| format!("提取输出张量失败: {e}"))?;
    let n = out.len().min(data.len());
    out[..n].copy_from_slice(&data[..n]);
    Ok(())
}

// ============================================================================
// ONNX 评估器（泛型，适配任意变体）
// ============================================================================

/// 基于 ONNX 模型的批量评估器，与 `crate::inference::torchscript::LocalEvaluator` /
/// `TchEvaluator` 等价（后端为 ONNX Runtime，不依赖 libtorch）。
pub struct OnnxEvaluator<G: GameEnv> {
    pub model: Arc<OnnxModel>,
    pub _marker: PhantomData<G>,
}

impl<G: GameEnv> OnnxEvaluator<G> {
    pub fn new(model: Arc<OnnxModel>) -> Self {
        Self {
            model,
            _marker: PhantomData,
        }
    }
}

impl<G: GameEnv> Evaluator<G> for OnnxEvaluator<G> {
    fn evaluate(&self, envs: &[G]) -> (Vec<Vec<f32>>, Vec<f32>) {
        if envs.is_empty() {
            return (Vec::new(), Vec::new());
        }

        // 维度从首个环境运行时观测推导（兼容 4x8 / 4x4 / 4x2 三种棋盘）。
        let ref_obs = envs[0].get_state();
        let board_channels = ref_obs.board.shape()[0];
        let board_rows = ref_obs.board.shape()[1];
        let board_cols = ref_obs.board.shape()[2];
        let scalar_count = ref_obs.scalars.len();
        let batch_size = envs.len();
        let action_space = G::action_space_size();

        let mut board_data = Vec::with_capacity(batch_size * board_channels * board_rows * board_cols);
        let mut scalars_data = Vec::with_capacity(batch_size * scalar_count);
        for env in envs {
            env.encode_features_flat_into(&mut board_data, &mut scalars_data);
        }

        let (raw_logits, values) = match self.model.run(
            &board_data,
            &scalars_data,
            batch_size,
            board_channels,
            board_rows,
            board_cols,
            scalar_count,
        ) {
            Ok(x) => x,
            Err(e) => {
                // Evaluator 接口无 Result；推理失败时退化为均匀 logits（受合法
                // 动作掩码约束），记录日志避免静默。
                eprintln!("[onnx] 推理失败: {e}");
                let logits = vec![vec![0.0f32; action_space]; batch_size];
                return (logits, vec![0.0f32; batch_size]);
            }
        };

        // 模型输出动作维度可能小于环境动作空间（如 4x4 模型 112 vs 192），
        // 不足部分补 -inf（在合法动作掩码下无效，不影响搜索）。
        let model_action = raw_logits.first().map_or(0, |r| r.len());
        let copy_n = model_action.min(action_space);
        let logits: Vec<Vec<f32>> = raw_logits
            .into_iter()
            .map(|row| {
                let mut padded = vec![f32::NEG_INFINITY; action_space];
                padded[..copy_n].copy_from_slice(&row[..copy_n]);
                padded
            })
            .collect();

        (logits, values)
    }

    fn evaluate_logits(&self, envs: &[G]) -> (Vec<Vec<f32>>, Vec<f32>) {
        self.evaluate(envs)
    }
}

// ============================================================================
// MCTS + ONNX 策略（供 banqi-tauri 等单步决策使用）
// ============================================================================

/// MCTS + ONNX 深度学习策略，每次调用创建新 MCTS 实例（与 MctsDlPolicy 一致）。
pub struct OnnxMctsPolicy<G: GameEnv> {
    model: Arc<OnnxModel>,
    num_simulations: usize,
    _marker: PhantomData<G>,
}

impl<G: GameEnv> OnnxMctsPolicy<G> {
    pub fn new(model: Arc<OnnxModel>, _env: &G, num_simulations: usize) -> Self {
        Self {
            model,
            num_simulations,
            _marker: PhantomData,
        }
    }

    pub fn set_iterations(&mut self, sims: usize) {
        self.num_simulations = sims.max(1);
    }

    pub fn choose_action(&self, env: &G) -> Option<usize> {
        onnx_choose_action_once(&self.model, env, self.num_simulations)
    }
}

/// 为给定环境选择最佳动作（每次创建新 MCTS）。
pub fn onnx_choose_action_once<G: GameEnv>(
    model: &Arc<OnnxModel>,
    env: &G,
    num_simulations: usize,
) -> Option<usize> {
    let evaluator = OnnxEvaluator::<G>::new(model.clone());
    let config = GumbelConfig {
        num_simulations,
        max_considered_actions: 16,
        c_scale: 1.0,
        gumbel_scale: 1.0,
    };
    let mut mcts = GumbelMCTS::new(env, &evaluator, config);
    mcts.run().map(|result| result.action)
}
