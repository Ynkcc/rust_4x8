// src/py/onnx_collector.rs
//
// Rust 持有模型的 ONNX 数据收集器（pyo3 绑定）。
//
// 与 src/py/rust_collector.rs（RustTorchCollector，后端 tch-rs）等价，但推理
// 后端为 ONNX Runtime（ort crate）：
//   - 模型在 Rust 侧加载（OnnxEvaluator），推理不经过 Python / GIL；
//   - 批量 / rayon 多线程自对弈真正并行，且模型只加载一份；
//   - 不依赖 libtorch，只依赖 onnxruntime（更轻量）。
//
// 需同时启用 `onnx` 与 `pyo3` 两个 feature（`rust-onnx-collector`）。

use std::sync::{Arc, RwLock};

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use crate::core::env::{DarkChessEnv, Game4x4Env, GameEnv, MiniDarkChessEnv};
use crate::inference::onnx::{OnnxEvaluator, OnnxModel};
use crate::bridge::python::{PyGameEpisode, PySelfPlayConfig};
use crate::pipeline::self_play::{self, SelfPlayConfig};

// ============================================================================
// 类型擦除：按变体保存对应的泛型 OnnxEvaluator
// ============================================================================

/// 按变体分派的 Rust 侧 ONNX 评估器（0=4x8 暗棋、1=4x2 迷你、2=4x4）。
enum RustOnnxEvaluator {
    Dark(OnnxEvaluator<DarkChessEnv>),
    Mini(OnnxEvaluator<MiniDarkChessEnv>),
    Game4x4(OnnxEvaluator<Game4x4Env>),
}

impl RustOnnxEvaluator {
    fn variant(&self) -> u8 {
        match self {
            RustOnnxEvaluator::Dark(_) => 0,
            RustOnnxEvaluator::Mini(_) => 1,
            RustOnnxEvaluator::Game4x4(_) => 2,
        }
    }
}

// OnnxEvaluator 仅含 Arc<OnnxModel>（Mutex<Session> 为 Send + Sync）与
// PhantomData，因此枚举可安全跨线程共享（Arc + Sync）。

// ============================================================================
// 批量子对弈：每个变体的具体调用（在纯 Rust 内运行，无 GIL）
// ============================================================================

/// 运行批量自对弈，返回非空 episode。
fn run_batched_by_variant(
    py: Python<'_>,
    eval: &RustOnnxEvaluator,
    cfg: &SelfPlayConfig,
    num_games: usize,
    concurrency: usize,
    _worker_id: usize,
) -> Vec<PyGameEpisode> {
    let variant = eval.variant();
    let batch: Vec<self_play::GameEpisode> = py.detach(|| match eval {
        RustOnnxEvaluator::Dark(e) => {
            self_play::run_batched_self_play(e, cfg, num_games, concurrency, DarkChessEnv::new)
        }
        RustOnnxEvaluator::Mini(e) => {
            self_play::run_batched_self_play(e, cfg, num_games, concurrency, MiniDarkChessEnv::new)
        }
        RustOnnxEvaluator::Game4x4(e) => {
            self_play::run_batched_self_play(e, cfg, num_games, concurrency, Game4x4Env::new)
        }
    });
    batch
        .into_iter()
        .filter(|ep| !ep.samples.is_empty())
        .map(|ep| PyGameEpisode { inner: ep, variant })
        .collect()
}

// ============================================================================
// pyclass：RustOnnxCollector
// ============================================================================

/// 在 Rust 侧持有 ONNX 模型的多线程数据收集器（后端 ONNX Runtime）。
///
/// 与 `RustTorchCollector` 等价，但推理不经过 GIL 且不依赖 libtorch。
/// 用法：
/// ```python
/// c = banqi_4x8.RustOnnxCollector("banqi_model_latest.onnx", variant="4x8", device="cuda")
/// ep = c.run_batched(cfg, num_games=100, concurrency=8, worker_id=0)
/// ```
#[pyclass(name = "RustOnnxCollector", module = "banqi_4x8")]
pub struct RustOnnxCollector {
    /// 按变体分派的 Rust 侧 ONNX 评估器（模型只加载一份，跨线程共享）。
    eval: RwLock<Arc<RustOnnxEvaluator>>,
    /// 设备描述（供日志/调试）。
    device: String,
}

#[pymethods]
impl RustOnnxCollector {
    /// 构造收集器：一次性加载 ONNX 模型到 Rust（onnxruntime）。
    ///
    /// 参数：
    ///   - model_path: .onnx 文件路径（由 banqi/checkpoint.py 的 export_onnx 导出）
    ///   - variant: "4x8"（默认）| "4x2" | "4x4"
    ///   - device: "cuda"（启用 onnx-cuda 且可用时）/ "cpu" / "auto"（默认，自动探测）
    #[new]
    #[pyo3(signature = (model_path, variant = "4x8", device = "auto"))]
    fn new(model_path: &str, variant: &str, device: &str) -> PyResult<Self> {
        let eval = match variant {
            "4x8" => RustOnnxEvaluator::Dark(load_eval::<DarkChessEnv>(model_path, device)?),
            "4x2" => RustOnnxEvaluator::Mini(load_eval::<MiniDarkChessEnv>(model_path, device)?),
            "4x4" => RustOnnxEvaluator::Game4x4(load_eval::<Game4x4Env>(model_path, device)?),
            other => {
                return Err(PyValueError::new_err(format!(
                    "未知 variant: {other:?}（应为 '4x8' | '4x2' | '4x4'）"
                )));
            }
        };
        println!(
            "[RustOnnxCollector] ONNX 模型已加载到 Rust: {} (variant={}, device={})",
            model_path, variant, device
        );
        Ok(Self {
            eval: RwLock::new(Arc::new(eval)),
            device: device.to_string(),
        })
    }

    /// 当前变体 id："4x8" | "4x2" | "4x4"。
    #[getter]
    fn variant(&self) -> &'static str {
        match self.eval.read().unwrap().variant() {
            0 => "4x8",
            1 => "4x2",
            _ => "4x4",
        }
    }

    /// 推理设备描述。
    #[getter]
    fn device(&self) -> String {
        self.device.clone()
    }

    /// 重新加载 ONNX 模型（用于训练中权重热更新）。
    #[pyo3(signature = (model_path))]
    fn reload(&self, model_path: Option<&str>) -> PyResult<()> {
        let path = match model_path {
            Some(p) if !p.is_empty() => p,
            _ => return Ok(()),
        };
        let device = self.device.clone();
        let new_eval = match self.eval.read().unwrap().variant() {
            0 => RustOnnxEvaluator::Dark(load_eval::<DarkChessEnv>(path, &device)?),
            1 => RustOnnxEvaluator::Mini(load_eval::<MiniDarkChessEnv>(path, &device)?),
            _ => RustOnnxEvaluator::Game4x4(load_eval::<Game4x4Env>(path, &device)?),
        };
        // 原子替换：读取中的 run_* 继续用旧模型跑完当前批，写完后新批用新模型。
        let mut guard = self.eval.write().unwrap();
        *guard = Arc::new(new_eval);
        drop(guard);
        println!("[RustOnnxCollector] ONNX 模型已热更新: {}", path);
        Ok(())
    }

    /// 批量自对弈（流水线，推荐）：Rust 内起后台评估线程合并大 batch 推理，
    /// 不经过 GIL，多局并发并行推进。
    #[pyo3(signature = (config, num_games=100, concurrency=8, worker_id=0))]
    fn run_batched(
        &self,
        py: Python<'_>,
        config: Option<PyRef<PySelfPlayConfig>>,
        num_games: usize,
        concurrency: usize,
        worker_id: usize,
    ) -> PyResult<Vec<PyGameEpisode>> {
        let cfg: SelfPlayConfig = match config {
            Some(c) => c.inner.clone(),
            None => SelfPlayConfig::default(),
        };
        let eval = self.eval.read().unwrap().clone();
        Ok(run_batched_by_variant(
            py, eval.as_ref(), &cfg, num_games, concurrency, worker_id,
        ))
    }
}

// ============================================================================
// 辅助函数
// ============================================================================

fn load_eval<G: GameEnv>(model_path: &str, device: &str) -> PyResult<OnnxEvaluator<G>> {
    let model = OnnxModel::new(model_path, device).map_err(|e| {
        PyRuntimeError::new_err(format!(
            "Rust 侧加载 ONNX 模型失败 ({model_path}): {e}"
        ))
    })?;
    Ok(OnnxEvaluator::<G>::new(Arc::new(model)))
}
