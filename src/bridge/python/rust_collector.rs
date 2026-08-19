// src/py/rust_collector.rs
//
// Rust 持有模型的 Torch 数据收集器（pyo3 绑定）。
//
// 背景 / 动机：
//   现有 Python 绑定（run_*_self_play_with_predictor）把 `predict_fn` 传给 Rust，
//   MCTS 评估时 `PyEvaluator::evaluate` 通过 `Python::with_gil` 调用 Python 预测函数。
//   GIL 导致即使 Rust 侧开了多线程（rayon / batched eval worker），所有推理调用
//   仍被串行化——多线程自对弈无法真正并行。
//
//   为了绕开 GIL，有人改用 `multiprocessing`（spawn）让每个 worker 拥有独立解释器，
//   但每个子进程都要重新加载一份 libtorch + 模型权重，内存成倍上涨。
//
// 本模块的解法：
//   把模型**留在 Rust 侧**（用 tch-rs 的 `CModule` 加载 TorchScript），
//   `LocalEvaluator` 是纯 Rust 结构、推理不经过 Python / GIL，且为 `Send + Sync`，
//   因此可以在 Rust 内用 `run_batched_self_play` / rayon 多线程**真正并行**地评估，
//   同时模型只在构造时加载一份（单份内存）。
//
//   对外暴露一个 pyclass：`RustTorchCollector`，Python 侧只需：
//     collector = RustTorchCollector(model_path, variant)   # 加载一次
//     episodes = collector.run_batched(cfg, num_games, concurrency, worker_id)
//
// 本文件需同时启用 `torch` 与 `pyo3` 两个 feature。

use std::sync::{Arc, RwLock};

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use crate::core::env::{DarkChessEnv, Game4x4Env, GameEnv, MiniDarkChessEnv};
use crate::inference::torchscript::LocalEvaluator;
use crate::bridge::python::{PyGameEpisode, PySelfPlayConfig};
use crate::pipeline::self_play::{self, SelfPlayConfig};
use tch::Device;

// ============================================================================
// 类型擦除：按变体保存对应的泛型 LocalEvaluator
// ============================================================================

/// 按变体分派的 Rust 侧 Torch 评估器（0=4x8 暗棋、1=4x2 迷你、2=4x4）。
///
/// 用枚举而非泛型，是因为 `pyclass` 不能持有泛型参数；实际运行时按构造时的
/// variant 选择具体环境类型。三个变体都共享同一套 `LocalEvaluator<G>` 逻辑。
enum RustEvaluator {
    Dark(LocalEvaluator<DarkChessEnv>),
    Mini(LocalEvaluator<MiniDarkChessEnv>),
    Game4x4(LocalEvaluator<Game4x4Env>),
}

impl RustEvaluator {
    fn variant(&self) -> u8 {
        match self {
            RustEvaluator::Dark(_) => 0,
            RustEvaluator::Mini(_) => 1,
            RustEvaluator::Game4x4(_) => 2,
        }
    }
}

// 三个环境均为 Copy + Send + Sync，LocalEvaluator 亦为 Send + Sync，
// 因此枚举本身可安全跨线程共享（Arc + Sync）。
// （此处不需要手写 unsafe impl：LocalEvaluator 只含 CModule(实现 Send+Sync)
//   与 Device / PhantomData，均线程安全。）

// ============================================================================
// 批量子对弈：每个变体的具体调用（在纯 Rust 内运行，无 GIL）
// ============================================================================

/// 运行批量自对弈，返回非空 episode。
///
/// `py`：用于 `allow_threads` 释放 GIL。批量自对弈内部起后台评估线程并可能
/// 长时间阻塞主线程，若此时仍持有 GIL，Python 主线程 / 其他线程将被饿死；
/// 释放 GIL 让出解释器。推理本身在 Rust 内完成，不受 GIL 限制。
fn run_batched_by_variant(
    py: Python<'_>,
    eval: &RustEvaluator,
    cfg: &SelfPlayConfig,
    num_games: usize,
    concurrency: usize,
    _worker_id: usize,
) -> Vec<PyGameEpisode> {
    let variant = eval.variant();
    let batch: Vec<self_play::GameEpisode> = py.detach(|| match eval {
        RustEvaluator::Dark(e) => {
            self_play::run_batched_self_play(e, cfg, num_games, concurrency, DarkChessEnv::new)
        }
        RustEvaluator::Mini(e) => {
            self_play::run_batched_self_play(e, cfg, num_games, concurrency, MiniDarkChessEnv::new)
        }
        RustEvaluator::Game4x4(e) => {
            self_play::run_batched_self_play(e, cfg, num_games, concurrency, Game4x4Env::new)
        }
    });

    // 过滤空局：与既有 py/mod.rs 的批量版语义一致（空局不占用目标局数，
    // 但为简化绑定，此处仅丢弃空局；若需严格凑满 num_games，可循环重跑）。
    batch
        .into_iter()
        .filter(|ep| !ep.samples.is_empty())
        .map(|ep| PyGameEpisode { inner: ep, variant })
        .collect()
}

// ============================================================================
// pyclass：RustTorchCollector
// ============================================================================

/// 在 Rust 侧持有 TorchScript 模型的多线程数据收集器。
///
/// 与 `run_*_self_play_with_predictor`（传 Python predict_fn，受 GIL 串行化）不同，
/// 本类在构造时把模型加载进 Rust（`LocalEvaluator`，tch-rs），推理完全在 Rust 内、
/// 不经过 GIL，因此多线程 / 批量自对弈能真正并行，且模型只加载一份。
///
/// 用法：
/// ```python
/// c = banqi_4x8.RustTorchCollector("banqi_model_latest.pt", variant="4x8")
/// ep = c.run_batched(cfg, num_games=100, concurrency=8, worker_id=0)
/// ```
#[pyclass(name = "RustTorchCollector", module = "banqi_4x8")]
pub struct RustTorchCollector {
    /// 按变体分派的 Rust 侧 Torch 评估器（模型只加载一份，跨线程共享）。
    ///
    /// `RwLock<Arc<...>>`：读锁用于 run_* 取当前模型；写锁用于 `reload` 热更新
    /// 权重——加载成功后再整体替换 Arc，读取中的 run_* 继续用旧模型跑完当前批，
    /// 之后的新批自动用新模型，无锁竞争、不阻塞推理。
    eval: RwLock<Arc<RustEvaluator>>,
    /// 设备描述（供日志/调试）。
    device: String,
}

#[pymethods]
impl RustTorchCollector {
    /// 构造收集器：一次性加载 TorchScript 模型到 Rust（tch-rs CModule）。
    ///
    /// 参数：
    ///   - model_path: TorchScript .pt 文件路径（由 banqi/checkpoint.py 导出）
    ///   - variant: "4x8"（默认）| "4x2" | "4x4"
    ///   - device: "cuda"（可用时）/ "cpu" / "auto"（默认，自动探测）
    ///
    /// 若本机无 GPU（如本设备），device="cuda" 会回退到 CPU。
    #[new]
    #[pyo3(signature = (model_path, variant = "4x8", device = "auto"))]
    fn new(model_path: &str, variant: &str, device: &str) -> PyResult<Self> {
        let dev = resolve_device(device)?;
        let eval = match variant {
            "4x8" => RustEvaluator::Dark(load_eval::<DarkChessEnv>(model_path, dev)?),
            "4x2" => RustEvaluator::Mini(load_eval::<MiniDarkChessEnv>(model_path, dev)?),
            "4x4" => RustEvaluator::Game4x4(load_eval::<Game4x4Env>(model_path, dev)?),
            other => {
                return Err(PyValueError::new_err(format!(
                    "未知 variant: {other:?}（应为 '4x8' | '4x2' | '4x4'）"
                )));
            }
        };
        println!(
            "[RustTorchCollector] 模型已加载到 Rust: {} (variant={}, device={:?})",
            model_path, variant, dev
        );
        Ok(Self {
            eval: RwLock::new(Arc::new(eval)),
            device: format!("{:?}", dev),
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

    /// 重新加载 TorchScript 模型（用于训练中权重热更新）。
    ///
    /// 训练侧每轮保存 checkpoint（含导出的 .pt）后调用本方法刷新 Rust 侧模型，
    /// 之后 `run_batched` / `run_parallel` 会用新权重继续收集。
    /// 传 `None` 或空串则保持现有模型不变。
    #[pyo3(signature = (model_path))]
    fn reload(&self, model_path: Option<&str>) -> PyResult<()> {
        let path = match model_path {
            Some(p) if !p.is_empty() => p,
            _ => return Ok(()),
        };
        let dev = resolve_device(&self.device)?;
        let new_eval = match self.eval.read().unwrap().variant() {
            0 => RustEvaluator::Dark(load_eval::<DarkChessEnv>(path, dev)?),
            1 => RustEvaluator::Mini(load_eval::<MiniDarkChessEnv>(path, dev)?),
            _ => RustEvaluator::Game4x4(load_eval::<Game4x4Env>(path, dev)?),
        };
        // 原子替换：读取中的 run_* 继续用旧模型跑完当前批，写完后新批用新模型。
        let mut guard = self.eval.write().unwrap();
        *guard = Arc::new(new_eval);
        drop(guard);
        println!("[RustTorchCollector] 模型已热更新: {}", path);
        Ok(())
    }

    /// 批量自对弈（流水线，推荐）：Rust 内起后台评估线程合并大 batch 推理，
    /// 不经过 GIL，多局并发并行推进。
    ///
    /// 参数与 `run_batched_self_play_with_predictor` 一致：
    ///   - config: SelfPlayConfig
    ///   - num_games: 目标对局数
    ///   - concurrency: 同时推进的并发局数
    ///   - worker_id: worker 编号（写入 episode）
    ///
    /// 返回非空 episode 列表。
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

fn load_eval<G: GameEnv>(model_path: &str, device: Device) -> PyResult<LocalEvaluator<G>> {
    LocalEvaluator::<G>::new(model_path, device).map_err(|e| {
        PyRuntimeError::new_err(format!(
            "Rust 侧加载 TorchScript 模型失败 ({:?}): {}",
            model_path, e
        ))
    })
}

/// 解析设备描述。本机无 GPU 时 cuda/auto 回退 CPU。
fn resolve_device(device: &str) -> PyResult<Device> {
    let dev = match device {
        "cuda" => {
            if tch::Cuda::is_available() {
                Device::Cuda(0)
            } else {
                eprintln!("[RustTorchCollector] 请求 cuda 但本机无 GPU，回退 CPU");
                Device::Cpu
            }
        }
        "cpu" => Device::Cpu,
        "auto" => {
            if tch::Cuda::is_available() {
                Device::Cuda(0)
            } else {
                Device::Cpu
            }
        }
        other => {
            return Err(PyValueError::new_err(format!(
                "未知 device: {other:?}（应为 'cuda' | 'cpu' | 'auto'）"
            )));
        }
    };
    Ok(dev)
}
