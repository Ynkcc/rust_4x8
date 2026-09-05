// src/py/py_evaluator.rs
// 基于 Python 回调的通用评估器（泛型化：G = 游戏环境）

use pyo3::buffer::PyBuffer;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::{PyByteArray, PyModule, PyTuple};
use std::marker::PhantomData;

use crate::core::env::GameEnv;
use crate::core::mcts::{Evaluator, EvaluatorOutput};

/// 通用 Python 评估器：把 `Vec<G>` 编码为 numpy 批量特征后调用 Python 预测函数。
///
/// 预测函数约定：`predict_fn(boards_np, scalars_np) -> (policy_logits, values)`
/// - `boards_np` shape `[batch, G::RESNET_BOARD_CHANNELS, G::BOARD_ROWS, G::BOARD_COLS]`
/// - `scalars_np` shape `[batch, G::RESNET_SCALAR_FEATURE_COUNT]`
/// - `policy_logits` shape `[batch, G::action_space_size()]`
pub struct PyEvaluator<G: GameEnv> {
    predict_fn: Py<PyAny>,
    /// 缓存的 numpy 模块引用，避免每次 call_python 都重新 import
    numpy_module: std::sync::OnceLock<Py<PyModule>>,
    /// 游戏环境类型标记
    _marker: PhantomData<G>,
}

// Safety: PyEvaluator 只持有 PyObject (=Py<PyAny>) 和 Py<PyModule>。
// Py<T> 实现了 Send + Sync (只要 T: Send + Sync；PyAny/PyModule 就是 Send+Sync)。
// OnceLock<Py<PyModule>> 是 Sync。访问 Python 侧对象时必须先通过 Python::with_gil
// 获取 GIL，因此没有数据竞争。
unsafe impl<G: GameEnv> Send for PyEvaluator<G> {}
unsafe impl<G: GameEnv> Sync for PyEvaluator<G> {}

impl<G: GameEnv> PyEvaluator<G> {
    pub fn new(predict_fn: Py<PyAny>) -> Self {
        Self {
            predict_fn,
            numpy_module: std::sync::OnceLock::new(),
            _marker: PhantomData,
        }
    }

    /// 获取或初始化 numpy 模块引用（懒加载，线程安全）
    fn get_numpy<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyModule>> {
        // OnceLock 尚未初始化时执行 import
        if self.numpy_module.get().is_none() {
            let np = py
                .import("numpy")
                .map_err(|e| PyRuntimeError::new_err(format!("numpy import failed: {}", e)))?;
            // clone_ref 后存入 OnceLock；竞争条件下多个线程可能同时 import，
            // 但 OnceLock::set 对第一个调用者生效，后续忽略，行为正确
            let _ = self.numpy_module.set(np.clone().unbind());
        }
        // 已初始化则直接获取绑定引用
        Ok(self
            .numpy_module
            .get()
            .expect("numpy_module should be initialized")
            .bind(py)
            .clone())
    }

    fn call_python(
        &self,
        boards_flat: Vec<f32>,
        scalars_flat: Vec<f32>,
        batch_size: usize,
    ) -> PyResult<(Vec<Vec<f32>>, Vec<f32>, Option<Vec<Vec<f32>>>)> {
        Python::attach(|py| {
            let np = self.get_numpy(py)?;

            let boards_np = Self::to_numpy(
                py,
                &np,
                &boards_flat,
                &[batch_size, G::RESNET_BOARD_CHANNELS, G::BOARD_ROWS, G::BOARD_COLS],
            )?;
            let scalars_np = Self::to_numpy(
                py,
                &np,
                &scalars_flat,
                &[batch_size, G::RESNET_SCALAR_FEATURE_COUNT],
            )?;

            let result = self
                .predict_fn
                .call1(py, (boards_np, scalars_np))
                .map_err(|e| {
                    eprintln!("Python predictor call failed: {}", e);
                    PyRuntimeError::new_err(format!("predictor call failed: {}", e))
                })?;

            // 兼容 2 输出（旧模型）与 3 输出（带血量差异头）。
            let tup = result.bind(py).cast::<PyTuple>().map_err(|e| {
                eprintln!("Failed to extract tuple from predictor result: {}", e);
                PyRuntimeError::new_err(format!(
                    "predictor should return (policy_logits, values[, health]) tuple: {}",
                    e
                ))
            })?;
            let n = tup.len();
            if n != 2 && n != 3 {
                return Err(PyRuntimeError::new_err(format!(
                    "predictor should return a 2- or 3-tuple, got {n} elements"
                )));
            }
            let policy_logits_py = tup.get_item(0)?.unbind();
            let values_py = tup.get_item(1)?.unbind();
            let health_py = if n == 3 {
                Some(tup.get_item(2)?.unbind())
            } else {
                None
            };

            // 优先用 PyBuffer 零拷贝读取 numpy 输出（无 Python 对象装箱 / 逐元素转换）。
            // 若 predictor 返回的不是 buffer 协议对象（如 list-of-lists），回退逐元素提取。
            let policy_vec =
                match Self::extract_policy_via_buffer(py, policy_logits_py.bind(py), batch_size) {
                    Ok(v) => v,
                    Err(_) => policy_logits_py.extract(py).map_err(|e| {
                        eprintln!("Failed to extract policy logits: {}", e);
                        PyRuntimeError::new_err(format!("policy logits extraction failed: {}", e))
                    })?,
                };

            // values 兼容两种形状：
            // - 扁平 (batch,)：numpy 默认（当前暗棋 predictor 约定）
            // - 嵌套 (batch, 1)：PyTorch 网络输出的默认形状
            let values_vec_flat =
                match Self::extract_values_via_buffer(py, values_py.bind(py), batch_size) {
                    Ok(v) => v,
                    Err(_) => Self::extract_values_flat(py, &values_py, batch_size)?,
                };

            // 与 policy 的 normalize_policy_shape 一致：先截断到 batch_size，
            // 不足部分补 0.0，保证任何返回形状下都不会触发 chunks(0) panic。
            let mut values_vec = values_vec_flat;
            values_vec.truncate(batch_size);
            values_vec.resize(batch_size, 0.0);

            let policy_vec = Self::normalize_policy_shape(policy_vec, batch_size);

            let health_vec = match health_py {
                Some(h) => match Self::extract_health_via_buffer(py, h.bind(py), batch_size) {
                    Ok(v) => Some(v),
                    Err(_) => {
                        eprintln!("Failed to extract health logits (treated as absent)");
                        None
                    }
                },
                None => None,
            };

            Ok((policy_vec, values_vec, health_vec))
        })
    }

    /// 提取 values 并统一为扁平 `Vec<f32>`。
    ///
    /// 支持：
    /// - 扁平 `[batch]`：直接提取
    /// - 嵌套 `[[v0],[v1],...]`（shape `(batch, 1)`，PyTorch 默认）：取每行首元素
    fn extract_values_flat(py: Python<'_>, obj: &Py<PyAny>, batch_size: usize) -> PyResult<Vec<f32>> {
        let bound = obj.bind(py);
        if let Ok(flat) = bound.extract::<Vec<f32>>() {
            return Ok(flat);
        }
        let nested: Vec<Vec<f32>> = bound.extract().map_err(|e| {
            eprintln!(
                "Failed to extract values (neither flat [batch] nor nested [batch,1]): {}",
                e
            );
            PyRuntimeError::new_err(format!("values extraction failed: {}", e))
        })?;
        let mut out = Vec::with_capacity(nested.len());
        for row in nested.into_iter().take(batch_size) {
            out.push(row.first().copied().unwrap_or(0.0));
        }
        Ok(out)
    }

    /// 用 PyBuffer 从 numpy 输出中提取 policy（零拷贝，无逐元素 Python 装箱）。
    ///
    /// 返回 `Vec<Vec<f32>>`，每行长度为 `G::action_space_size()`；形状不足 / 多余时
    /// 自动截断补齐，与 `normalize_policy_shape` 语义保持一致。
    fn extract_policy_via_buffer(
        py: Python<'_>,
        obj: &Bound<'_, PyAny>,
        batch_size: usize,
    ) -> PyResult<Vec<Vec<f32>>> {
        let buf = PyBuffer::<f32>::get(obj)?;
        if !buf.is_c_contiguous() {
            return Err(PyRuntimeError::new_err("policy buffer is not C-contiguous"));
        }
        let action_space = G::action_space_size();
        let flat = buf.to_vec(py)?;
        let mut out: Vec<Vec<f32>> = Vec::with_capacity(batch_size);
        for i in 0..batch_size {
            let start = i * action_space;
            let end = (start + action_space).min(flat.len());
            let mut row = Vec::with_capacity(action_space);
            row.extend_from_slice(&flat[start..end]);
            row.resize(action_space, 0.0);
            out.push(row);
        }
        Ok(out)
    }

    /// 用 PyBuffer 从 numpy 输出中提取血量分桶 logits（零拷贝，shape `[batch, K]`）。
    fn extract_health_via_buffer(
        py: Python<'_>,
        obj: &Bound<'_, PyAny>,
        batch_size: usize,
    ) -> PyResult<Vec<Vec<f32>>> {
        let buf = PyBuffer::<f32>::get(obj)?;
        if !buf.is_c_contiguous() {
            return Err(PyRuntimeError::new_err("health buffer is not C-contiguous"));
        }
        let k = buf.shape().get(1).copied().unwrap_or(0);
        if k == 0 {
            return Err(PyRuntimeError::new_err("health output has no dim 1"));
        }
        let flat = buf.to_vec(py)?;
        let mut out: Vec<Vec<f32>> = Vec::with_capacity(batch_size);
        for i in 0..batch_size {
            let start = i * k;
            let end = (start + k).min(flat.len());
            let mut row = Vec::with_capacity(k);
            row.extend_from_slice(&flat[start..end]);
            row.resize(k, 0.0);
            out.push(row);
        }
        Ok(out)
    }

    /// 用 PyBuffer 从 numpy 输出中提取 values（零拷贝，兼容 `(batch,)` 与 `(batch, 1)`）。
    fn extract_values_via_buffer(
        py: Python<'_>,
        obj: &Bound<'_, PyAny>,
        batch_size: usize,
    ) -> PyResult<Vec<f32>> {
        let buf = PyBuffer::<f32>::get(obj)?;
        if !buf.is_c_contiguous() {
            return Err(PyRuntimeError::new_err("values buffer is not C-contiguous"));
        }
        let mut flat = buf.to_vec(py)?;
        flat.truncate(batch_size);
        flat.resize(batch_size, 0.0);
        Ok(flat)
    }

    /// 将 flat Vec<f32> 转换为指定 shape 的 float32 numpy 数组。
    ///
    /// 优化：直接把数据 memcpy 进 PyByteArray（一次拷贝），再 `np.frombuffer(...,
    /// dtype='float32').reshape(shape)` 构造视图。旧实现 `np.array(Vec<f32>)` 会先把
    /// 每个 f32 装箱成 Python float（约 8 倍内存膨胀），再生成 float64 数组、再
    /// `astype("float32")`，共三次全量转换 + 海量小对象分配，是跨进程拷贝的主要瓶颈。
    ///
    /// 返回数组 C 连续且可写（`torch.from_numpy` 要求可写），语义与原契约一致。
    fn to_numpy(
        py: Python<'_>,
        np: &Bound<'_, PyModule>,
        data: &[f32],
        shape: &[usize],
    ) -> PyResult<Py<PyAny>> {
        // f32 是无可变 padding 的 POD，按字节切片安全（rust-numpy 同样依赖此性质）。
        // 空 data 时 as_ptr() 返回悬空指针（NonNull::dangling），不能直接 from_raw_parts，
        // 否则属于潜在 UB；空切片交给 PyByteArray 空字节处理。
        let bytes = if data.is_empty() {
            &[][..]
        } else {
            unsafe {
                std::slice::from_raw_parts(
                    data.as_ptr() as *const u8,
                    data.len() * std::mem::size_of::<f32>(),
                )
            }
        };
        let bytearray = PyByteArray::new(py, bytes);
        let array = np
            .call_method1("frombuffer", (bytearray, "float32"))
            .map_err(|e| PyRuntimeError::new_err(format!("numpy.frombuffer failed: {}", e)))?;
        let reshaped = array
            .call_method1("reshape", (shape.to_vec(),))
            .map_err(|e| PyRuntimeError::new_err(format!("numpy reshape failed: {}", e)))?;
        Ok(reshaped.unbind())
    }

    fn normalize_policy_shape(mut policy: Vec<Vec<f32>>, batch_size: usize) -> Vec<Vec<f32>> {
        if policy.len() != batch_size {
            if policy.len() == 1 && batch_size > 1 {
                let single = policy.remove(0);
                policy = std::iter::repeat(single).take(batch_size).collect();
            } else if policy.len() > batch_size {
                policy.truncate(batch_size);
            } else {
                while policy.len() < batch_size {
                    policy.push(vec![0.0; G::action_space_size()]);
                }
            }
        }
        for row in policy.iter_mut() {
            if row.len() != G::action_space_size() {
                row.resize(G::action_space_size(), 0.0);
            }
        }
        policy
    }
}

impl<G: GameEnv> Evaluator<G> for PyEvaluator<G> {
    fn evaluate(&self, envs: &[G]) -> EvaluatorOutput {
        if envs.is_empty() {
            return EvaluatorOutput {
                logits: Vec::new(),
                values: Vec::new(),
                health: None,
            };
        }

        let batch_size = envs.len();
        let mut boards_flat: Vec<f32> =
            Vec::with_capacity(batch_size * G::RESNET_BOARD_CHANNELS * G::BOARD_ROWS * G::BOARD_COLS);
        let mut scalars_flat: Vec<f32> = Vec::with_capacity(batch_size * G::RESNET_SCALAR_FEATURE_COUNT);

        // 直接写入 flat buffer，避免创建 ResNetObservation（ndarray 分配 + clone）的开销。
        // 注意：`encode_resnet_features_flat_into` 内部会 `clear()` 后重写，因此为每个 env
        // 复用同一组临时 Vec，再 extend 进目标 batch buffer（相比旧实现每 env 新分配，
        // 省去 batch 规模次数的堆分配）。
        let mut board_buf = Vec::new();
        let mut scalar_buf = Vec::new();
        for env in envs {
            env.encode_resnet_features_flat_into(&mut board_buf, &mut scalar_buf);
            boards_flat.extend_from_slice(&board_buf);
            scalars_flat.extend_from_slice(&scalar_buf);
        }

        match self.call_python(boards_flat, scalars_flat, batch_size) {
            Ok((logits, values, health)) => EvaluatorOutput {
                logits,
                values,
                health,
            },
            // 错误尽早暴露：Python 推理端一旦出错（网络结构/权重不匹配、PyTorch 维度
            // 报错等），说明模型输出已损坏。若静默回退 Uniform 会让 MCTS 在假数据下
            // 进行昂贵且无意义的搜索，违背"错误尽早暴露"原则。这里直接 panic。
            Err(e) => panic!(
                "PyEvaluator: Python predictor 调用失败，中止以避免在损坏模型输出下搜索: {}",
                e
            ),
        }
    }

    fn evaluate_logits(&self, envs: &[G]) -> EvaluatorOutput {
        self.evaluate(envs)
    }
}
