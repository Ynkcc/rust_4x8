// src/py/py_evaluator.rs
// 基于 Python 回调的通用评估器（泛型化：G = 游戏环境）

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyModule;
use std::marker::PhantomData;
use std::sync::atomic::{AtomicU32, Ordering};

use crate::game_env::GameEnv;
use crate::mcts::Evaluator;

/// 连续失败阈值：超过此值时 panic，避免用垃圾数据持续搜索
const MAX_CONSECUTIVE_FAILURES: u32 = 10;

/// 通用 Python 评估器：把 `Vec<G>` 编码为 numpy 批量特征后调用 Python 预测函数。
///
/// 预测函数约定：`predict_fn(boards_np, scalars_np) -> (policy_logits, values)`
/// - `boards_np` shape `[batch, G::BOARD_CHANNELS, G::BOARD_ROWS, G::BOARD_COLS]`
/// - `scalars_np` shape `[batch, G::SCALAR_FEATURE_COUNT]`
/// - `policy_logits` shape `[batch, G::action_space_size()]`
pub struct PyEvaluator<G: GameEnv> {
    predict_fn: PyObject,
    /// 缓存的 numpy 模块引用，避免每次 call_python 都重新 import
    numpy_module: std::sync::OnceLock<Py<PyModule>>,
    /// 连续失败计数器（AtomicU32 允许在多线程下安全递增）
    consecutive_failures: AtomicU32,
    /// 游戏环境类型标记
    _marker: PhantomData<G>,
}

// Safety: PyEvaluator 只持有 PyObject (=Py<PyAny>) 和 Py<PyModule>。
// Py<T> 实现了 Send + Sync (只要 T: Send + Sync；PyAny/PyModule 就是 Send+Sync)。
// OnceLock<Py<PyModule>> 和 AtomicU32 都是 Sync。
// 访问 Python 侧对象时必须先通过 Python::with_gil 获取 GIL，因此没有数据竞争。
unsafe impl<G: GameEnv> Send for PyEvaluator<G> {}
unsafe impl<G: GameEnv> Sync for PyEvaluator<G> {}

impl<G: GameEnv> PyEvaluator<G> {
    pub fn new(predict_fn: PyObject) -> Self {
        Self {
            predict_fn,
            numpy_module: std::sync::OnceLock::new(),
            consecutive_failures: AtomicU32::new(0),
            _marker: PhantomData,
        }
    }

    /// 获取或初始化 numpy 模块引用（懒加载，线程安全）
    fn get_numpy<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyModule>> {
        // OnceLock 尚未初始化时执行 import
        if self.numpy_module.get().is_none() {
            let np = py
                .import_bound("numpy")
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
    ) -> PyResult<(Vec<Vec<f32>>, Vec<f32>)> {
        Python::with_gil(|py| {
            let np = self.get_numpy(py)?;

            let boards_np = Self::to_numpy(
                &np,
                boards_flat,
                &[batch_size, G::BOARD_CHANNELS, G::BOARD_ROWS, G::BOARD_COLS],
            )?;
            let scalars_np =
                Self::to_numpy(&np, scalars_flat, &[batch_size, G::SCALAR_FEATURE_COUNT])?;

            let result = self
                .predict_fn
                .call1(py, (boards_np, scalars_np))
                .map_err(|e| {
                    eprintln!("Python predictor call failed: {}", e);
                    PyRuntimeError::new_err(format!("predictor call failed: {}", e))
                })?;

            let (policy_logits_py, values_py) = result.extract::<(PyObject, PyObject)>(py).map_err(|e| {
                eprintln!("Failed to extract (policy, value) from predictor result: {}", e);
                PyRuntimeError::new_err(format!(
                    "predictor should return (policy_logits, values) tuple: {}",
                    e
                ))
            })?;

            let policy_vec: Vec<Vec<f32>> = policy_logits_py.extract(py).map_err(|e| {
                eprintln!("Failed to extract policy logits: {}", e);
                PyRuntimeError::new_err(format!("policy logits extraction failed: {}", e))
            })?;

            // values 兼容两种形状：
            // - 扁平 (batch,)：numpy 默认（当前暗棋 predictor 约定）
            // - 嵌套 (batch, 1)：PyTorch 网络输出的默认形状
            let values_vec_flat = Self::extract_values_flat(py, &values_py, batch_size)?;

            // 与 policy 的 normalize_policy_shape 一致：先截断到 batch_size，
            // 不足部分补 0.0，保证任何返回形状下都不会触发 chunks(0) panic。
            let mut values_vec = values_vec_flat;
            values_vec.truncate(batch_size);
            values_vec.resize(batch_size, 0.0);

            let policy_vec = Self::normalize_policy_shape(policy_vec, batch_size);

            Ok((policy_vec, values_vec))
        })
    }

    /// 提取 values 并统一为扁平 `Vec<f32>`。
    ///
    /// 支持：
    /// - 扁平 `[batch]`：直接提取
    /// - 嵌套 `[[v0],[v1],...]`（shape `(batch, 1)`，PyTorch 默认）：取每行首元素
    fn extract_values_flat(py: Python<'_>, obj: &PyObject, batch_size: usize) -> PyResult<Vec<f32>> {
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

    /// 将 flat Vec<f32> 转换为指定 shape 的 float32 numpy 数组。
    fn to_numpy(np: &Bound<'_, PyModule>, data: Vec<f32>, shape: &[usize]) -> PyResult<PyObject> {
        let array = np
            .call_method1("array", (data,))
            .map_err(|e| PyRuntimeError::new_err(format!("numpy.array failed: {}", e)))?;
        let reshaped = array
            .call_method1("reshape", (shape.to_vec(),))
            .map_err(|e| PyRuntimeError::new_err(format!("numpy reshape failed: {}", e)))?;
        let astype = reshaped
            .call_method1("astype", ("float32",))
            .map_err(|e| PyRuntimeError::new_err(format!("numpy astype failed: {}", e)))?;
        Ok(astype.into())
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

    /// 记录一次连续失败，超过阈值时 panic
    fn record_failure(&self) {
        let count = self.consecutive_failures.fetch_add(1, Ordering::Relaxed) + 1;
        if count >= MAX_CONSECUTIVE_FAILURES {
            panic!(
                "PyEvaluator: Python predictor 连续失败 {} 次 (阈值 {})，终止以避免垃圾数据污染搜索",
                count, MAX_CONSECUTIVE_FAILURES
            );
        }
    }

    /// 记录一次成功，重置连续失败计数
    fn record_success(&self) {
        self.consecutive_failures.store(0, Ordering::Relaxed);
    }
}

impl<G: GameEnv> Evaluator<G> for PyEvaluator<G> {
    fn evaluate(&self, envs: &[G]) -> (Vec<Vec<f32>>, Vec<f32>) {
        if envs.is_empty() {
            return (Vec::new(), Vec::new());
        }

        let batch_size = envs.len();
        let mut boards_flat: Vec<f32> =
            Vec::with_capacity(batch_size * G::BOARD_CHANNELS * G::BOARD_ROWS * G::BOARD_COLS);
        let mut scalars_flat: Vec<f32> = Vec::with_capacity(batch_size * G::SCALAR_FEATURE_COUNT);

        // 直接写入 flat buffer，避免创建 Observation（ndarray 分配 + clone）的开销
        let mut board_buf = Vec::new();
        let mut scalar_buf = Vec::new();
        for env in envs {
            env.encode_features_flat_into(&mut board_buf, &mut scalar_buf);
            boards_flat.extend_from_slice(&board_buf);
            scalars_flat.extend_from_slice(&scalar_buf);
        }

        match self.call_python(boards_flat, scalars_flat, batch_size) {
            Ok((logits, values)) => {
                self.record_success();
                (logits, values)
            }
            Err(e) => {
                let fail_count = self.consecutive_failures.load(Ordering::Relaxed) + 1;
                eprintln!(
                    "Python predictor error (falling back to uniform) [连续失败 {}/{}]: {}",
                    fail_count, MAX_CONSECUTIVE_FAILURES, e
                );
                self.record_failure();
                (
                    vec![vec![0.0; G::action_space_size()]; batch_size],
                    vec![0.0; batch_size],
                )
            }
        }
    }

    fn evaluate_logits(&self, envs: &[G]) -> (Vec<Vec<f32>>, Vec<f32>) {
        self.evaluate(envs)
    }
}
