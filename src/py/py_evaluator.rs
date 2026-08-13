use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

use crate::game_env::{
    ACTION_SPACE_SIZE, BOARD_CHANNELS, BOARD_COLS, BOARD_ROWS, DarkChessEnv, SCALAR_FEATURE_COUNT,
};
use crate::mcts::Evaluator;

pub struct PyEvaluator {
    predict_fn: PyObject,
}

// Safety: PyEvaluator 只持有 PyObject (=Py<PyAny>)。
// Py<T> 实现了 Send + Sync (只要 T: Send + Sync；PyAny 就是 Send+Sync)。
// 访问 Python 侧对象时必须先通过 Python::with_gil 获取 GIL，因此没有数据竞争。
unsafe impl Send for PyEvaluator {}
unsafe impl Sync for PyEvaluator {}

impl PyEvaluator {
    pub fn new(predict_fn: PyObject) -> Self {
        Self { predict_fn }
    }

    fn call_python(
        &self,
        boards_flat: Vec<f32>,
        scalars_flat: Vec<f32>,
        batch_size: usize,
    ) -> PyResult<(Vec<Vec<f32>>, Vec<f32>)> {
        Python::with_gil(|py| {
            let boards_np = Self::to_numpy_4d(
                py,
                boards_flat,
                &[batch_size, BOARD_CHANNELS, BOARD_ROWS, BOARD_COLS],
            )?;
            let scalars_np = Self::to_numpy_2d(py, scalars_flat, &[batch_size, SCALAR_FEATURE_COUNT])?;

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

            let values_vec_flat: Vec<f32> = values_py.extract(py).map_err(|e| {
                eprintln!("Failed to extract values: {}", e);
                PyRuntimeError::new_err(format!("values extraction failed: {}", e))
            })?;

            let values_vec: Vec<f32> = if values_vec_flat.len() == batch_size {
                values_vec_flat
            } else {
                values_vec_flat
                    .chunks(values_vec_flat.len() / batch_size.max(1))
                    .map(|c| c[0])
                    .take(batch_size)
                    .collect()
            };

            let policy_vec = Self::normalize_policy_shape(policy_vec, batch_size);

            Ok((policy_vec, values_vec))
        })
    }

    fn to_numpy_4d(py: Python<'_>, data: Vec<f32>, shape: &[usize]) -> PyResult<PyObject> {
        let np = py
            .import_bound("numpy")
            .map_err(|e| PyRuntimeError::new_err(format!("numpy import failed: {}", e)))?;
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

    fn to_numpy_2d(py: Python<'_>, data: Vec<f32>, shape: &[usize]) -> PyResult<PyObject> {
        let np = py
            .import_bound("numpy")
            .map_err(|e| PyRuntimeError::new_err(format!("numpy import failed: {}", e)))?;
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
                    policy.push(vec![0.0; ACTION_SPACE_SIZE]);
                }
            }
        }
        for row in policy.iter_mut() {
            if row.len() != ACTION_SPACE_SIZE {
                row.resize(ACTION_SPACE_SIZE, 0.0);
            }
        }
        policy
    }
}

impl Evaluator for PyEvaluator {
    fn evaluate(&self, envs: &[DarkChessEnv]) -> (Vec<Vec<f32>>, Vec<f32>) {
        if envs.is_empty() {
            return (Vec::new(), Vec::new());
        }

        let batch_size = envs.len();
        let mut boards_flat: Vec<f32> =
            Vec::with_capacity(batch_size * BOARD_CHANNELS * BOARD_ROWS * BOARD_COLS);
        let mut scalars_flat: Vec<f32> = Vec::with_capacity(batch_size * SCALAR_FEATURE_COUNT);

        let mut board_buf = Vec::new();
        let mut scalar_buf = Vec::new();
        for env in envs {
            env.get_state_into(&mut board_buf, &mut scalar_buf);
            boards_flat.extend_from_slice(&board_buf);
            scalars_flat.extend_from_slice(&scalar_buf);
        }

        match self.call_python(boards_flat, scalars_flat, batch_size) {
            Ok((logits, values)) => (logits, values),
            Err(e) => {
                eprintln!(
                    "Python predictor error (falling back to uniform): {}",
                    e
                );
                (
                    vec![vec![0.0; ACTION_SPACE_SIZE]; batch_size],
                    vec![0.0; batch_size],
                )
            }
        }
    }

    fn evaluate_logits(&self, envs: &[DarkChessEnv]) -> (Vec<Vec<f32>>, Vec<f32>) {
        self.evaluate(envs)
    }
}
