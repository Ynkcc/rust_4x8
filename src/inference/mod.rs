//! 神经网络模型推理模块 (Neural Network Inference)
//!
//! 包含 PyTorch LibTorch / TorchScript 评估器、ONNX Runtime 推理引擎以及 PyO3 Python 预估器适配。

#[cfg(feature = "torch")]
pub mod torchscript;

#[cfg(feature = "onnx")]
pub mod onnx;

#[cfg(feature = "pyo3")]
pub use crate::bridge::python::py_evaluator::PyEvaluator;
