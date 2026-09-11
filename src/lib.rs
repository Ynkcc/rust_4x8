//! # Banqi 4x8 - 暗棋强化学习与博弈搜索库 (DDD 领域驱动架构)
//!
//! ## 领域分层架构
//! - `core`:      领域核心模块（来自 banqi-core crate：暗棋逻辑、MCTS、Expectimax、NNUE）
//! - `engine`:    策略引擎模块（走子生成来自 banqi-core；基础策略与 DL-MCTS 在此实现）
//! - `inference`: 神经网络推理层（LibTorch / TorchScript 评估器、ONNX Runtime 推理引擎）
//! - `pipeline`:  数据与自对弈管线（Rust 原生多线程/批量自对弈、Episode 序列化及持久化存储）
//! - `bridge`:    跨语言交互桥梁（PyO3 Python 扩展模块导出）
//! - `utils`:     通用基础设施（内存占用计算等）

pub mod bridge;
pub mod engine;
pub mod inference;
pub mod pipeline;
pub mod registry;
pub mod utils;

// 领域核心已拆出为独立 crate（规划迁至 banqi-core 仓库）。
// 以同名包装模块保持 `crate::core::...` / `banqi_4x8::core::...` 路径不变。
pub mod core {
    pub use banqi_core::core::*;
}
