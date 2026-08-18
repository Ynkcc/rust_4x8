//! # Banqi 4x8 - 暗棋强化学习与博弈搜索库 (DDD 领域驱动架构)
//!
//! ## 领域分层架构
//! - `core`:      领域核心模块（暗棋逻辑、物理规则、变体环境及 Gumbel MCTS 核心算法）
//! - `engine`:    策略引擎模块（Expectiminimax、Alpha-Beta 搜索强引擎、启发式评估及走子生成）
//! - `inference`: 神经网络推理层（LibTorch / TorchScript 评估器、ONNX Runtime 推理引擎）
//! - `pipeline`:  数据与自对弈管线（Rust 原生多线程/批量自对弈、Episode 序列化及持久化存储）
//! - `bridge`:    跨语言交互桥梁（PyO3 Python 扩展模块导出）
//! - `utils`:     通用基础设施（内存占用计算等）

pub mod bridge;
pub mod core;
pub mod engine;
pub mod inference;
pub mod pipeline;
pub mod utils;
