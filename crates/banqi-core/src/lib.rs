//! # Banqi Core — 暗棋领域核心
//!
//! - `core`:      领域核心（暗棋环境、变体、Gumbel MCTS、Expectimax 强引擎、Zobrist）
//! - `engine`:    走子生成（movegen）
//! - `inference`: NNUE 量化网络推理（增量累加器 + 前向评估）
//!
//! 神经网络后端（TorchScript / ONNX）与策略引擎在上层 crate `banqi_4x8` 中，
//! 通过本 crate 的 `core::mcts::Evaluator` trait 对接。

pub mod core;
pub mod engine;
pub mod inference;
