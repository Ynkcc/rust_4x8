//! 数据处理与强化学习自对弈管线模块 (Pipeline)
//!
//! 包含 Rust 原生并行自对弈驱动程序 (`self_play`)、训练 Episode 解码与 Replay Buffer (`replay`)、以及存储层 (`storage`)。

pub mod replay;
pub mod self_play;
pub mod storage;
