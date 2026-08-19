// src/self_play/mod.rs - 自对弈与数据生成模块（同步版，泛型化 G = 游戏环境）
//
// 子模块划分：
// - `types`：数据结构（GameEpisode/ScenarioType/SelfPlayConfig）、同步运行器与单局/批量自对弈 API
// - `batched`：批量流水线自对弈（多树并发 + 后台评估线程）
// - `finalize`：样本回填与动作选择辅助函数

pub mod types;
pub mod batched;
pub mod finalize;
pub mod match_core;
pub mod serialize;

// 对外重新导出，保持 `crate::pipeline::self_play::*` 命名空间兼容旧调用方。
pub use types::{
    GameEpisode, GameStats, ScenarioType, SelfPlayConfig, SelfPlayRunner, run_batch_self_play,
    run_self_play,
};
pub use batched::run_batched_self_play;
pub use finalize::{finalize_episode, get_top_k_actions, select_completed_q_action};
pub use match_core::{
    AsDarkChessRef, MatchParams, MatchResult, PlayerSpec, SeedableEnv, run_match_core,
};
