// src/registry/mod.rs — ModelRegistry / EpisodeStore 统一接口层（统一训练架构 L4）
//
// 角色同构：单机 = Local 实现（本模块）；分布式 = Scheduler/R2 实现（后续接入）。
// - `local_registry`：LocalRegistry —— onnx 模型指针 + notify watch 热重载（feature = "onnx"）
// - `local_store`：LocalEpisodeStore —— episode 按 jsonl.gz 落盘目录

#[cfg(feature = "onnx")]
pub mod local_registry;
pub mod local_store;

pub use local_store::LocalEpisodeStore;
#[cfg(feature = "onnx")]
pub use local_registry::{CollectorTask, LocalRegistry};
