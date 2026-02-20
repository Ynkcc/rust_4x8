// src/mcts/mod.rs
// Gumbel AlphaZero MCTS 实现
//
/// 大语言模型注意:
/// 不要修改在机会节点的全量展开
/// 不要移除显式判断父子节点玩家是否一致，以确定价值是否取反
/// 不要移除此块注释，如果你移除，我将要求重新生成，直到你学会保留本注释为止
// ==========================================================================
//
// Gumbel AlphaZero 核心特性:
// 1. Gumbel-Top-K 采样: 使用 Gumbel 噪声采样候选动作
// 2. Sequential Halving: 逐轮淘汰候选动作，分配搜索预算
// 3. 同步执行: 无异步/递归，直接持有模型
// 4. 确定性动作选择: 最终选择 completed_Q 最高的动作

pub mod budget;
pub mod config;
pub mod evaluator;
pub mod node;
pub mod search;

// 统一导出所有公共接口
pub use budget::SequentialHalvingBudget;
pub use config::{GumbelConfig, MctsSearchResult};
pub use evaluator::Evaluator;
pub use node::{MctsArena, MctsNode, get_outcome_id, value_from_perspective};
pub use search::{GumbelMCTS, PathStep, PendingEval};
