//! 策略引擎模块 (Strategy & Engine)
//!
//! 归集 Alpha-Beta 强引擎（core/expectimax）、走子生成及基础策略。

pub mod movegen;
pub mod policies;

#[cfg(feature = "torch")]
pub mod mcts_dl;

pub use movegen::generate_moves;
pub use policies::{Policy, RandomPolicy, RevealFirstPolicy};


#[cfg(feature = "torch")]
pub use mcts_dl::{MctsDlPolicy, ModelWrapper, TchEvaluator};
