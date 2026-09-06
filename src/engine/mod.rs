//! 策略引擎与博弈搜索算法模块 (Strategy & Engine)
//!
//! 归集 Alpha-Beta 强引擎（core/expectimax）、启发式评估、启发式 MCTS 及各种开局/基础策略。

pub mod evaluation;
pub mod mcts_heuristic;
pub mod movegen;
pub mod policies;

#[cfg(feature = "torch")]
pub mod mcts_dl;

pub use evaluation::{
    CORRECTED_VALUES, EVAL_SCALE, EvalParams, evaluate, evaluate_for,
};
pub use mcts_heuristic::{HeuristicEvaluator, HeuristicMctsPolicy};
pub use movegen::generate_moves;
pub use policies::{Policy, RandomPolicy, RevealFirstPolicy};


#[cfg(feature = "torch")]
pub use mcts_dl::{MctsDlPolicy, ModelWrapper, TchEvaluator};
