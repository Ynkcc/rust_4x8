//! 策略引擎与博弈搜索算法模块 (Strategy & Engine)
//!
//! 归集传统 Minimax、Alpha-Beta 搜索强引擎、启发式 MCTS 以及各种开局/基础策略。

pub mod alpha_beta;
pub mod evaluation;
pub mod mcts_heuristic;
pub mod minimax;
pub mod movegen;
pub mod policies;

#[cfg(feature = "torch")]
pub mod mcts_dl;

pub use alpha_beta::{EngineConfig, EngineResult, best_move};
pub use evaluation::{
    CORRECTED_VALUES, EVAL_SCALE, EvalParams, evaluate, evaluate_for,
};
pub use mcts_heuristic::{HeuristicEvaluator, HeuristicMctsPolicy};
pub use minimax::{
    MinimaxConfig, MinimaxResult, heuristic_value, minimax_best_action,
    minimax_best_action_with_config, minimax_choose_action,
};
pub use movegen::generate_moves;
pub use policies::{Policy, RandomPolicy, RevealFirstPolicy};


#[cfg(feature = "torch")]
pub use mcts_dl::{MctsDlPolicy, ModelWrapper, TchEvaluator};
