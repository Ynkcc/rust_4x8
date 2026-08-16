//! 简单 AI 策略模块
//!
//! - `Policy`：策略接口
//! - `RandomPolicy`：随机选择任何有效动作
//! - `RevealFirstPolicy`：若可翻棋则优先翻棋，否则随机有效动作
//! - `minimax`：expectiminimax + alpha-beta（已升级：多特征评估/置换表/走子排序/静态搜索）
//! - `engine`：纯计算强引擎（αβ + Star1 + 置换表 + 迭代加深，移植 misty-banqi）
//! - `eval`：共享多特征启发式评估（校正价值表 + 覆盖物质 + 将帅情境 + 支配价值 + 机动性 + 将帅危险度）
//! - `heuristic_mcts`：纯计算启发式 Gumbel MCTS 对手（无需 torch）

use crate::DarkChessEnv;

#[cfg(feature = "torch")]
mod mcts_dl;
pub mod engine;
pub mod eval;
pub mod heuristic_mcts;
pub mod minimax;
pub(crate) mod movegen;
mod random;
mod reveal_first;

pub use engine::{EngineConfig, EngineResult, best_move};
pub use eval::{
    CORRECTED_VALUES, EVAL_SCALE, EvalParams, evaluate, evaluate_for,
};
pub use heuristic_mcts::{HeuristicEvaluator, HeuristicMctsPolicy};
pub use minimax::{
    MinimaxConfig, MinimaxResult, heuristic_value, minimax_best_action,
    minimax_best_action_with_config, minimax_choose_action,
};

#[cfg(feature = "torch")]
pub use mcts_dl::{MctsDlPolicy, ModelWrapper};
pub use random::RandomPolicy;
pub use reveal_first::RevealFirstPolicy;

/// 策略接口：给定环境，返回一个有效动作编号
pub trait Policy {
    fn choose_action(env: &DarkChessEnv) -> Option<usize>;
}
