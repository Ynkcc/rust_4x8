//! 简单 AI 策略模块
//!
//! - `Policy`：策略接口
//! - `RandomPolicy`：随机选择任何有效动作
//! - `RevealFirstPolicy`：若可翻棋则优先翻棋，否则随机有效动作

use crate::DarkChessEnv;

mod random;
mod reveal_first;
mod mcts_dl;

pub use random::RandomPolicy;
pub use reveal_first::RevealFirstPolicy;
pub use mcts_dl::{MctsDlPolicy, ModelWrapper};

/// 策略接口：给定环境，返回一个有效动作编号
pub trait Policy {
    fn choose_action(env: &DarkChessEnv) -> Option<usize>;
}
