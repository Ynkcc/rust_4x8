//! 基础规则策略模块

use crate::core::env::DarkChessEnv;

pub mod random;
pub mod reveal_first;

pub use random::RandomPolicy;
pub use reveal_first::RevealFirstPolicy;

/// 策略接口：给定环境，返回一个有效动作编号
pub trait Policy {
    fn choose_action(env: &DarkChessEnv) -> Option<usize>;
}
