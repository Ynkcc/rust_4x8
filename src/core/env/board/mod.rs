// src/game_env/board/mod.rs
//
// 暗棋环境 DarkChessEnv：Copy 语义、config 驱动、支持机会节点（翻棋 / 吃暗子）。
//
// 子模块分层（同一类型的多个 impl 块分散，私有辅助以 pub(crate) 跨文件可见）：
//   - struct_def:  结构体定义 + 构造器 (new / with_config / from_board / get_coords_for_action)
//   - reset:       内部状态复位、棋盘初始化、翻子、翻棋概率表
//   - step:        核心 step 逻辑、走子应用、机会节点扩展、棋盘打印
//   - accessors:   公共访问器 + 供其他模块使用的内部辅助
//   - tests:       单元测试

use rand::Rng;
use rand::seq::SliceRandom;
use rand::thread_rng;
use rand::SeedableRng;

use super::actions::{action_lookup_tables, pack_coords};
use super::bitboard::{board_mask, ray_attacks, ull};
use super::config::{
    GameConfig, MAX_PIECES_PER_PLAYER, MAX_POSITIONS, MAX_REVEAL_PROBABILITY_SIZE,
    NUM_PIECE_TYPES_MAX, darkchess_config, game_4x4_config, mini_config,
};
use super::types::*;

pub use struct_def::DarkChessEnv;

#[cfg(test)]
mod tests;

mod accessors;
mod reset;
mod step;
mod struct_def;

impl DarkChessEnv {
    pub fn reset(&mut self) {
        self.reset_internal_state();
        self.initialize_board();
    }
}
