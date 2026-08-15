// ==============================================================================
// --- 4x4 暗棋环境 (Game4x4Env) ---
//
// 复用共享的 DarkChessEnv 核心逻辑（config 驱动），仅以 game_4x4_config() 区分：
// - 棋盘 4x4（16 格）
// - 7 类棋子全激活，每方：兵2 炮1 马1 车1 象1 士1 将1（共 8 子填满棋盘）
// - 分值：兵4 / 炮10 / 马10 / 车10 / 象10 / 士20 / 将30
// - 血量上限 = 60（由变体指定，独立于分值总和）
//
// 本类型仅提供与 `DarkChessEnv` 不同的 `GameEnv` 关联常量
// （16 通道 / 4x4 / 19 标量），其余全部委托给 `inner`。
// ==============================================================================

use super::board::DarkChessEnv;
use super::config::game_4x4_config;
use super::types::{Observation, Player};

#[derive(Clone, Copy, Debug)]
pub struct Game4x4Env {
    pub inner: DarkChessEnv,
}

/// 4x4 暗棋动作空间大小 = 翻棋16 + 常规48 + 炮击48 = 112。
pub const GAME4X4_ACTION_SPACE_SIZE: usize = 112;

impl Game4x4Env {
    /// 创建标准 4x4 开局。
    pub fn new() -> Self {
        Self {
            inner: DarkChessEnv::with_config(game_4x4_config()),
        }
    }

    pub fn action_space_size() -> usize {
        game_4x4_config().action_space_size
    }

    pub fn max_steps() -> usize {
        game_4x4_config().max_steps_per_episode
    }

    pub fn get_current_player(&self) -> Player {
        self.inner.get_current_player()
    }

    /// 切换当前玩家（flip_player：不改变棋盘/棋子归属，仅改变编码视角）。
    pub fn flip_player(&mut self) {
        self.inner.flip_player();
    }

    pub fn action_masks_into(&self, masks: &mut [i32]) {
        self.inner.action_masks_into(masks);
    }

    pub fn step(
        &mut self,
        action: usize,
    ) -> Result<(Observation, f32, bool, bool, Option<i32>), String> {
        self.inner.step(action, None)
    }

    pub fn get_state(&self) -> Observation {
        self.inner.get_state()
    }

    pub fn check_game_over_conditions(&self) -> (bool, bool, Option<i32>) {
        self.inner.check_game_over_conditions()
    }

    pub fn encode_features_flat_into(&self, board_data: &mut Vec<f32>, scalars_data: &mut Vec<f32>) {
        self.inner.get_state_flat_into(board_data, scalars_data);
    }

    pub fn is_chance_action(&self, action: usize) -> bool {
        self.inner.is_chance_action(action)
    }

    pub fn chance_outcomes(&self, action: usize) -> Vec<(usize, f32, Self)> {
        self.inner
            .chance_outcomes(action)
            .into_iter()
            .map(|(id, p, env)| (id, p, Self { inner: env }))
            .collect()
    }

    pub fn step_outcome_id(&self, action: usize) -> Option<usize> {
        self.inner.step_outcome_id(action)
    }

    /// 打印棋盘（方便演示/调试）。
    pub fn print_board(&self) {
        self.inner.print_board();
    }
}

impl Default for Game4x4Env {
    fn default() -> Self {
        Self::new()
    }
}
