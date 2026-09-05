// ==============================================================================
// --- 4x2 迷你暗棋环境 (MiniDarkChessEnv) ---
//
// 复用共享的 DarkChessEnv 核心逻辑（config 驱动），仅以 mini_config() 区分：
// - 棋盘 4x2（8 格）
// - 仅 兵 / 炮 / 士 / 将 四种棋子，每方各 1 子（共 8 子填满棋盘）
// - 血量上限 = 2 + 5 + 10 + 30 = 47（= 单方棋子价值总和），全灭敌方即判胜
//
// 本类型仅提供与 `DarkChessEnv` 不同的 `GameEnv` 关联常量（10 通道 / 4x2 / 11 标量），
// 其余全部委托给 `inner`。
// ==============================================================================

use crate::core::env::board::DarkChessEnv;
use crate::core::env::config::mini_config;
use crate::core::env::types::{ResNetObservation, Player};

#[derive(Clone, Copy, Debug)]
pub struct MiniDarkChessEnv {
    pub inner: DarkChessEnv,
}

/// 4x2 迷你暗棋：动作空间 = 翻8 + 常规20 + 炮12 = 40。
pub const MINI_ACTION_SPACE_SIZE: usize = 40;
/// 4x2 迷你暗棋：棋盘通道数 = 2*4(激活类型) + 2 = 10
pub const MINI_RESNET_BOARD_CHANNELS: usize = 10;
/// 4x2 迷你暗棋：棋盘行数
pub const MINI_BOARD_ROWS: usize = 4;
/// 4x2 迷你暗棋：棋盘列数
pub const MINI_BOARD_COLS: usize = 2;
/// 4x2 迷你暗棋：标量特征数 = 3 + 2*4 = 11
pub const MINI_RESNET_SCALAR_FEATURE_COUNT: usize = 11;

impl MiniDarkChessEnv {
    /// 创建标准 4x2 迷你开局。
    pub fn new() -> Self {
        Self {
            inner: DarkChessEnv::with_config(mini_config()),
        }
    }

    pub fn action_space_size() -> usize {
        mini_config().action_space_size
    }

    pub fn max_steps() -> usize {
        mini_config().max_steps_per_episode
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
    ) -> Result<(f32, bool, bool, Option<i32>), String> {
        self.inner.step(action, None)
    }

    pub fn get_resnet_state(&self) -> ResNetObservation {
        self.inner.get_resnet_state()
    }

    pub fn check_game_over_conditions(&self) -> (bool, bool, Option<i32>) {
        self.inner.check_game_over_conditions()
    }

    pub fn encode_resnet_features_flat_into(&self, board_data: &mut Vec<f32>, scalars_data: &mut Vec<f32>) {
        self.inner.resnet_features_flat_into(board_data, scalars_data);
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

    pub fn nnue_active_features(&self) -> Vec<usize> {
        self.inner.nnue_active_features()
    }

    /// 打印棋盘（方便演示/调试）。
    pub fn print_board(&self) {
        self.inner.print_board();
    }
}

impl Default for MiniDarkChessEnv {
    fn default() -> Self {
        Self::new()
    }
}
