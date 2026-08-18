// src/game_env/traits.rs
// 泛型游戏环境抽象：Gumbel MCTS 依赖的最小接口集。
//
// 背景：
// - 本项目最初只支持暗棋（DarkChessEnv），MCTS（mcts/）被硬编码绑定到
//   具体的 DarkChessEnv / Slot / Piece 类型上。
// - 为了复用同一套搜索核心做井字棋（Tic-Tac-Toe）与 4x2 迷你暗棋验证，
//   这里抽出 `GameEnv` trait。
// - 机会节点（翻牌随机性）是暗棋特有语义：trait 提供 `is_chance_action` /
//   `chance_outcomes` / `step_outcome_id` 三个扩展点，默认实现为「无机会节点」，
//   DarkChessEnv 与 MiniDarkChessEnv 覆盖实现、TicTacToeEnv 保持默认。

use super::board::DarkChessEnv;
use super::config::GameConfig;
use super::constants::MAX_STEPS_PER_EPISODE;
use super::variants::game4x4::Game4x4Env;
use super::variants::mini_darkchess::MiniDarkChessEnv;
use super::types::{Observation, Piece, Player};

/// 泛型游戏环境：Gumbel MCTS 对其施加的全部约束。
///
/// 要求 `Copy`：MCTS 节点以值语义保存环境快照（与既有 DarkChessEnv 的 Copy 设计一致）。
pub trait GameEnv: Copy + Clone + Send + Sync + 'static {
    /// 动作空间大小
    fn action_space_size() -> usize;

    /// 当前玩家
    fn get_current_player(&self) -> Player;

    /// 将合法动作掩码写入 `masks`（合法位置 1，其余位置 0）。
    /// 调用方保证 `masks.len() >= Self::action_space_size()`。
    fn action_masks_into(&self, masks: &mut [i32]);

    /// 执行动作，返回 `(观测, 奖励, 是否终止, 是否截断, 胜者)`。
    ///
    /// `winner` 使用全局视角：`Some(1)` = 红方/先手胜，`Some(-1)` = 黑方/后手胜，
    /// `Some(0)` = 平局，`None` = 未结束。
    fn step(
        &mut self,
        action: usize,
    ) -> Result<(Observation, f32, bool, bool, Option<i32>), String>;

    /// 获取当前观测（神经网络输入）
    fn get_state(&self) -> Observation;

    /// 终局检测：`(terminated, truncated, winner)`
    fn check_game_over_conditions(&self) -> (bool, bool, Option<i32>);

    /// 每局最大步数（步数上限截断）。
    fn max_steps() -> usize {
        MAX_STEPS_PER_EPISODE
    }

    // ------------------------------------------------------------------------
    // 神经网络特征形状（供批量推理 / Python 绑定使用）
    // ------------------------------------------------------------------------

    /// 棋盘特征通道数
    const BOARD_CHANNELS: usize;
    /// 棋盘行数
    const BOARD_ROWS: usize;
    /// 棋盘列数
    const BOARD_COLS: usize;
    /// 标量特征数
    const SCALAR_FEATURE_COUNT: usize;

    /// 将环境编码为扁平特征写入外部缓冲区。
    fn encode_features_flat_into(&self, board_data: &mut Vec<f32>, scalars_data: &mut Vec<f32>);

    // ------------------------------------------------------------------------
    // 机会节点扩展点
    // ------------------------------------------------------------------------

    fn is_chance_action(&self, _action: usize) -> bool {
        false
    }

    fn chance_outcomes(&self, _action: usize) -> Vec<(usize, f32, Self)> {
        Vec::new()
    }

    fn step_outcome_id(&self, _action: usize) -> Option<usize> {
        None
    }

    // ------------------------------------------------------------------------
    // 终局血量差（训练/归档辅助数据）
    // ------------------------------------------------------------------------

    /// 终局归一化血量差（红方视角为正）。
    ///
    /// 公式：`(红方HP - 黑方HP) / (初始总HP + 最大子力分值)`，大致落在 [-1, 1]。
    /// 在终局（与获取游戏真实结果同一时机）调用。无血量机制的游戏（如井字棋）返回 None。
    fn terminal_health_diff_red(&self) -> Option<f32> {
        None
    }
}

// ============================================================================
// 暗棋实现
// ============================================================================

/// 获取棋子的唯一结果 ID（暗棋机会节点的可能结果标识）。
///
/// ID 计算方式：按 config 的激活类型紧凑索引 + 玩家偏移（红方 0，黑方 num_active）。
pub fn get_outcome_id(cfg: &GameConfig, piece: &Piece) -> usize {
    cfg.outcome_id_for(piece.piece_type, piece.player == Player::Black)
}

impl GameEnv for DarkChessEnv {
    fn action_space_size() -> usize {
        super::constants::ACTION_SPACE_SIZE
    }

    fn get_current_player(&self) -> Player {
        DarkChessEnv::get_current_player(self)
    }

    fn action_masks_into(&self, masks: &mut [i32]) {
        DarkChessEnv::action_masks_into(self, masks);
    }

    fn step(
        &mut self,
        action: usize,
    ) -> Result<(Observation, f32, bool, bool, Option<i32>), String> {
        DarkChessEnv::step(self, action, None)
    }

    fn get_state(&self) -> Observation {
        DarkChessEnv::get_state(self)
    }

    fn check_game_over_conditions(&self) -> (bool, bool, Option<i32>) {
        DarkChessEnv::check_game_over_conditions(self)
    }

    fn max_steps() -> usize {
        super::constants::MAX_STEPS_PER_EPISODE
    }

    const BOARD_CHANNELS: usize = super::constants::BOARD_CHANNELS;
    const BOARD_ROWS: usize = super::constants::BOARD_ROWS;
    const BOARD_COLS: usize = super::constants::BOARD_COLS;
    const SCALAR_FEATURE_COUNT: usize = super::constants::SCALAR_FEATURE_COUNT;

    fn encode_features_flat_into(&self, board_data: &mut Vec<f32>, scalars_data: &mut Vec<f32>) {
        DarkChessEnv::get_state_flat_into(self, board_data, scalars_data);
    }

    // --- 机会节点 ---

    fn is_chance_action(&self, action: usize) -> bool {
        DarkChessEnv::is_chance_action(self, action)
    }

    fn chance_outcomes(&self, action: usize) -> Vec<(usize, f32, Self)> {
        DarkChessEnv::chance_outcomes(self, action)
    }

    fn step_outcome_id(&self, action: usize) -> Option<usize> {
        DarkChessEnv::step_outcome_id(self, action)
    }

    fn terminal_health_diff_red(&self) -> Option<f32> {
        let denom = self.config.initial_health as f32
            + self.config.piece_values.iter().copied().max().unwrap_or(0) as f32;
        if denom <= 0.0 {
            None
        } else {
            Some((self.get_hp(Player::Red) - self.get_hp(Player::Black)) as f32 / denom)
        }
    }
}

// ============================================================================
// 4x4 暗棋实现
// ============================================================================

impl GameEnv for Game4x4Env {
    fn action_space_size() -> usize {
        Game4x4Env::action_space_size()
    }

    fn get_current_player(&self) -> Player {
        Game4x4Env::get_current_player(self)
    }

    fn action_masks_into(&self, masks: &mut [i32]) {
        Game4x4Env::action_masks_into(self, masks);
    }

    fn step(
        &mut self,
        action: usize,
    ) -> Result<(Observation, f32, bool, bool, Option<i32>), String> {
        Game4x4Env::step(self, action)
    }

    fn get_state(&self) -> Observation {
        Game4x4Env::get_state(self)
    }

    fn check_game_over_conditions(&self) -> (bool, bool, Option<i32>) {
        Game4x4Env::check_game_over_conditions(self)
    }

    fn max_steps() -> usize {
        Game4x4Env::max_steps()
    }

    const BOARD_CHANNELS: usize = 16; // 2*7(全激活) + 2
    const BOARD_ROWS: usize = 4;
    const BOARD_COLS: usize = 4;
    const SCALAR_FEATURE_COUNT: usize = 19; // 3 + 2*8

    fn encode_features_flat_into(&self, board_data: &mut Vec<f32>, scalars_data: &mut Vec<f32>) {
        Game4x4Env::encode_features_flat_into(self, board_data, scalars_data);
    }

    fn is_chance_action(&self, action: usize) -> bool {
        Game4x4Env::is_chance_action(self, action)
    }

    fn chance_outcomes(&self, action: usize) -> Vec<(usize, f32, Self)> {
        Game4x4Env::chance_outcomes(self, action)
    }

    fn step_outcome_id(&self, action: usize) -> Option<usize> {
        Game4x4Env::step_outcome_id(self, action)
    }

    fn terminal_health_diff_red(&self) -> Option<f32> {
        self.inner.terminal_health_diff_red()
    }
}

// ============================================================================
// 4x2 迷你暗棋实现
// ============================================================================

impl GameEnv for MiniDarkChessEnv {
    fn action_space_size() -> usize {
        MiniDarkChessEnv::action_space_size()
    }

    fn get_current_player(&self) -> Player {
        MiniDarkChessEnv::get_current_player(self)
    }

    fn action_masks_into(&self, masks: &mut [i32]) {
        MiniDarkChessEnv::action_masks_into(self, masks);
    }

    fn step(
        &mut self,
        action: usize,
    ) -> Result<(Observation, f32, bool, bool, Option<i32>), String> {
        MiniDarkChessEnv::step(self, action)
    }

    fn get_state(&self) -> Observation {
        MiniDarkChessEnv::get_state(self)
    }

    fn check_game_over_conditions(&self) -> (bool, bool, Option<i32>) {
        MiniDarkChessEnv::check_game_over_conditions(self)
    }

    fn max_steps() -> usize {
        MiniDarkChessEnv::max_steps()
    }

    const BOARD_CHANNELS: usize = 10;
    const BOARD_ROWS: usize = 4;
    const BOARD_COLS: usize = 2;
    const SCALAR_FEATURE_COUNT: usize = 11;

    fn encode_features_flat_into(&self, board_data: &mut Vec<f32>, scalars_data: &mut Vec<f32>) {
        MiniDarkChessEnv::encode_features_flat_into(self, board_data, scalars_data);
    }

    fn is_chance_action(&self, action: usize) -> bool {
        MiniDarkChessEnv::is_chance_action(self, action)
    }

    fn chance_outcomes(&self, action: usize) -> Vec<(usize, f32, Self)> {
        MiniDarkChessEnv::chance_outcomes(self, action)
    }

    fn step_outcome_id(&self, action: usize) -> Option<usize> {
        MiniDarkChessEnv::step_outcome_id(self, action)
    }

    fn terminal_health_diff_red(&self) -> Option<f32> {
        self.inner.terminal_health_diff_red()
    }
}
