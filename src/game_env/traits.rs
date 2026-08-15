// src/game_env/traits.rs
// 泛型游戏环境抽象：Gumbel MCTS 依赖的最小接口集。
//
// 背景：
// - 本项目最初只支持暗棋（DarkChessEnv），MCTS（mcts/）被硬编码绑定到
//   具体的 DarkChessEnv / Slot / Piece 类型上。
// - 为了复用同一套搜索核心做井字棋（Tic-Tac-Toe）验证，这里抽出 `GameEnv` trait。
// - 机会节点（翻牌随机性）是暗棋特有语义：trait 提供 `is_chance_action` /
//   `chance_outcomes` / `step_outcome_id` 三个扩展点，默认实现为「无机会节点」，
//   DarkChessEnv 覆盖实现、TicTacToeEnv 保持默认（所有节点均为常规节点）。

use super::board::DarkChessEnv;
use super::constants::MAX_STEPS_PER_EPISODE;
use super::types::{Observation, Piece, PieceType, Player};

/// 泛型游戏环境：Gumbel MCTS 对其施加的全部约束。
///
/// 要求 `Copy`：MCTS 节点以值语义保存环境快照（与既有 DarkChessEnv 的 Copy 设计一致）。
pub trait GameEnv: Copy + Clone + Send + Sync + 'static {
    /// 动作空间大小（暗棋 352，井字棋 9）
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

    /// 每局最大步数（步数上限截断）。暗棋使用 100；井字棋 9 步内必结束，用 9。
    fn max_steps() -> usize {
        MAX_STEPS_PER_EPISODE
    }

    // ------------------------------------------------------------------------
    // 神经网络特征形状（供批量推理 / Python 绑定使用）
    // ------------------------------------------------------------------------

    /// 棋盘特征通道数（暗棋 16，井字棋 2）
    const BOARD_CHANNELS: usize;
    /// 棋盘行数
    const BOARD_ROWS: usize;
    /// 棋盘列数
    const BOARD_COLS: usize;
    /// 标量特征数（暗棋 35，井字棋 0）
    const SCALAR_FEATURE_COUNT: usize;

    /// 将环境编码为扁平特征写入外部缓冲区（避免每次推理重复分配）。
    ///
    /// `board_data` 长度 = `BOARD_CHANNELS * BOARD_ROWS * BOARD_COLS`，
    /// `scalars_data` 长度 = `SCALAR_FEATURE_COUNT`。调用方须在写前 `clear()`。
    fn encode_features_flat_into(&self, board_data: &mut Vec<f32>, scalars_data: &mut Vec<f32>);

    // ------------------------------------------------------------------------
    // 机会节点扩展点（暗棋特有；井字棋「所有节点均为常规节点」，保持默认实现）
    // ------------------------------------------------------------------------

    /// 该动作是否会产生机会节点（暗棋：翻牌且目标格为 Hidden；井字棋：恒为 false）
    fn is_chance_action(&self, _action: usize) -> bool {
        false
    }

    /// 枚举机会动作的所有可能结果：`(outcome_id, 概率, 结果环境)`。
    ///
    /// 在「执行该动作之前」的环境上调用。默认实现返回空（无机会节点）。
    fn chance_outcomes(&self, _action: usize) -> Vec<(usize, f32, Self)> {
        Vec::new()
    }

    /// 执行动作后，若该动作产生了机会结果，返回其 `outcome_id`
    /// （用于 MCTS 子树复用匹配）。普通动作返回 `None`。
    fn step_outcome_id(&self, _action: usize) -> Option<usize> {
        None
    }
}

// ============================================================================
// 暗棋实现
// ============================================================================

/// 获取棋子的唯一结果 ID（暗棋机会节点的可能结果标识）。
///
/// ID 计算方式：棋子类型索引 + 玩家偏移量（红方 0，黑方 7），范围 0-13。
pub fn get_outcome_id(piece: &Piece) -> usize {
    let type_idx = match piece.piece_type {
        PieceType::Soldier => 0,
        PieceType::Cannon => 1,
        PieceType::Horse => 2,
        PieceType::Chariot => 3,
        PieceType::Elephant => 4,
        PieceType::Advisor => 5,
        PieceType::General => 6,
    };
    let player_offset = match piece.player {
        Player::Red => 0,
        Player::Black => 7,
    };
    type_idx + player_offset
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
        MAX_STEPS_PER_EPISODE
    }

    const BOARD_CHANNELS: usize = super::constants::BOARD_CHANNELS;
    const BOARD_ROWS: usize = super::constants::BOARD_ROWS;
    const BOARD_COLS: usize = super::constants::BOARD_COLS;
    const SCALAR_FEATURE_COUNT: usize = super::constants::SCALAR_FEATURE_COUNT;

    fn encode_features_flat_into(&self, board_data: &mut Vec<f32>, scalars_data: &mut Vec<f32>) {
        DarkChessEnv::get_state_flat_into(self, board_data, scalars_data);
    }

    // --- 机会节点（实现已下沉到 DarkChessEnv，见 board.rs） ---

    fn is_chance_action(&self, action: usize) -> bool {
        DarkChessEnv::is_chance_action(self, action)
    }

    fn chance_outcomes(&self, action: usize) -> Vec<(usize, f32, Self)> {
        DarkChessEnv::chance_outcomes(self, action)
    }

    fn step_outcome_id(&self, action: usize) -> Option<usize> {
        DarkChessEnv::step_outcome_id(self, action)
    }
}
