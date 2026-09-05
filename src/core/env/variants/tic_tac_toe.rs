// src/game_env/tic_tac_toe.rs
// 井字棋（Tic-Tac-Toe）环境实现。
//
// 设计要点（与暗棋共用同一套 Gumbel MCTS）：
// - `Copy` 类型：cells 用定长数组 `[i8; 9]`，MCTS 节点以值语义保存环境快照。
// - 动作空间 = 9（每个格子一个动作，动作索引即格子索引 0..=8）。
// - 所有节点均为常规节点，无机会节点（`GameEnv` 机会节点扩展点保持默认关闭）。
// - 玩家视角与暗棋一致：红方 Red=1 先手执 X，黑方 Black=-1 后手执 O。
// - 特征编码遵循暗棋 features.rs 约定：通道0=当前方棋子，通道1=对手棋子。

use ndarray::{Array1, Array3};

use crate::core::env::traits::GameEnv;
use crate::core::env::types::{ResNetObservation, Player};

/// 井字棋动作空间大小（= 格子数）
pub const TTT_ACTION_SPACE_SIZE: usize = 9;
/// 棋盘尺寸（3x3）
pub const TTT_BOARD_ROWS: usize = 3;
pub const TTT_BOARD_COLS: usize = 3;
/// 特征通道数：0=当前方，1=对手
pub const TTT_RESNET_BOARD_CHANNELS: usize = 2;
/// 井字棋无标量特征
pub const TTT_RESNET_SCALAR_FEATURE_COUNT: usize = 0;

/// 井字棋环境。
///
/// `cells` 取值：0=空格，1=X（红方/先手），-1=O（黑方/后手）。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TicTacToeEnv {
    cells: [i8; 9],
    current_player: Player,
    total_steps: usize,
}

/// 胜负判定所需的全部连线（3 行 + 3 列 + 2 对角线）
const WIN_LINES: [[usize; 3]; 8] = [
    [0, 1, 2],
    [3, 4, 5],
    [6, 7, 8],
    [0, 3, 6],
    [1, 4, 7],
    [2, 5, 8],
    [0, 4, 8],
    [2, 4, 6],
];

impl TicTacToeEnv {
    /// 创建标准开局（红方/先手执 X，先行）
    pub fn new() -> Self {
        Self {
            cells: [0; 9],
            current_player: Player::Red,
            total_steps: 0,
        }
    }

    /// 从给定格子布局与当前玩家重建环境（供测试/棋谱还原）。
    pub fn from_cells(cells: [i8; 9], current_player: Player) -> Self {
        Self {
            cells,
            current_player,
            total_steps: cells.iter().filter(|&&c| c != 0).count(),
        }
    }

    /// 读取格子（0=空，1=X，-1=O）
    pub fn cell(&self, idx: usize) -> i8 {
        self.cells[idx]
    }

    /// 当前所有格子（副本）
    pub fn cells(&self) -> [i8; 9] {
        self.cells
    }

    /// 已走的步数
    pub fn total_steps(&self) -> usize {
        self.total_steps
    }

    /// 纯函数胜者判定：返回 `Some(1)`=X 胜、`Some(-1)`=O 胜、`Some(0)`=平局、`None`=未结束。
    fn winner_from(cells: &[i8; 9]) -> Option<i32> {
        for line in WIN_LINES {
            let a = cells[line[0]];
            let b = cells[line[1]];
            let c = cells[line[2]];
            if a != 0 && a == b && b == c {
                return if a == 1 { Some(1) } else { Some(-1) };
            }
        }
        if cells.iter().all(|&c| c != 0) {
            Some(0)
        } else {
            None
        }
    }

    /// 将当前棋盘编码为特征张量写入 `board`（通道0=当前方，通道1=对手）。
    fn encode_into(&self, board: &mut Vec<f32>) {
        board.clear();
        board.reserve(TTT_RESNET_BOARD_CHANNELS * TTT_ACTION_SPACE_SIZE);
        let my = self.current_player;
        let opp = my.opposite();
        for ch in 0..TTT_RESNET_BOARD_CHANNELS {
            let target = if ch == 0 { my.val() as i8 } else { opp.val() as i8 };
            for i in 0..TTT_ACTION_SPACE_SIZE {
                board.push(if self.cells[i] == target { 1.0 } else { 0.0 });
            }
        }
    }
}

impl Default for TicTacToeEnv {
    fn default() -> Self {
        Self::new()
    }
}

impl GameEnv for TicTacToeEnv {
    fn action_space_size() -> usize {
        TTT_ACTION_SPACE_SIZE
    }

    fn get_current_player(&self) -> Player {
        self.current_player
    }

    fn action_masks_into(&self, masks: &mut [i32]) {
        for i in 0..TTT_ACTION_SPACE_SIZE {
            masks[i] = if self.cells[i] == 0 { 1 } else { 0 };
        }
    }

    fn step(&mut self, action: usize) -> Result<(f32, bool, bool, Option<i32>), String> {
        if action >= TTT_ACTION_SPACE_SIZE || self.cells[action] != 0 {
            return Err(format!("无效动作: {}", action));
        }
        self.cells[action] = self.current_player.val() as i8;
        self.total_steps += 1;

        let winner = Self::winner_from(&self.cells);
        let terminated = winner.is_some();
        // 始终切换玩家（与暗棋一致）：即使游戏结束也切换，保证「交替行动」
        // 语义一致，使 minimax / MCTS 的视角取反逻辑在终局节点上依然成立。
        self.current_player = self.current_player.opposite();
        Ok((0.0, terminated, false, winner))
    }

    fn get_resnet_state(&self) -> ResNetObservation {
        let mut board_data = Vec::with_capacity(TTT_RESNET_BOARD_CHANNELS * TTT_ACTION_SPACE_SIZE);
        self.encode_into(&mut board_data);
        let board =
            Array3::from_shape_vec((TTT_RESNET_BOARD_CHANNELS, TTT_BOARD_ROWS, TTT_BOARD_COLS), board_data)
                .expect("Failed to reshape ttt board array");
        ResNetObservation {
            board,
            scalars: Array1::from_vec(Vec::new()),
        }
    }

    fn check_game_over_conditions(&self) -> (bool, bool, Option<i32>) {
        let winner = Self::winner_from(&self.cells);
        (winner.is_some(), false, winner)
    }

    fn max_steps() -> usize {
        TTT_ACTION_SPACE_SIZE
    }

    const RESNET_BOARD_CHANNELS: usize = TTT_RESNET_BOARD_CHANNELS;
    const BOARD_ROWS: usize = TTT_BOARD_ROWS;
    const BOARD_COLS: usize = TTT_BOARD_COLS;
    const RESNET_SCALAR_FEATURE_COUNT: usize = TTT_RESNET_SCALAR_FEATURE_COUNT;

    fn encode_resnet_features_flat_into(&self, board_data: &mut Vec<f32>, scalars_data: &mut Vec<f32>) {
        self.encode_into(board_data);
        scalars_data.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_winner_lines() {
        // X 对角胜利
        let cells = [1, -1, 0, -1, 1, 0, 0, 0, 1];
        assert_eq!(TicTacToeEnv::winner_from(&cells), Some(1));
        // O 行胜利
        let cells = [0, 0, 0, -1, -1, -1, 1, 0, 1];
        assert_eq!(TicTacToeEnv::winner_from(&cells), Some(-1));
        // X 列胜利
        let cells = [1, 0, -1, 1, -1, 0, 1, 0, 0];
        assert_eq!(TicTacToeEnv::winner_from(&cells), Some(1));
        // 平局
        let cells = [1, -1, 1, 1, -1, 1, -1, 1, -1];
        assert_eq!(TicTacToeEnv::winner_from(&cells), Some(0));
        // 未结束
        let cells = [1, -1, 0, 0, 0, 0, 0, 0, 0];
        assert_eq!(TicTacToeEnv::winner_from(&cells), None);
    }

    #[test]
    fn test_step_and_action_mask() {
        let mut env = TicTacToeEnv::new();
        assert_eq!(env.get_current_player(), Player::Red);
        let mut masks = [0i32; TTT_ACTION_SPACE_SIZE];
        env.action_masks_into(&mut masks);
        assert_eq!(masks, [1; 9]);

        let (_, term, trunc, winner) = env.step(4).unwrap();
        assert!(!term && !trunc && winner.is_none());
        assert_eq!(env.cell(4), 1);
        assert_eq!(env.get_current_player(), Player::Black);

        // 非法落子
        assert!(env.step(4).is_err());
        // 掩码更新
        let mut masks = [0i32; TTT_ACTION_SPACE_SIZE];
        env.action_masks_into(&mut masks);
        assert_eq!(masks[4], 0);
        assert_eq!(masks[0], 1);
    }

    #[test]
    fn test_full_game_x_wins() {
        let mut env = TicTacToeEnv::new();
        // X 走 0,3,6 → 列胜
        for &a in &[0, 1, 3, 2, 6] {
            let (_, term, _, winner) = env.step(a).unwrap();
            if a == 6 {
                assert!(term);
                assert_eq!(winner, Some(1));
            }
        }
    }

    #[test]
    fn test_draw_game() {
        let mut env = TicTacToeEnv::new();
        // 标准平局序列（双方均不失误）
        for &a in &[0, 1, 2, 4, 3, 5, 7, 6, 8] {
            let (_, term, _, winner) = env.step(a).unwrap();
            if a == 8 {
                assert!(term);
                assert_eq!(winner, Some(0));
            }
        }
    }

    #[test]
    fn test_encode_perspective() {
        // X 在中心、O 在左上角，轮到 Black（O 方）时视角翻转
        let env = TicTacToeEnv::from_cells(
            [-1, 0, 0, 0, 1, 0, 0, 0, 0],
            Player::Black,
        );
        let obs = env.get_resnet_state();
        let board = obs.board.as_slice().unwrap();
        assert_eq!(board.len(), TTT_RESNET_BOARD_CHANNELS * TTT_ACTION_SPACE_SIZE);
        // 通道0=O(当前方)，通道1=X(对手)
        assert_eq!(board[0], 1.0); // 通道0 左上角 O
        assert_eq!(board[4], 0.0); // 通道0 中心（X 在对手通道）
        assert_eq!(board[9 + 0], 0.0); // 通道1 左上角
        assert_eq!(board[9 + 4], 1.0); // 通道1 中心 X
    }
}
