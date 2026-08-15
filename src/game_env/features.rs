use ndarray::{Array1, Array3};

use super::bitboard::ull;
use super::board::DarkChessEnv;
use super::types::*;

// ==============================================================================
// --- 特征提取扩展块 (Neural Network Input) ---
// 全部维度由 self.config 决定，支持 4x8 与 4x2 两个变体。
// ==============================================================================

impl DarkChessEnv {
    /// 把棋盘特征张量直接写入传入的缓冲区，避免在调用链上重复分配 Vec。
    fn get_board_state_tensor_into(&self, tensor: &mut Vec<f32>) {
        let cfg = &self.config;
        tensor.clear();
        tensor.reserve(cfg.board_channels * cfg.total_positions);
        let my = self.get_current_player();
        let opp = my.opposite();

        let mut push_bitboard = |bb: u64| {
            for sq in 0..cfg.total_positions {
                tensor.push(if (bb & ull(sq)) != 0 { 1.0 } else { 0.0 });
            }
        };

        let piece_bbs = self.get_piece_bitboards();
        // 己方激活棋子类型
        for &pt in cfg.active_types.iter().take(cfg.num_active) {
            push_bitboard(piece_bbs[my.idx()][pt]);
        }
        // 敌方激活棋子类型
        for &pt in cfg.active_types.iter().take(cfg.num_active) {
            push_bitboard(piece_bbs[opp.idx()][pt]);
        }
        push_bitboard(self.get_hidden_bitboard());
        push_bitboard(self.get_empty_bitboard());
    }

    fn get_board_state_tensor(&self) -> Vec<f32> {
        let mut tensor = Vec::with_capacity(self.config.board_channels * self.config.total_positions);
        self.get_board_state_tensor_into(&mut tensor);
        tensor
    }

    fn get_scalar_state_vector(&self) -> Vec<f32> {
        let mut vec = Vec::with_capacity(self.config.scalar_feature_count);
        self.get_scalar_state_vector_into(&mut vec);
        vec
    }

    fn get_scalar_state_vector_into(&self, vec: &mut Vec<f32>) {
        let cfg = &self.config;
        vec.clear();
        vec.reserve(cfg.scalar_feature_count);

        let my = self.get_current_player();
        let opp = my.opposite();

        vec.push(self.get_move_counter() as f32 / cfg.max_consecutive_moves_for_draw as f32);
        vec.push(self.get_hp(my) as f32 / cfg.initial_health as f32);
        vec.push(self.get_hp(opp) as f32 / cfg.initial_health as f32);

        let dead_counts = self.get_dead_piece_counts_by_type();
        for &player in &[my, opp] {
            for &pt in cfg.active_types.iter().take(cfg.num_active) {
                // 存活数 = 该类总数 - 该类阵亡数
                let dead = dead_counts[player.idx()][pt] as usize;
                let count = cfg.piece_counts[pt].saturating_sub(dead);
                vec.extend(std::iter::repeat(1.0).take(count));
                vec.extend(std::iter::repeat(0.0).take(cfg.piece_counts[pt] - count));
            }
        }
    }

    pub fn get_state(&self) -> Observation {
        let board_data = self.get_board_state_tensor();
        let board = Array3::from_shape_vec(
            (self.config.board_channels, self.config.rows, self.config.cols),
            board_data,
        )
        .expect("Failed to reshape board array");

        let scalars_data = self.get_scalar_state_vector();
        let scalars = Array1::from_vec(scalars_data);

        Observation { board, scalars }
    }

    /// 把特征写入外部缓冲区（避免每次分配临时 Vec）。
    pub fn get_state_into(
        &self,
        board_data: &mut Vec<f32>,
        scalars_data: &mut Vec<f32>,
    ) -> Observation {
        self.get_board_state_tensor_into(board_data);
        let board = Array3::from_shape_vec(
            (self.config.board_channels, self.config.rows, self.config.cols),
            board_data.clone(),
        )
        .expect("Failed to reshape board array");

        self.get_scalar_state_vector_into(scalars_data);
        let scalars = Array1::from_vec(scalars_data.clone());

        Observation { board, scalars }
    }

    /// 仅将扁平特征写入外部缓冲区，不创建 Observation。
    pub fn get_state_flat_into(&self, board_data: &mut Vec<f32>, scalars_data: &mut Vec<f32>) {
        self.get_board_state_tensor_into(board_data);
        self.get_scalar_state_vector_into(scalars_data);
    }
}
