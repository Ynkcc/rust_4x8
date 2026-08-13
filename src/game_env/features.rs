use ndarray::{Array1, Array3};

use super::bitboard::ull;
use super::board::DarkChessEnv;
use super::constants::*;
use super::types::*;

// ==============================================================================
// --- 特征提取扩展块 (Neural Network Input) ---
// ==============================================================================

impl DarkChessEnv {
    /// 把棋盘特征张量直接写入传入的缓冲区，避免在调用链上重复分配 Vec。
    fn get_board_state_tensor_into(&self, tensor: &mut Vec<f32>) {
        tensor.clear();
        tensor.reserve(BOARD_CHANNELS * TOTAL_POSITIONS);
        let my = self.get_current_player();
        let opp = my.opposite();

        let mut push_bitboard = |bb: u64| {
            for sq in 0..TOTAL_POSITIONS {
                tensor.push(if (bb & ull(sq)) != 0 { 1.0 } else { 0.0 });
            }
        };

        let piece_bbs = self.get_piece_bitboards();
        for pt in 0..NUM_PIECE_TYPES {
            push_bitboard(piece_bbs[my.idx()][pt]);
        }
        for pt in 0..NUM_PIECE_TYPES {
            push_bitboard(piece_bbs[opp.idx()][pt]);
        }
        push_bitboard(self.get_hidden_bitboard());
        push_bitboard(self.get_empty_bitboard());
    }

    fn get_board_state_tensor(&self) -> Vec<f32> {
        let mut tensor = Vec::with_capacity(BOARD_CHANNELS * TOTAL_POSITIONS);
        self.get_board_state_tensor_into(&mut tensor);
        tensor
    }

    fn get_scalar_state_vector(&self) -> Vec<f32> {
        let mut vec = Vec::with_capacity(SCALAR_FEATURE_COUNT);
        self.get_scalar_state_vector_into(&mut vec);
        vec
    }

    fn get_scalar_state_vector_into(&self, vec: &mut Vec<f32>) {
        vec.clear();
        vec.reserve(SCALAR_FEATURE_COUNT);

        let my = self.get_current_player();
        let opp = my.opposite();

        vec.push(self.get_move_counter() as f32 / MAX_CONSECUTIVE_MOVES_FOR_DRAW as f32);
        vec.push(self.get_hp(my) as f32 / INITIAL_HEALTH_POINTS as f32);
        vec.push(self.get_hp(opp) as f32 / INITIAL_HEALTH_POINTS as f32);

        for &player in &[my, opp] {
            let bitboards = &self.get_piece_bitboards()[player.idx()];
            for pt in 0..NUM_PIECE_TYPES {
                let count = bitboards[pt].count_ones() as usize;
                let max_count = PIECE_MAX_COUNTS[pt];
                vec.extend(std::iter::repeat(1.0).take(count));
                vec.extend(std::iter::repeat(0.0).take(max_count - count));
            }
        }
    }

    pub fn get_state(&self) -> Observation {
        let board_data = self.get_board_state_tensor();
        let board = Array3::from_shape_vec((BOARD_CHANNELS, BOARD_ROWS, BOARD_COLS), board_data)
            .expect("Failed to reshape board array");

        let scalars_data = self.get_scalar_state_vector();
        let scalars = Array1::from_vec(scalars_data);

        Observation { board, scalars }
    }

    /// 把特征写入外部缓冲区（避免每次分配临时 Vec）。
    ///
    /// 注意：`Array3`/`Array1` 需要持有自己的底层 Vec，因此这里仍需 clone 一次数据；
    /// 但缓冲区复用后可以避免在内部再分配临时 Vec，减少总分配次数。
    pub fn get_state_into(
        &self,
        board_data: &mut Vec<f32>,
        scalars_data: &mut Vec<f32>,
    ) -> Observation {
        self.get_board_state_tensor_into(board_data);
        let board = Array3::from_shape_vec(
            (BOARD_CHANNELS, BOARD_ROWS, BOARD_COLS),
            board_data.clone(),
        )
        .expect("Failed to reshape board array");

        self.get_scalar_state_vector_into(scalars_data);
        let scalars = Array1::from_vec(scalars_data.clone());

        Observation { board, scalars }
    }
}
