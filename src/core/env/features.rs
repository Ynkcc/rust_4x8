use ndarray::{Array1, Array3};

use super::bitboard::ull;
use super::board::DarkChessEnv;
use super::config::GameConfig;
use super::types::*;

// ==============================================================================
// --- 状态投影源 StateView（单次遍历，双架构共用）---
//
// ResNet 与 NNUE 的全部特征都从这一份快照派生：
// - env 内部布局只在此处（state_view）被读取一次；
// - 将来 env 状态重构（如真布局/信念分离）只需调整 state_view。
// ==============================================================================

/// 架构无关的只读状态快照（红黑绝对视角，无 my/opp 视角化）。
pub(crate) struct StateView {
    /// 明子位棋盘 [PlayerIdx][PieceType]
    pub piece_bbs: [[u64; super::config::NUM_PIECE_TYPES_MAX]; 2],
    /// 暗子位棋盘
    pub hidden_bb: u64,
    /// 空位位棋盘
    pub empty_bb: u64,
    /// 暗子包按型计数（红黑合并；暗子归属对双方均不可见）
    pub hidden_counts: [u8; super::config::NUM_PIECE_TYPES_MAX],
    /// 存活子力计数 [PlayerIdx][PieceType]
    pub alive_counts: [[u8; super::config::NUM_PIECE_TYPES_MAX]; 2],
    /// 血量 [Red, Black]
    pub hp: [i32; 2],
    pub current_player: Player,
    pub move_counter: usize,
}

impl StateView {
    /// 当前玩家视角的存活计数。
    #[inline]
    pub fn alive_counts_view(&self, my: Player) -> (&[u8; super::config::NUM_PIECE_TYPES_MAX], &[u8; super::config::NUM_PIECE_TYPES_MAX]) {
        (&self.alive_counts[my.idx()], &self.alive_counts[my.opposite().idx()])
    }
}

impl DarkChessEnv {
    /// 一次遍历内部状态，产出双架构共用的投影快照。
    pub(crate) fn state_view(&self) -> StateView {
        let cfg = &self.config;
        let mut hidden_counts = [0u8; super::config::NUM_PIECE_TYPES_MAX];
        for piece in self.get_hidden_pieces_raw() {
            hidden_counts[piece.piece_type as usize] += 1;
        }
        let mut alive_counts = [[0u8; super::config::NUM_PIECE_TYPES_MAX]; 2];
        let dead_counts = self.get_dead_piece_counts_by_type();
        for player in 0..2 {
            for pt in 0..super::config::NUM_PIECE_TYPES_MAX {
                alive_counts[player][pt] =
                    (cfg.piece_counts[pt].saturating_sub(dead_counts[player][pt] as usize)) as u8;
            }
        }
        StateView {
            piece_bbs: *self.get_piece_bitboards(),
            hidden_bb: self.get_hidden_bitboard(),
            empty_bb: self.get_empty_bitboard(),
            hidden_counts,
            alive_counts,
            hp: [self.get_hp(Player::Red), self.get_hp(Player::Black)],
            current_player: self.get_current_player(),
            move_counter: self.get_move_counter(),
        }
    }
}

// ==============================================================================
// --- ResNet / CNN 稠密特征 (ResNetObservation) ---
// 全部维度由 self.config 决定，支持 4x8 与 4x2 等变体。
// ==============================================================================

impl DarkChessEnv {
    /// 把棋盘特征张量直接写入传入的缓冲区，避免在调用链上重复分配 Vec。
    fn resnet_board_tensor_into(view: &StateView, cfg: &GameConfig, tensor: &mut Vec<f32>) {
        tensor.clear();
        tensor.reserve(cfg.resnet_board_channels * cfg.total_positions);
        let my = view.current_player;
        let opp = my.opposite();

        let mut push_bitboard = |bb: u64| {
            for sq in 0..cfg.total_positions {
                tensor.push(if (bb & ull(sq)) != 0 { 1.0 } else { 0.0 });
            }
        };

        // 己方激活棋子类型
        for &pt in cfg.active_types.iter().take(cfg.num_active) {
            push_bitboard(view.piece_bbs[my.idx()][pt]);
        }
        // 敌方激活棋子类型
        for &pt in cfg.active_types.iter().take(cfg.num_active) {
            push_bitboard(view.piece_bbs[opp.idx()][pt]);
        }
        push_bitboard(view.hidden_bb);
        push_bitboard(view.empty_bb);
    }

    fn resnet_scalar_vector_into(view: &StateView, cfg: &GameConfig, vec: &mut Vec<f32>) {
        vec.clear();
        vec.reserve(cfg.resnet_scalar_feature_count);

        let my = view.current_player;
        let opp = my.opposite();

        vec.push(view.move_counter as f32 / cfg.max_consecutive_moves_for_draw as f32);
        vec.push(view.hp[my.idx()] as f32 / cfg.initial_health as f32);
        vec.push(view.hp[opp.idx()] as f32 / cfg.initial_health as f32);

        let (mine, theirs) = view.alive_counts_view(my);
        for counts in [mine, theirs] {
            for &pt in cfg.active_types.iter().take(cfg.num_active) {
                let count = counts[pt] as usize;
                vec.extend(std::iter::repeat(1.0).take(count));
                vec.extend(std::iter::repeat(0.0).take(cfg.piece_counts[pt] - count));
            }
        }
    }

    /// 当前玩家视角的 ResNet 观测（稠密棋盘张量 + 标量向量）。
    pub fn get_resnet_state(&self) -> ResNetObservation {
        let cfg = &self.config;
        let view = self.state_view();
        let mut board_data = Vec::with_capacity(cfg.resnet_board_channels * cfg.total_positions);
        Self::resnet_board_tensor_into(&view, cfg, &mut board_data);
        let board = Array3::from_shape_vec(
            (cfg.resnet_board_channels, cfg.rows, cfg.cols),
            board_data,
        )
        .expect("Failed to reshape board array");

        let mut scalars_data = Vec::with_capacity(cfg.resnet_scalar_feature_count);
        Self::resnet_scalar_vector_into(&view, cfg, &mut scalars_data);
        let scalars = Array1::from_vec(scalars_data);

        ResNetObservation { board, scalars }
    }

    /// 仅将 ResNet 扁平特征写入外部缓冲区，不创建 ResNetObservation。
    pub fn resnet_features_flat_into(&self, board_data: &mut Vec<f32>, scalars_data: &mut Vec<f32>) {
        let cfg = &self.config;
        let view = self.state_view();
        Self::resnet_board_tensor_into(&view, cfg, board_data);
        Self::resnet_scalar_vector_into(&view, cfg, scalars_data);
    }
}

// ==============================================================================
// --- NNUE 稀疏特征 (active feature indices) ---
//
// 布局完全由 config 推导（4x8 = 555）：
//   [格位段 total_positions * states_per_square] 状态按相对行棋方视角编码：
//     0=空 1=暗子 2..2+num_active=己方明子 2+num_active..=对方明子（无红黑身份）
//   [暗子包段 num_active * bag_stride]（暗子归属不可见，无身份）
//   [标量段 1]：无吃子标记（总步数不进观测）
// ==============================================================================

impl DarkChessEnv {
    /// 计算指定格子槽位在特定玩家视角下的 NNUE 特征索引。
    #[inline]
    pub fn nnue_slot_feature_index(cfg: &GameConfig, perspective: Player, sq: usize, slot: Slot) -> usize {
        let states = cfg.nnue_states_per_square();
        match slot {
            Slot::Empty => sq * states,
            Slot::Hidden => sq * states + 1,
            Slot::Revealed(piece) => {
                let compact = cfg.compact_index(piece.piece_type as usize);
                let base_offset = if piece.player == perspective {
                    2usize
                } else {
                    2 + cfg.num_active
                };
                sq * states + base_offset + compact
            }
        }
    }

    /// 把当前行棋方视角的 NNUE 活性稀疏特征索引写入 `out`（追加，不清空）。
    pub fn nnue_active_features_into(&self, out: &mut Vec<usize>) {
        self.nnue_active_features_for_player_into(self.get_current_player(), out);
    }

    /// 把指定玩家视角的 NNUE 活性稀疏特征索引写入 `out`（追加，不清空）。
    pub fn nnue_active_features_for_player_into(&self, perspective: Player, out: &mut Vec<usize>) {
        let cfg = &self.config;
        let view = self.state_view();
        let my = perspective;
        let opp = my.opposite();

        // --- 格位段：从位棋盘逐位直取（empty/hidden/revealed 互斥完备，
        //     O(明子数) 替代逐格扫描） ---
        let states = cfg.nnue_states_per_square();
        let mut bb = view.empty_bb;
        while bb != 0 {
            let sq = bb.trailing_zeros() as usize;
            out.push(sq * states); // 空位
            bb &= bb - 1;
        }
        let mut bb = view.hidden_bb;
        while bb != 0 {
            let sq = bb.trailing_zeros() as usize;
            out.push(sq * states + 1);
            bb &= bb - 1;
        }
        for (player, base_offset) in [(my, 2usize), (opp, 2 + cfg.num_active)] {
            for &pt in cfg.active_types.iter().take(cfg.num_active) {
                let compact = cfg.compact_index(pt);
                let mut pb = view.piece_bbs[player.idx()][pt];
                while pb != 0 {
                    let sq = pb.trailing_zeros() as usize;
                    out.push(sq * states + base_offset + compact);
                    pb &= pb - 1;
                }
            }
        }

        // --- 暗子包段：每型计数桶 ---
        let bag_base = cfg.total_positions * states;
        let stride = cfg.nnue_bag_stride();
        for &pt in cfg.active_types.iter().take(cfg.num_active) {
            let compact = cfg.compact_index(pt);
            let count = (view.hidden_counts[pt] as usize).min(stride - 1);
            out.push(bag_base + compact * stride + count);
        }

        // --- 标量段：无吃子标记（行棋方已由格位段视角内建） ---
        let scalar_base = bag_base + cfg.num_active * stride;
        if view.move_counter.min(8) > 0 {
            out.push(scalar_base);
        }
    }

    /// NNUE 活性稀疏特征索引列表。
    pub fn nnue_active_features(&self) -> Vec<usize> {
        let mut out = Vec::with_capacity(self.config.total_positions + 16);
        self.nnue_active_features_into(&mut out);
        out
    }
}
