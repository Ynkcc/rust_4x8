// src/game_env/board/accessors.rs
// 公共访问器与供其他模块调用的内部辅助。

use super::*;

impl DarkChessEnv {
    pub fn get_board_slots(&self) -> &[Slot] {
        &self.board[0..self.config.total_positions]
    }

    pub fn get_current_player(&self) -> Player {
        self.current_player
    }

    /// 切换当前玩家（视角反转验证专用）。
    pub fn flip_player(&mut self) {
        self.current_player = self.current_player.opposite();
    }

    pub fn get_move_counter(&self) -> usize {
        self.move_counter
    }

    pub fn get_total_steps(&self) -> usize {
        self.total_step_counter
    }

    pub fn get_score(&self, player: Player) -> i32 {
        self.scores[player.idx()]
    }

    pub fn get_scores(&self) -> (i32, i32) {
        (self.get_score(Player::Red), self.get_score(Player::Black))
    }

    pub fn get_hp(&self, player: Player) -> i32 {
        self.get_score(player)
    }

    /// 返回死亡棋子的切片视图。
    pub fn get_dead_pieces(&self, player: Player) -> &[PieceType] {
        let count = self.dead_pieces_count[player.idx()];
        &self.dead_pieces_pool[player.idx()][0..count]
    }

    /// 返回隐藏棋子中属于指定玩家的类型列表。
    pub fn get_hidden_pieces(&self, player: Player) -> Vec<PieceType> {
        self.hidden_pieces_pool[0..self.hidden_pieces_count]
            .iter()
            .filter(|p| p.player == player)
            .map(|p| p.piece_type)
            .collect()
    }

    pub fn get_hidden_pieces_raw(&self) -> &[Piece] {
        &self.hidden_pieces_pool[0..self.hidden_pieces_count]
    }

    pub fn get_action_for_coords(&self, coords: &[usize]) -> Option<usize> {
        action_lookup_tables(&self.config)
            .coords_to_action
            .get(&pack_coords(coords))
            .copied()
    }

    pub fn get_bitboards(&self) -> std::collections::HashMap<String, Vec<bool>> {
        let cfg = &self.config;
        let mut bitboards = std::collections::HashMap::new();

        let bb_to_vec =
            |bb: u64| -> Vec<bool> { (0..cfg.total_positions).map(|sq| (bb & ull(sq)) != 0).collect() };

        bitboards.insert("hidden".to_string(), bb_to_vec(self.hidden_bitboard));
        bitboards.insert("empty".to_string(), bb_to_vec(self.empty_bitboard));

        const PIECE_NAMES: [&str; NUM_PIECE_TYPES_MAX] = [
            "soldier", "cannon", "horse", "chariot", "elephant", "advisor", "general",
        ];

        for &player in &[Player::Red, Player::Black] {
            let prefix = match player {
                Player::Red => "red",
                Player::Black => "black",
            };

            bitboards.insert(
                format!("{}_revealed", prefix),
                bb_to_vec(self.revealed_bitboards[player.idx()]),
            );

            for &pt in cfg.active_types.iter().take(cfg.num_active) {
                bitboards.insert(
                    format!("{}_{}", prefix, PIECE_NAMES[pt]),
                    bb_to_vec(self.piece_bitboards[player.idx()][pt]),
                );
            }
        }

        bitboards
    }

    // === 内部辅助方法 (供其他模块调用) ===

    pub(crate) fn get_piece_bitboards(&self) -> &[[u64; NUM_PIECE_TYPES_MAX]; 2] {
        &self.piece_bitboards
    }

    pub(crate) fn get_dead_piece_counts_by_type(&self) -> &[[u8; NUM_PIECE_TYPES_MAX]; 2] {
        &self.dead_piece_counts_by_type
    }

    pub(crate) fn get_revealed_bitboards(&self) -> &[u64; 2] {
        &self.revealed_bitboards
    }

    pub(crate) fn get_hidden_bitboard(&self) -> u64 {
        self.hidden_bitboard
    }

    pub(crate) fn get_empty_bitboard(&self) -> u64 {
        self.empty_bitboard
    }
}
