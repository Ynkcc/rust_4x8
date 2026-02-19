use rand::seq::SliceRandom;
use rand::thread_rng;
use rand::Rng;

use super::actions::action_lookup_tables;
use super::bitboard::{ray_attacks, ull, BOARD_MASK};
use super::constants::*;
use super::types::*;

// ==============================================================================
// --- 环境结构体 (DarkChessEnv) ---
// ==============================================================================

/// 支持 Copy 的暗棋环境
/// 所有 Vec 已替换为定长数组 + 计数器
#[derive(Clone, Copy, Debug)]
pub struct DarkChessEnv {
    // --- 游戏核心状态 ---
    /// 棋盘格子状态
    board: [Slot; TOTAL_POSITIONS],
    /// 当前玩家
    current_player: Player,
    /// 连续无吃子步数
    move_counter: usize,
    /// 游戏总步数
    total_step_counter: usize,

    // --- 位棋盘 (Bitboards) ---
    piece_bitboards: [[u64; NUM_PIECE_TYPES]; 2],
    revealed_bitboards: [u64; 2],
    hidden_bitboard: u64,
    empty_bitboard: u64,

    // --- 游戏统计与记录 (Copy Refactor) ---
    /// 阵亡棋子池 [PlayerIdx][Idx]
    dead_pieces_pool: [[PieceType; TOTAL_PIECES_PER_PLAYER]; 2],
    /// 阵亡棋子计数 [PlayerIdx]
    dead_pieces_count: [usize; 2],
    
    /// 玩家分数/血量
    scores: [i32; 2],
    /// 上一步动作
    last_action: i32,

    // --- 概率相关 (Bag Model - Copy Refactor) ---
    /// 隐藏棋子池: 使用定长数组代替 Vec
    hidden_pieces_pool: [Piece; TOTAL_POSITIONS],
    /// 当前隐藏棋子数量
    hidden_pieces_count: usize,
    
    /// 翻棋概率表
    reveal_probabilities: [f32; REVEAL_PROBABILITY_SIZE],
}

impl DarkChessEnv {
    pub fn new() -> Self {
        let mut env = Self {
            board: [Slot::Empty; TOTAL_POSITIONS],
            current_player: Player::Red,
            move_counter: 0,
            total_step_counter: 0,

            piece_bitboards: [[0; NUM_PIECE_TYPES]; 2],
            revealed_bitboards: [0; 2],
            hidden_bitboard: 0,
            empty_bitboard: 0,

            // 初始化阵亡列表
            dead_pieces_pool: [[PieceType::default(); TOTAL_PIECES_PER_PLAYER]; 2],
            dead_pieces_count: [0; 2],
            
            scores: [0; 2],
            last_action: -1,

            // 初始化隐藏池
            hidden_pieces_pool: [Piece::default(); TOTAL_POSITIONS],
            hidden_pieces_count: 0,
            
            reveal_probabilities: [0.0; REVEAL_PROBABILITY_SIZE],
        };

        action_lookup_tables();
        ray_attacks();
        
        env.reset();
        env
    }

    pub fn get_coords_for_action(&self, action: usize) -> Option<&Vec<usize>> {
        action_lookup_tables().action_to_coords.get(action)
    }

    fn reset_internal_state(&mut self) {
        self.board = [Slot::Empty; TOTAL_POSITIONS];

        self.piece_bitboards = [[0; NUM_PIECE_TYPES]; 2];
        self.revealed_bitboards = [0; 2];

        self.hidden_bitboard = 0;
        self.empty_bitboard = 0;

        // 重置阵亡计数，无需清空 pool 内容，依靠 count 即可
        self.dead_pieces_count = [0; 2];

        self.scores = [INITIAL_HEALTH_POINTS, INITIAL_HEALTH_POINTS];

        self.current_player = Player::Red;
        self.move_counter = 0;
        self.total_step_counter = 0;
        self.last_action = -1;

        self.hidden_pieces_count = 0;
        self.reveal_probabilities = [0.0; REVEAL_PROBABILITY_SIZE];
    }

    /// 初始化棋盘布局 (Shuffle Bag Model)
    fn initialize_board(&mut self) {
        let mut rng = thread_rng();

        // 1. 生成实际棋子池 (写入 Buffer)
        let mut idx = 0;
        for &player in &[Player::Red, Player::Black] {
            for _ in 0..GENERALS_COUNT { self.hidden_pieces_pool[idx] = Piece::new(PieceType::General, player); idx += 1; }
            for _ in 0..ADVISORS_COUNT { self.hidden_pieces_pool[idx] = Piece::new(PieceType::Advisor, player); idx += 1; }
            for _ in 0..ELEPHANTS_COUNT { self.hidden_pieces_pool[idx] = Piece::new(PieceType::Elephant, player); idx += 1; }
            for _ in 0..CHARIOTS_COUNT { self.hidden_pieces_pool[idx] = Piece::new(PieceType::Chariot, player); idx += 1; }
            for _ in 0..HORSES_COUNT { self.hidden_pieces_pool[idx] = Piece::new(PieceType::Horse, player); idx += 1; }
            for _ in 0..CANNONS_COUNT { self.hidden_pieces_pool[idx] = Piece::new(PieceType::Cannon, player); idx += 1; }
            for _ in 0..SOLDIERS_COUNT { self.hidden_pieces_pool[idx] = Piece::new(PieceType::Soldier, player); idx += 1; }
        }
        self.hidden_pieces_count = idx;

        // 打乱 slice
        self.hidden_pieces_pool[0..self.hidden_pieces_count].shuffle(&mut rng);

        // 2. 填充棋盘
        self.empty_bitboard = 0; 
        self.hidden_bitboard = BOARD_MASK; 

        for sq in 0..TOTAL_POSITIONS {
            self.board[sq] = Slot::Hidden;
        }

        self.update_reveal_probabilities();

        // 3. 随机翻开 N 个 Hidden 位置
        if TOTAL_POSITIONS > 0 {
            let mut hidden_indices: Vec<usize> = (0..TOTAL_POSITIONS).collect();
            hidden_indices.shuffle(&mut rng);
            let reveal_count = std::cmp::min(hidden_indices.len(), INITIAL_REVEALED_PIECES);

            for &idx in hidden_indices.iter().take(reveal_count) {
                self.reveal_piece_at(idx, None);
            }
        }
    }

    pub fn reset(&mut self) -> Observation {
        self.reset_internal_state();
        self.initialize_board();
        self.get_state()
    }

    // --- 翻子逻辑 ---

    /// 翻开指定位置的棋子并更新 Bitboards
    fn reveal_piece_at(&mut self, sq: usize, specified_piece: Option<Piece>) {
        if !matches!(self.board[sq], Slot::Hidden) {
            panic!("尝试翻开非 Hidden 位置: {}", sq);
        }

        if self.hidden_pieces_count == 0 {
            panic!("逻辑错误：棋盘上有 Hidden 位置，但 hidden_pieces 池已空");
        }

        // 获取 slice 视图
        let active_slice = &self.hidden_pieces_pool[0..self.hidden_pieces_count];

        let idx = if let Some(target) = specified_piece {
            active_slice
                .iter()
                .position(|p| *p == target)
                .expect("指定的棋子不在隐藏棋子池中")
        } else {
            let mut rng = thread_rng();
            rng.gen_range(0..self.hidden_pieces_count)
        };

        // Swap Remove 逻辑 (Copy version)
        let last_idx = self.hidden_pieces_count - 1;
        self.hidden_pieces_pool.swap(idx, last_idx); // 将选中的棋子交换到末尾
        let piece = self.hidden_pieces_pool[last_idx]; // 取出
        self.hidden_pieces_count -= 1; // 缩小有效范围

        // 更新 Bitboards
        let mask = ull(sq);
        self.hidden_bitboard &= !mask;

        let p_bb = &mut self.revealed_bitboards[piece.player.idx()];
        *p_bb |= mask;

        let pt_bb =
            &mut self.piece_bitboards[piece.player.idx()][piece.piece_type as usize];
        *pt_bb |= mask;

        self.board[sq] = Slot::Revealed(piece);
        self.update_reveal_probabilities();
    }

    fn update_reveal_probabilities(&mut self) {
        let total_hidden = self.hidden_pieces_count;

        if total_hidden == 0 {
            self.reveal_probabilities = [0.0; REVEAL_PROBABILITY_SIZE];
            return;
        }

        let mut counts = vec![0; REVEAL_PROBABILITY_SIZE];
        for i in 0..total_hidden {
            let piece = self.hidden_pieces_pool[i];
            let idx = match (piece.player, piece.piece_type) {
                (Player::Red, PieceType::Soldier) => 0,
                (Player::Red, PieceType::Cannon) => 1,
                (Player::Red, PieceType::Horse) => 2,
                (Player::Red, PieceType::Chariot) => 3,
                (Player::Red, PieceType::Elephant) => 4,
                (Player::Red, PieceType::Advisor) => 5,
                (Player::Red, PieceType::General) => 6,
                (Player::Black, PieceType::Soldier) => 7,
                (Player::Black, PieceType::Cannon) => 8,
                (Player::Black, PieceType::Horse) => 9,
                (Player::Black, PieceType::Chariot) => 10,
                (Player::Black, PieceType::Elephant) => 11,
                (Player::Black, PieceType::Advisor) => 12,
                (Player::Black, PieceType::General) => 13,
            };
            counts[idx] += 1;
        }

        for i in 0..REVEAL_PROBABILITY_SIZE {
            self.reveal_probabilities[i] = counts[i] as f32 / total_hidden as f32;
        }
    }

    pub fn get_reveal_probabilities(&self) -> &[f32] {
        &self.reveal_probabilities
    }

    // --- 核心 Step 逻辑 ---

    pub fn step(
        &mut self,
        action: usize,
        reveal_piece: Option<Piece>,
    ) -> Result<(Observation, f32, bool, bool, Option<i32>), String> {
        let masks = self.action_masks();
        if masks[action] == 0 {
            return Err(format!("无效动作: {}", action));
        }

        self.last_action = action as i32;
        self.total_step_counter += 1;

        let lookup = action_lookup_tables();

        if action < REVEAL_ACTIONS_COUNT {
            let sq = lookup.action_to_coords[action][0];
            self.reveal_piece_at(sq, reveal_piece);
            self.move_counter = 0;
        } else {
            let coords = &lookup.action_to_coords[action];
            let from_sq = coords[0];
            let to_sq = coords[1];
            self.apply_move_action(from_sq, to_sq, reveal_piece);
        }

        self.current_player = self.current_player.opposite();
        let (terminated, truncated, winner) = self.check_game_over_conditions();
        Ok((self.get_state(), 0.0, terminated, truncated, winner))
    }

    fn apply_move_action(&mut self, from_sq: usize, to_sq: usize, reveal_piece: Option<Piece>) {
        let attacker = match std::mem::replace(&mut self.board[from_sq], Slot::Empty) {
            Slot::Revealed(p) => p,
            _ => panic!("Move action source is not a revealed piece!"),
        };

        if matches!(self.board[to_sq], Slot::Hidden) {
            self.reveal_piece_at(to_sq, reveal_piece);
        }

        let target_slot =
            std::mem::replace(&mut self.board[to_sq], Slot::Revealed(attacker.clone()));

        let attacker_mask = ull(from_sq);
        let defender_mask = ull(to_sq);
        let p = attacker.player;
        let pt = attacker.piece_type as usize;

        let my_revealed_bb = &mut self.revealed_bitboards[p.idx()];
        *my_revealed_bb &= !attacker_mask;

        let my_pt_bb = &mut self.piece_bitboards[p.idx()][pt];
        *my_pt_bb &= !attacker_mask;

        self.empty_bitboard |= attacker_mask;

        *my_revealed_bb |= defender_mask;
        *my_pt_bb |= defender_mask;
        self.empty_bitboard &= !defender_mask;

        match target_slot {
            Slot::Empty => {
                self.move_counter += 1;
            }
            Slot::Revealed(defender) => {
                let opp = defender.player;
                let opp_pt = defender.piece_type as usize;

                let opp_revealed_bb = &mut self.revealed_bitboards[opp.idx()];
                *opp_revealed_bb &= !defender_mask;

                let opp_pt_bb = &mut self.piece_bitboards[opp.idx()][opp_pt];
                *opp_pt_bb &= !defender_mask;

                // 记录被吃子 (使用 Array + Count 模拟 push)
                let opp_idx = defender.player.idx();
                let dead_idx = self.dead_pieces_count[opp_idx];
                if dead_idx < TOTAL_PIECES_PER_PLAYER {
                    self.dead_pieces_pool[opp_idx][dead_idx] = defender.piece_type;
                    self.dead_pieces_count[opp_idx] += 1;
                } else {
                    // 理论上不可能发生，除非逻辑错误
                    panic!("Dead pieces buffer overflow!");
                }

                let score = &mut self.scores[defender.player.idx()];
                *score = score.saturating_sub(defender.piece_type.value());

                self.move_counter = 0;
            }
            Slot::Hidden => {
                panic!("Unexpected Hidden slot after reveal");
            }
        }
    }

    pub fn get_target_slot(&self, action: usize) -> Slot {
        let coords = &action_lookup_tables().action_to_coords[action];
        
        if action < REVEAL_ACTIONS_COUNT {
            let sq = coords[0];
            self.board[sq].clone()
        } else {
            let to_sq = coords[1];
            self.board[to_sq].clone()
        }
    }

    pub fn print_board(&self) {
        println!("\n      0         1         2         3");
        println!("   +---------+---------+---------+---------+");
        for r in 0..BOARD_ROWS {
            print!(" {} |", (b'A' + r as u8) as char);
            for c in 0..BOARD_COLS {
                let idx = r * BOARD_COLS + c;
                match &self.board[idx] {
                    Slot::Empty => print!("   .     |"),
                    Slot::Hidden => print!("    ?    |"),
                    Slot::Revealed(p) => print!(" {:^7} |", p.short_name()),
                }
            }
            println!("\n   +---------+---------+---------+---------+");
        }
        println!("当前玩家: {}", self.current_player);
        println!(
            "Total Steps: {}, Move Counter: {}",
            self.total_step_counter, self.move_counter
        );
        println!("Dead (Red): {:?}", self.get_dead_pieces(Player::Red));
        println!("Dead (Black): {:?}", self.get_dead_pieces(Player::Black));
        println!("---------------------------------------------");
    }

    // === 公共访问器方法 ===

    pub fn get_board_slots(&self) -> &[Slot] {
        &self.board
    }

    pub fn get_current_player(&self) -> Player {
        self.current_player
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

    /// 返回死亡棋子的切片视图（替代 Vec 返回）
    pub fn get_dead_pieces(&self, player: Player) -> &[PieceType] {
        let count = self.dead_pieces_count[player.idx()];
        &self.dead_pieces_pool[player.idx()][0..count]
    }

    /// 返回隐藏棋子的 Vec（此处需要分配内存来收集，或返回迭代器）
    /// 为了保持兼容性返回 Vec
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
        action_lookup_tables().coords_to_action.get(coords).copied()
    }

    pub fn get_bitboards(&self) -> std::collections::HashMap<String, Vec<bool>> {
        let mut bitboards = std::collections::HashMap::new();

        let bb_to_vec =
            |bb: u64| -> Vec<bool> { (0..TOTAL_POSITIONS).map(|sq| (bb & ull(sq)) != 0).collect() };

        bitboards.insert("hidden".to_string(), bb_to_vec(self.hidden_bitboard));
        bitboards.insert("empty".to_string(), bb_to_vec(self.empty_bitboard));

        const PIECE_NAMES: [&str; NUM_PIECE_TYPES] = [
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

            for (pt, &name) in PIECE_NAMES.iter().enumerate() {
                bitboards.insert(
                    format!("{}_{}", prefix, name),
                    bb_to_vec(self.piece_bitboards[player.idx()][pt]),
                );
            }
        }

        bitboards
    }

    // === 内部辅助方法 (供其他模块调用) ===

    /// 获取内部 bitboards (供 rules.rs 使用)
    pub(super) fn get_piece_bitboards(&self) -> &[[u64; NUM_PIECE_TYPES]; 2] {
        &self.piece_bitboards
    }

    pub(super) fn get_revealed_bitboards(&self) -> &[u64; 2] {
        &self.revealed_bitboards
    }

    pub(super) fn get_hidden_bitboard(&self) -> u64 {
        self.hidden_bitboard
    }

    pub(super) fn get_empty_bitboard(&self) -> u64 {
        self.empty_bitboard
    }
}
