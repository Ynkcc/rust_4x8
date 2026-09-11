// src/game_env/board/step.rs
// 核心 step 逻辑、走子应用、机会节点扩展、棋盘打印。

use super::*;
use crate::core::env::traits::get_outcome_id;

impl DarkChessEnv {
    /// 执行动作。观测不随步返回，按需调用 `get_resnet_state()` 获取。
    pub fn step(
        &mut self,
        action: usize,
        reveal_piece: Option<Piece>,
    ) -> Result<(f32, bool, bool, Option<i32>), String> {
        // 动作空间大小随 config 变化，使用 Vec（无法用编译期定长栈数组）
        let mut masks = vec![0i32; self.config.action_space_size];
        self.action_masks_into(&mut masks);
        if masks[action] == 0 {
            return Err(format!("无效动作: {}", action));
        }

        self.last_action = action as i32;
        self.total_step_counter += 1;

        let lookup = action_lookup_tables(&self.config);

        if action < self.config.reveal_actions_count {
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
        Ok((0.0, terminated, truncated, winner))
    }

    fn apply_move_action(&mut self, from_sq: usize, to_sq: usize, reveal_piece: Option<Piece>) {
        let attacker = match std::mem::replace(&mut self.board[from_sq], Slot::Empty) {
            Slot::Revealed(p) => p,
            _ => panic!("Move action source is not a revealed piece!"),
        };

        if matches!(self.board[to_sq], Slot::Hidden) {
            self.reveal_piece_at(to_sq, reveal_piece);
        }

        let attacker_mask = ull(from_sq);
        let defender_mask = ull(to_sq);
        let p = attacker.player;
        let pt = attacker.piece_type as usize;

        // --- 1. 清除攻击方在源格 from_sq 的 bitboard，源格标为空位 ---
        self.revealed_bitboards[p.idx()] &= !attacker_mask;
        self.piece_bitboards[p.idx()][pt] &= !attacker_mask;
        self.empty_bitboard |= attacker_mask;

        // --- 2. 目标格 to_sq：彻底清除所有既有归属，再写入攻击方 ---
        self.hidden_bitboard &= !defender_mask;
        self.empty_bitboard &= !defender_mask;
        for player_idx in 0..2 {
            self.revealed_bitboards[player_idx] &= !defender_mask;
            for t in 0..NUM_PIECE_TYPES_MAX {
                self.piece_bitboards[player_idx][t] &= !defender_mask;
            }
        }
        self.revealed_bitboards[p.idx()] |= defender_mask;
        self.piece_bitboards[p.idx()][pt] |= defender_mask;

        let target_slot = std::mem::replace(&mut self.board[to_sq], Slot::Revealed(attacker));

        match target_slot {
            Slot::Empty => {
                self.move_counter += 1;
            }
            Slot::Revealed(defender) => {
                // 吃子：移除被攻击棋子并扣其所属方血量。
                let victim_idx = defender.player.idx();
                let dead_idx = self.dead_pieces_count[victim_idx];
                if dead_idx < MAX_PIECES_PER_PLAYER {
                    self.dead_pieces_pool[victim_idx][dead_idx] = defender.piece_type;
                    self.dead_pieces_count[victim_idx] += 1;
                    self.dead_piece_counts_by_type[victim_idx][defender.piece_type as usize] += 1;
                } else {
                    panic!("Dead pieces buffer overflow!");
                }
                let score = &mut self.scores[defender.player.idx()];
                // 吃子扣血：分值为变体可配置（config.piece_values），不再用硬编码 value()。
                *score = score.saturating_sub(self.config.piece_values[defender.piece_type as usize]);
                self.move_counter = 0;
            }
            Slot::Hidden => {
                panic!("Unexpected Hidden slot after reveal");
            }
        }
    }

    pub fn get_target_slot(&self, action: usize) -> Slot {
        let coords = &action_lookup_tables(&self.config).action_to_coords[action];

        if action < self.config.reveal_actions_count {
            let sq = coords[0];
            self.board[sq]
        } else {
            let to_sq = coords[1];
            self.board[to_sq]
        }
    }

    // --- 机会节点扩展 ---

    /// 该动作是否会产生机会节点（翻棋动作或吃暗子动作，目标格为 Hidden 即随机翻出）。
    pub fn is_chance_action(&self, action: usize) -> bool {
        matches!(self.get_target_slot(action), Slot::Hidden)
    }

    /// 枚举机会动作的所有可能结果：`(outcome_id, 概率, 结果环境)`。
    pub fn chance_outcomes(&self, action: usize) -> Vec<(usize, f32, Self)> {
        let cfg = &self.config;
        let mut counts = vec![0usize; cfg.reveal_probability_size];
        for p in self.get_hidden_pieces_raw() {
            let id = cfg.outcome_id_for(p.piece_type, p.player == Player::Black);
            counts[id] += 1;
        }
        let total_hidden = self.get_hidden_pieces_raw().len() as f32;
        if total_hidden == 0.0 {
            return Vec::new();
        }
        let mut outcomes = Vec::new();
        for outcome_id in 0..cfg.reveal_probability_size {
            if counts[outcome_id] > 0 {
                let prob = counts[outcome_id] as f32 / total_hidden;
                let mut next_env = *self;
                let specific_piece = *self
                    .get_hidden_pieces_raw()
                    .iter()
                    .find(|p| cfg.outcome_id_for(p.piece_type, p.player == Player::Black) == outcome_id)
                    .expect("Piece not found");
                let _ = next_env.step(action, Some(specific_piece));
                outcomes.push((outcome_id, prob, next_env));
            }
        }
        outcomes
    }

    /// 执行动作后，若该动作产生了机会结果，返回其 `outcome_id`。
    pub fn step_outcome_id(&self, _action: usize) -> Option<usize> {
        if let Some(piece) = self.last_revealed_piece {
            return Some(get_outcome_id(&self.config, &piece));
        }
        None
    }

    pub fn print_board(&self) {
        let cfg = &self.config;
        println!("\n      {}", (0..cfg.cols).map(|c| format!("{:^9}", c)).collect::<Vec<_>>().join(""));
        println!("   +{}+", "---------+".repeat(cfg.cols));
        for r in 0..cfg.rows {
            print!(" {} |", (b'A' + r as u8) as char);
            for c in 0..cfg.cols {
                let idx = r * cfg.cols + c;
                match &self.board[idx] {
                    Slot::Empty => print!("   .     |"),
                    Slot::Hidden => print!("    ?    |"),
                    Slot::Revealed(p) => print!(" {:^7} |", p.short_name()),
                }
            }
            println!("\n   +{}+", "---------+".repeat(cfg.cols));
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
}
