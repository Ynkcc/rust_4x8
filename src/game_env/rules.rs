use super::actions::{action_lookup_tables, pack_coords};
use super::bitboard::{board_mask, msb_index, not_file_a, not_file_h, pop_lsb, ray_attacks, trailing_zeros, ull};
use super::board::DarkChessEnv;
use super::config::NUM_PIECE_TYPES_MAX;
use super::types::*;

// ==============================================================================
// --- 规则逻辑扩展块 (动作掩码、胜负判定) ---
// ==============================================================================

impl DarkChessEnv {
    /// 游戏终止条件检查
    pub fn check_game_over_conditions(&self) -> (bool, bool, Option<i32>) {
        let cfg = &self.config;
        if self.get_score(Player::Red) <= 0 {
            return (true, false, Some(Player::Black.val()));
        }
        if self.get_score(Player::Black) <= 0 {
            return (true, false, Some(Player::Red.val()));
        }

        // 全灭判定 (使用 count 判断)
        if self.get_dead_pieces(Player::Red).len() == cfg.total_pieces_per_player {
            return (true, false, Some(Player::Black.val()));
        }
        if self.get_dead_pieces(Player::Black).len() == cfg.total_pieces_per_player {
            return (true, false, Some(Player::Red.val()));
        }

        // 无合法走法
        let mut masks = vec![0i32; cfg.action_space_size];
        self.get_action_masks_for_player_into(self.get_current_player(), &mut masks);
        if masks.iter().all(|&x| x == 0) {
            return (
                true,
                false,
                Some(self.get_current_player().opposite().val()),
            );
        }

        if self.get_move_counter() >= cfg.max_consecutive_moves_for_draw {
            return (true, false, Some(0));
        }

        if self.get_total_steps() >= cfg.max_steps_per_episode {
            return (false, true, Some(0));
        }

        (false, false, None)
    }

    // --- 动作掩码计算 ---

    pub fn action_masks(&self) -> Vec<i32> {
        let mut mask = vec![0; self.config.action_space_size];
        self.action_masks_into(&mut mask);
        mask
    }

    pub fn action_masks_into(&self, mask: &mut [i32]) {
        self.get_action_masks_for_player_into(self.get_current_player(), mask);
    }

    pub(super) fn get_action_masks_for_player_into(&self, player: Player, mask: &mut [i32]) {
        let cfg = &self.config;
        for m in mask.iter_mut() {
            *m = 0;
        }
        let lookup = action_lookup_tables(cfg);

        // 1. 翻棋动作
        let mut temp_hidden = self.get_hidden_bitboard();
        while temp_hidden != 0 {
            let sq = pop_lsb(&mut temp_hidden);
            if let Some(&idx) = lookup.coords_to_action.get(&pack_coords(&[sq])) {
                mask[idx] = 1;
            }
        }

        let empty_bb = self.get_empty_bitboard();
        let my = player;
        let opp = player.opposite();

        let my_revealed_bb = self.get_revealed_bitboards()[my.idx()];
        let my_piece_bb = self.get_piece_bitboards()[my.idx()];
        let opp_piece_bb = self.get_piece_bitboards()[opp.idx()];

        // 2. 常规移动（仅遍历激活的棋子类型；未激活类型位棋盘为空，自然跳过）
        let mut target_bbs: [u64; NUM_PIECE_TYPES_MAX] = [0; NUM_PIECE_TYPES_MAX];
        let mut cumulative_targets: u64 = empty_bb;

        for &pt in cfg.active_types.iter().take(cfg.num_active) {
            cumulative_targets |= opp_piece_bb[pt];
            target_bbs[pt] = cumulative_targets;
        }

        // 兵克将 / 将怕兵 特例（类型索引固定：兵=0，将=6）
        let soldier = PieceType::Soldier as usize;
        let general = PieceType::General as usize;
        target_bbs[soldier] |= opp_piece_bb[general];
        target_bbs[general] &= !opp_piece_bb[soldier];

        let bmask = board_mask(cfg);
        let nfa = not_file_a(cfg);
        let nfh = not_file_h(cfg);
        let shifts = [-(cfg.cols as isize) as i32, (cfg.cols as i32), -1, 1];
        let wrap_checks = [bmask, bmask, nfa, nfh];

        for &pt in cfg.active_types.iter().take(cfg.num_active) {
            if pt == PieceType::Cannon as usize {
                continue;
            }

            let from_bb = my_piece_bb[pt];
            if from_bb == 0 {
                continue;
            }

            for (dir_idx, &shift) in shifts.iter().enumerate() {
                let wrap = wrap_checks[dir_idx];

                let temp_from_bb = from_bb & wrap;
                if temp_from_bb == 0 {
                    continue;
                }

                let potential_to_bb = if shift > 0 {
                    (temp_from_bb << (shift as u32)) & bmask
                } else {
                    (temp_from_bb >> ((-shift) as u32)) & bmask
                };

                let mut actual_to_bb = potential_to_bb & target_bbs[pt];

                while actual_to_bb != 0 {
                    let to_sq = pop_lsb(&mut actual_to_bb);

                    let from_sq = if shift > 0 {
                        (to_sq as isize - (shift as isize)) as usize
                    } else {
                        (to_sq as isize + ((-shift) as isize)) as usize
                    };

                    if let Some(&idx) =
                        lookup.coords_to_action.get(&pack_coords(&[from_sq, to_sq]))
                    {
                        mask[idx] = 1;
                    }
                }
            }
        }

        // 3. 炮击
        let my_cannons_bb = my_piece_bb[PieceType::Cannon as usize];
        if my_cannons_bb != 0 {
            let all_pieces_bb = bmask & !empty_bb;

            let valid_cannon_targets = bmask & (!my_revealed_bb);

            let mut temp_cannons = my_cannons_bb;
            let rays = ray_attacks(cfg);
            while temp_cannons != 0 {
                let from_sq = pop_lsb(&mut temp_cannons);

                for dir in 0..4 {
                    let ray_bb = rays[dir][from_sq];
                    let blockers = ray_bb & all_pieces_bb;

                    if blockers == 0 {
                        continue;
                    }

                    let screen_sq = match dir {
                        0 | 2 => msb_index(blockers), // UP | LEFT
                        _ => Some(trailing_zeros(blockers)),
                    };

                    if screen_sq.is_none() {
                        continue;
                    }
                    let screen_sq = screen_sq.unwrap();

                    let after_screen_ray = rays[dir][screen_sq];
                    let targets = after_screen_ray & all_pieces_bb;

                    if targets == 0 {
                        continue;
                    }

                    let target_sq = match dir {
                        0 | 2 => msb_index(targets),
                        _ => Some(trailing_zeros(targets)),
                    };

                    if target_sq.is_none() {
                        continue;
                    }
                    let target_sq = target_sq.unwrap();

                    if ((ull(target_sq)) & valid_cannon_targets) != 0 {
                        if let Some(&idx) =
                            lookup.coords_to_action.get(&pack_coords(&[from_sq, target_sq]))
                        {
                            mask[idx] = 1;
                        }
                    }
                }
            }
        }
    }
}
