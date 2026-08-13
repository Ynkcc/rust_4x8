use super::constants::*;
use std::sync::OnceLock;

// ==============================================================================
// --- Bitboard 辅助函数 ---
// ==============================================================================

pub const BOARD_MASK: u64 = (1u64 << TOTAL_POSITIONS) - 1;

#[inline]
pub const fn ull(x: usize) -> u64 {
    1u64 << x
}

#[inline]
pub fn trailing_zeros(bb: u64) -> usize {
    bb.trailing_zeros() as usize
}

#[inline]
pub fn msb_index(bb: u64) -> Option<usize> {
    if bb == 0 {
        None
    } else {
        Some(U64_BITS - 1 - bb.leading_zeros() as usize)
    }
}

#[inline]
pub fn pop_lsb(bb: &mut u64) -> usize {
    let tz = bb.trailing_zeros() as usize;
    *bb &= *bb - 1;
    tz
}

pub const fn file_mask(file_col: usize) -> u64 {
    let mut m: u64 = 0;
    let mut r = 0;
    while r < BOARD_ROWS {
        let sq = r * BOARD_COLS + file_col;
        m |= ull(sq);
        r += 1;
    }
    m
}

pub const NOT_FILE_A: u64 = BOARD_MASK & !(file_mask(0));
pub const NOT_FILE_H: u64 = BOARD_MASK & !(file_mask(BOARD_COLS - 1));

// ==============================================================================
// --- 射线攻击预计算表 ---
// ==============================================================================

static RAY_ATTACKS: OnceLock<Vec<Vec<u64>>> = OnceLock::new();

pub fn ray_attacks() -> &'static Vec<Vec<u64>> {
    RAY_ATTACKS.get_or_init(build_ray_attacks)
}

fn build_ray_attacks() -> Vec<Vec<u64>> {
    let mut ray_attacks = vec![vec![0u64; TOTAL_POSITIONS]; NUM_DIRECTIONS];
    for sq in 0..TOTAL_POSITIONS {
        let r = sq / BOARD_COLS;
        let c = sq % BOARD_COLS;

        // UP
        for i in (0..r).rev() {
            let target_sq = i * BOARD_COLS + c;
            ray_attacks[DIRECTION_UP][sq] |= ull(target_sq);
        }
        // DOWN
        for i in (r + 1)..BOARD_ROWS {
            let target_sq = i * BOARD_COLS + c;
            ray_attacks[DIRECTION_DOWN][sq] |= ull(target_sq);
        }
        // LEFT
        for i in (0..c).rev() {
            let target_sq = r * BOARD_COLS + i;
            ray_attacks[DIRECTION_LEFT][sq] |= ull(target_sq);
        }
        // RIGHT
        for i in (c + 1)..BOARD_COLS {
            let target_sq = r * BOARD_COLS + i;
            ray_attacks[DIRECTION_RIGHT][sq] |= ull(target_sq);
        }
    }
    ray_attacks
}
