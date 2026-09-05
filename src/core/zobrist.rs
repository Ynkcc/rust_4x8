//! Zobrist 局面哈希（core 层共享，变体无关）。
//!
//! 局面哈希 = 棋盘槽位 + 暗子袋（按颜色/类型计数，直读环境暗子池）+ 走子方。
//! 所有维度取自 `MAX_*` 上界常量，索引由各变体运行时 config / 棋子实际值驱动，
//! 4x8 / 4x4 / 4x2 通用；袋模型下暗子不预分配格位，该键精确刻画搜索状态。
//! 连续无吃子步数（和棋时钟）不参与哈希，以换取更多置换表命中。
//! 供 expectimax / alpha_beta / minimax 共用（跨算法一致性）。
//!
//! 随机数采用 SplitMix64（增量 0x9E37_79B9_7F4A_7C15 = 2^64/φ，
//! 终化乘数 0xBF58_476D_1CE4_E5B9 / 0x94D0_49BB_1331_11EB 为其标准终化常量），
//! 种子 0x1234_5678_9ABC_DEF0 仅为任选的非零初值，保证跨运行确定性。

use std::sync::OnceLock;

use crate::core::env::config::{MAX_PIECES_PER_PLAYER, MAX_POSITIONS, NUM_PIECE_TYPES_MAX};
use crate::core::env::symmetry::{Symmetry, sq_map};
use crate::core::env::types::Slot;
use crate::core::env::DarkChessEnv;

/// 槽位状态编码：0=空，1=暗，2+player*TYPE_NB+type=明子。
fn slot_state(slot: &Slot) -> usize {
    match slot {
        Slot::Empty => 0,
        Slot::Hidden => 1,
        Slot::Revealed(p) => 2 + p.player.idx() * NUM_PIECE_TYPES_MAX + p.piece_type as usize,
    }
}

struct Zobrist {
    sq: [[u64; 2 * NUM_PIECE_TYPES_MAX + 2]; MAX_POSITIONS],
    bag: [[[u64; MAX_PIECES_PER_PLAYER + 1]; NUM_PIECE_TYPES_MAX]; 2],
    side: [u64; 2],
}

static ZOB: OnceLock<Zobrist> = OnceLock::new();

fn zob() -> &'static Zobrist {
    ZOB.get_or_init(|| {
        let mut s: u64 = 0x1234_5678_9ABC_DEF0;
        let mut next = || {
            s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = s;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        };
        let mut zb = Zobrist {
            sq: [[0; 2 * NUM_PIECE_TYPES_MAX + 2]; MAX_POSITIONS],
            bag: [[[0; MAX_PIECES_PER_PLAYER + 1]; NUM_PIECE_TYPES_MAX]; 2],
            side: [0; 2],
        };
        for i in 0..MAX_POSITIONS {
            for c in 0..2 * NUM_PIECE_TYPES_MAX + 2 {
                zb.sq[i][c] = next();
            }
        }
        for pl in 0..2 {
            for t in 0..NUM_PIECE_TYPES_MAX {
                for c in 0..=MAX_PIECES_PER_PLAYER {
                    zb.bag[pl][t][c] = next();
                }
            }
        }
        for s2 in 0..2 {
            zb.side[s2] = next();
        }
        zb
    })
}

/// 局面哈希。供 expectimax / minimax 复用同一哈希（跨算法一致性）。
pub fn zkey(env: &DarkChessEnv) -> u64 {
    let z = zob();
    let mut h = 0u64;
    for sq in 0..env.config.total_positions {
        let st = slot_state(&env.get_board_slots()[sq]);
        if st != 0 {
            h ^= z.sq[sq][st];
        }
    }
    h ^ bag_side_hash(env, &z)
}

/// 袋计数 + 走子方哈希分量（空间几何变换下的不变量）。
fn bag_side_hash(env: &DarkChessEnv, z: &Zobrist) -> u64 {
    // 暗子袋计数：直读环境暗子池（含归属），自动适配任意变体。
    let mut counts = [[0usize; NUM_PIECE_TYPES_MAX]; 2];
    for piece in env.get_hidden_pieces_raw() {
        counts[piece.player.idx()][piece.piece_type as usize] += 1;
    }
    let mut h = 0u64;
    for (pl, row) in counts.iter().enumerate() {
        for (t, &c) in row.iter().enumerate() {
            h ^= z.bag[pl][t][c.min(MAX_PIECES_PER_PLAYER)];
        }
    }
    h ^ z.side[env.get_current_player().idx()]
}

/// 对称视角哈希：棋盘格位项按 `sym` 的格子重排表映射后计算，
/// 袋计数 / 走子方分量为空间不变量、直接复用。
/// `Symmetry::Identity` 时与 [`zkey`] 完全一致。
pub fn sym_zkey(env: &DarkChessEnv, sym: Symmetry) -> u64 {
    let z = zob();
    let map = sq_map(env.config.rows, env.config.cols, sym);
    let slots = env.get_board_slots();
    let mut h = 0u64;
    for sq in 0..env.config.total_positions {
        let st = slot_state(&slots[sq]);
        if st != 0 {
            h ^= z.sq[map[sq]][st];
        }
    }
    h ^ bag_side_hash(env, &z)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::env::symmetry::search_group;

    #[test]
    fn sym_zkey_identity_matches_zkey() {
        for mut env in [
            DarkChessEnv::default(),
            DarkChessEnv::new_4x4(),
            DarkChessEnv::new_mini(),
        ] {
            env.seed = Some(11);
            env.reset();
            assert_eq!(sym_zkey(&env, Symmetry::Identity), zkey(&env));
        }
    }

    #[test]
    fn sym_zkey_deterministic_and_orientation_sensitive() {
        // 初始局面全为暗子（各格非空），重排映射必然改变格位项 → 非恒等变换键应不同
        for mut env in [DarkChessEnv::default(), DarkChessEnv::new_4x4()] {
            env.seed = Some(3);
            env.reset();
            let raw = sym_zkey(&env, Symmetry::Identity);
            for &sym in search_group(env.config.rows, env.config.cols) {
                if sym == Symmetry::Identity {
                    continue;
                }
                let k = sym_zkey(&env, sym);
                assert_eq!(k, sym_zkey(&env, sym), "对称键必须确定");
                assert_ne!(k, raw, "非恒等变换应产生不同键");
            }
        }
    }

    #[test]
    fn zkey_variants_deterministic_and_state_sensitive() {
        for mut env in [
            DarkChessEnv::default(),
            DarkChessEnv::new_4x4(),
            DarkChessEnv::new_mini(),
        ] {
            env.seed = Some(7);
            env.reset();

            let k0 = zkey(&env);
            assert_eq!(k0, zkey(&env), "同一局面哈希必须确定");

            // 翻棋改变袋计数 / 槽位 → 键必须变化
            if let Some(&act) = env
                .legal_action_indices()
                .iter()
                .find(|&&a| env.is_chance_action(a))
            {
                let _ = env.step(act, None);
                assert_ne!(k0, zkey(&env), "翻棋后哈希必须变化");
            }
        }
    }
}
