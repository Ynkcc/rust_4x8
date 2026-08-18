//! Zobrist 哈希与置换表（决策节点）。
//!
//! 局面哈希 = 棋盘槽位 + 暗子袋（按颜色/类型计数）+ 走子方。
//! 连续无吃子步数（和棋时钟）不参与哈希，以换取更多置换表命中。
//! `zkey` 同时也被 `minimax.rs` 复用（跨算法一致性）。

use std::sync::OnceLock;

use crate::game_env::types::{Player, Slot};
use crate::DarkChessEnv;

/// 搜索值区间常量（节点走子方视角）。
pub const VMIN: f32 = -1.0;
pub const VMAX: f32 = 1.0;
pub const INF: f32 = f32::INFINITY;

/// 槽位状态编码：0=空，1=暗，2+player*7+type=明子。
fn slot_state(slot: &Slot) -> usize {
    match slot {
        Slot::Empty => 0,
        Slot::Hidden => 1,
        Slot::Revealed(p) => 2 + p.player.idx() * 7 + p.piece_type as usize,
    }
}

struct Zobrist {
    sq: [[u64; 16]; 32],
    bag: [[[u64; 17]; 7]; 2],
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
            sq: [[0; 16]; 32],
            bag: [[[0; 17]; 7]; 2],
            side: [0; 2],
        };
        for i in 0..32 {
            for c in 0..16 {
                zb.sq[i][c] = next();
            }
        }
        for pl in 0..2 {
            for t in 0..7 {
                for c in 0..17 {
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

/// 局面哈希。供 minimax 复用同一哈希（跨算法一致性）。
pub(crate) fn zkey(env: &DarkChessEnv) -> u64 {
    let z = zob();
    let mut h = 0u64;
    for sq in 0..env.config.total_positions {
        h ^= z.sq[sq][slot_state(&env.get_board_slots()[sq])];
    }
    let hidden = hidden_counts(env);
    for pl in 0..2 {
        for t in 0..7 {
            h ^= z.bag[pl][t][(hidden[pl][t] as usize).min(16)];
        }
    }
    h ^= z.side[env.get_current_player().idx()];
    h
}

/// 双方剩余暗子数 [player_idx][type_idx]（总数 - 已明 - 已死）。
fn hidden_counts(env: &DarkChessEnv) -> [[u32; 7]; 2] {
    let cfg = &env.config;
    let mut accounted = [[0u32; 7]; 2]; // 已明 + 已死
    for slot in env.get_board_slots() {
        if let Slot::Revealed(p) = slot {
            accounted[p.player.idx()][p.piece_type as usize] += 1;
        }
    }
    for &p in &[Player::Red, Player::Black] {
        for &t in env.get_dead_pieces(p) {
            accounted[p.idx()][t as usize] += 1;
        }
    }
    let mut hidden = [[0u32; 7]; 2];
    for pl in 0..2 {
        for t in 0..7 {
            hidden[pl][t] = (cfg.piece_counts[t] as u32).saturating_sub(accounted[pl][t]);
        }
    }
    hidden
}

/// 置换表项（决策节点）。
#[derive(Clone, Copy)]
pub struct TtEntry {
    pub key: u64,
    pub value: f32,
    pub depth: i16,
    /// 0 空, 1 exact, 2 下界(fail-high), 3 上界(fail-low)
    pub flag: u8,
    pub best: usize,
}

pub const TT_EMPTY: TtEntry = TtEntry {
    key: 0,
    value: 0.0,
    depth: 0,
    flag: 0,
    best: 0,
};
