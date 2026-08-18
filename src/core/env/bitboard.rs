use super::config::GameConfig;
use super::constants::{DIRECTION_DOWN, DIRECTION_LEFT, DIRECTION_RIGHT, DIRECTION_UP, NUM_DIRECTIONS};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};

// ==============================================================================
// --- Bitboard 辅助函数（config 驱动） ---
//
// BOARD_MASK / NOT_FILE_A / NOT_FILE_H / ray_attacks 均依赖棋盘尺寸，因此改为
// 按 config 计算并以 config 键缓存，支持 4x8 与 4x2 两个变体在同一进程共存。
// ==============================================================================

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
        Some(63 - bb.leading_zeros() as usize)
    }
}

#[inline]
pub fn pop_lsb(bb: &mut u64) -> usize {
    let tz = bb.trailing_zeros() as usize;
    *bb &= *bb - 1;
    tz
}

/// 全棋盘掩码（低 rows*cols 位为 1）。
///
/// 注意：`total_positions` 必须小于 64，否则 `1u64 << n` 是未定义行为。
#[inline]
pub fn board_mask(cfg: &GameConfig) -> u64 {
    debug_assert!(
        cfg.total_positions < 64,
        "total_positions 必须小于 64，实际为 {}",
        cfg.total_positions
    );
    (1u64 << cfg.total_positions) - 1
}

/// 非最左列掩码（用于左移 wrap 检查）。
#[inline]
pub fn not_file_a(cfg: &GameConfig) -> u64 {
    let m = board_mask(cfg);
    let mut col_a: u64 = 0;
    for r in 0..cfg.rows {
        col_a |= ull(r * cfg.cols);
    }
    m & !col_a
}

/// 非最右列掩码（用于右移 wrap 检查）。
#[inline]
pub fn not_file_h(cfg: &GameConfig) -> u64 {
    let m = board_mask(cfg);
    let mut col_h: u64 = 0;
    for r in 0..cfg.rows {
        col_h |= ull(r * cfg.cols + (cfg.cols - 1));
    }
    m & !col_h
}

/// 缓存键：rows 与 cols 的组合即可唯一确定一张射线表。
fn ray_key(cfg: &GameConfig) -> u64 {
    ((cfg.rows as u64) << 16) | (cfg.cols as u64)
}

static RAY_CACHE: OnceLock<Mutex<HashMap<u64, Arc<Vec<Vec<u64>>>>>> = OnceLock::new();

fn ray_cache() -> &'static Mutex<HashMap<u64, Arc<Vec<Vec<u64>>>>> {
    RAY_CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// 加锁并容忍 poison（若持有锁的线程 panic，仍可继续使用缓存内容）。
fn lock_cache() -> MutexGuard<'static, HashMap<u64, Arc<Vec<Vec<u64>>>>> {
    ray_cache().lock().unwrap_or_else(|e| e.into_inner())
}

/// 射线攻击预计算表：`ray_attacks[dir][sq]` 表示从 sq 沿 dir 方向所有可达格。
/// dir 约定：0=上, 1=下, 2=左, 3=右。
pub fn ray_attacks(cfg: &GameConfig) -> Arc<Vec<Vec<u64>>> {
    let key = ray_key(cfg);
    {
        let cache = lock_cache();
        if let Some(t) = cache.get(&key) {
            return Arc::clone(t);
        }
    }
    let table = build_ray_attacks(cfg);
    let mut cache = lock_cache();
    cache.entry(key).or_insert_with(|| Arc::new(table)).clone()
}

fn build_ray_attacks(cfg: &GameConfig) -> Vec<Vec<u64>> {
    let mut rays = vec![vec![0u64; cfg.total_positions]; NUM_DIRECTIONS];
    for sq in 0..cfg.total_positions {
        let r = sq / cfg.cols;
        let c = sq % cfg.cols;
        // UP
        for i in (0..r).rev() {
            rays[DIRECTION_UP][sq] |= ull(i * cfg.cols + c);
        }
        // DOWN
        for i in (r + 1)..cfg.rows {
            rays[DIRECTION_DOWN][sq] |= ull(i * cfg.cols + c);
        }
        // LEFT
        for i in (0..c).rev() {
            rays[DIRECTION_LEFT][sq] |= ull(r * cfg.cols + i);
        }
        // RIGHT
        for i in (c + 1)..cfg.cols {
            rays[DIRECTION_RIGHT][sq] |= ull(r * cfg.cols + i);
        }
    }
    rays
}
