//! 无锁共享置换表（Lazy SMP 用）。
//!
//! 表项打包为单个 `AtomicU64`：(value f32 bits | depth u8 | flag u2 | key_check u22)，
//! 键校验位与值同字原子读写，杜绝跨局面脏命中；最佳走子提示存于独立 `AtomicU32`，
//! 允许读到过期值（仅影响走子排序，不影响正确性）。写入为 last-write-wins，
//! 允许极小概率的覆盖竞争（标准引擎做法，不做完全串行一致性保证）。

use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

use super::zobrist::TtEntry;

/// key_check 位宽（u64 高 22 位）。
const KC_BITS: u32 = 22;
const KC_MASK: u64 = (1 << KC_BITS) - 1;

/// 探测结果：(value, depth, flag, best 提示)。
pub type TtProbe = (f32, i32, u8, u32);

pub struct SharedTT {
    entries: Vec<AtomicU64>,
    bests: Vec<AtomicU32>,
    mask: usize,
    /// 量化统计（`SearchConfig::tt_sym_probe` 开启时记录）：
    /// [0]=TT 探测节点数, [1]=原始键命中, [2]=对称键二探命中, [3]=对称命中且深度足够
    stats: [AtomicU64; 4],
}

impl SharedTT {
    /// `tt_bits = 2^n` 表项数；`tt_bits = 0` 时为禁用表（探测恒空）。
    pub fn new(tt_bits: u32) -> Self {
        let size = if tt_bits == 0 { 0 } else { 1usize << tt_bits };
        Self {
            entries: (0..size).map(|_| AtomicU64::new(0)).collect(),
            bests: (0..size).map(|_| AtomicU32::new(0)).collect(),
            mask: size.wrapping_sub(1),
            stats: std::array::from_fn(|_| AtomicU64::new(0)),
        }
    }

    /// 统计计数自增（Relaxed，仅作量化参考）。
    #[inline]
    pub fn bump(&self, slot: usize) {
        self.stats[slot].fetch_add(1, Ordering::Relaxed);
    }

    /// 读取量化统计快照。
    pub fn tt_stats(&self) -> [u64; 4] {
        let mut out = [0u64; 4];
        for (o, c) in out.iter_mut().zip(&self.stats) {
            *o = c.load(Ordering::Relaxed);
        }
        out
    }

    #[inline]
    fn pack(key: u64, entry: &TtEntry) -> u64 {
        let kc = (key >> (64 - KC_BITS)) & KC_MASK;
        (entry.value.to_bits() as u64)
            | ((entry.depth as u64 & 0xFF) << 32)
            | ((entry.flag as u64 & 0x3) << 40)
            | (kc << 42)
    }

    #[inline]
    fn unpack(raw: u64, key: u64) -> Option<TtEntry> {
        let flag = ((raw >> 40) & 0x3) as u8;
        if flag == 0 {
            return None;
        }
        let kc = key >> (64 - KC_BITS);
        if (raw >> 42) != kc {
            return None;
        }
        Some(TtEntry {
            key,
            value: f32::from_bits(raw as u32),
            depth: ((raw >> 32) & 0xFF) as i16,
            flag,
            best: 0,
        })
    }

    /// 探测。返回 `Some((value, depth, flag, best 提示))`；表禁用或键不匹配返回 None。
    #[inline]
    pub fn probe(&self, key: u64) -> Option<TtProbe> {
        if self.entries.is_empty() {
            return None;
        }
        let idx = (key as usize) & self.mask;
        let raw = self.entries[idx].load(Ordering::Relaxed);
        let e = Self::unpack(raw, key)?;
        let best = self.bests[idx].load(Ordering::Relaxed);
        Some((e.value, e.depth as i32, e.flag, best))
    }

    /// 存储表项（last-write-wins；禁用表为 no-op）。
    #[inline]
    pub fn store(&self, key: u64, entry: &TtEntry) {
        if self.entries.is_empty() {
            return;
        }
        let idx = (key as usize) & self.mask;
        self.entries[idx].store(Self::pack(key, entry), Ordering::Relaxed);
        self.bests[idx].store(entry.best as u32, Ordering::Relaxed);
    }

    /// 条件存储：仅在空槽 / 异键 / 更深搜索时替换（深度优先），语义与本地表一致。
    #[inline]
    pub fn store_cond(&self, key: u64, entry: &TtEntry) {
        if self.entries.is_empty() {
            return;
        }
        let idx = (key as usize) & self.mask;
        let cur_raw = self.entries[idx].load(Ordering::Relaxed);
        if let Some(cur) = Self::unpack(cur_raw, key) {
            if (cur.depth as i32) > entry.depth as i32 {
                return;
            }
        }
        self.store(key, entry);
    }
}
