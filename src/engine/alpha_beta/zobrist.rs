//! Zobrist 哈希与置换表（决策节点）。
//!
//! 哈希实现已下沉至 [`crate::core::zobrist`]，供 expectimax / minimax / alpha_beta 共用。

/// 搜索值区间常量（节点走子方视角）。
pub const VMIN: f32 = -1.0;
pub const VMAX: f32 = 1.0;
pub const INF: f32 = f32::INFINITY;

pub(crate) use crate::core::zobrist::zkey;

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
