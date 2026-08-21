//! Expectimax 搜索引擎置换表 (TT) 与 Zobrist 算法

pub const INF: f32 = 1_000_000.0;
pub const VMAX: f32 = 10_000.0;
pub const VMIN: f32 = -10_000.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TtFlag {
    Exact = 0,
    LowerBound = 1,
    UpperBound = 2,
}

#[derive(Clone, Copy, Debug)]
pub struct TtEntry {
    pub key: u64,
    pub depth: i8,
    pub flag: u8,
    pub value: f32,
    pub best_action: u16,
}

pub const TT_EMPTY: TtEntry = TtEntry {
    key: 0,
    depth: -1,
    flag: 0,
    value: 0.0,
    best_action: 0xFFFF,
};
