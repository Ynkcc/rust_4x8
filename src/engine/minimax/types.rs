// src/ai/minimax/types.rs - 数据结构与常量

use crate::engine::evaluation::EvalParams;
use crate::engine::movegen::Move;
use crate::core::env::board::DarkChessEnv;

/// 机会节点（翻棋/吃暗子）消耗的搜索深度。
///
/// 机会动作每个要枚举 2*num_active 种翻棋结果，若不惩罚会指数爆炸
/// （alpha-beta 对期望节点无效，无法剪枝）。设 2：机会层等价于两倍普通层代价。
pub(super) const CHANCE_DEPTH_PENALTY: usize = 2;

pub(super) const VMIN: f32 = -1.0;
pub(super) const VMAX: f32 = 1.0;
pub(super) const INF: f32 = f32::INFINITY;

/// 单次搜索的结果。
#[derive(Debug, Clone, Copy)]
pub struct MinimaxResult {
    /// 从当前玩家视角选择的最优动作。
    pub action: usize,
    /// 该动作的期望效用（当前玩家视角，[-1, 1]）。
    pub value: f32,
    /// 搜索展开的节点数（不含根）。
    pub nodes: u64,
}

/// Minimax 搜索配置。
#[derive(Clone, Copy, Debug)]
pub struct MinimaxConfig {
    /// 是否使用多特征启发式评估（关闭则退回 HP 差评估）
    pub rich_eval: bool,
    /// 是否启用置换表
    pub use_tt: bool,
    /// 是否启用走子排序
    pub use_ordering: bool,
    /// 是否启用静态搜索
    pub use_quiescence: bool,
    /// 置换表大小（2^tt_bits 项）
    pub tt_bits: u32,
    /// 评估参数（rich_eval 时生效）
    pub params: EvalParams,
}

impl Default for MinimaxConfig {
    fn default() -> Self {
        Self {
            rich_eval: true,
            use_tt: true,
            use_ordering: true,
            use_quiescence: true,
            tt_bits: 16,
            params: EvalParams::default(),
        }
    }
}

// --- 置换表 ---

#[derive(Clone, Copy)]
pub(super) struct TtEntry {
    pub(super) key: u64,
    pub(super) value: f32,
    pub(super) depth: i16,
    pub(super) flag: u8, // 0 空, 1 exact, 2 下界, 3 上界
    pub(super) best: usize,
}

const TT_EMPTY: TtEntry = TtEntry {
    key: 0,
    value: 0.0,
    depth: 0,
    flag: 0,
    best: 0,
};

pub(super) struct SearchState {
    pub(super) nodes: u64,
    pub(super) tt: Vec<TtEntry>,
    pub(super) tt_mask: usize,
    pub(super) killers: Vec<[usize; 2]>,
    pub(super) history: Vec<i32>,
}

impl SearchState {
    pub(super) fn new(cfg: &MinimaxConfig, env: &DarkChessEnv, max_depth: usize) -> Self {
        let total = env.config.total_positions;
        let tt_size = if cfg.use_tt { 1usize << cfg.tt_bits } else { 0 };
        Self {
            nodes: 0,
            tt: vec![TT_EMPTY; tt_size],
            tt_mask: tt_size.wrapping_sub(1),
            killers: vec![[0; 2]; max_depth + 2],
            history: vec![0; total * total],
        }
    }

    pub(super) fn record_cutoff(&mut self, m: &Move, depth: usize, total: usize) {
        let d = depth as usize;
        if d < self.killers.len() && self.killers[d][0] != m.action {
            self.killers[d][1] = self.killers[d][0];
            self.killers[d][0] = m.action;
        }
        let key = m.from * total + m.to;
        if key < self.history.len() {
            self.history[key] += (depth * depth) as i32;
        }
    }
}
