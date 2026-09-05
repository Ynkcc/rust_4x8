//! # Expectimax 核心搜索引擎 (参照 core/mcts 风格立为 core 一级模块)
//!
//! 包含 Star1 概率节点剪枝 Expecti-Alpha-Beta 强搜索核心 `ExpectimaxEngine`
//! （置换表 + 走子排序 + 静态搜索 + LMR + 迭代加深），叶评估以 NNUE 为唯一来源。
//!
//! 子模块分层：
//!   - search:   Expecti-Alpha-Beta 主搜索（negamax + Star1 机会节点 + quiescence + LMR）
//!   - ordering: 走子排序（MVV-LVA + 杀手 + 历史）+ 终局检测/价值 + 根层送子检测
//!   - zobrist:  值域常量 + 置换表 TtEntry（哈希下沉至 `core::zobrist`）
//!
//! 值约定：所有搜索值均为“当前节点走子方视角”，范围约 [-1, 1]。

use std::sync::Arc;
use std::time::Instant;

use crate::core::env::DarkChessEnv;
use crate::engine::movegen::Move;
use crate::inference::nnue::NnueEvaluator;

use zobrist::{TT_EMPTY, TtEntry};

pub mod ordering;
pub mod search;
pub mod zobrist;

#[cfg(test)]
mod tests;

pub use search::{SearchConfig, SearchResult, search};

// 搜索特性标志位
pub const FEAT_ORDERING: u32 = 1 << 0; // 走子排序（MVV-LVA + 杀手 + 历史）
pub const FEAT_TT: u32 = 1 << 1; // 置换表（决策节点）
pub const FEAT_LMR: u32 = 1 << 2; // 晚走子减深（late move reductions）
pub const FEAT_REP: u32 = 1 << 3; // 重复局面检测（路径 zkey）

/// 正交方向（用于送子检测）。
pub(crate) const ORTHO: [(i32, i32); 4] = [(1, 0), (-1, 0), (0, 1), (0, -1)];

/// 搜索上下文。
pub struct Ctx {
    nodes: u64,
    budget: u64,
    start: Instant,
    time_limit_ms: u64,
    killers: Vec<[usize; 2]>, // 每个剩余深度的两个杀手动作
    history: Vec<i32>,        // [total_positions*total_positions] 静走子截断历史
    tt: Vec<TtEntry>,
    tt_mask: usize,
    path: Vec<u64>, // 当前搜索路径上的祖先 zkey（重复检测）
    root: usize,    // 根走子方 idx（contempt 方向）
}

impl Ctx {
    fn new(cfg: &SearchConfig, env: &DarkChessEnv) -> Self {
        let total = env.config.total_positions;
        let kd = (cfg.max_depth.max(1) + 2) as usize;
        let tt_size = if cfg.feat(FEAT_TT) {
            1usize << cfg.tt_bits
        } else {
            0
        };
        Self {
            nodes: 0,
            budget: cfg.node_budget.max(1),
            start: Instant::now(),
            time_limit_ms: cfg.time_limit_ms,
            killers: vec![[0; 2]; kd],
            history: vec![0; total * total],
            tt: vec![TT_EMPTY; tt_size],
            tt_mask: tt_size.wrapping_sub(1),
            path: Vec::with_capacity(64),
            root: env.get_current_player().idx(),
        }
    }

    #[inline]
    fn tick(&mut self) -> Result<(), ()> {
        self.nodes += 1;
        if self.nodes > self.budget {
            return Err(());
        }
        if self.time_limit_ms > 0
            && (self.nodes & 1023) == 0
            && self.start.elapsed().as_millis() as u64 >= self.time_limit_ms
        {
            return Err(());
        }
        Ok(())
    }

    /// 记录静走子截断：提升为杀手走并累加历史分（深度²）。
    #[inline]
    fn record_cutoff(&mut self, m: &Move, depth: i32, total: usize) {
        let d = depth as usize;
        if d < self.killers.len() && self.killers[d][0] != m.action {
            self.killers[d][1] = self.killers[d][0];
            self.killers[d][0] = m.action;
        }
        let key = m.from * total + m.to;
        if key < self.history.len() {
            self.history[key] += depth * depth;
        }
    }
}

/// Expectimax 独立搜索引擎实体
pub struct ExpectimaxEngine {
    pub config: SearchConfig,
}

impl Default for ExpectimaxEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl ExpectimaxEngine {
    /// 创建默认 Expectimax 引擎实体
    pub fn new() -> Self {
        let mut config = SearchConfig::default();
        config.nnue_evaluator = Some(Arc::new(NnueEvaluator::new_dummy(crate::core::env::darkchess_config().nnue_feature_dim())));
        Self { config }
    }

    /// 从指定 `.nnue` 权重量化文件加载并创建引擎
    pub fn from_nnue_file(path: &str) -> Result<Self, String> {
        let evaluator = NnueEvaluator::load_from_file(path)
            .map_err(|e| format!("加载 NNUE 权重文件失败 {}: {}", path, e))?;

        let mut config = SearchConfig::default();
        config.nnue_evaluator = Some(Arc::new(evaluator));

        Ok(Self { config })
    }

    /// 设置搜索最大深度
    pub fn set_max_depth(&mut self, depth: i32) {
        self.config.max_depth = depth;
    }

    /// 设置节点预算
    pub fn set_node_budget(&mut self, budget: u64) {
        self.config.node_budget = budget;
    }

    /// 搜寻最佳走子
    pub fn search(&self, env: &DarkChessEnv) -> Option<SearchResult> {
        search(env, &self.config)
    }

    /// 搜寻最佳动作编号
    pub fn best_action(&self, env: &DarkChessEnv) -> Option<usize> {
        self.search(env).map(|res| res.action)
    }
}

#[cfg(test)]
mod engine_entity_tests {
    use super::*;
    use crate::core::env::DarkChessEnv;

    #[test]
    fn test_expectimax_engine_standalone() {
        let env = DarkChessEnv::default();
        let mut engine = ExpectimaxEngine::new();
        engine.set_max_depth(4);
        engine.set_node_budget(5_000);

        let res = engine.search(&env);
        assert!(res.is_some(), "ExpectimaxEngine 独立搜索应顺利产出最佳走子");
        let result = res.unwrap();
        assert!(result.depth > 0);
    }
}
