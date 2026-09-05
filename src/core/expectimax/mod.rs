//! # Expectimax 核心搜索引擎 (参照 core/mcts 风格立为 core 一级模块)
//!
//! 包含 Star1 概率节点剪枝 Expecti-Alpha-Beta 搜索核心 `ExpectimaxEngine`，
//! 属于 `src/core/` 基础领域算法实体，零依赖 `src/engine/`。

use std::sync::Arc;

use crate::core::env::DarkChessEnv;
use crate::inference::nnue::NnueEvaluator;

pub mod ordering;
pub mod search;
pub mod zobrist;

pub use search::{SearchConfig, SearchResult, search};

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
mod tests {
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
