// src/mcts/path.rs
// MCTS 路径类型定义：路径步骤、路径选择结果、待评估项（泛型化：G = 游戏环境）
//
// 分层说明：
// - 本模块只包含路径相关的类型定义，不包含任何搜索逻辑；
// - 路径遍历 / 搜索逻辑见 search.rs，树构建与回溯见 tree.rs。

use crate::game_env::GameEnv;
use crate::Player;

/// 路径步骤
///
/// 在 MCTS 路径遍历中，表示每一步是选择了一个动作还是发生是一个随机机会结果。
#[derive(Clone, Copy, Debug)]
pub enum PathStep {
    /// 选择动作 (Action Index)
    Action(usize),
    /// 机会结果 (Outcome ID)
    ChanceOutcome(usize),
}

/// `select_path_collect` 单次调用的结果，供空转防护诊断。
///
/// 用于区分"预算自然耗尽"（正常退出）与"所有候选路径命中终局"（退化但良性）
/// 两类空转，避免日志误导。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SelectPathOutcome {
    /// 正常：产出了待评估项，或完成了有效回传
    Normal,
    /// 路径命中终局节点并静默回传（未产出评估项）
    TerminalBackprop,
    /// 其他静默早退（根缺子节点 / chance 无结果 / 无子节点 / 步数超限）
    EarlyReturn,
}

/// 待评估项 (Pending Evaluation)
///
/// 表示在模拟过程中到达叶子节点后，需要进行网络评估的状态。
pub struct PendingEval<G: GameEnv> {
    /// 到达该叶子节点的路径
    pub path: Vec<PathStep>,
    /// 叶子节点对应的游戏环境
    pub env: G,
    /// 叶子节点的当前玩家
    pub leaf_player: Player,
}
