// src/mcts/path.rs
// MCTS 路径类型定义：路径步骤、路径选择结果、待评估项（泛型化：G = 游戏环境）
//
// 分层说明：
// - 本模块只包含路径相关的类型定义，不包含任何搜索逻辑；
// - 路径遍历 / 搜索逻辑见 search.rs，树构建与回溯见 tree.rs。

use crate::core::env::GameEnv;
use crate::core::env::Player;

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

/// 机会节点展开爆发的回传种子标记。
///
/// 展开瞬间批量评估的全部 outcome 共享同一机会节点，其价值回传不能逐路径
/// 整数回溯（会使机会节点初始 Q 退化为无权均值），须按概率加权 \sum p_i v_i
/// 单次回传。`prefix_len` 为路径中到达机会节点为止的前缀长度
/// （即去掉末尾 `ChanceOutcome` 步后的长度）。
#[derive(Clone, Copy, Debug)]
pub struct ChanceSeed {
    /// 所属机会节点在 Arena 中的索引
    pub chance_idx: usize,
    /// 该 outcome 的发生概率
    pub prob: f32,
    /// 路径前缀长度（到达机会节点，含其全部祖先步）
    pub prefix_len: usize,
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
    /// 机会节点展开爆发标记：普通叶子为 None；展开爆发的 outcome 项为 Some。
    pub chance_seed: Option<ChanceSeed>,
}
