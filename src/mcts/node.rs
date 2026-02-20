// src/mcts/node.rs
// MCTS 树节点定义与内存池管理

use crate::{DarkChessEnv, Observation, Piece, PieceType, Player};
use slab::Slab;

// ============================================================================
// 内存池架构 (基于 Slab)
// ============================================================================

/// MCTS 树节点的内存池
/// 
/// 使用 Slab<MctsNode> 存储所有节点，通过 usize 索引引用。
/// Slab 提供了高效的内存池管理，大幅减少堆碎片化和改善缓存局部性。
pub struct MctsArena {
    /// 所有节点的紧凑存储 (基于 Slab)
    nodes: Slab<MctsNode>,
}

impl MctsArena {
    /// 创建一个新的内存池
    pub fn new() -> Self {
        Self { nodes: Slab::with_capacity(10000) }
    }

    /// 为节点分配内存并返回索引
    #[inline]
    pub fn allocate(&mut self, node: MctsNode) -> usize {
        self.nodes.insert(node)
    }

    /// 直接访问节点（不可变）
    #[inline]
    pub fn get(&self, idx: usize) -> &MctsNode {
        &self.nodes[idx]
    }

    /// 直接访问节点（可变）
    #[inline]
    pub fn get_mut(&mut self, idx: usize) -> &mut MctsNode {
        &mut self.nodes[idx]
    }
}

impl Default for MctsArena {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// 节点定义 (Node Definition)
// ============================================================================

/// MCTS 树节点 (Slab 版本)
///
/// 该结构体表示蒙特卡洛树搜索中的一个节点。
/// 使用 usize 索引而非 Box 来引用子节点，以减少堆碎片化。
#[derive(Debug, Clone)]
pub struct MctsNode {
    /// 访问次数 (N)
    /// 表示该节点被访问的次数。
    pub visit_count: u32,
    /// 价值总和 (W)
    /// 表示从该节点开始的所有模拟的累计价值。
    pub value_sum: f32,
    /// 先验概率 (P)
    /// 由神经网络输出的在该状态下采取某动作的概率。
    pub prior: f32,
    /// 策略 Logit
    /// 神经网络输出的原始 Logit 值，用于 Gumbel 采样。
    pub logit: f32,
    /// 子节点映射 (紧凑存储)
    /// Vec 元组: (action_index, node_arena_index)
    /// 存储当前节点在采取不同动作后到达的子节点。
    /// 使用 Vec 而非 HashMap 以改善缓存局部性。
    pub children: Vec<(usize, usize)>,
    /// 是否已扩展
    /// 如果为 true，表示该节点的子节点已经被初始化。
    pub is_expanded: bool,
    /// 是否是机会节点 (Chance Node)
    /// 在翻开盖棋或某些随机事件发生时，节点为机会节点。
    pub is_chance_node: bool,
    /// 是否是根节点
    /// 标记该节点是否为当前搜索树的根节点。
    pub is_root_node: bool,
    /// 机会节点的可能状态 (紧凑存储)
    /// Vec 元组: (outcome_id, probability, node_arena_index)
    pub possible_states: Vec<(usize, f32, usize)>,
    /// 游戏环境
    /// 该节点对应的游戏状态环境。
    pub env: Option<Box<DarkChessEnv>>,
    /// 当前玩家
    pub player: Player,
    /// 节点对应的观测状态
    pub state: Option<Observation>,
    /// 节点的初始价值 (来自网络预测的 V)
    /// 当访问次数为 0 时，用于初始化 Q 值。
    pub initial_value: f32,
    /// 是否为终局状态
    pub is_terminal: bool,
}

impl MctsNode {
    /// 创建一个新的 MctsNode 实例
    ///
    /// # 参数
    ///
    /// * `prior` - 先验概率
    /// * `logit` - 策略 Logit
    /// * `is_chance_node` - 是否为机会节点
    /// * `env` - 游戏环境状态 (Optional)
    /// * `state` - 观测状态 (Optional)
    /// * `is_root_node` - 是否为根节点
    pub fn new(prior: f32, logit: f32, is_chance_node: bool, env: Option<DarkChessEnv>, state: Option<Observation>, is_root_node: bool) -> Self {
        let (player, is_terminal) = if let Some(ref e) = env {
            let (terminated, truncated, _) = e.check_game_over_conditions();
            (e.get_current_player(), terminated || truncated)
        } else {
            (Player::Red, false)
        };

        Self {
            visit_count: 0,
            value_sum: 0.0,
            prior,
            logit,
            children: Vec::new(),
            is_expanded: false,
            is_chance_node,
            is_root_node,
            possible_states: Vec::new(),
            env: env.map(Box::new),
            player,
            state,
            initial_value: 0.0,
            is_terminal,
        }
    }

    /// 获取当前节点对应的玩家
    pub fn player(&self) -> Player {
        self.player
    }

    /// 计算当前节点的平均 Q 值 (动作价值)
    ///
    /// 公式: Q = W / N
    /// 如果访问次数为 0，则返回 0.0。
    pub fn q_value(&self) -> f32 {
        if self.visit_count == 0 { 0.0 } else { self.value_sum / self.visit_count as f32 }
    }
}

// ============================================================================
// 助手函数
// ============================================================================

/// 获取棋子的唯一结果 ID
///
/// 用于机会节点中区分不同的可能结果。
/// ID 计算方式: 棋子类型索引 + 玩家偏移量 (红方: 0, 黑方: 7)。
/// 范围: 0-13 (7种棋子 * 2个玩家)
pub fn get_outcome_id(piece: &Piece) -> usize {
    let type_idx = match piece.piece_type {
        PieceType::Soldier => 0, PieceType::Cannon => 1, PieceType::Horse => 2,
        PieceType::Chariot => 3, PieceType::Elephant => 4, PieceType::Advisor => 5, PieceType::General => 6,
    };
    let player_offset = match piece.player { Player::Red => 0, Player::Black => 7 };
    type_idx + player_offset
}

/// 从父节点视角转换价值
///
/// 由于 AlphaZero 通常使用双人零和博弈假设（或交替行动），
/// 如果父节点和子节点的玩家不同，价值通常需要取反。
pub fn value_from_perspective(parent_player: Player, child_player: Player, child_value: f32) -> f32 {
    if parent_player == child_player { child_value } else { -child_value }
}
