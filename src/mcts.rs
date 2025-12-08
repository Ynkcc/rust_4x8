// code_files/src/mcts.rs
///大语言模型注意:
///不要修改在机会节点的全量展开，以及后续在机会节点的所有子节点模拟
///不要移除显式判断父子节点玩家是否一致，以确定价值是否取反
///不要移除此块注释，如果你移除，我将要求重新生成，直到你学会保留本注释为止
/// ==========================================================================
use crate::{
    DarkChessEnv, Piece, PieceType, Player, Slot, ACTION_SPACE_SIZE,
};
use std::collections::HashMap;
use std::sync::Arc;

// ============================================================================
// 1. 节点定义 (Node Definition)
// ============================================================================

/// MCTS 树节点
///
/// 包含搜索树节点的统计信息、子节点指针以及对应的游戏状态。
/// 支持两种节点类型：
/// - **决策节点 (State Node)**: 玩家采取行动的节点。
/// - **机会节点 (Chance Node)**: 处理随机事件（如翻棋）的节点，其分支代表不同的随机结果。
#[derive(Debug, Clone)]
pub struct MctsNode {
    /// 访问次数 (N)
    pub visit_count: u32,
    /// 价值总和 (W) - 从该节点对应玩家的视角累积
    pub value_sum: f32,
    /// 先验概率 (P) - 由神经网络策略头输出
    pub prior: f32,
    /// 子节点映射 (针对 State Node)
    /// Key: 动作索引 (Action Index), Value: 对应的子节点
    pub children: HashMap<usize, MctsNode>,
    /// 标记该节点是否已经扩展过（即是否已经计算过子节点的先验概率）
    pub is_expanded: bool,

    // --- Chance Node 相关属性 ---
    /// 是否为机会节点 (Chance Node)
    /// 当上一步动作包含不确定性（如翻开暗子）时，当前节点为机会节点。
    pub is_chance_node: bool,
    /// 可能的后续状态映射 (针对 Chance Node)
    /// Key: 结果 ID (Outcome ID, 代表具体的棋子类型), Value: (该结果的概率, 对应的子节点)
    pub possible_states: HashMap<usize, (f32, MctsNode)>,

    // --- 游戏环境 ---
    /// 存储该节点对应的游戏环境状态
    /// State Node 通常持有环境快照。
    /// 使用 Box 将大对象移至堆内存，防止深层递归导致栈溢出。
    pub env: Option<Box<DarkChessEnv>>,
}

impl MctsNode {
    /// 创建新节点
    pub fn new(prior: f32, is_chance_node: bool, env: Option<DarkChessEnv>) -> Self {
        Self {
            visit_count: 0,
            value_sum: 0.0,
            prior,
            children: HashMap::new(),
            is_expanded: false,
            is_chance_node,
            possible_states: HashMap::new(),
            env: env.map(Box::new),
        }
    }

    /// 获取当前节点对应的行动玩家
    pub fn player(&self) -> Player {
        self.env
            .as_ref()
            .expect("Node must have environment")
            .as_ref()
            .get_current_player()
    }

    /// 获取节点的平均价值 Q(s, a)
    /// 计算公式: W / N
    pub fn q_value(&self) -> f32 {
        if self.visit_count == 0 {
            0.0
        } else {
            self.value_sum / self.visit_count as f32
        }
    }
}

/// 辅助函数：为翻开的棋子生成唯一 ID
/// 用于在 Chance Node 中区分不同的翻棋结果 (Outcome)。
/// 映射规则:
/// 0-6: 红方 [兵, 炮, 马, 车, 象, 士, 将]
/// 7-13: 黑方 [兵, 炮, 马, 车, 象, 士, 将]
fn get_outcome_id(piece: &Piece) -> usize {
    let type_idx = match piece.piece_type {
        PieceType::Soldier => 0,
        PieceType::Cannon => 1,
        PieceType::Horse => 2,
        PieceType::Chariot => 3,
        PieceType::Elephant => 4,
        PieceType::Advisor => 5,
        PieceType::General => 6,
    };
    let player_offset = match piece.player {
        Player::Red => 0,
        Player::Black => 7,
    };
    type_idx + player_offset
}

// ============================================================================
// 2. 评估接口 (Evaluation Interface)
// ============================================================================

/// 状态评估器 trait
/// 用于抽象神经网络或其他估值函数的接口。
pub trait Evaluator {
    /// 评估给定状态，返回 (策略概率分布, 状态价值)
    fn evaluate(&self, env: &DarkChessEnv) -> (Vec<f32>, f32);
}

/// 随机评估器 (用于测试或无模型情况)
pub struct RandomEvaluator;

impl Evaluator for RandomEvaluator {
    fn evaluate(&self, env: &DarkChessEnv) -> (Vec<f32>, f32) {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        let mut probs = vec![0.0; ACTION_SPACE_SIZE];
        let mut masks = vec![0; ACTION_SPACE_SIZE];
        env.action_masks_into(&mut masks);
        let valid_count = masks.iter().sum::<i32>() as f32;

        if valid_count > 0.0 {
            for (i, &m) in masks.iter().enumerate() {
                if m == 1 {
                    probs[i] = 1.0 / valid_count;
                }
            }
        }
        let value: f32 = rng.gen_range(-1.0..1.0);
        (probs, value)
    }
}

// ============================================================================
// MCTS 主逻辑
// ============================================================================

/// MCTS 配置参数
pub struct MCTSConfig {
    /// PUCT 探索常数 (C_puct)
    pub cpuct: f32,
    /// 每次搜索的模拟次数
    pub num_simulations: usize,
    /// 虚拟损失值（用于异步MCTS防止过度探索同一路径，这里主要预留）
    pub virtual_loss: f32,
    /// 并行 Worker 数量 (预留)
    pub num_mcts_workers: usize,
    /// Dirichlet 噪声参数 alpha (控制噪声分布的集中程度)
    pub dirichlet_alpha: f32,
    /// Dirichlet 噪声权重 epsilon (混合比例)
    pub dirichlet_epsilon: f32,
    /// 是否为训练模式 (训练模式下会在根节点添加噪声)
    pub train: bool,
}

impl Default for MCTSConfig {
    fn default() -> Self {
        Self {
            cpuct: 1.0,
            num_simulations: 50,
            virtual_loss: 1.0,
            num_mcts_workers: 8,
            dirichlet_alpha: 0.3,
            dirichlet_epsilon: 0.25,
            train: false,
        }
    }
}

/// 蒙特卡洛树搜索 (MCTS) 主结构
pub struct MCTS<E: Evaluator> {
    pub root: MctsNode, // 公开以便于调试或提取统计信息
    evaluator: Arc<E>,
    config: MCTSConfig,
}

impl<E: Evaluator> MCTS<E> {
    /// 创建新的 MCTS 实例
    pub fn new(env: &DarkChessEnv, evaluator: Arc<E>, config: MCTSConfig) -> Self {
        let root = MctsNode::new(1.0, false, Some(env.clone()));
        Self {
            root,
            evaluator,
            config,
        }
    }

    /// 推进搜索树 (Tree Reuse)
    /// 当实际游戏执行了某个动作后，将根节点移动到对应的子节点，保留子树统计信息。
    /// 对于机会节点，需要根据实际翻出的棋子来选择正确的分支。
    pub fn step_next(&mut self, env: &DarkChessEnv, action: usize) {
        if let Some(mut child) = self.root.children.remove(&action) {
            if child.is_chance_node {
                // 如果子节点是 Chance Node，说明上一步动作触发了不确定性事件（如翻棋）
                // 我们需要检查当前环境实际翻出了什么棋子，从而选择正确的后续状态节点

                // 使用 env 获取动作目标位置的实际 Slot
                let slot = env.get_target_slot(action);

                match slot {
                    Slot::Revealed(piece) => {
                        let outcome_id = get_outcome_id(&piece);
                        if let Some((_, next_node)) = child.possible_states.remove(&outcome_id) {
                            // 成功找到对应的后续状态节点，将其设为新的根
                            self.root = next_node;
                            return;
                        }
                    }
                    _ => {
                        // 理论上不会进入这里，除非外部状态同步错误或逻辑异常
                        panic!("Expected revealed piece at action position in Chance Node");
                    }
                }
                // 如果没找到对应分支（例如之前搜索未覆盖到该结果），则无法复用，重置树
                self.root = MctsNode::new(1.0, false, Some(env.clone()));
            } else {
                // 确定性节点（如普通移动），直接复用该子节点
                self.root = child;
            }
        } else {
            // 树中没有该动作的分支，无法复用，重置树
            self.root = MctsNode::new(1.0, false, Some(env.clone()));
        }
    }

    /// 执行 MCTS 搜索
    /// 进行 num_simulations 次模拟，并返回访问次数最多的动作。
    pub fn run(&mut self) -> Option<usize> {
        let mut total_used = 0;

        while total_used < self.config.num_simulations {
            // 执行一次从根节点开始的模拟
            let (cost, _value) =
                Self::simulate(&mut self.root, None, &self.evaluator, &self.config);

            // simulate 内部已经更新了路径上所有节点的统计信息
            // cost 表示本次模拟消耗的评估次数（通常为1，除非在机会节点展开了多个分支）
            total_used += cost;
        }

        // 搜索结束，选择访问次数 (N) 最大的动作作为最佳动作（鲁棒性最强）
        self.root
            .children
            .iter()
            .max_by_key(|(_, node)| node.visit_count)
            .map(|(action, _)| *action)
    }

    /// 递归模拟函数
    ///
    /// # 参数
    /// - `node`: 当前访问的节点
    /// - `incoming_action`: 进入该节点的前置动作（用于 Chance Node 确定翻棋位置）
    /// - `evaluator`: 评估器
    /// - `config`: 配置
    ///
    /// # 返回
    /// (cost, value)
    /// - cost: 本次递归消耗的计算量（评估次数）
    /// - value: 叶节点相对于当前节点行动方的价值 [-1, 1]
    fn simulate(
        node: &mut MctsNode,
        incoming_action: Option<usize>,
        evaluator: &Arc<E>,
        config: &MCTSConfig,
    ) -> (usize, f32) {
        // 获取当前节点的环境（只在需要时克隆到栈上，避免不必要的开销）
        let env = node
            .env
            .as_ref()
            .expect("Node must have environment")
            .as_ref()
            .clone();

        // 检查游戏是否结束
        let mut masks = vec![0; ACTION_SPACE_SIZE];
        env.action_masks_into(&mut masks);
        if masks.iter().all(|&x| x == 0) {
            // 游戏结束（无子可走），判负
            node.visit_count += 1;
            node.value_sum += -1.0;
            return (1, -1.0);
        }

        // ========================================================================
        // Case A: Chance Node (机会节点)
        // 这里的逻辑处理翻棋后的不确定性
        // ========================================================================
        if node.is_chance_node {
            let reveal_pos = incoming_action.expect("Chance node must have incoming action");

            // 1. 如果尚未扩展，则进行全量扩展 (Full Expansion)
            // 即枚举所有可能翻出的棋子，并计算它们的概率
            if !node.is_expanded {
                // 统计剩余隐藏棋子种类和数量（7种棋子 x 2方 = 14种 Outcome）
                let mut counts = [0; 14];
                for p in &env.hidden_pieces {
                    counts[get_outcome_id(p)] += 1;
                }
                let total_hidden = env.hidden_pieces.len() as f32;

                let mut total_eval_cost = 0;
                let mut total_weighted_value = 0.0;

                // 对每一种可能的 Outcome 进行扩展和评估
                for outcome_id in 0..14 {
                    if counts[outcome_id] > 0 {
                        // 计算该结果的概率
                        let prob = counts[outcome_id] as f32 / total_hidden;

                        // 构造该 Outcome 对应的确定性环境
                        let mut next_env = env.clone();
                        let specific_piece = next_env
                            .hidden_pieces
                            .iter()
                            .find(|p| get_outcome_id(p) == outcome_id)
                            .expect("指定类型的棋子不在隐藏池中")
                            .clone();
                        // 在环境中强制执行翻出该特定棋子
                        let _ = next_env.step(reveal_pos, Some(specific_piece));

                        // 创建子节点（确定性状态节点）
                        let mut child_node = MctsNode::new(1.0, false, Some(next_env));

                        // 递归模拟子节点
                        let (child_cost, child_value) =
                            Self::simulate(&mut child_node, None, evaluator, config);

                        total_eval_cost += child_cost;

                        // 关键：价值对齐
                        // 计算子节点价值相对于当前节点视角的价值
                        let aligned_value = Self::value_from_child_perspective(
                            node.player(),
                            child_node.player(),
                            child_value,
                        );
                        // 机会节点的价值是所有可能子节点价值的加权平均
                        total_weighted_value += prob * aligned_value;

                        node.possible_states.insert(outcome_id, (prob, child_node));
                    }
                }

                node.is_expanded = true;

                // 更新机会节点的统计信息
                node.visit_count += 1;
                node.value_sum += total_weighted_value;

                return (total_eval_cost, total_weighted_value);
            }

            // 2. 如果已扩展，则继续向下搜索
            // 对于 Chance Node，我们通常需要遍历所有可能的分支来获得准确的期望值
            let mut total_cost = 0;
            let mut total_weighted_value = 0.0;

            // 先获取父节点玩家，避免后续借用检查冲突
            let parent_player = node.player();

            // 遍历所有已展开的可能结果
            for (_, (prob, child_node)) in &mut node.possible_states {
                // 递归搜索该子节点
                let (child_cost, child_value) = Self::simulate(child_node, None, evaluator, config);

                total_cost += child_cost;
                // 加权累加价值
                let aligned_value = Self::value_from_child_perspective(
                    parent_player,
                    child_node.player(),
                    child_value,
                );
                total_weighted_value += *prob * aligned_value;
            }

            // 更新机会节点的统计信息
            node.visit_count += 1;
            node.value_sum += total_weighted_value;

            // 返回加权平均价值
            return (total_cost, total_weighted_value);
        }

        // ========================================================================
        // Case B: State Node (决策节点)
        // 这里的逻辑处理普通的动作选择 (PUCT)
        // ========================================================================

        // 1. 扩展 (Expansion)
        if !node.is_expanded {
            // 使用评估器（神经网络）获取当前状态的策略和价值
            let (mut policy_probs, value) = evaluator.evaluate(&env);

            // 如果是训练模式且是根节点，添加 Dirichlet 噪声以增加探索性
            if config.train && incoming_action.is_none() {
                use rand::distributions::Distribution;
                use rand_distr::Dirichlet as DirichletDist;
                
                // 统计有效动作
                let valid_actions: Vec<usize> = masks
                    .iter()
                    .enumerate()
                    .filter_map(|(idx, &mask)| if mask == 1 { Some(idx) } else { None })
                    .collect();
                
                let num_valid = valid_actions.len();
                // Dirichlet 分布至少需要 2 个元素
                if num_valid > 1 {
                    // 生成噪声
                    let alpha = vec![config.dirichlet_alpha; num_valid];
                    let dirichlet = DirichletDist::new(&alpha).expect("Invalid Dirichlet alpha");
                    let noise = dirichlet.sample(&mut rand::thread_rng());
                    
                    // 混合先验概率和噪声: P(a) = (1-ε)*P(a) + ε*Noise
                    for (i, &action_idx) in valid_actions.iter().enumerate() {
                        policy_probs[action_idx] = (1.0 - config.dirichlet_epsilon) * policy_probs[action_idx]
                            + config.dirichlet_epsilon * noise[i] as f32;
                    }
                }
            }

            // 为每个合法动作创建子节点
            for (action_idx, &mask) in masks.iter().enumerate() {
                if mask == 1 {
                    let prior = policy_probs[action_idx];

                    // 判断该动作是否会导致进入 Chance Node
                    // 如果目标位置是暗子 (Hidden)，则该动作的结果是不确定的
                    let target_is_hidden = matches!(env.get_target_slot(action_idx), Slot::Hidden);
                    let is_chance_node = target_is_hidden;

                    // 准备子节点的环境
                    let child_env = if is_chance_node {
                        // 机会节点需要存储父节点的环境（未执行动作前），以便后续扩展时可以穷举所有可能的翻棋结果
                        Some(env.clone())
                    } else {
                        // 确定性节点（移动/吃子），直接执行动作并存储新环境
                        let mut temp_env = env.clone();
                        let _ = temp_env.step(action_idx, None);
                        Some(temp_env)
                    };

                    // 创建并插入子节点
                    let child_node = MctsNode::new(prior, is_chance_node, child_env);
                    node.children.insert(action_idx, child_node);
                }
            }
            node.is_expanded = true;

            // 更新当前节点统计信息
            node.visit_count += 1;
            node.value_sum += value;

            return (1, value);
        }

        // 2. 选择 (Selection) - 使用 PUCT 算法
        let parent_player = node.player(); // 获取当前节点行动方
        let (action, best_child) = {
            let sqrt_total_visits = (node.visit_count as f32).sqrt();
            let mut best_action = None;
            let mut best_score = f32::NEG_INFINITY;

            for (&action, child) in &node.children {
                let child_q = child.q_value();
                let child_player = child.player();

                // 关键：视角转换
                // MCTS 树中每层可能由不同玩家行动（红/黑交替）
                // 子节点的 Q 值是相对于子节点行动方的，需要转换回父节点行动方的视角
                let adjusted_q =
                    Self::value_from_child_perspective(parent_player, child_player, child_q);

                // PUCT 公式: Q + U
                // U = c_puct * P * sqrt(N_parent) / (1 + N_child)
                let u_score = config.cpuct * child.prior * sqrt_total_visits
                    / (1.0 + child.visit_count as f32);
                let score = adjusted_q + u_score;

                if score > best_score {
                    best_score = score;
                    best_action = Some(action);
                }
            }

            let best_action = best_action.expect("No valid child found");
            (best_action, node.children.get_mut(&best_action).unwrap())
        };

        // 3. 递归 (Recursion)
        let (cost, child_v) = Self::simulate(best_child, Some(action), evaluator, config);

        // 4. 反向传播 (Backpropagation)
        // 将子节点返回的价值转换回当前节点视角
        let my_value =
            Self::value_from_child_perspective(parent_player, best_child.player(), child_v);

        // 更新统计信息
        node.visit_count += 1;
        node.value_sum += my_value;

        (cost, my_value)
    }

    /// 获取根节点的访问概率分布（即策略 π）
    pub fn get_root_probabilities(&self) -> Vec<f32> {
        let mut probs = vec![0.0; ACTION_SPACE_SIZE];
        let total = self.root.visit_count as f32;
        if total == 0.0 {
            return probs;
        }

        for (&action, child) in &self.root.children {
            if action < probs.len() {
                probs[action] = child.visit_count as f32 / total;
            }
        }
        probs
    }
}

impl<E: Evaluator> MCTS<E> {
    /// 价值视角转换辅助函数
    ///
    /// 在零和博弈中，如果不改变视角，子节点的“好”对父节点来说就是“坏”。
    /// 如果父子节点是同一玩家（例如连续行动），价值不变。
    /// 如果父子节点是不同玩家（红/黑交替），价值取反。
    fn value_from_child_perspective(
        parent_player: Player,
        child_player: Player,
        child_value: f32,
    ) -> f32 {
        if parent_player == child_player {
            child_value
        } else {
            -child_value
        }
    }
}