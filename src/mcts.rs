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
#[derive(Debug, Clone)]
pub struct MctsNode {
    /// 访问次数 (N)
    pub visit_count: u32,
    /// 价值总和 (W)
    pub value_sum: f32,
    /// 先验概率 (P)
    pub prior: f32,
    /// 当前节点的动作-子节点映射 (针对 State Node)
    /// Key: Action Index
    pub children: HashMap<usize, MctsNode>,
    /// 标记是否已扩展
    pub is_expanded: bool,

    // --- Chance Node 相关属性 ---
    /// 是否为机会节点 (Chance Node)
    pub is_chance_node: bool,
    /// 可能的状态映射 (针对 Chance Node)
    /// Key: Outcome ID (表示具体的翻棋结果), Value: (Probability, ChildNode)
    pub possible_states: HashMap<usize, (f32, MctsNode)>,

    // --- 游戏环境 ---
    /// 存储该节点对应的游戏环境状态 (State Node 包含，Chance Node 不包含)
    /// 使用 Box 将大对象移至堆内存，避免栈溢出
    pub env: Option<Box<DarkChessEnv>>,
}

impl MctsNode {
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

    /// 获取当前节点对应的玩家
    pub fn player(&self) -> Player {
        self.env
            .as_ref()
            .expect("Node must have environment")
            .as_ref()
            .get_current_player()
    }

    /// 获取平均价值 Q(s, a)
    pub fn q_value(&self) -> f32 {
        if self.visit_count == 0 {
            0.0
        } else {
            self.value_sum / self.visit_count as f32
        }
    }
}

// 辅助函数：为翻开的棋子生成唯一 ID
// 0-6: Red [Sol, Can, Hor, Cha, Ele, Adv, Gen]; 7-13: Black [Sol, Can, Hor, Cha, Ele, Adv, Gen]
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

pub trait Evaluator {
    fn evaluate(&self, env: &DarkChessEnv) -> (Vec<f32>, f32);
}

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

pub struct MCTSConfig {
    pub cpuct: f32,
    pub num_simulations: usize,
    /// 虚拟损失值（用于异步MCTS）
    pub virtual_loss: f32,
   
    pub num_mcts_workers: usize,
    /// Dirichlet 噪声 alpha 参数
    pub dirichlet_alpha: f32,
    /// Dirichlet 噪声权重（与先验策略的混合比例）
    pub dirichlet_epsilon: f32,
    /// 是否为训练模式（训练时添加噪声，对弈时不添加）
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

pub struct MCTS<E: Evaluator> {
    pub root: MctsNode, // made public for debug access if needed
    evaluator: Arc<E>,
    config: MCTSConfig,
}

impl<E: Evaluator> MCTS<E> {
    pub fn new(env: &DarkChessEnv, evaluator: Arc<E>, config: MCTSConfig) -> Self {
        let root = MctsNode::new(1.0, false, Some(env.clone()));
        Self {
            root,
            evaluator,
            config,
        }
    }

    /// 支持搜索树复用：根据动作将根节点推进一步
    pub fn step_next(&mut self, env: &DarkChessEnv, action: usize) {
        if let Some(mut child) = self.root.children.remove(&action) {
            if child.is_chance_node {
                // 如果是 Chance Node，说明上一步动作是翻棋或炮攻击暗子
                // 我们需要检查当前环境实际翻出了什么棋子，从而选择正确的子节点

                // 使用 get_target_slot 获取动作目标位置的 Slot
                let slot = env.get_target_slot(action);

                match slot {
                    Slot::Revealed(piece) => {
                        let outcome_id = get_outcome_id(&piece);
                        if let Some((_, next_node)) = child.possible_states.remove(&outcome_id) {
                            // 成功找到对应的后续状态节点
                            self.root = next_node;
                            return;
                        }
                    }
                    _ => {
                        // 理论上不会进入这里，除非外部状态同步错误
                    }
                }
                // 如果没找到对应分支（比如之前没探索到），则重置
                self.root = MctsNode::new(1.0, false, Some(env.clone()));
            } else {
                // 确定性节点（移动），直接复用
                self.root = child;
            }
        } else {
            // 树中没有该动作，重置
            self.root = MctsNode::new(1.0, false, Some(env.clone()));
        }
    }

    pub fn run(&mut self) -> Option<usize> {
        let mut total_used = 0;

        while total_used < self.config.num_simulations {
            let (cost, _value) =
                Self::simulate(&mut self.root, None, &self.evaluator, &self.config);

            // simulate内部已经更新了所有节点的统计信息
            total_used += cost;
        }

        self.root
            .children
            .iter()
            .max_by_key(|(_, node)| node.visit_count)
            .map(|(action, _)| *action)
    }

    /// 递归模拟
    /// incoming_action: 进入该节点的前置动作（用于 Chance Node 确定位置）
    /// 返回值: (cost, value) - cost 是消耗的评估次数，value 是相对于当前节点行动方的价值
    fn simulate(
        node: &mut MctsNode,
        incoming_action: Option<usize>,
        evaluator: &Arc<E>,
        config: &MCTSConfig,
    ) -> (usize, f32) {
        // 获取当前节点的环境（只在需要时克隆到栈上）
        let env = node
            .env
            .as_ref()
            .expect("Node must have environment")
            .as_ref()
            .clone();

        let mut masks = vec![0; ACTION_SPACE_SIZE];
        env.action_masks_into(&mut masks);
        if masks.iter().all(|&x| x == 0) {
            // 游戏结束（无子可走），判负
            node.visit_count += 1;
            node.value_sum += -1.0;
            return (1, -1.0);
        }

        // ========================================================================
        // Case A: Chance Node (上一步是翻棋)
        // ========================================================================
        if node.is_chance_node {
            let reveal_pos = incoming_action.expect("Chance node must have incoming action");

            // 1. 如果尚未扩展，则进行全量扩展
            if !node.is_expanded {
                // 统计剩余棋子种类和数量（7种棋子 x 2方 = 14）
                let mut counts = [0; 14];
                for p in &env.hidden_pieces {
                    counts[get_outcome_id(p)] += 1;
                }
                let total_hidden = env.hidden_pieces.len() as f32;

                let mut total_eval_cost = 0;
                let mut total_weighted_value = 0.0;

                // 对每一种可能的 outcome 进行扩展和评估
                for outcome_id in 0..14 {
                    if counts[outcome_id] > 0 {
                        let prob = counts[outcome_id] as f32 / total_hidden;

                        // 构造该 outcome 对应的环境
                        let mut next_env = env.clone();
                        let specific_piece = next_env
                            .hidden_pieces
                            .iter()
                            .find(|p| get_outcome_id(p) == outcome_id)
                            .expect("指定类型的棋子不在隐藏池中")
                            .clone();
                        let _ = next_env.step(reveal_pos, Some(specific_piece));

                        let mut child_node = MctsNode::new(1.0, false, Some(next_env));

                        // 递归模拟子节点（子节点已保存环境，不需要传入）
                        let (child_cost, child_value) =
                            Self::simulate(&mut child_node, None, evaluator, config);

                        total_eval_cost += child_cost;
                        let aligned_value = Self::value_from_child_perspective(
                            node.player(),
                            child_node.player(),
                            child_value,
                        );
                        // 机会节点的价值是加权平均（根据玩家关系决定是否取反）
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

            // 2. 如果已扩展，则对字典中所有可能的子节点进行MCTS搜索
            let mut total_cost = 0;
            let mut total_weighted_value = 0.0;

            // 先获取父节点玩家，避免后续借用冲突
            let parent_player = node.player();

            // 对每个可能的 outcome 进行搜索
            for (_, (prob, child_node)) in &mut node.possible_states {
                // 递归搜索该子节点（子节点已保存环境，直接使用）
                let (child_cost, child_value) = Self::simulate(child_node, None, evaluator, config);

                total_cost += child_cost;
                // 加权平均价值（根据玩家关系决定是否取反）
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
        // Case B: State Node (普通节点)
        // ========================================================================

        // 1. 扩展 (Expansion)
        if !node.is_expanded {
            let (mut policy_probs, value) = evaluator.evaluate(&env);

            // 如果是训练模式且是根节点，添加 Dirichlet 噪声
            if config.train && incoming_action.is_none() {
                use rand::distributions::Distribution;
                use rand_distr::Dirichlet as DirichletDist;
                
                // 统计有效动作数量
                let valid_actions: Vec<usize> = masks
                    .iter()
                    .enumerate()
                    .filter_map(|(idx, &mask)| if mask == 1 { Some(idx) } else { None })
                    .collect();
                
                let num_valid = valid_actions.len();
                // Dirichlet 分布至少需要 2 个元素，且只有一个动作时添加噪声无意义
                if num_valid > 1 {
                    // 生成 Dirichlet 噪声
                    let alpha = vec![config.dirichlet_alpha; num_valid];
                    let dirichlet = DirichletDist::new(&alpha).expect("Invalid Dirichlet alpha");
                    let noise = dirichlet.sample(&mut rand::thread_rng());
                    
                    // 混合先验策略和噪声
                    for (i, &action_idx) in valid_actions.iter().enumerate() {
                        policy_probs[action_idx] = (1.0 - config.dirichlet_epsilon) * policy_probs[action_idx]
                            + config.dirichlet_epsilon * noise[i] as f32;
                    }
                }
            }

            for (action_idx, &mask) in masks.iter().enumerate() {
                if mask == 1 {
                    let prior = policy_probs[action_idx];

                    // 判断该动作是否会导致 Chance Node
                    let target_is_hidden = matches!(env.get_target_slot(action_idx), Slot::Hidden);
                    let is_chance_node = target_is_hidden;
                    // Chance Node 存储父节点环境用于扩展，State Node 存储执行动作后的环境
                    let child_env = if is_chance_node {
                        Some(env.clone()) // 机会节点存储父节点环境（用于扩展时获取隐藏棋子信息）
                    } else {
                        // 移动节点需要执行动作后存储环境
                        let mut temp_env = env.clone();
                        let _ = temp_env.step(action_idx, None);
                        Some(temp_env)
                    };

                    // 🔥 修复：这里必须传入 is_chance_node，而不是 is_reveal
                    // 之前的写法导致"炮击暗子"被错误标记为确定性节点，从而复用了错误的父环境
                    let child_node = MctsNode::new(prior, is_chance_node, child_env);
                    node.children.insert(action_idx, child_node);
                }
            }
            node.is_expanded = true;

            // 更新节点统计信息
            node.visit_count += 1;
            node.value_sum += value;

            return (1, value);
        }

        // 2. 选择 (Selection)
        let parent_player = node.player(); // 先获取父节点玩家，避免借用冲突
        let (action, best_child) = {
            let sqrt_total_visits = (node.visit_count as f32).sqrt();
            let mut best_action = None;
            let mut best_score = f32::NEG_INFINITY;

            for (&action, child) in &node.children {
                let child_q = child.q_value();
                let child_player = child.player();

                // 将子节点的 Q 值转换为父节点玩家视角
                // 如果父子玩家不同，需要取反
                let adjusted_q =
                    Self::value_from_child_perspective(parent_player, child_player, child_q);

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

        // 3. 递归到子节点（子节点已保存环境，直接递归）
        let (cost, child_v) = Self::simulate(best_child, Some(action), evaluator, config);

        // 根据父子节点的行动方关系决定是否取反
        let my_value =
            Self::value_from_child_perspective(parent_player, best_child.player(), child_v);

        // 更新当前节点的统计信息
        node.visit_count += 1;
        node.value_sum += my_value;

        (cost, my_value)
    }

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
    /// 将子节点价值转换为父节点玩家视角
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
