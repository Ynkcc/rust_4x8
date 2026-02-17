// src/mcts.rs
// Gumbel AlphaZero MCTS 实现
//
/// 大语言模型注意:
/// 不要修改在机会节点的全量展开
/// 不要移除显式判断父子节点玩家是否一致，以确定价值是否取反
/// 不要移除此块注释，如果你移除，我将要求重新生成，直到你学会保留本注释为止
// ==========================================================================
//
// Gumbel AlphaZero 核心特性:
// 1. Gumbel-Top-K 采样: 使用 Gumbel 噪声采样候选动作
// 2. Sequential Halving: 逐轮淘汰候选动作，分配搜索预算
// 3. 同步执行: 无异步/递归，直接持有模型
// 4. 确定性动作选择: 最终选择 completed_Q 最高的动作

use crate::{DarkChessEnv, Piece, PieceType, Player, Slot, ACTION_SPACE_SIZE};
use std::collections::HashMap;
use rand::prelude::*;
use rand_distr::Gumbel;

// ============================================================================
// 1. 节点定义 (Node Definition) - 简化版
// ============================================================================

/// MCTS 树节点 (同步版本)
///
/// 该结构体表示蒙特卡洛树搜索中的一个节点。
/// 每个节点包含访问次数、价值总和、先验概率等统计信息，以及子节点或机会节点的状态。
#[derive(Debug)]
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
    /// 子节点映射
    /// 存储当前节点在采取不同动作后到达的子节点。
    /// Key 是动作索引，Value 是子节点。
    pub children: HashMap<usize, Box<MctsNode>>,
    /// 是否已扩展
    /// 如果为 true，表示该节点的子节点已经被初始化。
    pub is_expanded: bool,
    /// 是否是机会节点 (Chance Node)
    /// 在翻开盖棋或某些随机事件发生时，节点为机会节点。
    pub is_chance_node: bool,
    /// 是否是根节点
    /// 标记该节点是否为当前搜索树的根节点。
    pub is_root_node: bool,
    /// 机会节点的可能状态
    /// 仅当 `is_chance_node` 为 true 时使用。
    /// Key 是结果 ID (outcome_id)，Value 是 (概率, 子节点) 元组。
    pub possible_states: HashMap<usize, (f32, Box<MctsNode>)>,
    /// 游戏环境
    /// 该节点对应的游戏状态环境。
    pub env: Option<Box<DarkChessEnv>>,
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
    /// * `is_root_node` - 是否为根节点
    pub fn new(prior: f32, logit: f32, is_chance_node: bool, env: Option<DarkChessEnv>, is_root_node: bool) -> Self {
        Self {
            visit_count: 0,
            value_sum: 0.0,
            prior,
            logit,
            children: HashMap::new(),
            is_expanded: false,
            is_chance_node,
            is_root_node,
            possible_states: HashMap::new(),
            env: env.map(Box::new),
        }
    }

    /// 获取当前节点对应的玩家
    ///
    /// # Panics
    ///
    /// 如果节点没有关联的环境 (`env` 为 None)，则会 panic。
    pub fn player(&self) -> Player {
        self.env.as_ref().expect("Node must have environment").as_ref().get_current_player()
    }

    /// 计算当前节点的平均 Q 值 (动作价值)
    ///
    /// 公式: Q = W / N
    /// 如果访问次数为 0，则返回 0.0。
    pub fn q_value(&self) -> f32 {
        if self.visit_count == 0 { 0.0 } else { self.value_sum / self.visit_count as f32 }
    }
}

/// 获取棋子的唯一结果 ID
///
/// 用于机会节点中区分不同的可能结果。
/// ID 计算方式: 棋子类型索引 + 玩家偏移量 (红方: 0, 黑方: 7)。
/// 范围: 0-13 (7种棋子 * 2个玩家)
fn get_outcome_id(piece: &Piece) -> usize {
    let type_idx = match piece.piece_type {
        PieceType::Soldier => 0, PieceType::Cannon => 1, PieceType::Horse => 2,
        PieceType::Chariot => 3, PieceType::Elephant => 4, PieceType::Advisor => 5, PieceType::General => 6,
    };
    let player_offset = match piece.player { Player::Red => 0, Player::Black => 7 };
    type_idx + player_offset
}

// ============================================================================
// 2. 评估接口
// ============================================================================

/// 评估器特征 (Trait)
///
/// 定义了评估游戏状态的接口。
/// 实现该特征的结构体 (如神经网络模型) 需要提供状态评估功能。
pub trait Evaluator {
    /// 评估给定的游戏环境批次
    ///
    /// # 参数
    ///
    /// * `envs` - 需要评估的 `DarkChessEnv` 列表
    ///
    /// # 返回
    ///
    /// 返回一个元组 `(logits, values)`:
    /// * `logits`: 每个环境的动作原始 Logits（未 mask/softmax）
    /// * `values`: 每个环境的状态价值
    fn evaluate(&self, envs: &[DarkChessEnv]) -> (Vec<Vec<f32>>, Vec<f32>);

    /// 评估并返回 Logits 和 Value
    ///
    /// 默认实现直接返回 `evaluate` 的结果。
    /// Logits 用于 Gumbel 分布的采样。
    ///
    /// # 参数
    ///
    /// * `envs` - 需要评估的 `DarkChessEnv` 列表
    ///
    /// # 返回
    ///
    /// * `logits`: 每个环境的动作对数概率
    /// * `values`: 每个环境的状态价值
    fn evaluate_logits(&self, envs: &[DarkChessEnv]) -> (Vec<Vec<f32>>, Vec<f32>) {
        self.evaluate(envs)
    }
}

// ============================================================================
// 3. Gumbel MCTS 配置
// ============================================================================

/// Gumbel MCTS 配置参数
///
/// 用于控制 MCTS 搜索过程的超参数。
#[derive(Clone)]
pub struct GumbelConfig {
    /// 模拟总次数 (N_sim)
    pub num_simulations: usize,
    /// 初始考虑的最大动作数 (m)
    /// 在 Gumbel Top-K 采样中，最多选择多少个动作进行评估。
    pub max_considered_actions: usize,
    /// 访问次数缩放因子 (c_visit)
    /// 用于计算 completed_Q 值中的权重项: n / (n + c_visit)
    pub c_visit: f32,
    /// Gumbel 噪声缩放因子 (c_scale)
    pub c_scale: f32,
    /// 是否处于训练模式
    /// 如果为 true，可能会影响某些行为 (如噪声注入)，目前主要作为标记。
    pub train: bool,
}

impl Default for GumbelConfig {
    /// 默认配置
    ///
    /// * simulations: 64
    /// * max_considered_actions: 16
    /// * c_visit: 50.0
    /// * c_scale: 1.0
    /// * train: false
    fn default() -> Self {
        Self {
            num_simulations: 64,
            max_considered_actions: 16,
            c_visit: 50.0,
            c_scale: 1.0,
            train: false,
        }
    }
}

// ============================================================================
// 4. Gumbel MCTS 主逻辑
// ============================================================================

/// Gumbel MCTS 搜索器
///
/// 管理 MCTS 树的构建、搜索和动作选择过程。
/// 泛型 `E` 必须实现 `Evaluator` 特征。
pub struct GumbelMCTS<'a, E: Evaluator> {
    /// 搜索树的根节点
    pub root: Box<MctsNode>,
    /// 状态评估器
    evaluator: &'a E,
    /// 搜索配置
    config: GumbelConfig,
}

/// 路径步骤
///
/// 在 MCTS 路径遍历中，表示每一步是选择了一个动作还是发生是一个随机机会结果。
#[derive(Clone, Debug)]
enum PathStep {
    /// 选择动作 (Action Index)
    Action(usize),
    /// 机会结果 (Outcome ID)
    ChanceOutcome(usize),
}

/// 待评估项 (Pending Evaluation)
///
/// 表示在模拟过程中到达叶子节点后，需要进行网络评估的状态。
struct PendingEval {
    /// 到达该叶子节点的路径
    path: Vec<PathStep>,
    /// 叶子节点对应的游戏环境
    env: DarkChessEnv,
    /// 叶子节点的当前玩家
    leaf_player: Player,
}

impl<'a, E: Evaluator> GumbelMCTS<'a, E> {
    /// 创建一个新的 GumbelMCTS 实例
    ///
    /// 初始化根节点并准备搜索。
    ///
    /// # 参数
    ///
    /// * `env` - 初始游戏环境
    /// * `evaluator` - 状态评估器
    /// * `config` - 搜索配置
    pub fn new(env: &DarkChessEnv, evaluator: &'a E, config: GumbelConfig) -> Self {
        let root = MctsNode::new(1.0, 0.0, false, Some(env.clone()), true);
        Self { root: Box::new(root), evaluator, config }
    }

    /// 将搜索树移动到下一个状态
    ///
    /// 当环境发生实际变动（例如玩家采取了某个动作）时调用。
    /// 该方法会尝试重用现有的子树，如果子节点不存在则创建新的根节点。
    ///
    /// # 参数
    ///
    /// * `env` - 新的游戏环境
    /// * `action` - 刚刚执行的动作
    pub fn step_next(&mut self, env: &DarkChessEnv, action: usize) {
        if let Some(mut child) = self.root.children.remove(&action) {
            if child.is_chance_node {
                // 如果是机会节点 (翻牌)，需要根据实际翻出的棋子选择对应的子节点
                let slot = env.get_target_slot(action);
                if let Slot::Revealed(piece) = slot {
                    let outcome_id = get_outcome_id(&piece);
                    if let Some((_, mut next_node)) = child.possible_states.remove(&outcome_id) {
                        next_node.is_root_node = true;
                        self.root = next_node;
                        return;
                    }
                }
            } else {
                // 普通节点，直接移动根节点
                child.is_root_node = true;
                self.root = child;
                return;
            }
        }
        // 如果无法重用子树，则重置根节点
        self.root = Box::new(MctsNode::new(1.0, 0.0, false, Some(env.clone()), true));
    }

    /// 执行 Gumbel-Top-K 采样
    ///
    /// 从 Logits 中添加 Gumbel 噪声并选择前 K 个动作。
    /// 这是 Gumbel AlphaZero 的核心机制，用于在不进行完全树搜索的情况下选择候选动作。
    fn sample_gumbel_top_k(&self, logits: &[f32], masks: &[i32], k: usize) -> Vec<usize> {
        let mut rng = thread_rng();
        let gumbel_dist = Gumbel::new(0.0, 1.0).unwrap();
        let mut gumbel_logits: Vec<(usize, f32)> = logits.iter()
            .enumerate()
            .filter(|(i, _)| masks[*i] == 1)
            .map(|(i, &l)| { let noise: f64 = gumbel_dist.sample(&mut rng); (i, l + noise as f32) })
            .collect();
        // 按加噪后的 Logits 降序排序
        gumbel_logits.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let actual_k = k.min(gumbel_logits.len());
        gumbel_logits.into_iter().take(actual_k).map(|(i, _)| i).collect()
    }

    /// 计算补全后的 Q 值 (Completed Q-value)
    ///
    /// 用于在 Sequential Halving 过程中评估动作优劣。
    /// 
    /// Formula: Q(s, a) = (N(s, a) / (N(s, a) + c_visit)) * Q_hat(s, a)
    ///
    /// 当 N 很大时，权重趋近于 1，Q 趋向于其实际 Q 值；
    /// 当 N 很小时，权重趋近于 0，Q 值被抑制，倾向于选择访问次数多的动作。
    fn completed_q(&self, action: usize) -> f32 {
        if let Some(child) = self.root.children.get(&action) {
            let n = child.visit_count as f32;
            if n > 0.0 { let weight = n / (n + self.config.c_visit); weight * child.q_value() } else { 0.0 }
        } else { 0.0 }
    }

    /// 获取根节点指定动作的 completed_Q
    pub fn get_root_completed_q(&self, action: usize) -> f32 {
        self.completed_q(action)
    }

    /// 根据 Logits 和动作掩码计算概率分布
    fn compute_probs_from_logits(&self, logits: &[f32], masks: &[i32]) -> Vec<f32> {
        let mut probs = vec![0.0; logits.len()];
        let mut max_logit = f32::NEG_INFINITY;

        for (i, &logit) in logits.iter().enumerate() {
            if masks.get(i).copied().unwrap_or(0) == 1 && logit > max_logit {
                max_logit = logit;
            }
        }

        if !max_logit.is_finite() {
            return probs;
        }

        let mut sum = 0.0;
        for (i, &logit) in logits.iter().enumerate() {
            if masks.get(i).copied().unwrap_or(0) != 1 {
                continue;
            }
            let value = (logit - max_logit).exp();
            probs[i] = value;
            sum += value;
        }

        if sum > 0.0 {
            for p in &mut probs {
                *p /= sum;
            }
        }

        probs
    }

    /// 根据路径获取可变节点引用
    ///
    /// 从根节点开始，沿着 `path` 遍历树，返回目标节点的引用。
    ///
    /// # Panics
    ///
    /// 如果路径中的任何一步在树中不存在，则会 panic。
    /// (在正常逻辑中，路径应该主要来自于树中已存在的节点，或者是刚刚扩展的节点)
    fn get_node_mut_by_path<'b>(root: &'b mut MctsNode, path: &[PathStep]) -> &'b mut MctsNode {
        let mut current = root;
        for step in path {
            match *step {
                PathStep::Action(action) => {
                    current = current.children.get_mut(&action).expect("Path action not found");
                }
                PathStep::ChanceOutcome(outcome_id) => {
                    let (_, next) = current.possible_states.get_mut(&outcome_id).expect("Path outcome not found");
                    current = next.as_mut();
                }
            }
        }
        current
    }

    /// 从叶子节点向上回溯更新价值
    ///
    /// 递归函数。更新路径上所有节点的访问次数和价值总和。
    ///
    /// # 参数
    ///
    /// * `node` - 当前递归到的节点
    /// * `path` - 剩余路径
    /// * `leaf_player` - 叶子节点（评估点）的当前玩家
    /// * `leaf_value` - 叶子节点的评估价值 (相对于 leaf_player)
    ///
    /// # 返回
    ///
    /// 返回从 `node` 视角看到的价值 (已根据玩家视角翻转)。
    fn backprop_from_path(
        node: &mut MctsNode,
        path: &[PathStep],
        leaf_player: Player,
        leaf_value: f32,
    ) -> f32 {
        if path.is_empty() {
            // 到达目标节点（叶子节点）
            let value = value_from_perspective(node.player(), leaf_player, leaf_value);
            node.visit_count += 1;
            node.value_sum += value;
            return value;
        }

        let (child_player, child_value) = match path[0] {
            PathStep::Action(action) => {
                let child = node.children.get_mut(&action).expect("Backprop child not found");
                let child_value = Self::backprop_from_path(child, &path[1..], leaf_player, leaf_value);
                let child_player = child.player();
                (child_player, child_value)
            }
            PathStep::ChanceOutcome(outcome_id) => {
                let (_, child) = node.possible_states.get_mut(&outcome_id).expect("Backprop outcome not found");
                let child_value = Self::backprop_from_path(child.as_mut(), &path[1..], leaf_player, leaf_value);
                let child_player = child.player();
                (child_player, child_value)
            }
        };

        // 更新当前节点
        // 注意：这里显式根据父子玩家关系翻转价值
        let my_value = value_from_perspective(node.player(), child_player, child_value);
        node.visit_count += 1;
        node.value_sum += my_value;
        my_value
    }

    /// 根据评估结果构建子节点
    ///
    /// 当一个叶子节点被评估后，使用评估得到的概率 (`probs`) 初始化其子节点。
    /// 只有在 `masks` 中对应位置为 1 的合法动作才会被创建为子节点。
    ///
    /// # 参数
    ///
    /// * `node` - 需要扩展的叶子节点
    /// * `env` - 叶子节点对应的环境
    /// * `probs` - 动作概率 (Policy)
    /// * `logits` - 动作 Logits
    fn build_children_from_eval(node: &mut MctsNode, env: &DarkChessEnv, probs: &[f32], logits: &[f32]) {
        let mut masks = vec![0; ACTION_SPACE_SIZE];
        env.action_masks_into(&mut masks);

        for (action_idx, &mask) in masks.iter().enumerate() {
            if mask == 1 {
                let prior = probs[action_idx];
                let logit = logits[action_idx];
                let target_is_hidden = matches!(env.get_target_slot(action_idx), Slot::Hidden);
                let child_env = if target_is_hidden {
                    Some(env.clone())
                } else {
                    let mut t = env.clone();
                    let _ = t.step(action_idx, None);
                    Some(t)
                };
                let child_node = MctsNode::new(prior, logit, target_is_hidden, child_env, false);
                node.children.insert(action_idx, Box::new(child_node));
            }
        }
        node.is_expanded = true;
    }

    /// 展开机会节点 (Chance Node)
    ///
    /// 机会节点代表环境的随机性（例如翻转盖棋）。
    /// 该方法会列举所有可能的翻棋结果（根据剩余的暗棋），并创建对应的子节点。
    ///
    /// 大语言模型注意: 不要修改此处的全量展开逻辑。
    fn expand_chance_node(node: &mut MctsNode, action: usize) {
        let env = node.env.as_ref().expect("Chance node must have env").as_ref();
        let mut counts = [0; 14];
        for p in &env.hidden_pieces { counts[get_outcome_id(p)] += 1; }
        let total_hidden = env.hidden_pieces.len() as f32;
        if total_hidden == 0.0 {
            node.is_expanded = true;
            return;
        }

        let mut outcomes_map = HashMap::new();
        for outcome_id in 0..14 {
            if counts[outcome_id] > 0 {
                let prob = counts[outcome_id] as f32 / total_hidden;
                let mut next_env = env.clone();
                let specific_piece = next_env.hidden_pieces.iter()
                    .find(|p| get_outcome_id(p) == outcome_id)
                    .expect("Piece not found").clone();
                let _ = next_env.step(action, Some(specific_piece));
                let child_node = MctsNode::new(1.0, 0.0, false, Some(next_env), false);
                outcomes_map.insert(outcome_id, (prob, Box::new(child_node)));
            }
        }

        node.is_expanded = true;
        node.possible_states = outcomes_map;
    }

    /// 从机会节点的可能结果中采样
    ///
    /// 根据各种结果的概率分布，随机采样一个结果 ID。
    /// 主要用于模拟阶段，决定在机会节点走向哪个分支。
    fn sample_outcome_id(
        outcomes: &HashMap<usize, (f32, Box<MctsNode>)>,
        rng: &mut impl Rng,
    ) -> Option<usize> {
        if outcomes.is_empty() { return None; }
        let total: f32 = outcomes.values().map(|(p, _)| *p).sum();
        if total <= 0.0 { return outcomes.keys().copied().next(); }
        let mut pick = rng.r#gen::<f32>() * total;
        for (outcome_id, (prob, _)) in outcomes {
            pick -= *prob;
            if pick <= 0.0 {
                return Some(*outcome_id);
            }
        }
        outcomes.keys().copied().next()
    }

    /// 选择路径并收集待评估项
    ///
    /// 从根节点的特定动作出发，执行模拟直到到达叶子节点或游戏结束。
    /// 如果到达未扩展的节点，将其加入 `batch` 等待后续评估。
    ///
    /// 模拟过程中使用 PUCT 公式 (Predictor + Upper Confidence Bound applied to Trees) 选择动作：
    /// Score = Q(s, a) + U(s, a)
    /// U(s, a) = c_puct * P(s, a) * sqrt(N(parent)) / (1 + N(child))
    fn select_path_collect(
        &mut self,
        action: usize,
        batch: &mut Vec<PendingEval>,
        rng: &mut impl Rng,
    ) {
        let mut path = vec![PathStep::Action(action)];
        let mut current_action = action;
        let mut node = match self.root.children.get_mut(&action) {
            Some(child) => child.as_mut(),
            None => return,
        };

        loop {
            if node.is_chance_node {
                if !node.is_expanded {
                    Self::expand_chance_node(node, current_action);
                    if node.possible_states.is_empty() {
                        return;
                    }

                    let base_path = path.clone();
                    for (&outcome_id, (_, child)) in node.possible_states.iter() {
                        let child_env = child.env.as_ref().expect("Chance outcome must have env").as_ref().clone();
                        let mut outcome_path = base_path.clone();
                        outcome_path.push(PathStep::ChanceOutcome(outcome_id));
                        let leaf_player = child.player();
                        batch.push(PendingEval { path: outcome_path, env: child_env, leaf_player });
                    }
                    return;
                }

                let outcome_id = match Self::sample_outcome_id(&node.possible_states, rng) {
                    Some(id) => id,
                    None => return,
                };
                path.push(PathStep::ChanceOutcome(outcome_id));
                let (_, next_node) = node.possible_states.get_mut(&outcome_id).expect("Outcome not found");
                node = next_node.as_mut();
                continue;
            }

            let env = node.env.as_ref().expect("Node must have env").as_ref();
            let mut masks = vec![0; ACTION_SPACE_SIZE];
            env.action_masks_into(&mut masks);
            if masks.iter().all(|&x| x == 0) {
                let leaf_player = node.player();
                let path_clone = path.clone();
                drop(node);
                Self::backprop_from_path(self.root.as_mut(), &path_clone, leaf_player, -1.0);
                return;
            }

            if !node.is_expanded {
                let leaf_player = node.player();
                batch.push(PendingEval { path, env: env.clone(), leaf_player });
                return;
            }

            let sqrt_total = (node.visit_count as f32).sqrt();
            let mut best_action = None;
            let mut best_score = f32::NEG_INFINITY;
            let parent_player = node.player();

            for (&act, child) in &node.children {
                let child_q = child.q_value();
                let child_player = child.player();
                let adjusted_q = value_from_perspective(parent_player, child_player, child_q);
                let u_score = 1.0 * child.prior * sqrt_total / (1.0 + child.visit_count as f32);
                let score = adjusted_q + u_score;
                if score > best_score {
                    best_score = score;
                    best_action = Some(act);
                }
            }

            let act = match best_action {
                Some(a) => a,
                None => return,
            };
            path.push(PathStep::Action(act));
            current_action = act;
            node = node.children.get_mut(&act).expect("Selected child missing").as_mut();
        }
    }

    /// 展开根节点
    ///
    /// 在搜索开始前，确保根节点已经被评估和扩展。
    fn expand_root(&mut self) {
        if self.root.is_expanded { return; }
        let env = self.root.env.as_ref().expect("Root must have env").as_ref().clone();
        let (logits_batch, values) = self.evaluator.evaluate(std::slice::from_ref(&env));
        let logits = &logits_batch[0];
        let value = values[0];
        let mut masks = vec![0; ACTION_SPACE_SIZE];
        env.action_masks_into(&mut masks);
        let probs = self.compute_probs_from_logits(logits, &masks);

        for (action_idx, &mask) in masks.iter().enumerate() {
            if mask == 1 {
                let prior = probs[action_idx];
                let logit = logits[action_idx];
                let target_is_hidden = matches!(env.get_target_slot(action_idx), Slot::Hidden);
                let child_env = if target_is_hidden {
                    Some(env.clone())
                } else {
                    let mut t = env.clone();
                    let _ = t.step(action_idx, None);
                    Some(t)
                };
                let child_node = MctsNode::new(prior, logit, target_is_hidden, child_env, false);
                self.root.children.insert(action_idx, Box::new(child_node));
            }
        }
        self.root.is_expanded = true;
        self.root.visit_count += 1;
        self.root.value_sum += value;
    }

    /// 执行 Gumbel MCTS 搜索主循环
    ///
    /// 1. 扩展根节点。
    /// 2. 收集根节点 Logits 并进行 Gumbel Top-K 采样，选出候选动作。
    /// 3. 使用 Sequential Halving 算法，分阶段分配搜索预算，淘汰表现不佳的候选动作。
    /// 4. 最终返回访问次数最多的动作，或在剩余候选中选择最佳动作。
    ///
    /// # 返回
    ///
    /// * `Option<usize>` - 如果是 None 表示无合法动作；否则返回选择的动作索引。
    pub fn run(&mut self) -> Option<usize> {
        // 1. 扩展根节点
        self.expand_root();

        let env = self.root.env.as_ref().expect("Root must have env").as_ref();
        let mut masks = vec![0; ACTION_SPACE_SIZE];
        env.action_masks_into(&mut masks);
        if masks.iter().all(|&x| x == 0) { return None; }

        // 2. 收集 logits
        let logits: Vec<f32> = (0..ACTION_SPACE_SIZE)
            .map(|i| self.root.children.get(&i).map(|c| c.logit).unwrap_or(-1e6))
            .collect();

        // 3. Gumbel-Top-K 采样
        let candidates = self.sample_gumbel_top_k(&logits, &masks, self.config.max_considered_actions);
        if candidates.is_empty() { return None; }
        if candidates.len() == 1 { return Some(candidates[0]); }

        // 4. Sequential Halving
        let mut remaining = candidates;
        let num_phases = (remaining.len() as f32).log2().ceil() as usize;
        let sims_per_phase = self.config.num_simulations.saturating_div(num_phases.max(1));

        let mut rng = thread_rng();

        for _phase in 0..num_phases {
            if remaining.len() <= 1 { break; }
            let visits_per_action = sims_per_phase.saturating_div(remaining.len()).max(1);
            let mut batch: Vec<PendingEval> = Vec::new();
            for _ in 0..visits_per_action {
                for &action in &remaining {
                    self.select_path_collect(action, &mut batch, &mut rng);
                }
            }

            if !batch.is_empty() {
                let envs: Vec<DarkChessEnv> = batch.iter().map(|pending| pending.env.clone()).collect();
                let (logits_batch, values) = self.evaluator.evaluate(&envs);
                for (idx, pending) in batch.into_iter().enumerate() {
                    let logits = &logits_batch[idx];
                    let value = values[idx];
                    let mut masks = vec![0; ACTION_SPACE_SIZE];
                    pending.env.action_masks_into(&mut masks);
                    let probs = self.compute_probs_from_logits(logits, &masks);
                    let leaf = Self::get_node_mut_by_path(self.root.as_mut(), &pending.path);
                    if !leaf.is_expanded {
                        Self::build_children_from_eval(leaf, &pending.env, &probs, logits);
                    }
                    Self::backprop_from_path(self.root.as_mut(), &pending.path, pending.leaf_player, value);
                }
            }
            // 淘汰下半部分
            if remaining.len() > 1 {
                let mut scored: Vec<(usize, f32)> = remaining.iter()
                    .map(|&a| (a, self.completed_q(a))).collect();
                scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                let keep_count = (remaining.len() + 1) / 2;
                remaining = scored.into_iter().take(keep_count).map(|(a, _)| a).collect();
            }
        }

        // 5. 返回结果
        if remaining.is_empty() {
            self.root.children.iter()
                .max_by_key(|(_, child)| child.visit_count)
                .map(|(&action, _)| action)
        } else {
            Some(remaining[0])
        }
    }

    /// 获取根节点的访问概率分布
    ///
    /// 返回基于访问次数归一化的概率分布，可用于训练策略网络。已弃用，建议使用 `get_improved_policy` 获取 Gumbel AlphaZero 的改进策略。
    pub fn get_root_probabilities(&self) -> Vec<f32> {
        let mut probs = vec![0.0; ACTION_SPACE_SIZE];
        let total = self.root.visit_count as f32;
        if total == 0.0 { return probs; }
        for (&action, child) in &self.root.children {
            if action < probs.len() { probs[action] = child.visit_count as f32 / total; }
        }
        probs
    }

    /// 获取 Gumbel AlphaZero 的改进策略 (pi_target)
    ///
    /// 使用 root 的先验 logit 与 completed_Q 组合，计算 softmax 概率。
    pub fn get_improved_policy(&self) -> Vec<f32> {
        let mut policy = vec![0.0; ACTION_SPACE_SIZE];
        let env = match self.root.env.as_ref() {
            Some(env) => env.as_ref(),
            None => return policy,
        };

        let mut masks = vec![0; ACTION_SPACE_SIZE];
        env.action_masks_into(&mut masks);

        let mut scores = vec![f32::NEG_INFINITY; ACTION_SPACE_SIZE];
        let mut max_score = f32::NEG_INFINITY;

        for action in 0..ACTION_SPACE_SIZE {
            if masks[action] != 1 { continue; }
            if let Some(child) = self.root.children.get(&action) {
                let score = child.logit + self.config.c_scale * self.completed_q(action);
                scores[action] = score;
                if score > max_score { max_score = score; }
            }
        }

        if !max_score.is_finite() { return policy; }

        let mut sum = 0.0;
        for action in 0..ACTION_SPACE_SIZE {
            let score = scores[action];
            if score.is_finite() {
                let value = (score - max_score).exp();
                policy[action] = value;
                sum += value;
            }
        }

        if sum > 0.0 {
            for p in policy.iter_mut() { *p /= sum; }
        }

        policy
    }
}

/// 从父节点视角转换价值
///
/// 由于 AlphaZero 通常使用双人零和博弈假设（或交替行动），
/// 如果父节点和子节点的玩家不同，价值通常需要取反。
fn value_from_perspective(parent_player: Player, child_player: Player, child_value: f32) -> f32 {
    if parent_player == child_player { child_value } else { -child_value }
}