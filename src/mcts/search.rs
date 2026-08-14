// src/mcts/search.rs
// Gumbel AlphaZero MCTS 核心搜索与采样算法

use crate::{ACTION_SPACE_SIZE, DarkChessEnv, Player, Slot};
use rand::prelude::*;
use rand_distr::Gumbel;

use super::budget::SequentialHalvingBudget;
use super::config::{GumbelConfig, MctsSearchResult};
use super::evaluator::Evaluator;
use super::node::{MctsArena, MctsNode, get_outcome_id, value_from_perspective};

/// 单次 `select_path_collect` 允许的最大路径步数。
///
/// 正常路径长度受棋盘规模与 MAX_STEPS_PER_EPISODE 约束，远小于该值；
/// 该上限仅用于防御极端情况（如树结构损坏导致的无限循环）。
/// 超限时按当前节点已有 Q 值回传兜底，避免静默丢弃路径。
const MAX_SELECT_STEPS: usize = 512;

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
pub struct PendingEval {
    /// 到达该叶子节点的路径
    pub path: Vec<PathStep>,
    /// 叶子节点对应的游戏环境
    pub env: DarkChessEnv,
    /// 叶子节点的当前玩家
    pub leaf_player: Player,
}

/// Gumbel MCTS 搜索器
///
/// 管理 MCTS 树的构建、搜索和动作选择过程。
/// 泛型 `E` 必须实现 `Evaluator` 特征。
pub struct GumbelMCTS<'a, E: Evaluator> {
    /// Arena 内存池
    pub arena: MctsArena,
    /// 搜索树的根节点在 Arena 中的索引
    pub root_idx: usize,
    /// 状态评估器
    pub(crate) evaluator: &'a E,
    /// 搜索配置
    pub(crate) config: GumbelConfig,
    /// Scratch pad: 用于 Gumbel 采样阶段的临时存储，避免反复堆分配
    /// Vec<(action_index, gumbel_noise_logit)>
    pub(crate) scratch_gumbel: Vec<(usize, f32)>,
    /// 根节点合法动作掩码：在 run() 入口处计算一次，搜索全程不可变。
    /// 作为根节点合法动作的权威来源，直接用于结果返回。
    pub(crate) root_action_mask: Vec<i32>,
    /// 遍历临时缓冲：select_path_collect 中沿路径向下时复用，
    /// 存储当前遍历节点的 action mask。与 root_action_mask 物理隔离。
    pub(crate) traversal_action_mask: Vec<i32>,
    /// 复用的随机数生成器，避免每次搜索/采样重建 thread_rng
    pub(crate) rng: StdRng,
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
        let mut arena = MctsArena::new();
        let state = env.get_state();
        let root_node = MctsNode::new(1.0, 0.0, false, Some(*env), Some(state), true);
        let root_idx = arena.allocate(root_node);

        Self {
            arena,
            root_idx,
            evaluator,
            config,
            scratch_gumbel: Vec::with_capacity(32),
            root_action_mask: vec![0; ACTION_SPACE_SIZE],
            traversal_action_mask: vec![0; ACTION_SPACE_SIZE],
            rng: StdRng::from_entropy(),
        }
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
        let root_node = self.arena.get(self.root_idx);

        // 查找子节点
        let child_idx = root_node
            .children
            .iter()
            .find(|(act, _)| *act == action)
            .map(|(_, idx)| *idx);

        if let Some(idx) = child_idx {
            let child = self.arena.get(idx);
            if child.is_chance_node {
                // 如果是机会节点 (翻牌)，需要根据实际翻出的棋子选择对应的子节点
                let slot = env.get_target_slot(action);
                if let Slot::Revealed(piece) = slot {
                    let outcome_id = get_outcome_id(&piece);
                    if let Some((_, _, next_idx)) = child
                        .possible_states
                        .iter()
                        .find(|(id, _, _)| *id == outcome_id)
                        .map(|x| *x)
                    {
                        self.root_idx = next_idx;
                        let next_node = self.arena.get_mut(next_idx);
                        next_node.is_root_node = true;
                        return;
                    }
                }
            } else {
                // 普通节点，直接移动根节点
                self.root_idx = idx;
                let next_node = self.arena.get_mut(idx);
                next_node.is_root_node = true;
                return;
            }
        }

        // 如果无法重用子树，则重置根节点
        let state = env.get_state();
        let mut new_root = MctsNode::new(1.0, 0.0, false, Some(*env), Some(state), true);
        new_root.is_root_node = true;
        self.root_idx = self.arena.allocate(new_root);
    }

    /// 执行 Gumbel-Top-K 采样
    ///
    /// 从 Logits 中添加 Gumbel 噪声并选择前 K 个动作。
    /// 这是 Gumbel AlphaZero 的核心机制，用于在不进行完全树搜索的情况下选择候选动作。
    /// 使用内部 scratch_gumbel 缓存以避免重复堆分配。
    pub(crate) fn sample_gumbel_top_k(&mut self, logits: &[f32], masks: &[i32], k: usize) -> Vec<usize> {
        let gumbel_dist = Gumbel::new(0.0, 1.0).unwrap();

        // 清空并复用 scratch_gumbel
        self.scratch_gumbel.clear();
        for (i, &logit) in logits.iter().enumerate() {
            if masks[i] == 1 {
                let noise: f64 = gumbel_dist.sample(&mut self.rng);
                self.scratch_gumbel.push((i, logit + noise as f32));
            }
        }

        // 按加噪后的 Logits 降序排序
        self.scratch_gumbel
            .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let actual_k = k.min(self.scratch_gumbel.len());
        self.scratch_gumbel
            .iter()
            .take(actual_k)
            .map(|(i, _)| *i)
            .collect()
    }

    /// 计算补全后的 Q 值 (Completed Q-value)
    ///
    /// 用于在 Sequential Halving 过程中评估动作优劣。
    ///
    /// 规则:
    /// - N > 0 时：使用 W / N
    /// - N = 0 时：使用网络预测的 initial_value，或已访问兄弟子节点的平均 Q
    /// - 根节点不存在该子动作时：返回 0.0（中性）
    pub(crate) fn completed_q(&self, action: usize) -> f32 {
        let root = self.arena.get(self.root_idx);
        if let Some((_, child_idx)) = root
            .children
            .iter()
            .find(|(act, _)| *act == action)
            .map(|(act, idx)| (*act, *idx))
        {
            let child_player = self.arena.get(child_idx).player();
            let q = self.node_q_value(child_idx);
            // 统一到根玩家视角：翻子动作的 child 为机会节点（未执行 step，
            // player == root.player），视角天然一致；移动/炮击动作的 child
            // 已执行 step（player 为对手），Q 符号需取反。
            value_from_perspective(root.player, child_player, q)
        } else {
            0.0
        }
    }

    /// 获取根节点指定动作的 completed_Q
    pub fn get_root_completed_q(&self, action: usize) -> f32 {
        self.completed_q(action)
    }

    /// 根据 Logits 和动作掩码计算概率分布
    pub(crate) fn compute_probs_from_logits(&self, logits: &[f32], masks: &[i32]) -> Vec<f32> {
        let mut probs = vec![0.0; logits.len()];
        let mut max_logit = f32::NEG_INFINITY;

        // 第一遍：找到最大 logit（数值稳定性）
        for (i, &logit) in logits.iter().enumerate() {
            if masks[i] == 1 && logit > max_logit {
                max_logit = logit;
            }
        }

        if !max_logit.is_finite() {
            return probs;
        }

        // 第二遍：计算指数并求和
        let mut sum = 0.0;
        for (i, &logit) in logits.iter().enumerate() {
            if masks[i] == 1 {
                let value = (logit - max_logit).exp();
                probs[i] = value;
                sum += value;
            }
        }

        // 第三遍：归一化
        if sum > 0.0 {
            for p in &mut probs {
                *p /= sum;
            }
        }

        probs
    }

    /// 根据路径获取可变节点引用
    ///
    /// 从根节点开始，沿着 `path` 遍历树，返回目标节点的索引。
    ///
    /// # Panics
    ///
    /// 如果路径中的任何一步在树中不存在，则会 panic。
    /// (在正常逻辑中，路径应该主要来自于树中已存在的节点，或者是刚刚扩展的节点)
    pub(crate) fn get_node_idx_by_path(arena: &MctsArena, mut current_idx: usize, path: &[PathStep]) -> usize {
        for step in path {
            let current = arena.get(current_idx);
            match *step {
                PathStep::Action(action) => {
                    let next_idx = current
                        .children
                        .iter()
                        .find(|(act, _)| *act == action)
                        .map(|(_, idx)| *idx)
                        .expect("Path action not found");
                    current_idx = next_idx;
                }
                PathStep::ChanceOutcome(outcome_id) => {
                    let next_idx = current
                        .possible_states
                        .iter()
                        .find(|(id, _, _)| *id == outcome_id)
                        .map(|(_, _, idx)| *idx)
                        .expect("Path outcome not found");
                    current_idx = next_idx;
                }
            }
        }
        current_idx
    }

    /// 从叶子节点向上回溯更新价值
    ///
    /// 迭代函数，使用 Arena 结构更新路径上所有节点的访问次数和价值总和。
    ///
    /// # 参数
    ///
    /// * `arena` - MCTS 内存池对象
    /// * `node_idx` - 当前节点索引
    /// * `path` - 剩余路径
    /// * `leaf_player` - 叶子节点（评估点）的当前玩家
    /// * `leaf_value` - 叶子节点的评估价值 (相对于 leaf_player)
    ///
    /// # 返回
    ///
    /// 返回从当前节点视角看到的价值 (已根据玩家视角翻转)。
    pub(crate) fn backprop_from_path(
        arena: &mut MctsArena,
        node_idx: usize,
        path: &[PathStep],
        leaf_player: Player,
        leaf_value: f32,
    ) -> f32 {
        if path.is_empty() {
            // 到达目标节点（叶子节点）
            let node = arena.get_mut(node_idx);
            let value = value_from_perspective(node.player(), leaf_player, leaf_value);
            node.visit_count += 1;
            node.value_sum += value;
            return value;
        }

        let first_step = path[0];
        let rest_path = &path[1..];

        // 在分支内一次性取得子节点索引与子玩家，避免事后用
        // get_node_idx_by_path 再次沿路径查找同一子节点。
        let (child_value, child_player) = match first_step {
            PathStep::Action(action) => {
                let current = arena.get(node_idx);
                let child_idx = current
                    .children
                    .iter()
                    .find(|(act, _)| *act == action)
                    .map(|(_, idx)| *idx)
                    .expect("Backprop child not found");
                let child_player = arena.get(child_idx).player();
                let v = Self::backprop_from_path(arena, child_idx, rest_path, leaf_player, leaf_value);
                (v, child_player)
            }
            PathStep::ChanceOutcome(outcome_id) => {
                let current = arena.get(node_idx);
                let child_idx = current
                    .possible_states
                    .iter()
                    .find(|(id, _, _)| *id == outcome_id)
                    .map(|(_, _, idx)| *idx)
                    .expect("Backprop outcome not found");
                let child_player = arena.get(child_idx).player();
                let v = Self::backprop_from_path(arena, child_idx, rest_path, leaf_player, leaf_value);
                (v, child_player)
            }
        };

        // 更新当前节点
        let my_value =
            value_from_perspective(arena.get(node_idx).player(), child_player, child_value);
        let node = arena.get_mut(node_idx);
        node.visit_count += 1;
        node.value_sum += my_value;
        my_value
    }

    /// 获取节点的 Q 值（包含 N=0 的初始化规则）
    fn node_q_value(&self, node_idx: usize) -> f32 {
        let node = self.arena.get(node_idx);
        if node.visit_count > 0 {
            return node.value_sum / node.visit_count as f32;
        }

        // N=0：优先使用已访问子节点的平均值
        let mut sum = 0.0;
        let mut count = 0u32;
        for (_, child_idx) in node.children.iter() {
            let child = self.arena.get(*child_idx);
            if child.visit_count > 0 {
                let child_q = child.value_sum / child.visit_count as f32;
                let adjusted = value_from_perspective(node.player, child.player, child_q);
                sum += adjusted;
                count += 1;
            }
        }

        if count > 0 {
            sum / count as f32
        } else {
            node.initial_value
        }
    }

    /// 根据评估结果构建子节点
    ///
    /// 当一个叶子节点被评估后，使用评估得到的概率 (`probs`) 初始化其子节点。
    /// 只有在 `masks` 中对应位置为 1 的合法动作才会被创建为子节点。
    ///
    /// # 参数
    ///
    /// * `arena` - MCTS 内存池对象
    /// * `node_idx` - 需要扩展的叶子节点的索引
    /// * `env` - 叶子节点对应的环境
    /// * `probs` - 动作概率 (Policy)
    /// * `logits` - 动作 Logits
    pub(crate) fn build_children_from_eval(
        arena: &mut MctsArena,
        node_idx: usize,
        env: &DarkChessEnv,
        probs: &[f32],
        logits: &[f32],
    ) {
        let mut masks = vec![0; ACTION_SPACE_SIZE];
        env.action_masks_into(&mut masks);

        let mut children_to_add = Vec::new();

        for (action_idx, &mask) in masks.iter().enumerate() {
            if mask == 1 {
                let prior = probs[action_idx];
                let logit = logits[action_idx];
                let target_is_hidden = matches!(env.get_target_slot(action_idx), Slot::Hidden);
                let child_env = if target_is_hidden {
                    Some(*env)
                } else {
                    let mut t = *env;
                    let _ = t.step(action_idx, None);
                    Some(t)
                };
                let child_state = child_env.as_ref().map(|e| e.get_state());
                let child_node = MctsNode::new(
                    prior,
                    logit,
                    target_is_hidden,
                    child_env,
                    child_state,
                    false,
                );
                let child_idx = arena.allocate(child_node);
                children_to_add.push((action_idx, child_idx));
            }
        }

        let node = arena.get_mut(node_idx);
        for (action, child_idx) in children_to_add {
            node.children.push((action, child_idx));
        }
        node.is_expanded = true;
    }

    /// 展开机会节点 (Chance Node)
    ///
    /// 机会节点代表环境的随机性（例如翻转盖棋）。
    /// 该方法会列举所有可能的翻棋结果（根据剩余的暗棋），并创建对应的子节点。
    ///
    /// 大语言模型注意: 不要修改此处的全量展开逻辑。
    pub(crate) fn expand_chance_node(arena: &mut MctsArena, node_idx: usize, action: usize) {
        let (env_clone, hidden_pieces) = {
            let env = arena
                .get(node_idx)
                .env
                .as_ref()
                .expect("Chance node must have env")
                .as_ref();
            let mut counts = [0; 14];
            for p in env.get_hidden_pieces_raw() {
                counts[get_outcome_id(p)] += 1;
            }
            (*env, counts)
        };

        let total_hidden = {
            let env = arena
                .get(node_idx)
                .env
                .as_ref()
                .expect("Chance node must have env")
                .as_ref();
            env.get_hidden_pieces_raw().len() as f32
        };

        if total_hidden == 0.0 {
            arena.get_mut(node_idx).is_expanded = true;
            return;
        }

        let mut outcomes_to_add = Vec::new();
        for outcome_id in 0..14 {
            if hidden_pieces[outcome_id] > 0 {
                let prob = hidden_pieces[outcome_id] as f32 / total_hidden;
                let mut next_env = env_clone;
                let hidden_raw = next_env.get_hidden_pieces_raw();
                let specific_piece = *hidden_raw
                    .iter()
                    .find(|p| get_outcome_id(p) == outcome_id)
                    .expect("Piece not found");
                let _ = next_env.step(action, Some(specific_piece));
                let child_state = next_env.get_state();
                let child_node =
                    MctsNode::new(1.0, 0.0, false, Some(next_env), Some(child_state), false);
                let child_idx = arena.allocate(child_node);
                outcomes_to_add.push((outcome_id, prob, child_idx));
            }
        }

        let node = arena.get_mut(node_idx);
        node.is_expanded = true;
        for (outcome_id, prob, child_idx) in outcomes_to_add {
            node.possible_states.push((outcome_id, prob, child_idx));
        }
    }

    /// 从机会节点的可能结果中采样
    ///
    /// 根据各种结果的概率分布，随机采样一个结果 ID。
    /// 主要用于模拟阶段，决定在机会节点走向哪个分支。
    fn sample_outcome_id(outcomes: &[(usize, f32, usize)], rng: &mut impl Rng) -> Option<usize> {
        if outcomes.is_empty() {
            return None;
        }
        let total: f32 = outcomes.iter().map(|(_, p, _)| p).sum();
        if total <= 0.0 {
            return outcomes.first().map(|(id, _, _)| *id);
        }
        let mut pick = rng.gen_range(0.0..1.0) * total;
        for (outcome_id, prob, _) in outcomes {
            pick -= *prob;
            if pick <= 0.0 {
                return Some(*outcome_id);
            }
        }
        outcomes.first().map(|(id, _, _)| *id)
    }

    /// 选择路径并收集待评估项
    ///
    /// 从根节点的特定动作出发，执行模拟直到到达叶子节点或游戏结束。
    /// 如果到达未扩展的节点，将其加入 `batch` 等待后续评估。
    ///
    /// 模拟过程中使用 PUCT 公式 (Predictor + Upper Confidence Bound applied to Trees) 选择动作：
    /// Score = Q(s, a) + U(s, a)
    /// U(s, a) = c_puct * P(s, a) * sqrt(N(parent)) / (1 + N(child))
    pub(crate) fn select_path_collect(
        &mut self,
        action: usize,
        batch: &mut Vec<PendingEval>,
    ) -> SelectPathOutcome {
        let mut path = vec![PathStep::Action(action)];
        let current_idx = {
            let root = self.arena.get(self.root_idx);
            root.children
                .iter()
                .find(|(act, _)| *act == action)
                .map(|(_, idx)| *idx)
        };

        if current_idx.is_none() {
            eprintln!(
                "⚠️ MCTS: select_path_collect 根节点缺少候选动作 {} 的子节点",
                action
            );
            return SelectPathOutcome::EarlyReturn;
        }
        let mut current_idx = current_idx.unwrap();
        let mut current_action = action;
        let mut steps_taken = 0;

        loop {
            steps_taken += 1;
            if steps_taken > MAX_SELECT_STEPS {
                // 步数上限兜底：不再深入，按当前节点已有 Q 值回传，
                // 保证本次调用要么产出 batch、要么产生回传，不静默丢弃。
                let leaf_player = self.arena.get(current_idx).player();
                let leaf_value = self.node_q_value(current_idx);
                let path_clone = path.clone();
                Self::backprop_from_path(
                    &mut self.arena,
                    self.root_idx,
                    &path_clone,
                    leaf_player,
                    leaf_value,
                );
                return SelectPathOutcome::EarlyReturn;
            }

            let is_chance = self.arena.get(current_idx).is_chance_node;

            if is_chance {
                if !self.arena.get(current_idx).is_expanded {
                    Self::expand_chance_node(&mut self.arena, current_idx, current_action);
                    let possible_states = self.arena.get(current_idx).possible_states.clone();

                    if possible_states.is_empty() {
                        eprintln!(
                            "⚠️ MCTS: chance 节点展开后无可选结果 (node={}, action={})",
                            current_idx, current_action
                        );
                        return SelectPathOutcome::EarlyReturn;
                    }

                    let base_path = path.clone();
                    for (outcome_id, _, child_idx) in possible_states.iter() {
                        let child_env = *self
                            .arena
                            .get(*child_idx)
                            .env
                            .as_ref()
                            .expect("Chance outcome must have env")
                            .as_ref();
                        let mut outcome_path = base_path.clone();
                        outcome_path.push(PathStep::ChanceOutcome(*outcome_id));
                        let leaf_player = self.arena.get(*child_idx).player();
                        batch.push(PendingEval {
                            path: outcome_path,
                            env: child_env,
                            leaf_player,
                        });
                    }
                    return SelectPathOutcome::Normal;
                }

                let possible_states = self.arena.get(current_idx).possible_states.clone();
                let outcome_id = match Self::sample_outcome_id(&possible_states, &mut self.rng) {
                    Some(id) => id,
                    None => {
                        eprintln!(
                            "⚠️ MCTS: 已展开 chance 节点无结果可采样 (node={}, action={})",
                            current_idx, current_action
                        );
                        return SelectPathOutcome::EarlyReturn;
                    }
                };
                path.push(PathStep::ChanceOutcome(outcome_id));
                let next_idx = possible_states
                    .iter()
                    .find(|(id, _, _)| *id == outcome_id)
                    .map(|(_, _, idx)| *idx)
                    .expect("Outcome not found");
                current_idx = next_idx;
                continue;
            }

            let env = self
                .arena
                .get(current_idx)
                .env
                .as_ref()
                .expect("Node must have env")
                .as_ref();

            // 使用遍历缓冲 action_mask
            self.traversal_action_mask.iter_mut().for_each(|m| *m = 0);
            env.action_masks_into(&mut self.traversal_action_mask);

            // 终局检测：优先使用节点缓存的 is_terminal（覆盖分数归零/全灭/
            // 无合法动作/连续无吃子判和/步数截断），按真实胜负回传；
            // action_mask 全 0 作为兜底（理论上已包含在 is_terminal 中）。
            if self.arena.get(current_idx).is_terminal
                || self.traversal_action_mask.iter().all(|&x| x == 0)
            {
                let leaf_player = self.arena.get(current_idx).player();
                let (_, _, winner) = env.check_game_over_conditions();
                let leaf_value = match winner {
                    Some(w) if w == leaf_player.val() => 1.0,
                    Some(w) if w == leaf_player.opposite().val() => -1.0,
                    _ => 0.0, // 平局 (Some(0)) 或 winner=None
                };
                let path_clone = path.clone();
                Self::backprop_from_path(
                    &mut self.arena,
                    self.root_idx,
                    &path_clone,
                    leaf_player,
                    leaf_value,
                );
                return SelectPathOutcome::TerminalBackprop;
            }

            if !self.arena.get(current_idx).is_expanded {
                let leaf_player = self.arena.get(current_idx).player();
                batch.push(PendingEval {
                    path,
                    env: *env,
                    leaf_player,
                });
                return SelectPathOutcome::Normal;
            }

            let current = self.arena.get(current_idx);
            let sqrt_total = (current.visit_count as f32).sqrt();
            let parent_player = current.player();
            let children_clone = current.children.clone();

            let mut best_action = None;
            let mut best_score = f32::NEG_INFINITY;

            let puct_coeff = self.config.c_scale.max(0.1);
            for (act, child_idx) in children_clone.iter() {
                let child = self.arena.get(*child_idx);
                let child_q = self.node_q_value(*child_idx);
                let child_player = child.player();
                let adjusted_q = value_from_perspective(parent_player, child_player, child_q);
                let u_score = puct_coeff * child.prior * sqrt_total / (1.0 + child.visit_count as f32);
                let score = adjusted_q + u_score;
                if score > best_score {
                    best_score = score;
                    best_action = Some(*act);
                }
            }

            let act = match best_action {
                Some(a) => a,
                None => {
                    eprintln!(
                        "⚠️ MCTS: 已展开节点无可选子节点 (node={})",
                        current_idx
                    );
                    return SelectPathOutcome::EarlyReturn;
                }
            };
            path.push(PathStep::Action(act));
            current_action = act;
            let next_idx = children_clone
                .iter()
                .find(|(a, _)| *a == act)
                .map(|(_, idx)| *idx)
                .expect("Selected child missing");
            current_idx = next_idx;
        }
    }

    /// 展开根节点
    ///
    /// 在搜索开始前，确保根节点已经被评估和扩展。
    pub(crate) fn expand_root(&mut self) {
        let is_expanded = self.arena.get(self.root_idx).is_expanded;
        if is_expanded {
            return;
        }

        let env = *self
            .arena
            .get(self.root_idx)
            .env
            .as_ref()
            .expect("Root must have env")
            .as_ref();
        let (logits_batch, values) = self.evaluator.evaluate(std::slice::from_ref(&env));
        let logits = &logits_batch[0];
        let value = values[0];

        let mut masks = vec![0; ACTION_SPACE_SIZE];
        env.action_masks_into(&mut masks);
        let probs = self.compute_probs_from_logits(logits, &masks);

        Self::build_children_from_eval(&mut self.arena, self.root_idx, &env, &probs, logits);

        let root = self.arena.get_mut(self.root_idx);
        root.initial_value = value;
        root.visit_count += 1;
        root.value_sum += value;
    }

    /// 执行 Gumbel MCTS 搜索主循环
    ///
    /// 1. 扩展根节点。
    /// 2. 收集根节点 Logits 并进行 Gumbel Top-K 采样，选出候选动作。
    /// 3. 使用 Sequential Halving 算法，分阶段分配搜索预算，淘汰表现不佳的候选动作。
    /// 4. 最终返回搜索结果，包含选择的动作和所有相关数据。
    ///
    /// # 返回
    ///
    /// * `Option<MctsSearchResult>` - 如果是 None 表示无合法动作；否则返回完整的搜索结果。
    pub fn run(&mut self) -> Option<MctsSearchResult> {
        // 1. 扩展根节点
        self.expand_root();

        let env = self
            .arena
            .get(self.root_idx)
            .env
            .as_ref()
            .expect("Root must have env")
            .as_ref();
        self.root_action_mask.iter_mut().for_each(|m| *m = 0);
        env.action_masks_into(&mut self.root_action_mask);

        if self.root_action_mask.iter().all(|&x| x == 0) {
            return None;
        }

        // 2. 收集 logits
        let logits: Vec<f32> = (0..ACTION_SPACE_SIZE)
            .map(|i| {
                let root = self.arena.get(self.root_idx);
                root.children
                    .iter()
                    .find(|(act, _)| *act == i)
                    .map(|(_, idx)| self.arena.get(*idx).logit)
                    .unwrap_or(-1e6)
            })
            .collect();

        // 3. Gumbel-Top-K 采样 (克隆 mask 以避免借用冲突)
        let masks_cloned = self.root_action_mask.clone();
        let candidates =
            self.sample_gumbel_top_k(&logits, &masks_cloned, self.config.max_considered_actions);
        if candidates.is_empty() {
            return None;
        }
        if candidates.len() == 1 {
            // 只有一个候选动作，直接返回
            let action = candidates[0];
            let root = self.arena.get(self.root_idx);
            let state = root.state.clone()?;
            let player = root.player;
            let improved_policy = self.get_improved_policy();
            let mcts_value = root.q_value();
            let completed_q = self.completed_q(action);
            let root_visit_count = root.visit_count;
            let action_mask = self.root_action_mask.clone();

            return Some(MctsSearchResult {
                action,
                state,
                improved_policy,
                mcts_value,
                completed_q,
                root_visit_count,
                player,
                action_mask,
            });
        }

        // 4. Sequential Halving - 使用新的预算分配器
        let mut budget = SequentialHalvingBudget::new(
            candidates.len(),
            self.config.num_simulations,
            2, // eta = 2，表示每阶段淘汰 50% 的动作
        );

        let mut remaining = candidates;

        for phase in 0..budget.num_phases() {
            if remaining.len() <= 1 {
                break;
            }

            let visits_per_action = budget.visits_per_action_in_phase(phase);

            // 执行本阶段的搜索
            let mut total_phase_usage = 0;
            // 本阶段内所有候选路径中，命中终局节点静默回传的次数（供空转告警诊断）
            let mut terminal_hits = 0;
            for _ in 0..visits_per_action {
                let mut batch: Vec<PendingEval> = Vec::new();
                for &action in &remaining {
                    if self.select_path_collect(action, &mut batch)
                        == SelectPathOutcome::TerminalBackprop
                    {
                        terminal_hits += 1;
                    }
                }

                if !batch.is_empty() {
                    total_phase_usage += batch.len();
                    let envs: Vec<DarkChessEnv> =
                        batch.iter().map(|pending| pending.env).collect();
                    let (logits_batch, values) = self.evaluator.evaluate(&envs);

                    for (idx, pending) in batch.into_iter().enumerate() {
                        let logits = &logits_batch[idx];
                        let value = values[idx];
                        let mut masks = vec![0; ACTION_SPACE_SIZE];
                        pending.env.action_masks_into(&mut masks);
                        let probs = self.compute_probs_from_logits(logits, &masks);
                        let leaf_idx =
                            Self::get_node_idx_by_path(&self.arena, self.root_idx, &pending.path);
                        {
                            let leaf = self.arena.get_mut(leaf_idx);
                            leaf.initial_value = value;
                        }
                        Self::build_children_from_eval(
                            &mut self.arena,
                            leaf_idx,
                            &pending.env,
                            &probs,
                            logits,
                        );
                        Self::backprop_from_path(
                            &mut self.arena,
                            self.root_idx,
                            &pending.path,
                            pending.leaf_player,
                            value,
                        );
                    }
                }
            }

            budget.record_phase_usage(total_phase_usage);

            // 空转防护：本阶段没有任何模拟产生（候选动作子树全部静默早退，
            // 或 visits_per_action 为 0），继续按陈旧 completed_Q 剪枝没有意义，
            // 提前终止搜索循环，直接返回当前剩余候选。
            if total_phase_usage == 0 {
                // visits_per_action == 0：预算排程自然耗尽（remaining_budget <
                // num_actions），属正常退出，静默 break 不打印告警。
                if visits_per_action == 0 {
                    break;
                }
                // 有预算却无产出：候选路径全部静默早退。若终局回传占满全部调用，
                // 说明子树已全部命中终局（如接近判和/截断阈值的局面），退化但良性。
                let total_calls = visits_per_action * remaining.len();
                eprintln!(
                    "⚠️ MCTS: phase {} 实际模拟数为 0 (visits_per_action={}, remaining={}, 终局回传 {}/{})，提前终止搜索",
                    phase,
                    visits_per_action,
                    remaining.len(),
                    terminal_hits,
                    total_calls
                );
                break;
            }

            // 根据 completed_Q 排序并淘汰
            if remaining.len() > 1 {
                let mut scored: Vec<(usize, f32)> = remaining
                    .iter()
                    .map(|&a| (a, self.completed_q(a)))
                    .collect();
                scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

                let keep_count = budget.keep_count_after_phase();
                remaining = scored
                    .into_iter()
                    .take(keep_count)
                    .map(|(a, _)| a)
                    .collect();
            }

            budget.advance_phase();
        }

        // 5. 返回结果
        let action = if remaining.is_empty() {
            let root = self.arena.get(self.root_idx);
            root.children
                .iter()
                .max_by_key(|(_, child_idx)| self.arena.get(*child_idx).visit_count)
                .map(|(action, _)| *action)?
        } else {
            remaining[0]
        };

        // 6. 收集所有数据并返回
        let root = self.arena.get(self.root_idx);
        let state = root.state.clone()?;
        let player = root.player;
        let improved_policy = self.get_improved_policy();
        let mcts_value = root.q_value();
        let completed_q = self.completed_q(action);
        let root_visit_count = root.visit_count;
        let action_mask = self.root_action_mask.clone();

        Some(MctsSearchResult {
            action,
            state,
            improved_policy,
            mcts_value,
            completed_q,
            root_visit_count,
            player,
            action_mask,
        })
    }

    /// 获取根节点的访问概率分布
    ///
    /// 返回基于访问次数归一化的概率分布，可用于训练策略网络。已弃用，建议使用 `get_improved_policy` 获取 Gumbel AlphaZero 的改进策略。
    pub fn get_root_probabilities(&self) -> Vec<f32> {
        let mut probs = vec![0.0; ACTION_SPACE_SIZE];
        let root = self.arena.get(self.root_idx);
        let total = root.visit_count as f32;
        if total == 0.0 {
            return probs;
        }
        for (action, child_idx) in &root.children {
            let child = self.arena.get(*child_idx);
            if *action < probs.len() {
                probs[*action] = child.visit_count as f32 / total;
            }
        }
        probs
    }

    // ========================================================================
    // 注意：根节点 Dirichlet 噪声注入已被移除，请勿重新添加！
    //
    // 原因：本项目使用 Gumbel AlphaZero（Gumbel 论文探索方案），探索由以下
    // 机制提供，根节点先验 prior 不参与任何搜索决策：
    //   1. Gumbel Top-K 采样（sample_gumbel_top_k）使用子节点的 logit；
    //   2. Sequential Halving 淘汰基于 completed_q（Q 值）；
    //   3. 根节点第一跳由候选动作直接指定，不经过根节点 PUCT；
    //   4. 训练目标 get_improved_policy 使用 logit + σ·Q。
    // 因此修改根节点子节点的 prior（Dirichlet 注入）在搜索中是无效的空转，
    // 既不能提供探索，也不影响训练目标。
    // ========================================================================

    /// 基于根节点 completed Q 的温度策略（Gumbel AlphaZero 论文标准动作选择）
    ///
    /// π(a) ∝ exp(Q_comp(a) / τ)
    /// - τ = 1: 对 completed Q 做 softmax，鼓励探索
    /// - τ → 0: 趋向 argmax，确定性选择
    /// 仅对合法动作计算，非法动作保持 0。
    ///
    /// 注意：此处刻意使用 completed Q 而非访问计数 N^(1/τ)（经典 AlphaZero 做法）。
    /// Sequential Halving 结束后 surviving 候选的访问次数基本均分，基于 N 的策略
    /// 会退化为近似均匀采样、丢失动作质量信息；而 completed Q 保留了质量排序，
    /// 符合 Gumbel AlphaZero 论文（Policy improvement by planning with Gumbel）
    /// 的动作选择方式。请勿替换回基于 visit_count 的实现。
    pub fn get_root_completed_q_policy(&self, temperature: f32) -> Vec<f32> {
        let mut policy = vec![0.0; ACTION_SPACE_SIZE];

        let env = match self.arena.get(self.root_idx).env.as_ref() {
            Some(env) => env.as_ref(),
            None => return policy,
        };
        let mut masks = vec![0; ACTION_SPACE_SIZE];
        env.action_masks_into(&mut masks);

        let tau = temperature.max(1e-4);
        let inv_tau = 1.0 / tau;

        // 数值稳定性：先减去最大 completed Q 再做 exp，避免溢出
        let mut max_q = f32::NEG_INFINITY;
        for action in 0..ACTION_SPACE_SIZE {
            if masks[action] == 1 {
                max_q = max_q.max(self.completed_q(action));
            }
        }
        if !max_q.is_finite() {
            return policy;
        }

        let mut sum = 0.0;
        for action in 0..ACTION_SPACE_SIZE {
            if masks[action] == 1 {
                let value = ((self.completed_q(action) - max_q) * inv_tau).exp();
                policy[action] = value;
                sum += value;
            }
        }

        if sum > 0.0 {
            for p in policy.iter_mut() {
                *p /= sum;
            }
        }
        policy
    }

    /// 从离散概率分布中采样一个动作（仅合法动作）
    pub fn sample_action_from_policy(probs: &[f32], masks: &[i32]) -> usize {
        let mut rng = thread_rng();
        let mut sum = 0.0;
        for i in 0..probs.len() {
            if masks[i] == 1 {
                sum += probs[i];
            }
        }
        if sum <= 0.0 {
            for i in 0..masks.len() {
                if masks[i] == 1 {
                    return i;
                }
            }
            return 0;
        }
        let mut r: f32 = rng.gen_range(0.0..sum);
        for i in 0..probs.len() {
            if masks[i] == 1 {
                r -= probs[i];
                if r <= 0.0 {
                    return i;
                }
            }
        }
        for i in (0..probs.len()).rev() {
            if masks[i] == 1 {
                return i;
            }
        }
        0
    }

    /// 获取 Gumbel AlphaZero 的改进策略 (pi_target)
    ///
    /// 使用 root 的先验 logit 与 completed_Q 直接组合，计算 softmax 概率。
    pub fn get_improved_policy(&self) -> Vec<f32> {
        let mut policy = vec![0.0; ACTION_SPACE_SIZE];
        let env = match self.arena.get(self.root_idx).env.as_ref() {
            Some(env) => env.as_ref(),
            None => return policy,
        };

        let mut masks = vec![0; ACTION_SPACE_SIZE];
        env.action_masks_into(&mut masks);

        // 1. 计算打分: logit + sigma * completed_q
        // sigma = c_scale * ln(1 + N_root)  —— 按 Gumbel AlphaZero 论文
        let root = self.arena.get(self.root_idx);
        let root_visit_count = root.visit_count as f32;
        let sigma_scale = self.config.c_scale * (1.0 + root_visit_count).ln();

        let mut scores = vec![f32::NEG_INFINITY; ACTION_SPACE_SIZE];
        let mut max_score = f32::NEG_INFINITY;

        for action in 0..ACTION_SPACE_SIZE {
            if masks[action] != 1 {
                continue;
            }
            let child_idx = match root
                .children
                .iter()
                .find(|(act, _)| *act == action)
                .map(|(_, idx)| *idx)
            {
                Some(idx) => idx,
                None => continue,
            };
            let child = self.arena.get(child_idx);
            let q = self.completed_q(action);
            let score = child.logit + sigma_scale * q;
            scores[action] = score;
            if score > max_score {
                max_score = score;
            }
        }

        // 3. 计算 Softmax（带数值稳定性）
        if !max_score.is_finite() {
            return policy;
        }

        let mut sum = 0.0;
        for action in 0..ACTION_SPACE_SIZE {
            let score = scores[action];
            if score.is_finite() {
                let value = (score - max_score).exp();
                policy[action] = value;
                sum += value;
            }
        }

        // 4. 归一化概率，异常时回退到均匀分布
        if sum > 0.0 {
            for p in policy.iter_mut() {
                *p /= sum;
            }
        } else {
            let count = masks.iter().sum::<i32>() as f32;
            if count > 0.0 {
                for i in 0..ACTION_SPACE_SIZE {
                    if masks[i] == 1 {
                        policy[i] = 1.0 / count;
                    }
                }
            }
        }

        policy
    }
}
