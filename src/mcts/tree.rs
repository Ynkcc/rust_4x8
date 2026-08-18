// src/mcts/tree.rs
// 树构建与价值回溯层：节点扩展、机会节点展开、价值回传（泛型化：G = 游戏环境）
//
// 分层说明：
// - 本模块只负责 MCTS 树的「构建 / 回溯」操作（静态方法为主，不依赖搜索器状态）；
// - 搜索主循环与路径选择见 search.rs，策略计算见 policy.rs。

use super::evaluator::Evaluator;
use super::node::{value_from_perspective, MctsArena, MctsNode};
use super::path::PathStep;
use super::search::GumbelMCTS;
use crate::game_env::GameEnv;
use crate::Player;

impl<'a, G: GameEnv, E: Evaluator<G>> GumbelMCTS<'a, G, E> {
    /// 根据路径获取可变节点引用
    ///
    /// 从根节点开始，沿着 `path` 遍历树，返回目标节点的索引。
    ///
    /// # Panics
    ///
    /// 如果路径中的任何一步在树中不存在，则会 panic。
    /// (在正常逻辑中，路径应该主要来自于树中已存在的节点，或者是刚刚扩展的节点)
    pub(crate) fn get_node_idx_by_path(
        arena: &MctsArena<G>,
        mut current_idx: usize,
        path: &[PathStep],
    ) -> usize {
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
        arena: &mut MctsArena<G>,
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
    pub(crate) fn node_q_value(&self, node_idx: usize) -> f32 {
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
    /// * `parent_value` - 父节点（`node_idx`）的评估值（从父节点玩家视角）
    ///
    /// 子节点的 `initial_value` 继承父节点评估值（从子节点玩家视角换算），
    /// 即 Gumbel AlphaZero 标准的 Q(s,a) ≈ V(s) 先验。没有该先验时，
    /// 未访问子节点（N=0）的 completed_q 恒为 0，Sequential Halving 在
    /// 浅搜索下无法区分候选，可能淘汰最优动作。
    pub(crate) fn build_children_from_eval(
        arena: &mut MctsArena<G>,
        node_idx: usize,
        env: &G,
        probs: &[f32],
        logits: &[f32],
        parent_value: f32,
    ) {
        let mut masks = vec![0; G::action_space_size()];
        env.action_masks_into(&mut masks);
        let parent_player = arena.get(node_idx).player();

        let mut children_to_add = Vec::new();

        for (action_idx, &mask) in masks.iter().enumerate() {
            if mask == 1 {
                let prior = probs[action_idx];
                let logit = logits[action_idx];
                // 机会动作（暗棋翻牌）不执行 step，直接以当前环境作为机会节点；
                // 常规动作执行 step 得到后继环境。
                let is_chance = env.is_chance_action(action_idx);
                let child_env = if is_chance {
                    Some(*env)
                } else {
                    let mut t = *env;
                    let _ = t.step(action_idx);
                    Some(t)
                };
                let child_state = child_env.as_ref().map(|e| e.get_state());
                let mut child_node = MctsNode::new(
                    prior,
                    logit,
                    is_chance,
                    child_env,
                    child_state,
                    false,
                );
                // Q 先验：机会节点子玩家=父玩家，原样；常规节点子玩家=对手，取反。
                let child_player = child_node.player();
                child_node.initial_value =
                    value_from_perspective(child_player, parent_player, parent_value);
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
    /// 该方法枚举该机会动作的所有可能结果（委托给 `GameEnv::chance_outcomes`），
    /// 并为每个结果创建对应的子节点。
    ///
    /// 大语言模型注意: 不要修改此处的全量展开逻辑。
    pub(crate) fn expand_chance_node(arena: &mut MctsArena<G>, node_idx: usize, action: usize) {
        let env = *arena
            .get(node_idx)
            .env
            .as_ref()
            .expect("Chance node must have env");
        let outcomes = env.chance_outcomes(action);

        if outcomes.is_empty() {
            arena.get_mut(node_idx).is_expanded = true;
            return;
        }

        let mut outcomes_to_add = Vec::new();
        for (outcome_id, prob, next_env) in outcomes {
            let child_state = next_env.get_state();
            let child_node =
                MctsNode::new(1.0, 0.0, false, Some(next_env), Some(child_state), false);
            let child_idx = arena.allocate(child_node);
            outcomes_to_add.push((outcome_id, prob, child_idx));
        }

        let node = arena.get_mut(node_idx);
        node.is_expanded = true;
        for (outcome_id, prob, child_idx) in outcomes_to_add {
            node.possible_states.push((outcome_id, prob, child_idx));
        }
    }
}
