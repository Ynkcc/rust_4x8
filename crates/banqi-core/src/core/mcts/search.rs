// src/mcts/search.rs
// Gumbel AlphaZero MCTS 核心搜索与采样算法（泛型化：G = 游戏环境）
//
// 分层说明：
// - 本文件保留「搜索器结构体 + 主搜索循环（run / select_path_collect / expand_root）」；
// - 路径类型定义见 path.rs，Gumbel 采样见 sampling.rs，
//   树构建与价值回溯见 tree.rs，策略计算见 policy.rs。

use crate::core::env::GameEnv;

use super::budget::SequentialHalvingBudget;
use super::config::{GumbelConfig, MctsSearchResult};
use super::evaluator::Evaluator;
use super::node::{value_from_perspective, MctsArena, MctsNode};
use super::path::{ChanceSeed, PathStep, PendingEval, SelectPathOutcome};
use rand::prelude::*;

/// 单次 `select_path_collect` 允许的最大路径步数。
///
/// 正常路径长度受棋盘规模与 MAX_STEPS_PER_EPISODE 约束，远小于该值；
/// 该上限仅用于防御极端情况（如树结构损坏导致的无限循环）。
/// 超限时按当前节点已有 Q 值回传兜底，避免静默丢弃路径。
const MAX_SELECT_STEPS: usize = 512;

/// Gumbel MCTS 搜索器
///
/// 管理 MCTS 树的构建、搜索和动作选择过程。
/// 泛型 `G` 为游戏环境（须实现 `GameEnv`），泛型 `E` 必须实现 `Evaluator<G>`。
pub struct GumbelMCTS<'a, G: GameEnv, E: Evaluator<G>> {
    /// Arena 内存池
    pub arena: MctsArena<G>,
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

impl<'a, G: GameEnv, E: Evaluator<G>> GumbelMCTS<'a, G, E> {
    /// 创建一个新的 GumbelMCTS 实例
    ///
    /// 初始化根节点并准备搜索。
    ///
    /// # 参数
    ///
    /// * `env` - 初始游戏环境
    /// * `evaluator` - 状态评估器
    /// * `config` - 搜索配置
    pub fn new(env: &G, evaluator: &'a E, config: GumbelConfig) -> Self {
        let mut arena = MctsArena::new();
        let state = env.get_resnet_state();
        let root_node = MctsNode::new(1.0, 0.0, false, Some(*env), Some(state), true);
        let root_idx = arena.allocate(root_node);

        Self {
            arena,
            root_idx,
            evaluator,
            config,
            scratch_gumbel: Vec::with_capacity(32),
            root_action_mask: vec![0; G::action_space_size()],
            traversal_action_mask: vec![0; G::action_space_size()],
            rng: StdRng::from_entropy(),
        }
    }

    /// 当前根节点持有的环境引用（终局血量差等终局信息使用）。
    pub fn root_env(&self) -> Option<&G> {
        self.arena.get(self.root_idx).env.as_ref()
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
    pub fn step_next(&mut self, env: &G, action: usize) {
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
                if let Some(outcome_id) = env.step_outcome_id(action) {
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
        let state = env.get_resnet_state();
        let mut new_root = MctsNode::new(1.0, 0.0, false, Some(*env), Some(state), true);
        new_root.is_root_node = true;
        self.root_idx = self.arena.allocate(new_root);
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

    /// 计算根节点指定动作的补全复合效用（completed utility），用于 Sequential
    /// Halving 淘汰。与 `completed_q`（纯胜率，作为训练目标）解耦：health_enabled
    /// 时并入血量期望，否则与 `completed_q` 一致。
    pub(crate) fn completed_utility(&self, action: usize) -> f32 {
        let root = self.arena.get(self.root_idx);
        if let Some((_, child_idx)) = root
            .children
            .iter()
            .find(|(act, _)| *act == action)
            .map(|(act, idx)| (*act, *idx))
        {
            let child_player = self.arena.get(child_idx).player();
            let u = self.node_utility_value(child_idx);
            value_from_perspective(root.player, child_player, u)
        } else {
            0.0
        }
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
        batch: &mut Vec<PendingEval<G>>,
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
                let leaf_health = self.node_health_value(current_idx);
                let path_clone = path.clone();
                Self::backprop_from_path(
                    &mut self.arena,
                    self.root_idx,
                    &path_clone,
                    leaf_player,
                    leaf_value,
                    leaf_health,
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
                    for (outcome_id, prob, child_idx) in possible_states.iter() {
                        let child_env = *self
                            .arena
                            .get(*child_idx)
                            .env
                            .as_ref()
                            .expect("Chance outcome must have env");
                        let mut outcome_path = base_path.clone();
                        outcome_path.push(PathStep::ChanceOutcome(*outcome_id));
                        let leaf_player = self.arena.get(*child_idx).player();
                        batch.push(PendingEval {
                            path: outcome_path,
                            env: child_env,
                            leaf_player,
                            chance_seed: Some(ChanceSeed {
                                chance_idx: current_idx,
                                prob: *prob,
                                prefix_len: base_path.len(),
                            }),
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
                .expect("Node must have env");

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
                // 终局血量期望：整型血量差（红方视角）转到 leaf_player 视角后按 D 归一化。
                let leaf_health = if self.config.health_enabled {
                    match (env.terminal_health_diff_red_int(), env.health_diff_scale()) {
                        (Some(d), s) if s > 0.0 => {
                            let v = if leaf_player.val() == 1 {
                                d as f32
                            } else {
                                -(d as f32)
                            };
                            (v / s).clamp(-1.0, 1.0)
                        }
                        _ => 0.0,
                    }
                } else {
                    0.0
                };
                let path_clone = path.clone();
                Self::backprop_from_path(
                    &mut self.arena,
                    self.root_idx,
                    &path_clone,
                    leaf_player,
                    leaf_value,
                    leaf_health,
                );
                return SelectPathOutcome::TerminalBackprop;
            }

            if !self.arena.get(current_idx).is_expanded {
                let leaf_player = self.arena.get(current_idx).player();
                batch.push(PendingEval {
                    path,
                    env: *env,
                    leaf_player,
                    chance_seed: None,
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
                // 复合效用：health_enabled 时并入血量期望，否则退化为纯胜率 Q。
                let child_utility = self.node_utility_value(*child_idx);
                let child_player = child.player();
                let adjusted_q = value_from_perspective(parent_player, child_player, child_utility);
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
            .expect("Root must have env");
        let out = self.evaluator.evaluate(std::slice::from_ref(&env));
        let logits = &out.logits[0];
        let value = out.values[0];
        let health_mu = if self.config.health_enabled {
            out.health_expectation(0).unwrap_or(0.0)
        } else {
            0.0
        };

        let mut masks = vec![0; G::action_space_size()];
        env.action_masks_into(&mut masks);
        let probs = self.compute_probs_from_logits(logits, &masks);

        Self::build_children_from_eval(
            &mut self.arena,
            self.root_idx,
            &env,
            &probs,
            logits,
            value,
            health_mu,
        );

        let root = self.arena.get_mut(self.root_idx);
        root.initial_value = value;
        root.initial_health = health_mu;
        root.visit_count += 1;
        root.value_sum += value;
        root.health_sum += health_mu;
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
            .expect("Root must have env");
        self.root_action_mask.iter_mut().for_each(|m| *m = 0);
        env.action_masks_into(&mut self.root_action_mask);

        if self.root_action_mask.iter().all(|&x| x == 0) {
            return None;
        }

        // 2. 收集 logits
        let logits: Vec<f32> = (0..G::action_space_size())
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
                let mut batch: Vec<PendingEval<G>> = Vec::new();
                // 预算按模拟次数计量：一次 select_path_collect 调用 = 1 次模拟。
                // 机会节点展开爆发的 N 路批量评估真实发生，但不占用模拟预算
                // （否则 chance 子树密集的候选会窃取 Sequential Halving 后续阶段的预算）。
                let mut eval_calls = 0;
                for &action in &remaining {
                    match self.select_path_collect(action, &mut batch) {
                        SelectPathOutcome::TerminalBackprop => terminal_hits += 1,
                        SelectPathOutcome::Normal => eval_calls += 1,
                        SelectPathOutcome::EarlyReturn => {}
                    }
                }

                if !batch.is_empty() {
                    total_phase_usage += eval_calls;
                    let envs: Vec<G> = batch.iter().map(|pending| pending.env).collect();
                    let out = self.evaluator.evaluate(&envs);

                    let mut eval_values: Vec<(f32, f32)> = Vec::with_capacity(batch.len());
                    for (idx, pending) in batch.iter().enumerate() {
                        let logits = &out.logits[idx];
                        let value = out.values[idx];
                        let health_mu = if self.config.health_enabled {
                            out.health_expectation(idx).unwrap_or(0.0)
                        } else {
                            0.0
                        };
                        let mut masks = vec![0; G::action_space_size()];
                        pending.env.action_masks_into(&mut masks);
                        let probs = self.compute_probs_from_logits(logits, &masks);
                        let leaf_idx =
                            Self::get_node_idx_by_path(&self.arena, self.root_idx, &pending.path);
                        {
                            let leaf = self.arena.get_mut(leaf_idx);
                            leaf.initial_value = value;
                            leaf.initial_health = health_mu;
                        }
                        Self::build_children_from_eval(
                            &mut self.arena,
                            leaf_idx,
                            &pending.env,
                            &probs,
                            logits,
                            value,
                            health_mu,
                        );
                        eval_values.push((value, health_mu));
                    }
                    let evals: Vec<(&PendingEval<G>, f32, f32)> = batch
                        .iter()
                        .zip(eval_values)
                        .map(|(pending, (v, h))| (pending, v, h))
                        .collect();
                    Self::backprop_evals(&mut self.arena, self.root_idx, &evals);
                }
            }

            budget.record_phase_usage(total_phase_usage);

            // 空转防护：本阶段没有任何模拟产生（候选动作子树全部静默早退，
            // 或 visits_per_action 为 0）。
            if total_phase_usage == 0 {
                // visits_per_action == 0：预算排程自然耗尽（remaining_budget <
                // num_actions），属正常退出，静默 break 不打印告警。
                if visits_per_action == 0 {
                    break;
                }
                // 有预算却无产出：候选路径全部静默早退。若终局回传占满全部调用，
                // 说明子树已全部命中终局（如接近判和/截断阈值的局面），属退化但
                // 良性，静默继续不打印告警，避免训练循环刷屏。
                //
                // 注意：**不在此处 break**。若直接提前退出，剩余候选未按 completed_Q
                // 淘汰，最终会退回 remaining[0] = Gumbel 噪声采样的随机候选。
                // 这在确定性浅游戏（如井字棋）树复用后尤其常见：树完全展开后
                // 所有模拟都命中终局回传，必须仍基于现有 completed_Q 淘汰，
                // 才能选出最优动作。
                let total_calls = visits_per_action * remaining.len();
                // 仅当存在意外静默早退（终局回传未占满全部调用）时打印告警。
                if terminal_hits < total_calls {
                    eprintln!(
                        "⚠️ MCTS: phase {} 实际模拟数为 0 (visits_per_action={}, remaining={}, 终局回传 {}/{})，按现有 completed_Q 淘汰继续",
                        phase,
                        visits_per_action,
                        remaining.len(),
                        terminal_hits,
                        total_calls
                    );
                }
            }

            // 根据补全复合效用（completed utility）排序并淘汰
            if remaining.len() > 1 {
                let mut scored: Vec<(usize, f32)> = remaining
                    .iter()
                    .map(|&a| (a, self.completed_utility(a)))
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
}
