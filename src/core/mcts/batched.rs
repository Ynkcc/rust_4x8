// src/mcts/batched.rs
// 批量自对弈搜索协调器（泛型化：G = 游戏环境）
//
// 目标：显著提升自对弈吞吐。
//
// 瓶颈分析：
//   单棵 `GumbelMCTS` 在每次 `evaluate` 时只把「本轮收集到的待评估叶子」（通常 1~16 个）
//   一次性送给神经网络。对 GPU 推理而言 batch 太小，显存/算力严重利用不足。
//
// 方案：
//   同时驱动 B 棵游戏树（B = 并发对局数），让它们在「逐阶段、逐轮」上保持同步：
//   每轮先让所有树各自 select_path_collect 收集叶子，再把 B 棵树的叶子**合并成一个
//   大 batch** 送给 predictor，最后统一回填。这样单次推理 batch ≈ B × 每棵树每轮叶子数，
//   网络吞吐可成倍提升。
//
// 正确性保证：
//   - 任一树的叶子在被评估/回填之前，该树不会继续前进（严格 lockstep），
//     避免「对未扩展节点重复收集」导致路径漂移。
//   - 机会节点展开、终局回传等逻辑完全复用 self_play 的既有实现。
//   - 根节点价值（每步决策的开头评估）同样走同一 batch 通道，不产生额外 batch=1 推理。

use crate::core::env::GameEnv;
use crate::core::mcts::budget::SequentialHalvingBudget;
use crate::core::mcts::config::{GumbelConfig, MctsSearchResult};
use crate::core::mcts::evaluator::Evaluator;
use crate::core::mcts::path::PendingEval;
use crate::core::mcts::search::GumbelMCTS;
use crate::core::env::Player;

/// 本棵树的推进阶段（决定下一次 `collect` 应产生何种待评估项）
#[derive(PartialEq)]
enum Stage {
    /// 根节点尚未扩展，需要一次根评估（每步决策开头）
    Root,
    /// 根已扩展、候选已采样，处于 Sequential Halving 搜索中
    Searching,
    /// 搜索结束，等待产出动作并执行
    Ready,
    /// 本步决策完成，已执行动作、进入下一步（等下一次 collect 触发新的根评估）
    Idle,
}

/// 一棵正在进行批量搜索的游戏树及其推进状态。
pub struct BatchedTree<'a, G: GameEnv, E: Evaluator<G>> {
    /// 底层搜索树（持有 evaluator 借用）
    pub tree: GumbelMCTS<'a, G, E>,
    /// 当前阶段仍在考虑的候选动作
    candidates: Vec<usize>,
    /// Sequential Halving 预算分配器
    budget: SequentialHalvingBudget,
    /// 当前阶段索引
    phase: usize,
    /// 当前阶段剩余轮次（每轮 = 对每个候选 select 一次）
    phase_visits_left: usize,
    /// 当前推进阶段
    stage: Stage,
    /// 本次决策已完成的搜索结果
    pub result: Option<MctsSearchResult>,
    /// 本步决策的已执行动作（供外部记录）
    pub action: Option<usize>,
    /// 本次决策选中的根玩家
    pub player: Player,
    /// 游戏是否已结束
    pub game_over: bool,
    /// 本步的 (terminated, truncated, winner)
    pub step_outcome: (bool, bool, Option<i32>),
}

impl<'a, G: GameEnv, E: Evaluator<G>> BatchedTree<'a, G, E> {
    /// 创建一棵新树，尚未扩展根节点。
    pub fn new(env: &G, evaluator: &'a E, cfg: &GumbelConfig) -> Self {
        let tree = GumbelMCTS::new(env, evaluator, cfg.clone());
        Self {
            tree,
            candidates: Vec::new(),
            budget: SequentialHalvingBudget::new(0, 0, 2),
            phase: 0,
            phase_visits_left: 0,
            stage: Stage::Root,
            result: None,
            action: None,
            player: env.get_current_player(),
            game_over: env.check_game_over_conditions().2.is_some(),
            step_outcome: (false, false, None),
        }
    }

    /// 收集本棵树本轮需要的待评估项（根评估 或 叶子评估）到 `out`。
    /// 返回 false 表示本棵树本轮没有需要评估的项目。
    /// 每棵树至多产生 0 或 1 批待评估项；外部把所有树的合并成大 batch 后统一评估。
    pub fn collect(&mut self, out: &mut Vec<PendingEval<G>>) -> bool {
        match self.stage {
            Stage::Root => {
                let root_idx = self.tree.root_idx;
                if self.tree.arena.get(root_idx).is_expanded {
                    // 根已扩展（如复用于节点），直接进入搜索
                    self.stage = Stage::Searching;
                    self.ensure_search_prepared();
                    // 继续在 searching 阶段收集
                    self.collect_searching(out)
                } else {
                    let env = *self
                        .tree
                        .arena
                        .get(root_idx)
                        .env
                        .as_ref()
                        .expect("Root must have env");
                    let leaf_player = self.tree.arena.get(root_idx).player();
                    out.push(PendingEval {
                        path: Vec::new(),
                        env,
                        leaf_player,
                    });
                    true
                }
            }
            Stage::Searching => self.collect_searching(out),
            Stage::Ready | Stage::Idle => false,
        }
    }

    /// 处于搜索阶段时的叶子收集。
    fn collect_searching(&mut self, out: &mut Vec<PendingEval<G>>) -> bool {
        // 先执行一轮 select（对每个候选一次），收集叶子
        let mut batch: Vec<PendingEval<G>> = Vec::new();
        for &action in &self.candidates {
            self.tree.select_path_collect(action, &mut batch);
        }
        if batch.is_empty() {
            // 本阶段所有候选都命中终局/静默早退，无可评估叶子：推进阶段
            self.advance_phase();
            return false;
        }
        out.append(&mut batch);
        self.phase_visits_left = self.phase_visits_left.saturating_sub(1);
        true
    }

    /// 应用一批评估结果。`evals` 必须与最近一次 `collect` 产生顺序一致。
    pub fn apply(
        &mut self,
        evals: &[(&PendingEval<G>, &[f32], f32)],
    ) {
        match self.stage {
            Stage::Root => {
                // 根评估：构建子节点并准备搜索
                debug_assert_eq!(evals.len(), 1);
                let (pending, logits, value) = evals[0];
                let _ = pending;
                self.apply_root_eval(logits, value);
                self.stage = Stage::Searching;
                self.ensure_search_prepared();
            }
            Stage::Searching => {
                for (pending, logits, value) in evals {
                    let mut masks = vec![0; G::action_space_size()];
                    pending.env.action_masks_into(&mut masks);
                    let probs = self.tree.compute_probs_from_logits(logits, &masks);
                    let leaf_idx = GumbelMCTS::<G, E>::get_node_idx_by_path(
                        &self.tree.arena,
                        self.tree.root_idx,
                        &pending.path,
                    );
                    {
                        let leaf = self.tree.arena.get_mut(leaf_idx);
                        leaf.initial_value = *value;
                    }
                    GumbelMCTS::<G, E>::build_children_from_eval(
                        &mut self.tree.arena,
                        leaf_idx,
                        &pending.env,
                        &probs,
                        logits,
                        *value,
                    );
                    GumbelMCTS::<G, E>::backprop_from_path(
                        &mut self.tree.arena,
                        self.tree.root_idx,
                        &pending.path,
                        pending.leaf_player,
                        *value,
                    );
                }
                // 本轮收集已耗尽：若阶段轮次用完则推进阶段
                if self.phase_visits_left == 0 {
                    self.advance_phase();
                }
            }
            Stage::Ready | Stage::Idle => {}
        }
    }

    /// 阶段推进：淘汰候选、进入下一阶段，或结束搜索（置 Ready）。
    fn advance_phase(&mut self) {
        if self.stage == Stage::Searching && self.candidates.len() > 1 {
            let keep_count = self.budget.keep_count_after_phase();
            let mut scored: Vec<(usize, f32)> = self
                .candidates
                .iter()
                .map(|&a| (a, self.tree.completed_q(a)))
                .collect();
            scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            self.candidates = scored.into_iter().take(keep_count).map(|(a, _)| a).collect();
        }
        self.budget.advance_phase();
        self.phase += 1;

        if self.candidates.len() <= 1 {
            self.stage = Stage::Ready;
            return;
        }
        if self.phase >= self.budget.num_phases()
            || self.budget.visits_per_action_in_phase(self.phase) == 0
        {
            self.stage = Stage::Ready;
            return;
        }
        self.phase_visits_left = self.budget.visits_per_action_in_phase(self.phase);
    }

    /// 根已展开后：采样候选、初始化 Sequential Halving。根无合法动作时返回 false。
    fn ensure_search_prepared(&mut self) -> bool {
        let root_idx = self.tree.root_idx;
        self.tree.root_action_mask.iter_mut().for_each(|m| *m = 0);
        let env = *self
            .tree
            .arena
            .get(root_idx)
            .env
            .as_ref()
            .expect("Root must have env");
        env.action_masks_into(&mut self.tree.root_action_mask);
        if self.tree.root_action_mask.iter().all(|&x| x == 0) {
            self.stage = Stage::Ready;
            return false;
        }
        let logits: Vec<f32> = (0..G::action_space_size())
            .map(|i| {
                let root = self.tree.arena.get(root_idx);
                root.children
                    .iter()
                    .find(|(act, _)| *act == i)
                    .map(|(_, idx)| self.tree.arena.get(*idx).logit)
                    .unwrap_or(-1e6)
            })
            .collect();
        let masks_cloned = self.tree.root_action_mask.clone();
        self.candidates =
            self.tree.sample_gumbel_top_k(&logits, &masks_cloned, self.tree.config.max_considered_actions);
        if self.candidates.is_empty() {
            self.stage = Stage::Ready;
            return false;
        }
        self.budget = SequentialHalvingBudget::new(
            self.candidates.len(),
            self.tree.config.num_simulations,
            2,
        );
        self.phase = 0;
        self.phase_visits_left = if self.budget.num_phases() > 0 {
            self.budget.visits_per_action_in_phase(0)
        } else {
            0
        };
        if self.phase_visits_left == 0 {
            // 预算不足以给每个候选至少 1 次模拟：直接进入 Ready
            self.stage = Stage::Ready;
        } else if self.candidates.len() <= 1 {
            self.stage = Stage::Ready;
        } else {
            self.stage = Stage::Searching;
        }
        true
    }

    fn apply_root_eval(&mut self, logits: &[f32], value: f32) {
        let root_idx = self.tree.root_idx;
        let env = *self
            .tree
            .arena
            .get(root_idx)
            .env
            .as_ref()
            .expect("Root must have env");
        let mut masks = vec![0; G::action_space_size()];
        env.action_masks_into(&mut masks);
        let probs = self.tree.compute_probs_from_logits(logits, &masks);
        GumbelMCTS::<G, E>::build_children_from_eval(&mut self.tree.arena, root_idx, &env, &probs, logits, value);
        let root = self.tree.arena.get_mut(root_idx);
        root.initial_value = value;
        root.visit_count += 1;
        root.value_sum += value;
    }

    /// 尝试从 Ready 状态产出决策动作并执行，进入下一步（Idle → 下次 collect 触发新根评估）。
    /// 返回 true 表示本步决策完成并已推进（结果在 `self.result` / `self.action`）。
    pub fn finalize_step(&mut self) -> bool {
        if self.stage != Stage::Ready {
            return false;
        }
        let action = if self.candidates.is_empty() {
            let root = self.tree.arena.get(self.tree.root_idx);
            root.children
                .iter()
                .max_by_key(|(_, child_idx)| self.tree.arena.get(*child_idx).visit_count)
                .map(|(action, _)| *action)
                .unwrap_or(0)
        } else {
            self.candidates[0]
        };

        let root = self.tree.arena.get(self.tree.root_idx);
        let state = match &root.state {
            Some(s) => s.clone(),
            None => return false,
        };
        let player = root.player;
        let improved_policy = self.tree.get_improved_policy();
        let mcts_value = root.q_value();
        let completed_q = self.tree.completed_q(action);
        let root_visit_count = root.visit_count;
        let action_mask = self.tree.root_action_mask.clone();

        self.result = Some(MctsSearchResult {
            action,
            state,
            improved_policy,
            mcts_value,
            completed_q,
            root_visit_count,
            player,
            action_mask,
        });
        self.action = Some(action);
        self.player = player;

        // 执行动作并推进树
        let env_root = self
            .tree
            .arena
            .get(self.tree.root_idx)
            .env
            .as_ref()
            .expect("Root must have env");
        let mut env = *env_root;
        let step_res = env.step(action);
        let (terminated, truncated, winner) = match step_res {
            Ok((_, _, terminated, truncated, winner)) => (terminated, truncated, winner),
            Err(e) => {
                eprintln!("⚠️ batched_self_play 游戏错误 (action={}): {}", action, e);
                self.game_over = true;
                self.step_outcome = (true, true, None);
                self.tree.step_next(&env, action);
                self.stage = Stage::Idle;
                return false;
            }
        };
        self.step_outcome = (terminated, truncated, winner);
        self.tree.step_next(&env, action);

        // 重置本步驱动状态
        self.candidates.clear();
        self.budget = SequentialHalvingBudget::new(0, 0, 2);
        self.phase = 0;
        self.phase_visits_left = 0;
        self.stage = Stage::Idle;

        // 检查游戏是否结束（步数上限截断 / 终局）
        if terminated || truncated {
            self.game_over = true;
        } else if self.tree.arena.get(self.tree.root_idx).is_terminal {
            self.game_over = true;
        }
        true
    }

    /// 进入下一步决策（消费上次决策结果后，准备新的根评估）。
    /// 由协调器在记录本步样本后调用。
    pub fn start_next_step(&mut self) {
        if self.game_over {
            self.stage = Stage::Idle;
            return;
        }
        self.result = None;
        self.action = None;
        self.stage = Stage::Root;
    }

    /// 当前是否仍在推进中（可继续收集评估项）。
    pub fn is_active(&self) -> bool {
        !self.game_over && self.stage != Stage::Idle
    }
}
