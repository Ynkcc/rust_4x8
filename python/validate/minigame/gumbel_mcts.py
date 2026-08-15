"""
gumbel_mcts.py — 微型 Gumbel MCTS（与 Rust `src/mcts/search.rs` / `budget.rs` 语义同构）。

精确复刻以下 Rust 行为（供搜索提升率 / 视角反转测试依赖）：
  1. sample_gumbel_top_k : 仅对合法动作加 Gumbel(0,1) 噪声，k = min(k, 合法数)
  2. SequentialHalving    : num_phases = ceil(log_eta(K)) + 1；逐阶段贪心预算；
                           keep_count_after_phase = 下一阶段动作数（eta=2 淘汰 50%）
  3. completed_q          : N>0 用 W/N；N=0 用子节点均值或 initial_value；统一到根玩家视角
  4. value_from_perspective: 玩家相同原样返回，不同取负
  5. backprop             : 沿路径逐层用 value_from_perspective 翻转，路径上每个节点 visit+1
  6. get_improved_policy  : score = logit + sigma*Q, sigma = c_scale * ln(1 + N_root)，
                           softmax 数值稳定，sum<=0 回退合法均匀分布

与 Rust 的 `select_path_collect` 一样，本实现**批量收集待评估叶子**，一次性送给网络，
摊薄推理开销（本机 CPU 单次推理很贵，批处理是性能关键）。

本实现仅针对 Tic-Tac-Toe（无机会节点 / 翻棋），但核心搜索流程与 Rust 一致。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch

from .tic_tac_toe import TicTacToe, N_CELLS


# ============================================================================
# 节点
# ============================================================================

@dataclass
class Node:
    """对齐 Rust MctsNode 核心字段。"""
    env: TicTacToe
    prior: float = 0.0
    logit: float = 0.0
    visit_count: int = 0
    value_sum: float = 0.0
    initial_value: float = 0.0
    is_expanded: bool = False
    player: int = 1
    children: list = field(default_factory=list)  # list of (action_index, Node)


@dataclass
class PendingEval:
    """对齐 Rust PendingEval：待评估叶子及其路径。"""
    path: list
    node: Node


def value_from_perspective(parent_player: int, child_player: int, value: float) -> float:
    """对齐 Rust：玩家相同原样返回，不同取负。"""
    if parent_player == child_player:
        return value
    return -value


def compute_probs_from_logits(logits: np.ndarray, legal: list) -> np.ndarray:
    """对合法动作做数值稳定 softmax（对齐 Rust compute_probs_from_logits）。"""
    probs = np.zeros(N_CELLS, dtype=np.float32)
    if not legal:
        return probs
    max_logit = max(logits[a] for a in legal)
    if not np.isfinite(max_logit):
        return probs
    exps = np.zeros(N_CELLS, dtype=np.float32)
    total = 0.0
    for a in legal:
        v = math.exp(float(logits[a]) - max_logit)
        exps[a] = v
        total += v
    if total > 0.0:
        for a in legal:
            probs[a] = exps[a] / total
    return probs


# ============================================================================
# Sequential Halving 预算分配（对齐 budget.rs）
# ============================================================================

class SequentialHalvingBudget:
    """复刻 budget.rs 的 compute_budget_schedule 贪心逻辑。"""

    def __init__(self, num_candidates: int, total_budget: int, eta: int = 2):
        self.eta = eta
        self.total_budget = total_budget
        self.current_phase = 0
        self.used_budget = 0

        if num_candidates <= 1 or eta <= 1:
            self.num_phases = 1
        else:
            self.num_phases = math.ceil(math.log(num_candidates) / math.log(eta)) + 1

        self.actions_per_phase: list[int] = []
        self.visits_per_action_phase: list[int] = []
        if num_candidates > 0 and total_budget > 0:
            remaining = total_budget
            phases_left = self.num_phases
            for phase in range(self.num_phases):
                num_actions = max(1, math.ceil(num_candidates / (eta ** phase)))
                if remaining < num_actions:
                    break
                per_action = math.ceil(remaining / (phases_left * num_actions))
                per_action = min(per_action, remaining // num_actions)
                per_action = max(1, per_action)
                remaining -= num_actions * per_action
                phases_left -= 1
                self.actions_per_phase.append(num_actions)
                self.visits_per_action_phase.append(per_action)
                if num_actions <= 1:
                    break

    def current_actions(self) -> int:
        if self.current_phase < len(self.actions_per_phase):
            return self.actions_per_phase[self.current_phase]
        return 1

    def visits_per_action_in_phase(self, phase: int) -> int:
        if phase < len(self.visits_per_action_phase):
            return self.visits_per_action_phase[phase]
        return 0

    def num_actions_in_phase(self, phase: int) -> int:
        if phase < len(self.actions_per_phase):
            return self.actions_per_phase[phase]
        return 1

    def keep_count_after_phase(self) -> int:
        actions_now = self.current_actions()
        if actions_now <= 1:
            return actions_now
        if self.current_phase + 1 >= self.num_phases:
            return actions_now
        next_count = self.num_actions_in_phase(self.current_phase + 1)
        return min(max(next_count, 1), actions_now)

    def advance_phase(self) -> None:
        if self.current_phase < self.num_phases:
            self.current_phase += 1

    def record_phase_usage(self, used: int) -> None:
        self.used_budget += used

    def has_budget(self) -> bool:
        return self.used_budget < self.total_budget


# ============================================================================
# Gumbel MCTS 搜索器
# ============================================================================

class GumbelMCTS:
    """单次搜索：给定根局面与网络，返回 (动作, 改进策略, MCTS 值)。"""

    def __init__(self, env: TicTacToe, net, num_simulations: int = 64,
                 max_considered_actions: int = 16, c_scale: float = 1.0):
        self.root = Node(env=env.clone(), player=env.to_play)
        self.net = net
        self.num_simulations = num_simulations
        self.max_considered_actions = max_considered_actions
        self.c_scale = c_scale

    # ------------------------------------------------------------------
    # 批量网络评估
    # ------------------------------------------------------------------
    def _evaluate(self, envs: list):
        """返回 (list[logits_np], list[value_float])，批处理摊薄推理开销。"""
        if not envs:
            return [], []
        boards = torch.from_numpy(np.stack([e.encode() for e in envs])).float()
        with torch.inference_mode():
            logits, values = self.net(boards)
        return (logits.cpu().numpy(), values.cpu().numpy().reshape(-1))

    # ------------------------------------------------------------------
    # 根节点展开
    # ------------------------------------------------------------------
    def _expand_root(self) -> None:
        if self.root.is_expanded:
            return
        logits, values = self._evaluate([self.root.env])
        legal = self.root.env.legal_actions()
        probs = compute_probs_from_logits(logits[0], legal)
        self._build_children(self.root, legal, probs, logits[0])
        self.root.initial_value = float(values[0])
        self.root.visit_count += 1
        self.root.value_sum += float(values[0])

    def _build_children(self, node: Node, legal: list, probs: np.ndarray,
                        logits: np.ndarray) -> None:
        for a in legal:
            child_env, _term, _win = node.env.step(a)
            child = Node(env=child_env, prior=float(probs[a]),
                         logit=float(logits[a]), player=child_env.to_play)
            node.children.append((a, child))
        node.is_expanded = True

    # ------------------------------------------------------------------
    # Gumbel Top-K 采样（对齐 search.rs）
    # ------------------------------------------------------------------
    def _sample_gumbel_top_k(self, legal: list) -> list:
        gumbel = torch.distributions.Gumbel(0.0, 1.0)
        root_logits = {a: child.logit for a, child in self.root.children}
        scored = []
        for a in legal:
            noise = float(gumbel.sample())
            scored.append((a, root_logits.get(a, 0.0) + noise))
        scored.sort(key=lambda x: x[1], reverse=True)
        k = min(self.max_considered_actions, len(scored))
        return [a for a, _ in scored[:k]]

    # ------------------------------------------------------------------
    # completed Q（对齐 search.rs node_q_value + completed_q）
    # ------------------------------------------------------------------
    def _node_q_value(self, node: Node) -> float:
        if node.visit_count > 0:
            return node.value_sum / node.visit_count
        visited = []
        for _, child in node.children:
            if child.visit_count > 0:
                child_q = child.value_sum / child.visit_count
                visited.append(value_from_perspective(node.player, child.player, child_q))
        if visited:
            return float(np.mean(visited))
        return node.initial_value

    def _completed_q(self, action: int) -> float:
        for a, child in self.root.children:
            if a == action:
                q = self._node_q_value(child)
                return value_from_perspective(self.root.player, child.player, q)
        return 0.0

    # ------------------------------------------------------------------
    # PUCT 子节点选择
    # ------------------------------------------------------------------
    def _puct_select(self, node: Node):
        sqrt_total = math.sqrt(node.visit_count)
        best_a, best_child, best_score = None, None, float("-inf")
        for a, child in node.children:
            child_q = self._node_q_value(child)
            adjusted_q = value_from_perspective(node.player, child.player, child_q)
            u = 1.0 * child.prior * sqrt_total / (1.0 + child.visit_count)
            score = adjusted_q + u
            if score > best_score:
                best_score = score
                best_a, best_child = a, child
        return best_a, best_child

    @staticmethod
    def _terminal_value(node: Node) -> float:
        winner = node.env.winner()
        if winner is None or winner == 0:
            return 0.0
        return 1.0 if winner == node.player else -1.0

    # ------------------------------------------------------------------
    # 反传（对齐 Rust backprop_from_path）
    # ------------------------------------------------------------------
    def _apply_backprop(self, path: list, leaf_player: int, leaf_value: float) -> None:
        nodes = [self.root]
        for a in path:
            child = dict(nodes[-1].children)[a]
            nodes.append(child)
        leaf = nodes[-1]
        val = value_from_perspective(leaf.player, leaf_player, leaf_value)
        leaf.visit_count += 1
        leaf.value_sum += val
        for i in range(len(nodes) - 2, -1, -1):
            parent, child = nodes[i], nodes[i + 1]
            val = value_from_perspective(parent.player, child.player, val)
            parent.visit_count += 1
            parent.value_sum += val

    # ------------------------------------------------------------------
    # 单次模拟（批量收集待评估叶子）
    # ------------------------------------------------------------------
    def _select_collect(self, action: int, pending: list, terminal_paths: list) -> None:
        """
        对齐 Rust select_path_collect：从根节点特定候选动作出发，PUCT 下探到叶子。
        若到达未扩展叶子 → 加入 pending；若命中终局 → 直接反传进 terminal_paths。
        """
        child = dict(self.root.children).get(action)
        if child is None:
            return
        path = [action]
        node = child
        if node.env.is_terminal():
            terminal_paths.append((path, node.player, self._terminal_value(node)))
            return
        while node.is_expanded:
            best_a, best_child = self._puct_select(node)
            if best_a is None:
                break
            path.append(best_a)
            node = best_child
            if node.env.is_terminal():
                terminal_paths.append((path, node.player, self._terminal_value(node)))
                return
        pending.append(PendingEval(path=path, node=node))

    # ------------------------------------------------------------------
    # 改进策略（对齐 search.rs get_improved_policy）
    # ------------------------------------------------------------------
    def get_improved_policy(self) -> np.ndarray:
        policy = np.zeros(N_CELLS, dtype=np.float32)
        legal = self.root.env.legal_actions()
        sigma = self.c_scale * math.log(1.0 + self.root.visit_count)
        scores = np.full(N_CELLS, -np.inf, dtype=np.float32)
        max_score = float("-inf")
        root_logits = {a: child.logit for a, child in self.root.children}
        for a in legal:
            if a not in root_logits:
                continue
            score = root_logits[a] + sigma * self._completed_q(a)
            scores[a] = score
            max_score = max(max_score, score)
        if not np.isfinite(max_score):
            return policy
        total = 0.0
        for a in legal:
            s = float(scores[a])
            if np.isfinite(s):
                policy[a] = math.exp(s - max_score)
                total += policy[a]
        if total > 0.0:
            policy /= total
        else:
            n = len(legal)
            if n > 0:
                for a in legal:
                    policy[a] = 1.0 / n
        return policy

    def root_q_value(self) -> float:
        if self.root.visit_count > 0:
            return self.root.value_sum / self.root.visit_count
        return self.root.initial_value

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------
    def run(self):
        """返回 (action, improved_policy, root_q_value) 或 None（无合法动作）。"""
        self._expand_root()
        legal = self.root.env.legal_actions()
        if not legal:
            return None
        if len(legal) == 1:
            return legal[0], self.get_improved_policy(), self.root_q_value()

        candidates = self._sample_gumbel_top_k(legal)
        if not candidates:
            return None

        budget = SequentialHalvingBudget(len(candidates), self.num_simulations, eta=2)
        remaining = candidates

        phase_usage_total = 0
        for phase in range(budget.num_phases):
            if len(remaining) <= 1:
                break
            visits = budget.visits_per_action_in_phase(phase)
            phase_usage = 0
            for _ in range(visits):
                # 每次 visit 迭代立即评估并扩展，保证后续 visit 能下探到已扩展节点
                # （对齐 Rust：每次 select_path_collect 收集后立即 evaluate + 反传）
                pending: list = []
                terminal_paths: list = []
                for a in remaining:
                    self._select_collect(a, pending, terminal_paths)
                # 反传终局
                for path, leaf_player, leaf_value in terminal_paths:
                    self._apply_backprop(path, leaf_player, leaf_value)
                # 批量评估未扩展叶子 + 扩展 + 反传（按节点去重，避免同一叶子重复评估）
                if pending:
                    unique = []
                    seen_nodes = set()
                    for pe in pending:
                        if id(pe.node) not in seen_nodes:
                            seen_nodes.add(id(pe.node))
                            unique.append(pe)
                    envs = [pe.node.env for pe in unique]
                    logits_batch, values = self._evaluate(envs)
                    for pe, logits, value in zip(unique, logits_batch, values):
                        node = pe.node
                        legal_n = node.env.legal_actions()
                        probs = compute_probs_from_logits(logits, legal_n)
                        self._build_children(node, legal_n, probs, logits)
                        node.initial_value = float(value)
                        self._apply_backprop(pe.path, node.player, float(value))
                        phase_usage += 1
            phase_usage_total += phase_usage
            budget.record_phase_usage(phase_usage_total)
            if phase_usage == 0 and not terminal_paths:
                break
            if len(remaining) > 1:
                scored = [(a, self._completed_q(a)) for a in remaining]
                scored.sort(key=lambda x: x[1], reverse=True)
                keep = budget.keep_count_after_phase()
                remaining = [a for a, _ in scored[:keep]]
            budget.advance_phase()

        action = remaining[0] if remaining else legal[0]
        return action, self.get_improved_policy(), self.root_q_value()


def mcts_choose(env: TicTacToe, net, num_simulations: int = 64,
                max_considered_actions: int = 16, c_scale: float = 1.0,
                greedy: bool = False) -> int:
    """
    用 Gumbel MCTS 选动作：
      - greedy=False：返回搜索动作（训练 / 自对弈用）
      - greedy=True ：返回改进策略 argmax（评估最强走法用）
    """
    mcts = GumbelMCTS(env, net, num_simulations, max_considered_actions, c_scale)
    result = mcts.run()
    if result is None:
        legal = env.legal_actions()
        return legal[0] if legal else 0
    action, improved_policy, _ = result
    if greedy:
        return int(np.argmax(improved_policy))
    return action
