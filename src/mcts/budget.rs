// src/mcts/budget.rs
// Sequential Halving 预算分配模块

/// Sequential Halving 预算分配器
///
/// 负责在 Sequential Halving 的各个阶段中均匀、高效地分配搜索预算。
/// 典型流程：
/// 1. 初始化：指定候选动作数和总预算
/// 2. 每个阶段：获取该阶段的动作数和预算
/// 3. 淘汰：计算淘汰数量
/// 4. 进度：追踪已使用的预算
#[derive(Clone, Debug)]
pub struct SequentialHalvingBudget {
    /// 初始候选动作数
    initial_candidates: usize,
    /// 总预算（模拟次数）
    total_budget: usize,
    /// 淘汰率因子 (通常 2-4，表示每阶段淘汰比例)
    /// eta = 2 表示保留一半的动作
    /// eta = 4 表示保留 1/4 的动作
    eta: usize,
    /// 阶段总数
    num_phases: usize,
    /// 当前阶段索引 (0-based)
    current_phase: usize,
    /// 每个阶段应该分配的动作数
    actions_per_phase: Vec<usize>,
    /// 每个阶段应该分配的访问次数（每个动作）
    visits_per_action_per_phase: Vec<usize>,
    /// 总已使用预算
    used_budget: usize,
}

impl SequentialHalvingBudget {
    /// 创建新的 Sequential Halving 预算分配器
    ///
    /// # 参数
    ///
    /// * `num_candidates` - 初始候选动作数
    /// * `total_budget` - 总预算（模拟次数）
    /// * `eta` - 淘汰率因子 (默认2，表示保留50%)
    ///
    /// # 例子
    ///
    /// ```
    /// let mut budget = SequentialHalvingBudget::new(8, 1024, 2);
    /// assert_eq!(budget.num_phases(), 4); // phases = log_2(8) + 1
    /// assert_eq!(budget.actions_in_phase(0), 8); // 第1阶段：8个动作
    /// assert_eq!(budget.actions_in_phase(1), 4); // 第2阶段：4个动作
    /// ```
    pub fn new(num_candidates: usize, total_budget: usize, eta: usize) -> Self {
        if num_candidates == 0 || total_budget == 0 {
            return Self {
                initial_candidates: 0,
                total_budget: 0,
                eta: eta.max(2),
                num_phases: 0,
                current_phase: 0,
                actions_per_phase: Vec::new(),
                visits_per_action_per_phase: Vec::new(),
                used_budget: 0,
            };
        }

        let eta = eta.max(2);
        let num_phases = Self::compute_num_phases(num_candidates, eta);

        // 计算每个阶段的动作数和访问预算
        let (actions_per_phase, visits_per_action_per_phase) =
            Self::compute_budget_schedule(num_candidates, total_budget, eta, num_phases);

        Self {
            initial_candidates: num_candidates,
            total_budget,
            eta,
            num_phases,
            current_phase: 0,
            actions_per_phase,
            visits_per_action_per_phase,
            used_budget: 0,
        }
    }

    /// 计算所需的阶段数
    ///
    /// 对数级的阶段数：s = ceil(log_eta(K)) + 1
    /// 其中 K 是初始候选动作数
    fn compute_num_phases(num_candidates: usize, eta: usize) -> usize {
        if num_candidates <= 1 || eta <= 1 {
            return 1;
        }
        let log_val = (num_candidates as f32).log(eta as f32);
        log_val.ceil() as usize + 1
    }

    /// 计算完整的预算日程表
    ///
    /// 返回 (actions_per_phase, visits_per_action_per_phase)
    /// - actions_per_phase[i] = 第 i 阶段的动作数
    /// - visits_per_action_per_phase[i] = 第 i 阶段分配给每个动作的访问次数
    fn compute_budget_schedule(
        num_candidates: usize,
        total_budget: usize,
        eta: usize,
        num_phases: usize,
    ) -> (Vec<usize>, Vec<usize>) {
        let mut actions_per_phase = Vec::with_capacity(num_phases);
        let mut visits_per_action_per_phase = Vec::with_capacity(num_phases);

        // 计算初始参数
        // n_1 = ceil(N / K / log_eta(K))
        if num_phases == 0 || num_candidates == 0 {
            return (actions_per_phase, visits_per_action_per_phase);
        }

        let log_eta_k = if eta <= 1 {
            1.0
        } else {
            (num_candidates as f32).log(eta as f32).max(1.0)
        };

        let n1 = ((total_budget as f32) / (num_candidates as f32) / log_eta_k).ceil() as usize;
        let n1 = n1.max(1); // 至少 1 次访问

        for phase_idx in 0..num_phases {
            let num_actions =
                (num_candidates as f32 / (eta.pow(phase_idx as u32) as f32)).ceil() as usize;
            let num_actions = num_actions.max(1);
            let visits = (n1 as f32 / (eta.pow(phase_idx as u32) as f32)).ceil() as usize;
            let visits = visits.max(1);

            actions_per_phase.push(num_actions);
            visits_per_action_per_phase.push(visits);

            // 如果只剩 1 个动作，后续阶段不需要继续
            if num_actions <= 1 {
                break;
            }
        }

        (actions_per_phase, visits_per_action_per_phase)
    }

    /// 获取阶段数
    pub fn num_phases(&self) -> usize {
        self.num_phases
    }

    /// 获取当前阶段索引
    pub fn current_phase(&self) -> usize {
        self.current_phase
    }

    /// 获取指定阶段的动作数
    pub fn actions_in_phase(&self, phase: usize) -> usize {
        self.actions_per_phase.get(phase).copied().unwrap_or(1)
    }

    /// 获取指定阶段每个动作分配的访问次数
    pub fn visits_per_action_in_phase(&self, phase: usize) -> usize {
        self.visits_per_action_per_phase
            .get(phase)
            .copied()
            .unwrap_or(0)
    }

    /// 获取当前阶段的动作数
    pub fn current_actions(&self) -> usize {
        self.actions_in_phase(self.current_phase)
    }

    /// 获取当前阶段每个动作的访问次数
    pub fn current_visits_per_action(&self) -> usize {
        self.visits_per_action_in_phase(self.current_phase)
    }

    /// 获取当前阶段的总预算
    pub fn current_phase_budget(&self) -> usize {
        self.current_actions() * self.current_visits_per_action()
    }

    /// 计算该阶段后应该保留的动作数
    ///
    /// 语义: 当前阶段执行完成后，应返回下一阶段的候选动作数。
    /// - 若当前已经是末阶段（或已经只剩 1 个动作），保留全部当前候选数（不再淘汰）。
    /// - 否则按 eta 比例缩减，与 actions_in_phase(current_phase+1) 一致。
    pub fn keep_count_after_phase(&self) -> usize {
        let actions_now = self.current_actions();
        if actions_now <= 1 {
            return actions_now;
        }
        if self.current_phase + 1 >= self.num_phases {
            // 已经在末阶段或超出：不再淘汰，保留当前动作数
            return actions_now;
        }
        let next_phase_count = self.actions_in_phase(self.current_phase + 1);
        // 保证至少保留 1 个，且不超过当前动作数
        next_phase_count.min(actions_now).max(1)
    }

    /// 提前进入下一阶段
    pub fn advance_phase(&mut self) {
        if self.current_phase < self.num_phases {
            self.current_phase += 1;
        }
    }

    /// 记录本阶段已使用的预算
    pub fn record_phase_usage(&mut self, used: usize) {
        self.used_budget += used;
    }

    /// 获取总已使用预算
    pub fn total_used(&self) -> usize {
        self.used_budget
    }

    /// 获取总预算
    pub fn total_budget(&self) -> usize {
        self.total_budget
    }

    /// 检查预算是否充足
    pub fn has_budget(&self) -> bool {
        self.used_budget < self.total_budget
    }

    /// 获取剩余预算
    pub fn remaining_budget(&self) -> usize {
        self.total_budget.saturating_sub(self.used_budget)
    }

    /// 获取完整的预算摘要
    pub fn summary(&self) -> String {
        let mut s = format!(
            "📊 Sequential Halving 预算摘要\n\
             总预算: {}, 已用: {}/{}, 剩余: {}\n\
             初始候选数: {}, 淘汰率(eta): {}, 阶段数: {}\n",
            self.total_budget,
            self.used_budget,
            self.total_budget,
            self.remaining_budget(),
            self.initial_candidates,
            self.eta,
            self.num_phases,
        );

        s.push_str("预算日程表:\n");
        for (phase, &actions) in self.actions_per_phase.iter().enumerate() {
            let visits = self
                .visits_per_action_per_phase
                .get(phase)
                .copied()
                .unwrap_or(0);
            let total = actions * visits;
            s.push_str(&format!(
                "  阶段{}: {} 个动作 × {} 访问/动作 = {} 总访问\n",
                phase, actions, visits, total
            ));
        }

        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_initialization() {
        let budget = SequentialHalvingBudget::new(8, 1024, 2);
        assert_eq!(budget.num_phases(), 4);
        assert_eq!(budget.initial_candidates, 8);
        assert_eq!(budget.total_budget(), 1024);
    }

    #[test]
    fn test_actions_per_phase() {
        let budget = SequentialHalvingBudget::new(8, 1024, 2);
        assert_eq!(budget.actions_in_phase(0), 8);
        assert_eq!(budget.actions_in_phase(1), 4);
        assert_eq!(budget.actions_in_phase(2), 2);
        assert_eq!(budget.actions_in_phase(3), 1);
    }

    #[test]
    fn test_phase_progression() {
        let mut budget = SequentialHalvingBudget::new(16, 2048, 2);
        assert_eq!(budget.current_phase(), 0);
        budget.advance_phase();
        assert_eq!(budget.current_phase(), 1);
    }

    #[test]
    fn test_edge_cases() {
        let budget = SequentialHalvingBudget::new(1, 100, 2);
        assert_eq!(budget.actions_in_phase(0), 1);

        let budget = SequentialHalvingBudget::new(0, 100, 2);
        assert_eq!(budget.num_phases(), 0);
    }
}
