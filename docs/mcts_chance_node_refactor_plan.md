# MCTS 机会节点重构计划 — Single-Passage Outcome Sampling

> 状态：待实施（仅 Phase 1 排期）
> 关联文档：`docs/ARCHITECTURE.md` §3.1 `core/mcts/`
> 诊断来源：外部 agent 评审（2026-09-07），经代码核实后裁剪。

## 0. 诊断核实结论（与原评审的差异）

原评审的核心判断成立，但代码核实后有如下修正：

1. **预算稀释的真正来源**：chance node 首次展开时（`src/core/mcts/search.rs` 约 242–273 行）将全部 outcome（暗棋为 7 类棋子 × 2 色 = 至多 14 个）一次性 push 进 `batch` 做 NN 批量评估；而 `run()` 以 `total_phase_usage += batch.len()` 计量预算——一次"模拟"被计为 14 个预算单位，扭曲 Sequential Halving 的预算排程与空转防护统计。后续访问本已按概率采样、每次只评估 1 个叶子，并不稀释。
2. **全量展开的次生代价**：展开时 14 个 outcome 子节点经 `build_children_from_eval` 提前建树（各带一层孙子节点），放大树规模与后续 NN 调用量。
3. **备份结构**：现有回溯（`tree.rs::backprop_from_path`）对 chance 节点做普通访问均值。在 outcome sampling 下（采样频率 ∝ 概率）该均值无偏收敛于期望值，**无需改动**。原评审"按概率加权平均所有子节点统计量"的表述与代码不符，且不应照此实现（会引入双重加权）。
4. **原评审"层级2·等价 outcome 去重"不成立**：`board/step.rs::chance_outcomes` 已按 `(piece_type, 颜色)` 聚合，14 个 outcome 已是同构去重后的等价类，无进一步合并空间。该项从计划中移除。

## Phase 1（本次实施）：展开即采样，单路推进

### 目标

chance node 首次展开后不再批量评估全部 outcome，改为与后续访问一致：按概率采样一个 outcome 并继续深入。每次模拟经 chance node 只产生 1 个 `PendingEval`。

### 改动点（最小化，均集中在 `src/core/mcts/search.rs`）

1. **统一展开与采样路径**：`select_path_collect` 中 `is_chance && !is_expanded` 分支，`expand_chance_node` 后不再 `for` 循环 push 全部 outcome，改为 fall-through 到已有的"已展开"采样逻辑（`sample_outcome_id` → push `ChanceOutcome` 到 path → `continue`）。删除展开分支的批量 push 与提前 `return`。
   - 注意：`expand_chance_node` 本身**保持全量展开不动**（tree.rs 中有注释明确禁止修改；子节点 + 概率记录成本极低，是采样与 `step_next` 子树复用的基础）。
2. **备份不改**：`tree.rs::backprop_from_path` 保持普通访问均值。
3. **预算计数不改**：batch 恒为 1 后，`total_phase_usage` 语义自动修正。
4. **Gumbel 边界确认**：`sample_gumbel_top_k` 仅作用于根决策节点候选动作，chance node 不参与，现状已满足，不加改动。

### 已知影响（需在验证中确认，非阻塞）

- chance node 首次被访问后才有统计量；在此之前其 outcome 子节点 `N=0` 时 `node_q_value` fallback 到 `initial_value=0.0`（原实现为 14 路评估均值）。影响范围仅限该 chance node 未被访问期间的父节点 PUCT 比较，且父节点 `completed_q` 有兄弟均值兜底。预期无偏性不受影响，仅早期选择噪声略增。
- 单路采样使 chance 节点价值估计方差增大（无偏），收敛依赖访问次数。若训练/对局中出现可观测退化，再评估是否对根的直接翻子动作做"首访全量评估"特例——**当前不实现**。

### 验证

1. `cargo test -p banqi_4x8`（现有 mcts 单测；如有依赖"展开即全量评估"行为的测试需同步修正）。
2. 短程自对弈 A/B（同模拟数、同 checkpoint）：
   - 对比每局 NN 推理批次数 / 总推理次数（预期下降）；
   - 对局胜率与策略熵无异常漂移；
   - 井字棋（无 chance node）冒烟不受影响。
3. 训练闭环冒烟：`python -m banqi.trainer_cli 4x8` 跑通一个 iteration，确认 episode 产出与 NNUE 蒸馏旁路正常。

### 明确不做

- 不改 `expand_chance_node` 的全量展开；
- 不改 `tree.rs` 备份逻辑；
- 不做等价 outcome 去重（已无空间，见 §0.4）。

## Phase 2（观察后可选）：Star1 式界剪枝移植

仅当 Phase 1 落地后 profiling 显示 chance 子树仍明显浪费算力时启动：

- 为 chance node 各 outcome 子树维护置信区间 `[Q_i - cσ_i, Q_i + cσ_i]`；
- 当某 outcome 取上界也无法改变父决策节点的动作排序时，停止向其追加模拟；
- 阈值必须保守（MCTS 的 Q 区间比 expectimax 宽松），剪枝只停模拟、不改动备份的概率加权结构；
- 复用 `core/expectimax` 的 Star1 思想，封装为独立工具模块，避免侵入 `search.rs` 主循环。

## Phase 3（长期）：Progressive Widening / 方差冻结 / ISMCTS

维持原评审结论：Phase 1+2 稳定前不实施；ISMCTS 需重构搜索框架与 NN 输入表示，仅在棋力遇到天花板时立项。
