# MCTS 机会节点重构计划 — Single-Passage Outcome Sampling

> 状态：Phase 1 已实施（2026-09-07，`cargo test --lib` 32 通过）
> 关联文档：`docs/ARCHITECTURE.md` §3.1 `core/mcts/`
> 诊断来源：外部 agent 评审（2026-09-07），经代码核实后裁剪。

## 0. 诊断核实结论（与原评审的差异）

原评审的核心判断成立，但代码核实后有如下修正：

1. **预算稀释的真正来源**：chance node 首次展开时（`src/core/mcts/search.rs` 约 242–273 行）将全部 outcome（暗棋为 7 类棋子 × 2 色 = 至多 14 个）一次性 push 进 `batch` 做 NN 批量评估；而 `run()` 以 `total_phase_usage += batch.len()` 计量预算——一次"模拟"被计为 14 个预算单位，扭曲 Sequential Halving 的预算排程与空转防护统计。后续访问本已按概率采样、每次只评估 1 个叶子，并不稀释。
2. **全量展开的次生代价**：展开时 14 个 outcome 子节点经 `build_children_from_eval` 提前建树（各带一层孙子节点），放大树规模与后续 NN 调用量。
3. **备份结构**：现有回溯（`tree.rs::backprop_from_path`）对 chance 节点做普通访问均值。在 outcome sampling 下（采样频率 ∝ 概率）该均值无偏收敛于期望值，**无需改动**。原评审"按概率加权平均所有子节点统计量"的表述与代码不符，且不应照此实现（会引入双重加权）。
4. **原评审"层级2·等价 outcome 去重"不成立**：`board/step.rs::chance_outcomes` 已按 `(piece_type, 颜色)` 聚合，14 个 outcome 已是同构去重后的等价类，无进一步合并空间。该项从计划中移除。

## Phase 1（本次实施）：保留展开全量评估，修正预算计量与展开偏差

> 方案修订（2026-09-07，经用户确认）：**保留**展开时对全部 outcome 的批量评估（作为机会节点价值的初始化种子），后续访问维持按概率采样推进（现状已满足，无需改动）。Phase 1 聚焦修两处偏差。

### 目标

1. **预算计量修正**：展开爆发的 14 路 batch 不再被计为 14 次模拟；
2. **展开回溯去偏**：展开时 chance 节点初始 Q 由无权均值修正为概率加权期望。

### 改动点（最小化）

1. **预算计量**（`src/core/mcts/search.rs` `run()`）：
   `total_phase_usage += batch.len()` 改为按 `select_path_collect` 调用计量（每次调用 = 1 次模拟，无论其产出 1 个还是展开爆发的 N 个 `PendingEval`）。展开爆发的 NN 推理成本真实发生，但不占用 Sequential Halving 的模拟预算。
2. **展开回溯概率加权**（`src/core/mcts/tree.rs` `backprop_from_path` 或展开调用侧）：
   展开时对 14 条 outcome 路径的回溯，chance 节点按概率做分数访问：`visit_count += p_i`、`value_sum += p_i·v_i`（health 同理），使初始 Q 精确等于 \(\sum p_i V_i\)；后续采样访问维持整数计数。实现上需让展开侧回溯携带 outcome 概率权重（如给 `backprop_from_path` 增加可选权重参数，仅展开分支传 `Some(p_i)`，其余路径传 `None` 按整数 1 计）。
   - 注意：`expand_chance_node` 的全量展开逻辑本身**保持不动**（tree.rs 中有注释明确禁止修改）。
3. **采样推进不改**：后续访问经已展开 chance node 时按概率采样一个 outcome 深入，现状已满足。
4. **Gumbel 边界确认**：`sample_gumbel_top_k` 仅作用于根决策节点候选动作，chance node 不参与，现状已满足，不加改动。

### 采样结果不固化（决策记录）

**决策：不固化。** 每次经过已访问过的 chance node 独立按概率抽取，不锁定首次抽取结果：

- **无偏性**：独立采样是标准 outcome sampling，chance 节点均值无偏收敛于 \(\sum p_i V_i\)；固化后价值收敛到单一实现（某次具体翻子结果）的价值而非期望，偏差不随访问次数消失。
- **逻辑一致性**：固化会导致同一枚物理暗子在不同子树路径中被赋予不同身份（A 路径翻出"车"、B 路径同位置翻出"炮"），局面互相矛盾；"整局固定棋子位置"只有在根节点统一决定化时才自洽——那是 ISMCTS/PIMC（Phase 3）的范畴，且其正确做法是每次模拟重新洗牌并在根统计量中积分，而非每节点只随机一次。
- **子树复用**：独立采样下 `step_next` 按真实 outcome 定位子节点，复用统计量与实际局面一致；固化后若实际结果 ≠ 固化结果，复用的统计量基于错误局面。

固化唯一的收益是降低 chance 节点价值方差，代价为不可消除的偏差，不值得。

### 已知影响（需在验证中确认，非阻塞）

- 保留全量评估意味着展开瞬间仍有 14 路 NN 推理开销与 outcome 子节点提前建树；这是换取机会节点初始价值的已知代价，如 profiling 显示不可接受，再回退到"展开即采样"方案（本计划上一版）。

### 验证

1. `cargo test -p banqi_4x8`（现有 mcts 单测；分数访问改动需检查 visit_count 断言）。
2. 短程自对弈 A/B（同模拟数、同 checkpoint）：
   - 对比 Sequential Halving 各阶段实际预算消耗（修正后 chance 子树不再窃取预算）；
   - 机会节点初始 Q 与 \(\sum p_i V_i\) 一致性抽查（日志断言）；
   - 对局胜率与策略熵无异常漂移；
   - 井字棋（无 chance node）冒烟不受影响。
3. 训练闭环冒烟：`python -m banqi.trainer_cli 4x8` 跑通一个 iteration，确认 episode 产出与 NNUE 蒸馏旁路正常。

### 明确不做

- 不改 `expand_chance_node` 的全量展开；
- 不改后续访问的概率采样推进逻辑；
- 不固化机会节点采样结果（理由见上节）；
- 不做等价 outcome 去重（已无空间，见 §0.4）。

## Phase 2（观察后可选）：Star1 式界剪枝移植

仅当 Phase 1 落地后 profiling 显示 chance 子树仍明显浪费算力时启动：

- 为 chance node 各 outcome 子树维护置信区间 `[Q_i - cσ_i, Q_i + cσ_i]`；
- 当某 outcome 取上界也无法改变父决策节点的动作排序时，停止向其追加模拟；
- 阈值必须保守（MCTS 的 Q 区间比 expectimax 宽松），剪枝只停模拟、不改动备份的概率加权结构；
- 复用 `core/expectimax` 的 Star1 思想，封装为独立工具模块，避免侵入 `search.rs` 主循环。

## Phase 3（长期）：Progressive Widening / 方差冻结 / ISMCTS

维持原评审结论：Phase 1+2 稳定前不实施；ISMCTS 需重构搜索框架与 NN 输入表示，仅在棋力遇到天花板时立项。
