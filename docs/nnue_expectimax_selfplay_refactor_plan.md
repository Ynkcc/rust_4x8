# Expectimax + NNUE 自对弈重构计划

参照现有实现：`src/pipeline/self_play/`（batched.rs / match_core.rs / types.rs / serialize.rs）、`src/core/mcts/`、`src/core/expectimax/`。

目标：在不破坏现有 Gumbel MCTS 流水线的前提下，将 Expectimax + NNUE 接入统一自对弈/对局主干，形成独立的 NNUE 训练数据闭环。

---

## Phase 1 — Expectimax 引擎并发能力（Lazy SMP）

**现状**：`src/core/expectimax/search.rs` 的搜索引擎完全单线程；已有每线程 TT 配置位（`tt_bits`），无跨线程共享表。

**改动**：
1. 新增 `SharedTT`：`Arc<Vec<AtomicU64 简化 TtEntry>>`（或 `Arc<Mutex-chunked>`），替代当前 per-search 局部 TT。TT 写入全部无锁（原子 64bit packed entry），读到半旧数据无害。
2. `SearchConfig` 增加 `threads: usize`（默认 1，行为不变）。
3. 新增 Lazy SMP 搜索驱动：迭代加深主线程 + N 个 helper 线程，helper 在不同根走法排序偏移/aspiration 窗口下重复搜索同一迭代深度，共享 TT 交换信息；主线程到预算即停。
4. `expectimax/mod.rs` 暴露 `ExpectimaxEngine::search_par(&self, env, cfg) -> SearchResult`，单线程路径 `search` 保持不变（`threads=1` 时零开销转发）。

**验收**：固定局面下 `threads=4` 相对 `threads=1` 节点/秒接近线性，选择动作一致或更强（可复用 `python/validate/validate_smoke.py` 的对局校验）。

## Phase 2 — 统一对局主干抽象（match_core.rs）

**现状**：`PlayerSpec`（match_core.rs:110-119）缺 Expectimax 变体；`play_one_game_recorded`（L389+）硬绑定 `GumbelMCTS::run()`。

**改动**（最小化、优先重构而非新增平行实现）：
1. 扩展 `PlayerSpec`：
   ```rust
   Expectimax { engine: Arc<ExpectimaxEngine>, cfg: SearchConfig },
   ```
2. 在 `match_core.rs` 抽出"决策器"概念：将"给定 env + 配置 → (action, 附加记录)"抽象为现有 `PlayerEval` 分发逻辑的一个新分支，Expectimax 分支直接调用 `engine.search_par()`，不经过 Evaluator/MCTS。
3. 新增 `play_one_game_expectimax`（或作为 `play_one_game_recorded` 的 decision-provider 泛化重构）：每步只记录 NNUE 所需字段，产出 `NnueEpisode`（见 Phase 3），同时仍可产出 `GameEpisode` 兼容格式（policy/v 字段以搜索分数退化填充或留空，`is_full_search=true`）。
4. `SelfPlayConfig` 增加 `expectimax: Option<ExpectimaxSelfPlayConfig>`（node_budget / max_depth / quiesce / threads 等），避免污染 MCTS 字段。

**验收**：`run_match_core` 可调度 Expectimax vs MCTS / Expectimax vs Expectimax 对局；`validate_smoke.py` 扩展一条 Expectimax 自对弈 smoke case。

## Phase 3 — NNUE 专属数据契约（types.rs / serialize.rs）

**现状**：`GameEpisode` 9 元组完全为 MCTS 设计；`NnueStepFeatures`/`NnueEpisodeMeta` 已存在但寄生在 MCTS 收集路径（batched.rs 中固定 `None`）。

**改动**：
1. 新增轻量结构（不复用 9 元组）：
   ```rust
   pub struct NnueEpisode {
       pub meta: NnueEpisodeMeta,
       pub steps: Vec<NnueStepRecord>,  // { mover, opponent, search_value, player, result(终局回填) }
   }
   ```
   仅稀疏索引 + 标量，无 `ResNetObservation` 大张量。
2. `serialize.rs` 增加 `nnue_episode_to_jsonl` / 追加式 `.nnuejsonl` 写出器（每局一行或按步分行的 binpack 风格紧凑格式，JSONL 起步，字段与 Python 训练契约对齐）。
3. 搜索值来源：Expectimax 的 `SearchResult.value`（最深迭代、根视角）作为 `search_value`；alpha-beta 截断不影响根值，非最优走法不记录 Q 向量——NNUE 训练只需要局面值。

**验收**：生成的 JSONL 可直接被 `python/banqi/nnue/train.py::NnueSampleDataset` 消费（必要时在其上做薄适配，value_source 增加含 `search_value` 的分支）。

## Phase 4 — 局间并发自对弈运行器

**现状**：`run_batched_self_play`（self_play/batched.rs）为 EvalQueue batching 架构，对 Expectimax 不适用；局间并发目前只能靠 Rayon 逐局。

**改动**：
1. 新增 `src/pipeline/self_play/expectimax_batch.rs`：`run_expectimax_self_play(nnue: &NnueEvaluator, cfg, num_games, num_workers) -> Vec<NnueEpisode>`。
   - 局间：`rayon::par_iter` 或 scoped thread pool，每局独占 `ExpectimaxEngine`（各自线程局部累加器，零争用）。
   - 局内：由 `threads` 参数决定是否走 Phase 1 的 Lazy SMP（建议自对弈时局内 threads=1、加大 worker 数，机器核心利用率最优）。
2. NNUE 特征直接从 `env.nnue_active_features_for_player_into`（现 match_core 已用）增量收集，避免逐局面全量重算。
3. 边玩边写（每局完成即 flush JSONL），不整批驻留内存。

**验收**：N 局吞吐随 worker 数近线性；输出文件可被训练脚本直接加载。

## Phase 5 — PyO3 Bridge 与训练回环

**现状**：`src/bridge/python/eval.rs` 的 `run_native_match`（L165）只导向 MCTS；无 Expectimax 自对弈入口。

**改动**：
1. 新增 `#[pyfunction] run_expectimax_self_play(nnue_path, n_games, workers, node_budget, max_depth, out_jsonl, seed) -> stats dict`，内部走 Phase 4。
2. `run_native_match` 的 PlayerSpec 解析增加 `"expectimax"` 选手类型（评估对局用）。
3. Python 侧 `NnueSampleDataset` 增加 `value_source="search_value"` 分支 + 流式懒加载（当前是一次性载入内存的 `list`，样本量大时改 iterator-based Dataset，尽量少改 `train.py` 其他部分）。
4. `python/validate/validate_smoke.py` 增加端到端：生成 → 训练 1 epoch → 导出 `.nnue` → Rust 加载再自对弈一轮。

## 阶段依赖与提交切分

| 提交 | 内容 | 依赖 |
|---|---|---|
| C1 | Phase 1 SharedTT + Lazy SMP | 无 |
| C2 | Phase 2 match_core 抽象 + PlayerSpec::Expectimax | 无 |
| C3 | Phase 3 NnueEpisode 契约 + JSONL 写出 | C2 |
| C4 | Phase 4 expectimax_batch 运行器 | C1+C3 |
| C5 | Phase 5 PyO3 + Python 训练回环 | C4 |

## 风险与不做的事

- **不做** EvalQueue batching 化 Expectimax：DFS + CPU 向量评估与该架构拓扑不匹配，强行接入得不偿失。
- Lazy SMP 的 TT 为无锁原子写，允许极小概率的 race 覆盖（标准引擎做法），不做完全串行一致性保证。
- 序列化先 JSONL（与现有 Python 契约一致），binpack 二进制格式仅在 JSONL 成为吞吐瓶颈后再考虑，避免提前引入 failback/双格式维护。
- `GameEpisode` 9 元组保持不动；Expectimax 路径产出独立 `NnueEpisode`，不向旧元组里塞语义不符的字段。
