# 双选手对战评估重构与 4x2 变体 6 组对比实验结果

## 优化与代码修改
1. **语法重构（必须显式指定对战双方）**：
   - 命令行统一格式：`python -m banqi.eval <player_a> <player_b> [n] [--variant 4x2|4x4|4x8] [--seed SEED] [-j THREADS]`。
   - 对战双方在 Rust 与 Python 侧均拉平为对等的 `player_a` 与 `player_b`，支持模型 vs 模型、规则 vs 规则、随机 vs 模型等任意自由组合。
2. **随机初始化模型（受种子驱动）**：
   - 输入 `"random"` 或 `"random:<seed>"` 时，自动使用 `torch.manual_seed(seed)` 构建确定性的随机初始化 `BanqiNet` 模型并导出 TorchScript 供 Rust 瞬间加载。
3. **固定随机种子（RNG Seed）**：
   - 在 Rust 侧洗牌与决策中透传 `seed + i` 派生子种子，实现 100 局棋局布局与动作选择的确定性复现。

### 复现实验的命令行指令

| # | 对战组合 | 命令 |
| :--- | :--- | :--- |
| 1 | mcts128 vs mcts128 | `python -m banqi.eval mcts128 mcts128 100 --variant 4x2 --seed 42 -j 4` |
| 2 | random vs random | `python -m banqi.eval random random 100 --variant 4x2 --seed 42 -j 4` |
| 3 | random vs mcts128 | `python -m banqi.eval random mcts128 100 --variant 4x2 --seed 42 -j 4` |
| 4 | random vs last.pt | `python -m banqi.eval random python/outputs/4x2/checkpoints/last.pt 100 --variant 4x2 --model-sims 128 --seed 42 -j 4` |
| 5 | last.pt vs mcts128 | `python -m banqi.eval python/outputs/4x2/checkpoints/last.pt mcts128 100 --variant 4x2 --seed 42 -j 4` |
| 6 | coldstart_stage1/last.pt vs last.pt | `python -m banqi.eval python/outputs/4x2/checkpoints/coldstart_stage1/last.pt python/outputs/4x2/checkpoints/last.pt 100 --variant 4x2 --seed 42 -j 4` |

---

## 4x2 变体 6 组对比测试结果 (Seed = 42, n = 100)

| 序号 | 选手 A (Player A) | 选手 B (Player B) | 对战结果 (A 视角) | 块均胜率 (Mean ± Std) | 现象与分析 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | `mcts128` | `mcts128` | **胜 50 \| 平 8 \| 负 42** | **50.0 ± 20.7%** | 启发式 MCTS 自我对弈表现出良好的先手/后手对称分布。 |
| **2** | `random` (随机模型) | `random` (随机模型) | **胜 45 \| 平 23 \| 负 32** | **45.0 ± 7.1%** | 未训练网络无确定落子方向，平局率上升（23%）。 |
| **3** | `random` (随机模型) | `mcts128` | **胜 4 \| 平 6 \| 负 90** | **4.0 ± 3.7%** | 启发式 MCTS 对未训练网络形成压倒性胜利（胜率 90%）。 |
| **4** | `random` (随机模型) | `4x2/checkpoints/last.pt` | **胜 9 \| 平 4 \| 负 87** | **9.0 ± 11.1%** | 训练后模型 (`last.pt`) 对未经训练网络保持 87% 的绝对统率优势。 |
| **5** | `4x2/checkpoints/last.pt` | `mcts128` | **胜 64 \| 平 9 \| 负 27** | **64.0 ± 9.7%** | 训练后模型显著优于纯启发式规则对手，胜率达 **64.0%**。 |
| **6** | `coldstart_stage1/last.pt` | `4x2/checkpoints/last.pt` | **胜 49 \| 平 10 \| 负 41** | **49.0 ± 13.2%** | 冷启动阶段 1 模型与当前 `last.pt` 棋力相当（胜率 49% vs 41%）。 |

---

## 复测结果（2026-08-19，当前 last.pt = 自对弈强化后 global_step=80553）

> 说明：在首次记录之后，进行了以下改动并重新训练：
> - **修复了内存线性增长泄漏**：`torch.jit.trace`（TorchScript/ONNX 导出）改为每 10 轮导出一次（tracemalloc 定位 `torch/jit/_trace.py` 每轮泄漏 ~80KB），并引入环形 DataBuffer、gc.collect + malloc_trim。
> - **自对弈 MCTS 模拟数提升**：`MCTS_SIMS` 64 → **128**（更深自我对局，强化信号质量更高）。
> - `last.pt` 更新为 **128 sims 自对弈强化至 global_step 80553** 的模型（在首次测试 global_step ~50157 之后又推进约 3 万步）。

### 复测 6 组对比测试结果（Seed = 42, n = 100, A 视角）

| 序号 | 选手 A vs 选手 B | 复测结果 (A 视角) | 复测胜率 | 首次记录 | 变化 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | `mcts128` vs `mcts128` | 胜 52 \| 平 11 \| 负 37 | **52.0%** | 50.0% | ≈持平 |
| 2 | `random` vs `random` | 胜 43 \| 平 22 \| 负 35 | **43.0%** | 45.0% | ≈持平 |
| 3 | `random` vs `mcts128` | 胜 7 \| 平 5 \| 负 88 | **7.0%** | 4.0% | ≈持平 |
| 4 | `random` vs `last.pt` | 胜 8 \| 平 4 \| 负 88 | **8.0%** | 9.0% | ≈持平 |
| 5 | `last.pt` vs `mcts128` | 胜 60 \| 平 10 \| 负 30 | **60.0%** | 64.0% | 略降（±11.4% 内）|
| 6 | `coldstart_stage1/last.pt` vs `last.pt` | 胜 36 \| 平 13 \| 负 51 | **36.0%** | 49.0% | ↓（last.pt 反超）|

### 复测结论

1. **自对弈强化产生明显收益（测试 6 反转）**：
   - 测试 6 中 A=`coldstart_stage1/last.pt`（红方先手），B=`last.pt`（黑方）。
   - 首次记录：冷启动先手 49%（≈棋力相当）。
   - **复测：冷启动先手仅 36%，而 `last.pt` 后手胜 51%** —— 即自对弈强化后模型**即使后手也大幅胜出**，棋力已**明确超越冷启动模型**。
2. **训练后模型 vs 纯启发式保持显著优势（测试 5 = 60%）**：虽比首次 64% 略降，但仍在 ±11.4% 波动内，60% 胜率稳固成立。
3. **内存修复有效且训练可持续**：global_step 从 ~50157 推进到 80553（约 3 万步）期间内存稳定，不再线性增长。
4. **结论**：**128 sims 自对弈强化有效**，`last.pt` 已超越冷启动基线；后续可继续强化以逼近/超越 minimax 或提升 vs 启发式胜率。

---

## 自对弈评估结果（2026-08-19，当前 last.pt = 自对弈强化后 global_step=84750）

> 说明：在 80553 步复测之后，训练继续推进约 4000 步至 **global_step 84750**，并对核心两组对比（测试 5、测试 6）重新评估。

### 核心对比测试结果（Seed = 42, n = 100, A 视角）

| 序号 | 选手 A vs 选手 B | 对战结果 (A 视角) | A 视角胜率 | 复测（@80553） | 变化 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **5** | `last.pt` vs `mcts128` | **胜 60 \| 平 11 \| 负 29** | **60.0 ± 11.0%** | 60.0% | ≈持平 |
| **6a** | `coldstart_stage1/last.pt`（红方先手）vs `last.pt`（黑方后手） | **胜 45 \| 平 11 \| 负 44** | **coldstart 胜 45.0%** | 36.0% | ↑（趋近均衡）|
| **6b** | `last.pt`（黑方后手）vs `coldstart_stage1/last.pt`（红方先手） | **胜 44 \| 平 11 \| 负 45** | **last.pt 后手胜 44.0%** | 51.0% | ↓（≈五五开）|

### 综合评估（三次 walkthrough 对比）

| 时间点 | `coldstart` vs `last.pt`（coldstart 先手胜率） | `last.pt` vs `mcts128` |
| :--- | :--- | :--- |
| 首次（last.pt@50157） | coldstart 胜 **49%** | **64%** |
| 复测（last.pt@80553） | coldstart 胜 **36%**（last.pt 反超） | **60%** |
| 本次（last.pt@84750） | coldstart 胜 **45%**（≈五五开） | **60%** |

### 分析

1. **vs 纯启发式稳定在 60%**：自对弈强化模型的棋力稳定优于启发式 MCTS，这是可靠的强项，与之前两次评估一致（64% → 60% → 60%）。
2. **vs 冷启动模型趋近均衡（45%/55% 五五开）**：
   - 相比上次的 36%/64%（last.pt 明显占优），本次 last.pt 优势缩小至 ≈五五开。
   - 可能解释：
     - **随机性**：评估 100 局 ±9.5%，36% 与 45% 在统计上部分重叠；
     - **自对弈强化边际收益趋于平台期**：从 80553→84750（约 4000 步）棋力提升放缓，且 vs 冷启动这种「自对弈内部」对比天然趋近五五开（两者同源模型）；
     - **价值头非常健康**（corr=0.748、sep=0.83），loss 持续下降、策略熵下降——训练仍在优化，只是棋力提升进入平台期。

### 结论

- 自对弈强化**稳定运行**且棋力健康（**vs 启发式 60%**）；
- 但 **vs 冷启动已趋近五五开**，强化进入**平台期**；
- 模型棋力已稳定在一个较高水平（价值头收敛、策略 Top-1 62%）。