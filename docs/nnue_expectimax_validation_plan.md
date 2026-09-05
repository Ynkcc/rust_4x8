# NNUE + Expectimax 验证与追踪方案

本文档记录 NNUE 评估与 Expectimax 搜索整套训练与推理逻辑的验证方案、单项测试方法及当前追踪状态。

---

## 验证追踪总表

### Layer 0: 端到端一键冒烟（日常回归）

| 编号 | 验证项 | 验证脚本 / 命令 | 判定准则 | 状态 | 结果 / 记录 |
|---|---|---|---|---|---|
| 0.1 | 一键端到端完整闭环冒烟 | `python/validate/validate_smoke.py` | 自对弈(N=8,sims=16) -> 过拟合 -> 导出 -> Python/Rust误差<1e-5 -> Expectimax vs Random 20局胜率>70% | ✅ 已通过 | 耗时119s，误差0.00e+00，胜率95.0% (19/20) |

### Layer 1: 特征与模型结构一致性

| 编号 | 验证项 | 验证脚本 / 命令 | 判定准则 | 状态 | 结果 / 记录 |
|---|---|---|---|---|---|
| 1.1 | 维度回推一致性 | `cargo test test_infer_feature_dim_roundtrip` | 无 panic，断言成功 | ✅ 已通过 | pass |
| 1.2 | 特征索引范围与合法性 | `python/validate/unit/validate_nnue_l1.py` | 稀疏索引全在 [0, 555) 内 | ✅ 已通过 | 39 个特征全部合法 |
| 1.3 | 导出/加载跨语言评估等价 | `python/validate/validate_smoke.py` (Step 4) | 同一棋盘 Python forward 与 Rust evaluate 误差 < 1e-5 | ✅ 已通过 | 修复权重转置 bug 后误差达到 0.00e+00 |

### Layer 2: Expectimax 搜索单元验证

| 编号 | 验证项 | 验证脚本 / 命令 | 判定准则 | 状态 | 结果 / 记录 |
|---|---|---|---|---|---|
| 2.1 | 增量更新 = 全量重算 | `cargo test engine_incremental_nnue_matches_full_recompute` | 无 panic，断言成功 | ✅ 已通过 | pass |
| 2.2 | 特性消融基线一致性 | `cargo test engine_feature_ablation_consistency` | 关闭 TT/LMR/Ordering 后输出合法动作，根节点评估值有界偏差 | ✅ 已通过 | pass |

### Layer 3: 训练与反向传播闭环

| 编号 | 验证项 | 验证脚本 / 命令 | 判定准则 | 状态 | 结果 / 记录 |
|---|---|---|---|---|---|
| 3.1 | 单 Batch 极速过拟合 | `python/validate/e2e/validate_nnue_overfit.py` | 300 步优化后 loss < 1e-4 | ✅ 已通过 | Loss 降至 6.1e-5 |
| 3.2 | 对局数据真实训练 | `python/validate/validate_smoke.py` (Step 2) | 读取真实 JSONL 中的 nnue_features 进行训练收敛 | ✅ 已通过 | pass |

### Layer 4: 闭环行为验证

| 编号 | 验证项 | 验证脚本 / 命令 | 判定准则 | 状态 | 结果 / 记录 |
|---|---|---|---|---|---|
| 4.1 | Expectimax vs Random 对局 | `python/validate/validate_smoke.py` (Step 5) | 20 局对局胜率 > 70% | ✅ 已通过 | 胜率 95.0% (19/20) |
