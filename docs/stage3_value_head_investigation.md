# 4x4 Stage 3（8 翻子）训练停滞排查：价值头结论

日期：2026-09-10。背景：stage 3 自 20260910-061735 运行约 3.2h（~102 迭代 / 6120 局自对弈 /
36 训练 round），`eval/.../stage_12/win_rate` 全程 36~56%（均值 47.5%）无上升趋势，
且学习斜率仅为 stage 2 的约 1/3（同为前 36 round，loss 2.169→1.963 vs 1.518→1.434）。

## 结论

**价值头代码无 bug；value_loss "平" 是数学必然，不是停滞根因。**

- 静态检查全部通过：
  - `src/pipeline/self_play/finalize.rs`：game_result 按每步行棋方视角取号正确；
  - `src/core/mcts/tree.rs::backprop_from_path`：按 `value_from_perspective(node.player)` 逐节点翻转正确；
  - `src/core/mcts/search.rs`：`mcts_value = root.q_value()`，根节点价值累加为根玩家视角；
  - `python/banqi/training/buffer.py::_target_value`：mixed 目标 `(1-λ)*mcts + λ*game`（λ=0.5）实现正确。
- 动态实测（当前 `last.pt` 自对弈 100 局 / 2315 样本）：
  - `corr(mcts_value, 终局) = 0.738`（completed_q 为 0.760）；
  - `E[(mcts-game)²] = 0.378` → mixed 目标价值 loss 理论噪声底 ≈ `0.25×0.378 = 0.095`；
  - 训练中观测 value_loss = 0.106~0.11，**已贴噪声底**，价值头无可学信号剩余。
- 噪声来源按局内进度分桶：前 1/4 进程 corr 仅 0.526（E=0.599），残局 0.911（E=0.151）。
  早期局面预测力差是暗棋信息隐藏的固有随机性，非实现问题。

## 停滞的重新定性

1. 全局学习信号弱：8 翻子信息量骤减，policy_loss 仍缓降（1.44→1.33）但斜率小。
2. 自对弈自洽平台：policy_acc/top1≈0.32、value_drift/corr≈0.70 双双走平，搜索产出目标≈当前策略。
3. 样本量可能不足：stage 2 跑 ~118 round 才达 78%，stage 3 仅 36 round。

## 后续决策记录

- 价值头被排除后，先原配置继续训练观察（stage_8_1 续训起点，见下）。
- 若再跑 100~200 round 仍无趋势：优先提高探索打破自洽（如 FULL_SEARCH_PROB↑ / Gumbel 噪声），
  而非调整价值目标。
- 不建议 `VALUE_TARGET_MODE=game`（λ=1）：其噪声底（≈0.37）远高于 mixed，梯度更噪。

## 中断点归档

- 中断训练权重归档至 `python/outputs/4x4/checkpoints/stage_8_1/`（last.ckpt / last.pt / last.onnx）。
- `config.local.yaml` 的 `4x4.paths.INIT_FROM_CHECKPOINT` 改为指向 `stage_8_1/last.ckpt` 续训
  （加载 model+optimizer 权重、重置 global_step，非冷启动）。
