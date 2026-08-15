"""
minigame — 微型 Gumbel AlphaZero 验证引擎（纯 Python / 纯 CPU / 确定性）。

以 Tic-Tac-Toe 为载体，自包含实现与 Rust `src/mcts/search.rs` 语义同构的
Gumbel MCTS + 视角约定 + 自对弈训练环。仅用于验证训练闭环逻辑正确性，
不进入生产导入链。

模块：
- tic_tac_toe.py   : 井字棋环境（状态表示 / 合法动作 / step / 胜负判定 / 特征编码）
- network.py       : 微型 policy-value 网络
- gumbel_mcts.py   : Gumbel MCTS（sample_gumbel_top_k / SequentialHalving / completed_q
                     / value_from_perspective 反传 / get_improved_policy）
- train_loop.py    : 自对弈训练环（支持"固定 Batch 过拟合"与"完整自对弈"两种模式）
- minimax.py       : Minimax 完美裁判 + 纯随机对手策略
"""
