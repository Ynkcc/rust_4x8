# ARCHITECTURE — 4x8 暗棋 RL 平台全景

> **维护约定**：本文档是代码库的结构性快照，供 AI 助手与人类快速定位，避免每次需求变更都重新全库探索。
> **更新规则**：
> - 新增/删除模块、bin、feature、Python 子包、桥接 API（PyO3 / Tauri command / gRPC RPC）时，必须同步更新对应小节；
> - 阅读实际代码发现与本文档描述不一致时，更新本文档；
> - 仅函数级内部改动不需更新（本文只记录"结构性事实"：模块划分、关键类型、入口、数据流）；
> - 文末「变更记录」每次结构变更追加一行。

---

## 1. 项目定位

4x8 暗棋（Banqi/Flip Chess）强化学习平台：

- **Rust**：游戏环境 + 博弈搜索（Gumbel MCTS / Expectimax+NNUE）+ 神经网络推理（TorchScript / ONNX / 量化 NNUE）+ Tauri 桌面 GUI + gRPC 分布式自对弈 worker；
- **Python**：AlphaZero 式训练闭环、NNUE 蒸馏、验证与基准；
- **桥接**：PyO3/maturin（Rust wheel 给 Python）、Tauri 2（Rust 后端给前端）、tonic gRPC（分布式 worker）；
- 支持 **4x8 / 4x4 / 4x2（mini）** 变体（另有井字棋仅作验证）。变体由 Rust `GameConfig` 与 Python `banqi/variant.py` 双侧声明。

游戏规则详见 `README.md`（棋子价值、等级吃子、炮隔子攻击、60 分胜利、24 回合无吃子/100 步平局等）。

## 2. 顶层结构

| 条目 | 说明 |
|---|---|
| `Cargo.toml` | crate `banqi_4x8`，edition 2024，`default-run = "banqi-tauri"`；`[lib]` crate-type=`["lib","cdylib"]`（cdylib 供 maturin） |
| `build.rs` | ① 环境预检（pyo3 嵌入 bin 需 libpython 共享库；tauri 需 gtk3/webkit2gtk）；② `tauri_build::build()`；③ libtorch rpath/链接；④ `tonic_build` 编译 proto |
| `pyproject.toml` | maturin 构建，`features=["pyo3-extension"]`（cdylib wheel，不链接 libpython） |
| `tauri.conf.json` | productName「暗棋 4x8」，`frontendDist:"./frontend"`（纯静态前端），窗口 1600x1000 |
| `proto/banqi_service.proto` | 分布式自对弈 RPC 契约（4 个 RPC，见 §6.3） |
| `frontend/` | `index.html` + `main_tauri.js`（全部 UI/对局逻辑）+ `styles.css` |
| `icons/`、`capabilities/`、`gen/schemas/` | Tauri 图标 / 权限 / 生成的 ACL schema |
| `plot_lr_finder.py` | LR finder 绘图脚本 |
| `docs/` | 本文 + draft&archive + `mcts_chance_node_refactor_plan.md` |

### 2.1 feature 矩阵

`torch`、`pyo3`（嵌入 bin，链接 libpython）、`pyo3-extension`（maturin wheel，不链接 libpython）、`onnx`、`onnx-cuda`、`tauri`、`mongodb`；组合：`rust-torch-collector=[torch,pyo3]`、`rust-onnx-collector=[onnx,pyo3]`。default 为空。

工作区约定构建命令：
- Rust：`cargo build --features tauri,torch,onnx`
- Python wheel：`python -m maturin develop --features pyo3-extension,torch,onnx`

### 2.2 bin targets（`src/bin/`）

| bin | 文件 | required-features | 作用 |
|---|---|---|---|
| `banqi` | `banqi.rs` | — | 随机策略对局演示 |
| `banqi-tauri` | `banqi_tauri.rs` | `tauri` | 桌面 GUI 入口 |
| `banqi-data-collector` | `data_collector.rs` | `torch`,`mongodb` | Rust 持 TorchScript 模型自对弈 → MongoDB |
| `banqi-py-collector` | `py_data_collector.rs` | `pyo3` | 嵌入 Python 预测器自对弈 → JSONL |
| `banqi-selfplay-worker` | `selfplay_worker.rs` | — | gRPC 双角色（client+server）分布式自对弈 worker |
| `tmp_nnue_bench` / `tmp_reach` | `tmp_*.rs` | — | NNUE 吞吐/强度临时基准工具 |

## 3. Rust 源码 `src/`（DDD 分层，入口 `lib.rs` 声明 6 模块）

### 3.1 `core/` — 领域核心

- `core/zobrist.rs`：Zobrist 哈希（棋盘/暗袋/行棋方）。
- `core/env/`：暗棋环境
  - `types.rs`：`PieceType`(7 类)/`Player`/`Piece`/`Slot`/`ResNetObservation`；
  - `config.rs`：`GameConfig` + `darkchess_config`/`mini_config`/`game_4x4_config` + `nnue_feature_dim()`；
  - `constants.rs`：`MAX_POSITIONS=32`、`ACTION_SPACE_SIZE` 等；
  - `actions.rs`：动作↔坐标双向查找表（按 config 缓存）；
  - `rules.rs`：走子/吃子规则、`action_masks`；
  - `bitboard.rs`：u64 位棋盘工具；
  - `features.rs`：`StateView` 单次遍历快照 → ResNet 稠密特征（`get_resnet_state`）+ NNUE 稀疏特征（`nnue_active_features*`、`nnue_slot_feature_index`）；
  - `symmetry.rs`：8 种空间对称与动作置换表（数据增强用）；
  - `traits.rs`：`GameEnv` trait（Copy 语义；`is_chance_action` 等机会节点扩展点）；
  - `board/`：`DarkChessEnv` 实现拆分（`struct_def`/`reset`/`step`/`accessors`/`tests`）；
  - `variants/`：`game4x4.rs`、`mini_darkchess.rs`、`tic_tac_toe.rs`。
- `core/mcts/`：Gumbel MCTS（`tree`/`node`/`search`/`policy`/`sampling`/`batched`/`budget`/`evaluator`/`config`）。
- `core/expectimax/`：Expectimax 强引擎——Star1 机会节点剪枝 + 共享 TT（`zobrist.rs` 打包 TtEntry 原子无锁写）+ LMR + 静态搜索 + 迭代加深；`smp.rs` Lazy SMP 多线程（`SearchConfig.threads`）；`ordering.rs` 走法排序。

### 3.2 `engine/` — 策略引擎（弱基线/对照）

`minimax/`（alpha-beta + `eval.rs` 启发式评估）、`mcts_heuristic/`、`mcts_dl.rs`（DL 评估 MCTS）、`movegen/`、`evaluation/`、`policies/`（`random.rs`、`reveal_first.rs`）。

### 3.3 `inference/` — 神经网络推理

- `torchscript.rs`：TorchScript 模型加载与批量推理（`LocalEvaluator` 等）；
- `onnx/`：ONNX Runtime 推理（`ort` crate）；
- `nnue/feature.rs`：`Accumulator`/`DualAccumulator`（红黑双视角累加器）、`FeatureDiff`、`compute_step_diff` 增量更新；
- `nnue/network.rs`：`NnueEvaluator`（`load_from_file` 读量化 `.nnue`，`evaluate_dual` 双视角评估）、`NnueBoard`（增量评估包装，测试断言增量==全量）。

### 3.4 `pipeline/` — 自对弈与数据

- `self_play/`：统一对局主干——`match_core.rs`（`run_match_core`）、`batched.rs`（多线程批量自对弈）、`expectimax_batch.rs`（`run_expectimax_self_play`，NNUE Expectimax 产 episode）、`serialize.rs`、`types.rs`、`finalize.rs`；
- `replay/`：episode 反序列化/描述（`decode.rs`、`describe.rs`、`scalar.rs`）；
- `storage/mongodb.rs`：`MongoStorage` episode 持久化。

### 3.5 `bridge/` — PyO3 扩展

`bridge/mod.rs` 定义 `#[pymodule] fn banqi_4x8`。导出清单：
- 类：`PyGameEpisode`、`PySelfPlayConfig`、`PyTicTacToe`、`PyDarkChess`、`PyMiniDarkChess`、`PyGame4x4`；
- 函数：`run_native_match`、`run_python_match`（统一对局主干）、`run_expectimax_self_play`（NNUE 训练回环）、`describe_record`、`decode_scalar_state`、`variant_dims`、`ttt_mcts_search`、`run_ttt_self_play_with_predictor`、数据增强函数组（`augment.rs::register_augment_functions`）；
- 子模块：`chess_env.rs`（环境包装）、`eval.rs`（对局）、`expectimax.rs`、`self_play/`（配置）、`py_evaluator.rs`（Python 回调评估器）、`decode.rs`、`augment.rs`、`variant.rs`、`ttt.rs`。

### 3.6 `utils/`

`memory_estimator.rs` 等基础设施。

## 4. Python 源码 `python/`（包名 `banqi`）

### 4.1 核心训练闭环

- `trainer_cli/`：训练总入口。`cli.py`（`python -m banqi.trainer_cli <variant> [options]`：`--train-mode/--mcts-sims/--games-per-iter/--train-steps` 等）、`config_resolver.py`、`__main__.py`、`runners/`：`selfplay.py`（自对弈 runner，内常驻 `NnueDistillWorker`）、`expectimax_sidecar.py`（低频 NNUE 强自对弈 sidecar，监听 `ckpt_event`）、`archive_feeder.py`、`offline.py`、`context.py`。
- `config.py` + `config.default.yaml` / `config.local.yaml`：分层配置（local 覆盖 default）。
- `variant.py`：变体声明单一来源。
- `train.py` / `training_service.py` / `training/`：`worker.py`(`TrainWorker`)、`buffer.py`(`episode_to_samples`)、`losses.py`(`run_training_epochs`)、`lr_schedule.py`、`augment.py`、`eval.py`。
- `nn_model.py`：`BanqiNet`（ResNet 式策略+价值网络）。
- `predictor.py` / `selfplay/predictor.py`：推理预测器。
- `checkpoint.py`：ckpt 保存 + `export_model_isolated` 导出 TorchScript `.pt` / `.onnx`；`tools/export_ckpt.py` 命令行事后导出。
- `rust_bridge.py`：封装 `banqi_4x8` wheel 调用。
- `archiver.py` / `storage.py`：episode 归档（含 `TRAIN_MODE=archive` 冷数据复训）。

### 4.2 NNUE 蒸馏旁路

- `nnue/model.py`：`BanqiNNUE`（256→32→1）；
- `nnue/samples.py`：`NnueSampleBuffer` 流式累积（标签 `y = w·搜索价值 + (1-w)·终局回报`）；
- `nnue/distill.py`、`nnue/train.py`(`train_nnue`)、`nnue/exporter.py`（量化导出 `outputs/nnue/<variant>_latest.nnue`）。

### 4.3 其他

- `selfplay/worker.py`：gRPC worker 配套；`proto/`：grpc 生成代码；
- `benchmark/`：预测器/方案基准（`cli.py`、`predictors.py`、`runner.py`、`schemes.py`）；
- `tools/`：`benchmark_production.py`、`clear_db.py`、`play_and_record.py`、`run_baseline.py`；
- `validate/`：验证体系（见 `docs/nnue_expectimax_validation_plan.md`）——`validate_smoke.py`（一键五步冒烟：自对弈→过拟合→导出→Python/Rust 误差<1e-5→Expectimax vs Random 胜率>70%）、`e2e/`、`unit/`、`minigame/`（井字棋闭环验证）；
- `rule_teacher.py`、`memory_guard.py`、`system_monitor.py`、`tb_logger.py`、`constants.py`、`actions.py`、`eval.py`：辅助设施；
- `legacy/`：旧 4x4 训练存档，勿新增依赖。

## 5. Tauri 桌面端

`src/bin/banqi_tauri.rs` 持有 `TauriState`（环境 + 各引擎）。`#[tauri::command]` 列表（前端 `frontend/main_tauri.js` 经 `invoke` 调用）：

- 对局：`reset_game`、`step_game`、`bot_move`、`get_game_state`、`get_move_action`、`get_opponent_type`
- 模型：`list_models`、`load_model`
- 引擎参数：`set_minimax_depth`、`set_mcts_iterations`、`set_engine_budget`、`set_heuristic_sims`、`set_nnue_depth`、`set_nnue_budget`

## 6. 分布式与存储（可选路径）

### 6.1 gRPC（`banqi-selfplay-worker` ↔ 训练端）

`proto/banqi_service.proto` 四 RPC：`ReportGameMeta`（上报元信息）、`PullGameData`（流式拉样本）、`FetchLatestModel`（流式拉模型热更新）、`SyncControl`（心跳/模拟数/暂停/算力随机化控制）。

### 6.2 MongoDB

`data_collector.rs` 直接写 Mongo；训练侧 `archiver.py`/`storage.py` 归档冷存储供复训。

### 6.3 JSONL

`py_data_collector` 与 Expectimax 强自对弈产 JSONL episode 文件。

## 7. 端到端数据流（自对弈 → 训练 → NNUE → 搜索）

1. **编排**：`python -m banqi.trainer_cli 4x8` 启动训练端（`cli.py` → runners）。
2. **自对弈**：`rust_bridge` 调 Rust `run_match_core` + Gumbel MCTS（Python 预测器供 policy/value）产 episode；旁路 `run_expectimax_self_play` 用最新 `.nnue` + `ExpectimaxEngine` 产高质量 episode。
3. **训练**：`TrainWorker` 消费 episode → `buffer.py` 建样本 → `losses.py` 训 `BanqiNet` → `checkpoint.py` 导出 `.pt`/`.onnx` → 回灌自对弈（权重热更新闭环）。
4. **NNUE 蒸馏**：常驻 `NnueDistillWorker` 经 `TeeQueue` 分流含 `nnue_features` 的样本 → `NnueSampleBuffer` → 每 N 次 checkpoint 调 `train_nnue` → `exporter.py` 量化导出 `<variant>_latest.nnue`。
5. **Expectimax sidecar**：`expectimax_sidecar.py` 监听 `ckpt_event`（默认每 20 次），触发 Rust Expectimax+NNUE 强自对弈回流 JSONL，形成"蒸馏→强自对弈→精调"松耦合闭环。
6. **NNUE 推理（Rust）**：`inference/nnue/feature.rs` 增量累加器 + `network.rs` 量化前向；`NnueBoard` 步进评估。
7. **搜索消费**：`core/expectimax::ExpectimaxEngine`（Star1+共享 TT+LMR+静搜）以 NNUE 为唯一叶评估源；消费方：Tauri GUI（NNUE 对手）、validate 脚本、`tmp_*` 基准。

**一句话**：`trainer_cli` 编排 → Rust 自对弈（MCTS 或 Expectimax+NNUE）产 episode → `BanqiNet` 出 `.pt/.onnx` 回灌自对弈、`BanqiNNUE` 出 `.nnue` 回灌 Expectimax 强对弈与 GUI，双网络双引擎互为增强闭环。

## 8. 关键文档索引

- `README.md`：游戏规则；

## 变更记录

- 2026-09-06：初版，由全库探索固化。
- 2026-09-07：新增 `docs/mcts_chance_node_refactor_plan.md`（MCTS 机会节点 Single-Passage Outcome Sampling 重构计划）。
