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
| `Cargo.toml` | crate `banqi_4x8` + workspace root（members=`src-tauri`），edition 2024；`[lib]` crate-type=`["lib","cdylib"]`（cdylib 供 maturin） |
| `build.rs` | ① 环境预检（pyo3 嵌入 bin 需 libpython 共享库）；② libtorch rpath/链接；③ `tonic_build` 编译 proto |
| `pyproject.toml` | maturin 构建，`features=["pyo3-extension"]`（cdylib wheel，不链接 libpython） |
| `src-tauri/` | 桌面 GUI 独立 crate `banqi-tauri`：`Cargo.toml`、`build.rs`（GTK/WebKit 预检 + `tauri_build::build()`）、`src/main.rs`（入口）、`tauri.conf.json`、`frontend/`、`icons/` |
| `proto/banqi_service.proto` | 分布式自对弈 RPC 契约（4 个 RPC，见 §6.3） |
| `plot_lr_finder.py` | LR finder 绘图脚本 |
| `docs/` | 本文 + draft&archive + `mcts_chance_node_refactor_plan.md` + `distributed_training_reference_survey.md` |
| `proto/scheduler.proto` | 分布式调度器契约（worker↔中心调度器，见 §6.4） |
| `server/` | Go 中心调度器（module `banqi/server`，见 §6.4） |
| `deploy/` | 2C2G 云服务器部署物（占位） |

### 2.1 feature 矩阵

`torch`、`pyo3`（嵌入 bin，链接 libpython）、`pyo3-extension`（maturin wheel，不链接 libpython）、`onnx`、`onnx-cuda`、`tauri`、`mongodb`；组合：`rust-torch-collector=[torch,pyo3]`、`rust-onnx-collector=[onnx,pyo3]`。default 为空。

工作区约定构建命令：
- Rust（lib + 训练相关 bin）：`cargo build --features torch,onnx`
- 桌面 GUI（独立 crate）：`cargo build -p banqi-tauri`（在 `src-tauri/` 内直接 `cargo build` 亦可；torch/onnx 对手需加对应 feature，如 `-p banqi-tauri --features torch,onnx`）
- Python wheel：`python -m maturin develop --features pyo3-extension,torch,onnx`

### 2.2 bin targets（`src/bin/`）

| bin | 文件 | required-features | 作用 |
|---|---|---|---|
| `banqi` | `banqi.rs` | — | 随机策略对局演示 |
| `banqi-data-collector` | `data_collector.rs` | `torch`,`mongodb` | Rust 持 TorchScript 模型自对弈 → MongoDB |
| `banqi-py-collector` | `py_data_collector.rs` | `pyo3` | 嵌入 Python 预测器自对弈 → JSONL |
| `banqi-selfplay-worker` | `selfplay_worker.rs` | — | gRPC 双角色（client+server）分布式自对弈 worker |
| `tmp_resnet_dump` | `tmp_resnet_dump.rs` | — | ResNet 输入特征人工验证：4x4 随机对局，每手将 NN 输入解码为人类可读表述写入文件（默认 `outputs/resnet_decode_4x4.txt`） |
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
  - `features.rs`：`StateView` 单次遍历快照 → ResNet 稠密特征（`get_resnet_state`）+ NNUE 稀疏特征（`nnue_active_features*`、`nnue_slot_feature_index`）。
    - **ResNet 棋盘张量**（`resnet_board_channels × rows × cols`，视角化=当前行棋方）：通道 `[0..num_active)` 己方明子按型、`[num_active..2*num_active)` 对方明子按型、`[2*num_active]` 暗子、`[2*num_active+1]` 空位；
    - **ResNet 标量向量**（`resnet_scalar_vector_into`，维度 = `resnet_scalar_feature_count` = `3 + 4×total_pieces`；4x8=67 / 4x4=35 / 4x2=19）：`[0]` 步数/判和上限、`[1]` 己方 HP、`[2]` 对方 HP，随后 4 组计数向量（存活×2 + 暗子×2，均按当前行棋方视角，按 `active_types`/`piece_counts` 分块 one-hot）；
    - **解码（人类可读还原）**：`pipeline/replay/`——`decode.rs::decode_board_with_config`（张量→棋盘槽位）、`scalar.rs::decode_scalar_state`/`format_scalar_state`（标量→步数/HP/存活/暗子计数）、`util.rs::format_board`/`piece_name`（中文棋盘渲染）；入口工具 `tmp_resnet_dump`。
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

## 5. Tauri 桌面端（独立 crate `src-tauri/`）

`src-tauri/src/main.rs` 持有 `AppState`（环境 + 各引擎）。前端为 **Vite + Vue 3 + TypeScript** 工程（`src-tauri/frontend/`，源码 `src/`，构建产物 `dist/`，Tauri 加载 `frontendDist=./frontend/dist`，dev 经 `devUrl http://localhost:5173`）：

- `src/api/types.ts` + `src/api/client.ts`：command 返回类型定义与 `invoke` 封装（`api.*`）；
- `src/composables/`：`useGame`（对局状态/走子/高亮/日志/机器人回合）、`useSettings`（变体/对手/模型列表与参数设置）、`useMctsTree`（搜索树懒加载缓存 + 布局计算）、`useToast`、`useLogs`；
- `src/components/`：`App`（三栏布局 + 抽屉）、`BoardView`、`ControlPanel`、`StatusPanel`、`LogPanel`、`PieceTray`、`BitboardPanel`、`MctsTreePanel`、`ToastHost`；`src/domain/pieces.ts` 棋子/位板常量与棋盘尺寸换算。

`#[tauri::command]` 列表（前端经 `@tauri-apps/api/core::invoke` 调用）：

- 对局：`reset_game`、`step_game`、`bot_move`、`get_game_state`、`get_move_action`、`get_opponent_type`
- 模型：`list_models`、`load_model`
- 引擎参数：`set_minimax_depth`、`set_mcts_iterations`、`set_engine_budget`、`set_heuristic_sims`、`set_nnue_depth`、`set_nnue_budget`
- MCTS 树可视化（懒加载）：`mcts_get_root`、`mcts_get_children`、`mcts_get_node_detail`、`mcts_search`。MctsDL/MctsOnnx 落子后整棵 `MctsArena<DarkChessEnv>` 常驻 `AppState.mcts_tree`，前端按需逐节点拉取子边渲染（SVG 树面板，机会节点 outcome 亦懒展开）

前端构建/开发命令（在 `src-tauri/frontend/` 内）：`npm run dev`（Vite dev server，端口 5173）、`npm run build`（vue-tsc 类型检查 + vite build → `dist/`）。

## 6. 分布式与存储（可选路径）

### 6.1 gRPC（`banqi-selfplay-worker` ↔ 训练端）

`proto/banqi_service.proto` 四 RPC：`ReportGameMeta`（上报元信息）、`PullGameData`（流式拉样本）、`FetchLatestModel`（流式拉模型热更新）、`SyncControl`（心跳/模拟数/暂停/算力随机化控制）。

### 6.2 MongoDB

`data_collector.rs` 直接写 Mongo；训练侧 `archiver.py`/`storage.py` 归档冷存储供复训。

### 6.3 JSONL

`py_data_collector` 与 Expectimax 强自对弈产 JSONL episode 文件。

### 6.4 Go 中心调度器（`server/`，proto/scheduler.proto）

调研结论（`docs/distributed_training_reference_survey.md`）落地：lczero 拉取式调度 + KataGo URL 下发/预签名直传 + fishtest/pentanomial 五项 GSPRT 判停。技术栈：Go + tonic 对位的 grpc-go + SQLite（modernc 纯 Go 驱动，WAL）+ aws-sdk-go-v2 S3 预签名（R2 兼容，凭据走标准 `AWS_*` 环境变量）。

- `proto/scheduler.proto`：6 RPC——`GetTask`（worker 按机器规格拉任务：优先 gatekeeper rating，其次 best 网络 selfplay）、`ReportEpisode`（只收元数据，签发 R2 预签名 PUT，数据直传 R2）、`GetNetwork`（sha 或 best → 预签名 GET）、`RegisterNetwork`（trainer 登记新网络 → 自动创建 gatekeeper 对打；首个网络直接晋级）、`ReportMatchResult`（五项成对计数累计 → GSPRT 判停 → 晋级/拒绝 best 指针）、`Heartbeat`（worker 状态 + best sha 下发）。
- `server/cmd/scheduler/main.go`：入口，配置全走 `SCHEDULER_*` 环境变量（`-h` 列出）。
- `server/internal/store`：SQLite 元数据（networks/matches/episodes/workers，best 指针事务切换）。
- `server/internal/r2`：预签名 PUT/GET，键布局 `episodes/<sha>/*.jsonl.gz`、`networks/<sha>.bin`。
- `server/internal/sprt`：五项 GSPRT（正态近似 LLR，elo0/elo1/alpha/beta 可配，含单测）。
- `server/internal/scheduler`：gRPC 服务实现 + 任务表（内存 task_id 注册校验）。
- 生成方式：`protoc --proto_path=proto --go_out=server --go_opt=module=banqi/server --go-grpc_out=server --go-grpc_opt=module=banqi/server scheduler.proto`。

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
- 2026-09-09：Tauri 桌面端拆分为独立 crate `src-tauri/`（workspace member `banqi-tauri`，path 依赖 `banqi_4x8`；`tauri.conf.json`/`frontend/`/`icons/` 一并迁入；根 crate 移除 `tauri` feature 与 tauri 依赖；GUI 构建命令改为 `cargo build -p banqi-tauri`）。
- 2026-09-09：新增 `tmp_resnet_dump` bin 与 replay 标量解码暗子向量支持（`decode_scalar_state` 增 my/opp_hidden）；补充 ResNet 特征布局文档；修正 `resnet_scalar_feature_count` 为 `3 + 4×total_pieces`（4x8=67 / 4x4=35 / 4x2=19，Rust/Python 双侧同步，**旧 ckpt 标量维度不兼容需重训**）。
- 2026-09-09：调度器改用 Go 实现：新增 `proto/scheduler.proto`（6 RPC）与 Go module `server/`（cmd/scheduler + internal/{store,r2,sprt,scheduler}，SQLite 元数据 + R2 预签名直传 + 五项 GSPRT 判停，含单测，构建/冒烟通过）；`deploy/` 部署物占位。见 §6.4 与 `docs/distributed_training_reference_survey.md`。
- 2026-09-10：Tauri GUI 新增 MCTS 搜索树懒加载可视化：`GumbelConfig::with_search_scale` 公开构造器；`MctsDlPolicy`/`OnnxMctsPolicy` 落子路径改为 `bot_move` 内直接构造 `GumbelMCTS` 并将树常驻 `AppState.mcts_tree`；新增 4 个 command（§5）与前端 SVG 搜索树面板（点击展开逐节点拉取）。
- 2026-09-10：Tauri 前端由原生 HTML/JS 重写为 Vite + Vue 3 + TypeScript（`frontend/` 内源码 `src/`、组件/composables/api 分层、三栏布局重构，功能与 command 接口不变；`tauri.conf.json` 改用 `frontendDist=./frontend/dist` + devUrl:5173 + beforeDev/BuildCommand）。
