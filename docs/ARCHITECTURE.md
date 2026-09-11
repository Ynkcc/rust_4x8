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
| `src-tauri/` | 桌面 GUI 独立 crate `banqi-tauri`（规划拆出为独立仓库），架构见 `src-tauri/ARCHITECTURE.md` |
| `proto/banqi_service.proto` | 分布式自对弈 RPC 契约（4 个 RPC，见 §6.3） |
| `plot_lr_finder.py` | LR finder 绘图脚本 |
| `docs/` | 本文 + draft&archive + `mcts_chance_node_refactor_plan.md` + `distributed_training_reference_survey.md` |
| `proto/scheduler.proto` | 分布式调度器契约（worker↔中心调度器，见 §6.4；拆分时迁入调度器仓库） |
| `server/` | Go 中心调度器（module `banqi/server`，规划拆出为独立仓库 `banqi-scheduler`），架构见 `server/ARCHITECTURE.md` |

### 2.0 仓库拆分规划（方案 A）

按耦合边界拆为多个仓库，目录已预先聚集（各拆分单元自带 `.gitignore` 与 `ARCHITECTURE.md`，拆分时整体迁移即可）：

1. **`banqi-scheduler`**：`server/` + `proto/scheduler.proto`（+ `deploy/` 占位）。与其他代码零耦合；唯一跨界依赖是主仓库 `build.rs` 编译 `scheduler.proto`，拆分后主仓库改为引用该仓库的 proto 副本。
2. **`banqi-tauri`**：`src-tauri/`。已独立 crate，拆分时 path 依赖 `banqi_4x8` 改为 git 依赖。
3. **主体仓库（本仓库保留）**：Rust `src/` + `python/` + `proto/banqi_service.proto`。**后续将进一步拆分为训练（trainer）与数据收集（collector）两个仓库**——当前耦合点：`pipeline/`、`registry/`、`bridge/`（PyO3 episode 契约）与 Python `infra/`/`runners/`，拆分前先以 `registry/`（Rust）+ `infra/`（Python）为接口边界解耦，此文档届时再拆。

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
| `banqi-collector` | `collector.rs` | `onnx` | 统一 Collector 进程：`--backend local`（LocalRegistry 模型热重载 → Rust 推理自对弈 → LocalEpisodeStore 落盘）或 `--backend scheduler`（gRPC GetTask + R2 预签名直传，selfplay/rating 双任务） |
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

### 3.6 `registry/` — 协作接口层（统一训练架构 L4，见 `docs/unified_training_architecture.md`）

角色同构（Collector/Trainer/Registry）的单机本地实现；分布式 Scheduler/R2 实现后续接入。

- `local_registry.rs`（feature=`onnx`）：`LocalRegistry`——watch onnx 模型路径（notify），`ensure_model()` 惰性热重载（失败保留旧模型）；`CollectorTask` 描述一次采集任务；
- `local_store.rs`：`LocalEpisodeStore`——episode 按 `<dir>/<variant>/iter{N}_{ts}_w{id}.jsonl.gz` 落盘，字段复用 `serialize::episode_to_dict_json` 契约（与 PyO3/gRPC 一致）。

### 3.7 `utils/`

`memory_estimator.rs` 等基础设施。

## 4. Python 源码 `python/`（包名 `banqi`）

### 4.1 核心训练闭环

- `trainer_cli/`：训练总入口。`cli.py`（`python -m banqi.trainer_cli <variant> [options]`：`--train-mode/--mcts-sims/--games-per-iter/--train-steps` 等）、`config_resolver.py`、`__main__.py`、`runners/`：`selfplay.py`（自对弈 runner，内常驻 `NnueDistillWorker`）、`expectimax_sidecar.py`（低频 NNUE 强自对弈 sidecar，监听 `ckpt_event`）、`local_loop.py`（`TRAIN_MODE=local` 单机双进程闭环：collector 子进程 + TrainWorker 经 LocalEpisodeStore 消费 + RegistryPublisher 指针发布）、`archive_feeder.py`、`offline.py`、`context.py`。
- `infra/`：L4 协作接口（统一训练架构）——`episode_store.py`（`EpisodeStore` Protocol + `LocalEpisodeStore` 目录/jsonl.gz 扫描，队列语义 get/get_nowait/qsize 可直连 TrainWorker）、`model_registry.py`（`ModelRegistry` Protocol + `LocalModelRegistry` 指针发布）；r2/scheduler 实现后续接入。
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

已规划拆出为独立仓库（见 §2.0），完整架构（前端分层、`#[tauri::command]` 清单、构建命令）见 **`src-tauri/ARCHITECTURE.md`**。

## 6. 分布式与存储（可选路径）

### 6.1 gRPC（`banqi-selfplay-worker` ↔ 训练端）

`proto/banqi_service.proto` 四 RPC：`ReportGameMeta`（上报元信息）、`PullGameData`（流式拉样本）、`FetchLatestModel`（流式拉模型热更新）、`SyncControl`（心跳/模拟数/暂停/算力随机化控制）。

### 6.2 MongoDB

`data_collector.rs` 直接写 Mongo；训练侧 `archiver.py`/`storage.py` 归档冷存储供复训。

### 6.3 JSONL

`py_data_collector` 与 Expectimax 强自对弈产 JSONL episode 文件。

### 6.4 Go 中心调度器（`server/`）

已规划拆出为独立仓库 `banqi-scheduler`（见 §2.0），完整架构（9 RPC 契约、internal 分层、proto 生成命令、R2 预签名直传与 GSPRT 判停）见 **`server/ARCHITECTURE.md`**。

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
- 2026-09-11：新增 `docs/unified_training_architecture.md`（单机/分布式统一架构设计与实施计划：三角色 Collector/Trainer/Registry 同构、推理下沉 Rust、EpisodeStore/ModelRegistry 双实现、退役清单与进度表）。
- 2026-09-11：分布式训练改造落地（决策：优先分布式，单机后续在其基础上改造）：Rust `registry/` 新增 `scheduler_registry.rs`（SchedulerRegistry：tonic 客户端对接 `proto/scheduler.proto` + reqwest 预签名 URL 下载网络/直传 R2）；`build.rs` 增编 scheduler.proto（依赖新增 `reqwest`/`sha2`）；`MatchResult` 新增 `game_outcomes`（rating 五项成对计数推导）；`banqi-collector` 新增 `--backend scheduler`（selfplay/rating 双任务，无任务退避轮询）。Python：`infra/` 新增 `R2EpisodeStore`（boto3）/`SchedulerModelRegistry`（上传 R2 + RegisterNetwork）与 `TRAIN_MODE=distributed`（`runners/distributed.py`，生成 `proto/scheduler_pb2*`）。Go scheduler ↔ Rust collector gRPC 互通冒烟通过；含 R2 全链路联调待 MinIO。详见 `docs/unified_training_architecture.md` 进度表。
- 2026-09-10：Tauri 前端由原生 HTML/JS 重写为 Vite + Vue 3 + TypeScript（`frontend/` 内源码 `src/`、组件/composables/api 分层、三栏布局重构，功能与 command 接口不变；`tauri.conf.json` 改用 `frontendDist=./frontend/dist` + devUrl:5173 + beforeDev/BuildCommand）。
- 2026-09-11：仓库拆分规划落档（方案 A，见 §2.0）：`server/` 与 `src-tauri/` 各自新增 `ARCHITECTURE.md`（内容自本文 §5/§6.4 拆出，本文改为引用）与独立 `.gitignore`（server 特有忽略规则自根 `.gitignore` 下沉）；主体仓库后续将进一步拆分 trainer/collector，接口边界为 Rust `registry/` + Python `infra/`。
- 2026-09-11：分布式 worker 资源分配与心跳/完整性加固：Go 调度器 GetTask 按 worker 上报线程数缩放下发局数（`SCHEDULER_THREADS_BASELINE` 基准，0=不缩放）；`HeartbeatRequest` 增 `client_version`/`memory_mb`，workers 表记录版本声明（版本变更打日志，不做强校验）；ReportEpisode/ReportMatchResult 校验 task↔worker 归属；Rust `SchedulerRegistry` 后台 30s 心跳（版本声明/资源/累计局数/running_task_id，感知 best 换网与 pause）、GetTask 上报真实可用内存（/proc/meminfo）、网络文件下载与缓存命中均做 sha256 SRI 校验（不符删除缓存拒绝使用）。
