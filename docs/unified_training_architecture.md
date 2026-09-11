# 统一训练架构设计 — 单机/分布式闭环合流

> 日期：2026-09-11
> 前置文档：`docs/distributed_training_reference_survey.md`（调度器选型依据）、`docs/ARCHITECTURE.md`（现状快照）
> 目标：单机训练与分布式训练共用同一条 Collector/Trainer/Registry 代码路径，推理全部下沉 Rust，PyO3 推理回调退役。

---

## 1. 设计原则

1. **单机与分布式的差别只在协作方式**，不在功能：Rust 环境/搜索/推理与 Python 训练是两种模式完全共享的资产（L2）。
2. **推理只在 Rust**：Collector 自对弈用 Rust 原生推理器（TorchScript / ONNX），不再经 PyO3 回调传回 Python 推理。Python 预测器仅保留在 validate/benchmark 中做一致性校验。
3. **角色同构**：闭环 = 三个逻辑角色（Collector / Trainer / Registry），单机是它们的本地退化实现，分布式是远程实现；同一套接口，两套接线。
4. **指针推进语义统一**：两种模式的闭环推进都是"Registry 指针变更 → Collector 感知换网 → 产 episode → Trainer 消费 → 导出新模型 → 推进指针"。

## 2. 分层架构

```
L5 编排   单机: trainer_cli（顺序驱动，同机组装角色）
          分布式: Go scheduler（指针驱动，角色各自常驻）
              │ 差异仅"谁触发谁、节奏由谁定"
L4 协作   三个统一接口，各两实现：
          ├─ Inference（统一后仅 Rust 一侧）
          │    ├─ OnnxRunner / TorchRunner（现有 inference/，加模型热重载）
          │    └─ NnueEvaluator（Expectimax 旁路，不变）
          ├─ EpisodeStore
          │    ├─ LocalStore: 本地 jsonl.gz 目录（单机）
          │    └─ R2Store: ReportEpisode 签发预签名 PUT + R2 列举/下载（分布式）
          └─ ModelRegistry
               ├─ LocalRegistry: outputs/<variant>_latest.onnx + notify 文件 watch 热重载
               └─ SchedulerRegistry: GetTask / GetNetwork / RegisterNetwork（gRPC）
L3 角色   Collector（Rust，单一 bin）/ Trainer（Python）/ Registry（接口 + 两实现）
L2 功能   Rust: core/env + mcts + expectimax + inference（不动）
          Python: BanqiNet 训练 + NNUE 蒸馏 + ckpt 导出（不动）
```

## 3. 角色与接口

### 3.1 Collector（Rust，自对弈采集）

唯一新增的进程级代码，单机/分布式共用同一 bin，主循环同构：

```
loop:
    task     = registry.get_task()            # local: 读最新 onnx+参数; scheduler: GetTask
    model    = registry.ensure_model(task)    # local: watch 热重载; remote: 下载+sha 校验
    episodes = self_play::run(model, params)  # 复用 pipeline/self_play（MCTS / Expectimax）
    episode_store.put(episodes)               # local: 写目录; remote: ReportEpisode → R2 直传
    if task.kind == rating: registry.report_result(...)
```

- 演化来源：`banqi-selfplay-worker` / `data_collector` bin 与 `pipeline/self_play` 主干。
- Backend 通过启动配置切换：`--backend local|scheduler`。
- 模型热更新：LocalRegistry 用 `notify` crate watch 导出路径；SchedulerRegistry 由 GetTask 返回 best sha 触发。两种模式的"换网"语义一致：Registry 变更 → Collector 感知。

### 3.2 Trainer（Python，训练）

- episode 消费统一到 `EpisodeStore` Protocol：LocalStore 扫描目录（取代内存队列直喂）、R2Store 拉 R2。
- 新模型产出后：单机写 watch 路径（`outputs/<variant>_latest.onnx`）；分布式上传 R2 + `RegisterNetwork`。
- 入口仍为 `trainer_cli`（单机形态）；分布式 GPU 机上是薄壳 `trainer/gpu/`（消费 R2 + 注册网络），与调度器只耦合"拉数据 / 注册网络"两个接口。

### 3.3 Registry（模型版本与准入）

- 单机 = 本地文件指针（latest.onnx），无准入（每次导出即生效）。
- 分布式 = Go scheduler：best 指针 + 五项 GSPRT gatekeeper 晋级（已实现于 `server/`）。

## 4. 目标目录结构

```
4x8/
├── src/                      # L2 Rust 功能层（不动）
├── python/banqi/
│   ├── ...                   #   功能层：nn_model / train / nnue / checkpoint 等（不动）
│   ├── infra/                # L4（新）：episode_store.py + model_registry.py
│   │                         #   Protocol + local_* / r2_* / scheduler_* 实现
│   └── trainer_cli/          # 单机形态 Trainer+编排（保留原位，context.py 组装 infra）
├── collector/                # L3 Collector 分布式形态（Rust worker；由 worker/collector 升格）
├── trainer/gpu/              # L3 Trainer 分布式形态（薄壳；由 worker/trainer 升格）
├── scheduler/                # L3 Registry 分布式形态（现 server/ 迁入改名）
├── proto/                    # 契约（banqi_service.proto 退役，scheduler.proto 为准）
├── deploy/
└── docs/
```

目录迁移（`git mv`，无逻辑改动）：`server/ → scheduler/`，`worker/collector/ → collector/`，`worker/trainer/ → trainer/gpu/`；`ARCHITECTURE.md` §2/§6.4 路径同步。

## 5. 退役清单

| 现状组件 | 处置 | 原因 |
|---|---|---|
| `bridge/py_evaluator.rs`（自对弈推理回调） | 从自对弈链路移除 | 推理下沉 Rust |
| `python/banqi/predictor.py`、`selfplay/predictor.py` | 退出自对弈链路，仅 validate/benchmark 保留 | 同上 |
| `ckpt_event` 事件链（runners ↔ expectimax_sidecar） | 改为文件 watch（LocalRegistry） | 统一换网语义 |
| `proto/banqi_service.proto` + `selfplay/worker.py` + 旧 gRPC 生成代码 | 进 `legacy/` 或删除 | 被 scheduler.proto 取代 |
| `data_collector.rs`（MongoDB 直写） | 归档 | 数据出口统一为 EpisodeStore |
| 内存队列直喂 Trainer 的路径 | 改 LocalStore 目录扫描 | 统一 episode 消费接口 |

## 6. 实施计划与进度追踪

| # | 任务 | 产出 | 状态 |
|---|---|---|---|
| 1 | Rust LocalRegistry：onnx 加载 + notify 热重载，接入自对弈主干 | collector 模型热更新能力 | ✅ 已完成（`src/registry/local_registry.rs`） |
| 2 | 单机切 Rust 推理：进程内 Collector + LocalStore 替代 PyO3 回调路径；validate 冒烟验证一致性 | 单机闭环无 Python 推理 | 🟡 部分（`banqi-collector` 已全程 Rust 推理；validate 一致性冒烟待跑） |
| 3 | Python `infra/`：EpisodeStore/ModelRegistry Protocol + local 实现；Trainer 切 LocalStore | L4 接口显式化 | ✅ 已完成（`banqi/infra/` + `TRAIN_MODE=local` 走 `local_loop`） |
| 4 | Collector bin 化（`collector/`），支持 `--backend local` | 统一采集进程 | ✅ 已完成（`src/bin/collector.rs`，bin 名 `banqi-collector`） |
| 5 | Rust SchedulerBackend：gRPC(GetTask/ReportEpisode/ReportMatchResult/GetNetwork) + R2 直传 | 分布式 worker | 未开始 |
| 6 | R2Store + Trainer 分布式薄壳（`trainer/gpu/`：拉 R2 数据、注册网络） | 分布式 trainer | 未开始 |
| 7 | 目录迁移（§4）+ 退役清单执行 + ARCHITECTURE.md 改版 | 结构收敛 | 未开始 |
| 8 | 端到端联调：本地双进程冒烟 → scheduler+R2 沙箱闭环 | 验收 | 未开始（#1–#4 主干已具备单机双进程闭环） |

### 验收标准

- 单机：`trainer_cli` 全程无 Python 推理回调；Collector 与 Trainer 为独立进程经目录握手；`validate_smoke` 通过。
- 分布式：本地起 scheduler（SQLite 内存/临时库 + MinIO 模拟 R2）跑通"注册网络 → gatekeeper → 晋级 → selfplay 换网"闭环。
- 回归：两种模式下 episode 序列化格式一致，Trainer 无差别消费。

## 变更记录

- 2026-09-11：初版（统一架构设计 + 实施计划）。
- 2026-09-11：主干落地 #1/#3/#4——Rust 新增 `src/registry/`（`LocalRegistry`：onnx + notify 热重载；`LocalEpisodeStore`：jsonl.gz 目录落盘，复用 `episode_to_dict_json` 契约）与新 bin `banqi-collector`（`--backend local`，`run_match_core` + `OnnxEvaluator`，纯 Rust 推理）；Python 新增 `banqi/infra/`（EpisodeStore/ModelRegistry Protocol + local 实现）与 `TRAIN_MODE=local`（`runners/local_loop.py`：collector 子进程 + TrainWorker 经 LocalStore 消费 + RegistryPublisher 指针发布线程）。旧 selfplay 路径未动，退役清单延后执行。
