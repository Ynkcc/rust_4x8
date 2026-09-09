# 分布式训练移植参考调研 — 子 agent (code-explorer) 探索报告

> 调研日期：2026-09-09
> 范围：`/home/ynk/Projects/Games-Entertainment/engine-testing` 下各分布式项目的架构对比
> 背景：计划将 4x8 暗棋平台改为分布式训练（中心调度：2C2G 30Mbps 云服务器；存储：Cloudflare R2；worker：自有 + 朋友设备）

## 一图速览

| 项目 | 角色 | 服务端技术栈 | 协议 | 调度方式 | 数据/模型存储 |
|---|---|---|---|---|---|
| lczero-server | 自对弈调度+存储 | Go(gin)+PostgreSQL+nginx | HTTP表单/multipart+JSON | worker 拉取(`/next_game`) | 本地磁盘+Cloudflare CDN，S3仅备份 |
| lczero-client | 自对弈 worker | Go(单文件) | 同上 | 轮询+预取 | 本地缓存网络(sha+TTL) |
| lczero-training | 训练侧(闭环消费端) | Python(TF旧)/JAX+C++新管线 | 文件(tar/gz) | 人工/半自动 | 从 storage.lczero.org 拉数据 |
| fishtest | 引擎测试(验胜率) | FastAPI+MongoDB | JSON REST | worker 拉取+服务端判停 | MongoDB+磁盘 PGN |
| OpenBench | 引擎测试(多引擎) | Django+Postgres | HTTP表单+JSON | worker 拉取 | Postgres元数据+服务器文件 |
| KataGo+katago-server | 自对弈+评分 | Django/DRF+PG+Redis+Celery | REST/DRF+multipart | worker 拉取(任务随机配对) | django-storages 可插拔(本地/GCS) |
| pentanomial | 统计工具 | 无(纯Python) | 无 | 无 | 无 |

## 1. lczero-server + lczero-client（LC0 自对弈基础设施）

**架构要点**
- 服务端单文件 `lczero-server/main.go`（约1800行，gin+gorm+Postgres），核心路由 `setupRouter()`：`POST /next_game`（调度）、`POST /upload_game`（收数据）、`POST /upload_network`（收模型）、`GET /get_network`（发模型）、`POST /match_result`。
- 调度是**无状态拉取式**：worker 报 user+token，服务端查该 training run 的 `BestNetworkID` 下发"train"任务（sha+参数JSON+开局书URL）；若有未完成的 gatekeeper match（按 `target_slice` 切片抽样新网络），优先下发 match 任务对打。游戏编号靠 Postgres `UPDATE ... RETURNING last_game` 原子自增（`main.go:673`）。
- 模型分发：`getNetwork` 直接 **301 重定向到 Cloudflare CDN**（`config.URLs.NetworkLocation`，本地缺失时再回退备份地址）。上传支持与前一网络的 **XOR delta 差量上传**（`prev_delta_sha`，省带宽，`main.go:269-360`）。
- 数据存储：训练 chunk（gz）和 PGN 写服务器**本地磁盘** `games/runN/training.N.gz`，元数据进 Postgres；S3 只用于 pg_dump 备份；训练数据公开下载经 nginx cache + Cloudflare。
- 客户端 `lczero-client/lc0_main.go`：主循环 nextGame → 按 sha 下载网络（本地缓存+文件锁+keepTime 过期清理+**提前预取下一网络**）→ 起 lc0 子进程自对弈 → multipart 上传。版本管控：`MinClientVersion/MinEngineVersion` + 每 run 的 `PermissionExpr`。

**部署资源**：Go 单二进制 + Postgres + nginx + 本地大磁盘（数据全落盘），Cloudflare CDN 扛模型分发。

**移植评估（2C2G/30Mbps/R2/gRPC）：难度低-中，六个里最贴合目标场景**
- 架构同构（自对弈采集+模型分发+中心调度一体），worker 拉模式天然支持异地/NAT。
- 必改三处：① 数据上传经服务器中转 → 30Mbps 上行必爆，改成**服务端只收元数据+签发 R2 预签名 URL、worker 直传 R2**（其 getNetwork 已是"重定向到 CDN"思路，反向打通上传很顺）；② 模型放 R2 公开桶/CDN，替代本地磁盘；③ 协议换 gRPC：Go 端改造顺手，`NextGameResponse`（client_http.go）就是现成的 proto 蓝本。
- 2C2G 可跑（Go 进程内存占用极小，Postgres 可留或换轻量库）。

## 2. lczero-training（训练闭环）

**架构要点**：不含服务端，是闭环的"消费端"。两代管线并存：
- 旧管线 `tf/`：TensorFlow（chunkparser → shufflebuffer → tfprocess → train.py），YAML 配置，从 `storage.lczero.org` 下载按小时打包的 tar 数据训练，产出 weights.txt。
- 新管线 `src/`（2025，JAX/Orbax + meson 编译的 C++ dataloader + textproto 配置 + TUI/daemon）：支持 RL（`chunks_per_network` 滑动窗口、hanze 采样）与 SL 两种模式，watch 目录自动消费新 chunk，`leela2jax/jax2leela` 做权重格式转换。

**闭环组织**：worker 上传数据 → 服务器落盘/CDN → 训练机下载 → 训练出新网络 → `curl /upload_network` 传回服务器 → 服务端跑 gatekeeper match 验证 → `setBestNetwork` 推进 `BestNetworkID` 指针 → 全网 worker 自动换网。**训练与自对弈由服务器"指针"解耦**，无需服务端感知训练。

**移植评估**：与分布式场景基本正交——2C2G 无法训练，训练必须放独立 GPU 机器；只需让训练脚本对接"从 R2 拉数据、向服务器注册新网络"两个接口即可。难度：低（作为外围消费端）。

## 3. fishtest（Stockfish 测试框架）

**架构要点**
- worker（`fishtest/worker/worker.py`）：主循环 → `POST /api/request_task`（JSON：用户名+密码+worker_info 含 uname/并发/内存/编译器/UUID），服务端**按机器规格分配匹配的任务量**；无任务回 `task_waiting` 轮询。拿到任务后从 GitHub 拉引擎源码+fastchess 源码本地编译、下载开局书、跑 SPRT/SPSA/固定局数；期间 `/api/beat` 心跳、`/api/update_task` 增量上报、`/api/upload_pgn` 传棋谱。安全管控极严：`WORKER_VERSION=325`、SRI 哈希校验 worker 未被篡改、强制自动升级。
- server（`fishtest/server/fishtest/`）：FastAPI + **MongoDB**（本机）+ 主/备多实例 + `rundb.py`（任务队列、按 worker 资源拆分 task、**SPRT 服务端集中判停**（`stats/stat_util.py` 五项 GSPRT）、SPSA 参数流、Elo/置信区间）。开发者经 Web UI 或 GitHub PR 集成提交测试。
- 特点：**面向"验证改动是否变强"而非数据采集**——结果只上传胜负统计+PGN，训练数据闭环不存在。

**移植评估：难度高（不建议整体移植）**
- MongoDB、主备实例、GitHub PR 流、SPSA 全家在 2C2G 上不现实；gRPC 化要重写全部 API 层（worker 5 万行逻辑）。
- **但它是最该"抄算法"的项目**：`stats/` 下的 pentanomial GSPRT 判停、按 worker 资源分配任务、worker 心跳/版本/SRI 防篡改机制，可直接搬进自对弈调度器。

## 4. OpenBench

**架构要点**
- Django 单体 + Postgres + fastchess。客户端循环（`Client/worker.py`）：`POST clientGetWorkload`（machine_id+secret）拿 workload（test/tune/datagen，返回 dev/base 引擎 Git 分支、NNUE 网络 sha、时间控制、SPRT/局数参数、runner 分布）→ `clientGetNetwork` 下载网络（sha 校验）→ GitHub 拉引擎编译、ISA 检测选二进制 → 本地 bench 校验（节点数一致性）→ 按实测 NPS 缩放时间控制 → 多 runner 线程对打 → `clientSubmitResults` 增量上报，**判停用从 fishtest 移植的 Trinomial/Pentanomial SPRT**（`OpenBench/stats.py`），可选 `clientSubmitPGN`、心跳。
- 特色：单实例服务**十几个引擎**、SPSA 调参（`spsa_utils.py`）、SYZYGY 支持；网络文件是 Django FileField 存服务器本地。

**移植评估：难度中-高**。服务器侧最轻（Django+PG 能塞进 2C2G，模型改存 R2 只需换 storage backend），但要改成"自对弈数据采集"需新增数据上传管线与网络晋级机制，gRPC 化同样要重写 Client 通信层。适合当"调度+判停"参考骨架，不适合直接搬。

## 5. KataGo + katago-server

**架构要点**
- **客户端** `KataGo/cpp/distributed/client.cpp`（`katago contribute`）：通用 HTTP(S)（httplib），四件事：`getNextTask`（POST 返回 JSON，`kind=selfplay` 或 `rating`，含网络 downloadUrl、client config、startPoses、task_rep_factor）、`downloadModelIfNotPresent`（**URL 由服务器下发**，sha256 校验+并发节流+镜像加速+预取最新网络）、`uploadTrainingGameAndData`（multipart 传 sgf+npz）、`uploadRatingGame`。**客户端不直接碰 GCS**——存储完全由服务端决定的 URL 抽象。selfplay 本体是 `katago selfplay` 本地跑。
- **服务端** `katago-server/`：Cookiecutter Django+DRF+Postgres+Redis+Celery+nginx/Traefik，docker-compose（local/production.yml）+kubernetes+GCB。`POST /api/tasks/` 创建任务：按 `rating_game_probability` 概率让新网络与历史网络配对打评分（`RatingNetworkPairerService`），否则发 selfplay 任务+随机 startpos；git revision 白名单管控客户端版本。**存储是教科书级可插拔设计**：NETWORK/SGF/NPZ 三类文件各自用环境变量在 `FileSystemStorage` 与 `GoogleCloudStorage`（django-storages，5MB 分块）间切换（`settings/production.py:78-105`）。Celery 定时任务算 BayesElo、刷物化视图。

**移植评估：难度中-高，但协议设计最值得抄**
- 服务端全家桶在 2C2G 上跑不动，需砍掉 Redis/Celery/Traefik 精简为单 Postgres+单应用进程；
- 客户端协议（selfplay/rating 双任务、rating 配对采样、startpos 分发、sha 校验、重试/节流、预取）**可直接照抄进 gRPC proto**；
- GCS 对接换成 R2 = django-storages 换 S3 后端（R2 兼容 S3 API，配 endpoint 即可），或绕过 Django 用预签名 URL。

## 6. pentanomial

纯数学小工具（vdbergh 出品，`pentanomial/`：LLRsimulate.py、SPRT_pentanomial.py、sprta5.py、doc/ 推导 PDF）：用 BayesElo 模型蒙特卡洛模拟换色配对对局的五项分布（LL/LD/DD/DW/WW），实现**基于五项分布的广义 SPRT（GSPRT）**，并证明三项 SPRT 高估置信区间、五项模型省约 20% 对局数（README 实测：1809 vs 2191 局）。它是 fishtest/OpenBench 判停统计的理论参考实现，**不是调度系统**。

**移植评估：零适配，直接拿用**——即"胜率验证/判停模块"的现成实现（fishtest 的 `stats/` 和 OpenBench 的 `stats.py` 是其工程化版本）。

## 总结建议（面向"2C2G 30Mbps + R2 + 异地 worker + gRPC"）

1. **骨架选 lczero-server/client**：唯一同时具备"自对弈采集+模型分发+中心调度"三件套且服务端极轻的项目；改造成本最低（Go→gRPC 顺滑，网络分发本就是 CDN 重定向模式）。
2. **存储模式抄 KataGo server**：统一抽象为"URL 下发+对象存储直传"，R2（S3 API、零出口流量费）天然替代 GCS；上传务必走预签名直传，2C2G 服务器绝不能做数据中转（这是 lczero-server 现架构唯一必须动刀的地方）。
3. **调度与可靠性抄 fishtest**：按 worker 资源匹配任务量、心跳+判停、worker 版本/完整性校验（SRI）、无任务退避轮询。
4. **判停统计抄 pentanomial/fishtest stats**：五项 GSPRT 直接复用，gatekeeper match（lczero 的 target_slice 抽样 + KataGo 的 rating 配对）二选一或混合。
5. **不要整体移植** fishtest（MongoDB/多实例太重）与 katago-server（容器全家桶），二者均按"拆算法、弃框架"方式取材即可。训练侧（lczero-training）放独立 GPU 机，与中心服务仅通过"拉数据/注册网络"两个接口耦合。

## 目标架构（落到 banqi 4x8 仓库）

```
R2:  episodes/*.jsonl.gz  +  networks/<sha>.pt/.nnue
2C2G 服务器: 调度器（改造 proto 增任务分配/网络指针/预签名URL RPC）
            + gatekeeper 对打编排 + Postgres/SQLite 元数据
GPU 机(自有): trainer_cli 消费 R2 数据，出网注册到服务器
朋友设备:    banqi-selfplay-worker 拉任务/模型，直传 R2
```
