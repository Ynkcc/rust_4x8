// src/bin/selfplay_worker.rs
//
// Banqi 自对弈 Worker CLI 独立程序（自对弈 / 训练分离的分布式架构）。
//
// Worker 同时扮演两种 gRPC 角色：
//   1. **Client**（连接 Python 训练端）：
//      - `ReportGameMeta`：上报本批样本元信息（含 data_id），通知训练端可拉取。
//      - `SyncControl`：周期同步控制参数（模拟次数、暂停、算力随机化）。
//      - `FetchLatestModel`：周期拉取最新 TorchScript 模型并热更新。
//   2. **Server**（本地监听，供训练端拉取）：
//      - `PullGameData`：按 data_id 返回样本流（对局数据）。
//
// 自对弈使用 `SelfPlayRunner`（同步）。评估器通过 `WorkerEvaluator` enum
// 在「内置启发式」与「神经网络（TchEvaluator）」之间热切换，保持核心
// MCTS（GumbelMCTS）的 `E: Sized` 约束不变，零侵入。神经网络推理需
// `--features torch` 构建。

use anyhow::{Context, Result};
use clap::Parser;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::VecDeque;
use std::fs::File;
use std::io::Read;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use futures::Stream;
use rayon::prelude::*;
use tonic::transport::{Channel, Server};
use tonic::{Request, Response, Status};

use banqi_4x8::core::env::DarkChessEnv;
use banqi_4x8::core::mcts::Evaluator;
use banqi_4x8::engine::mcts_heuristic::HeuristicEvaluator;
use banqi_4x8::pipeline::self_play::serialize::episode_to_dict_json;
use banqi_4x8::pipeline::self_play::{GameEpisode, ScenarioType, SelfPlayConfig, run_self_play};

// 导入生成的 proto 定义
pub mod banqi_proto {
    tonic::include_proto!("banqi");
}

use banqi_proto::self_play_service_client::SelfPlayServiceClient;
use banqi_proto::self_play_service_server::{SelfPlayService, SelfPlayServiceServer};
use banqi_proto::{GameDataChunk, GameMeta, ModelRequest, PullDataRequest, WorkerStatus};

// ============================================================================
// CLI 参数与配置结构体
// ============================================================================

#[derive(Parser, Debug)]
#[command(name = "banqi-selfplay-worker", author = "Banqi Team", version = "0.1.0", about = "Banqi 自对弈分布式 Worker CLI")]
struct CliArgs {
    /// 配置文件路径（YAML 格式）
    #[arg(short, long)]
    config: Option<String>,

    /// Python 训练服务端 Host（Worker 作为 client 连接）
    #[arg(long, default_value = "127.0.0.1")]
    host: String,

    /// Python 训练服务端端口
    #[arg(short, long, default_value_t = 50051)]
    port: u16,

    /// Worker 作为 gRPC server 的监听地址
    #[arg(long, default_value = "0.0.0.0")]
    serve_host: String,

    /// Worker 作为 gRPC server 的监听端口（供训练端拉取样本）
    #[arg(long, default_value_t = 50052)]
    serve_port: u16,

    /// 并发自对弈工作线程数
    #[arg(short, long, default_value_t = 4)]
    threads: usize,

    /// MCTS 默认模拟次数
    #[arg(short, long, default_value_t = 64)]
    sims: usize,

    /// Worker 节点 ID 标识
    #[arg(short, long, default_value = "worker-rust-0")]
    worker_id: String,

    /// 拉取模型文件的临时落盘路径
    #[arg(long, default_value = "/tmp/banqi_worker_model.pt")]
    model_cache_path: String,
}

#[derive(Debug, Deserialize, Serialize)]
struct WorkerConfigFile {
    host: Option<String>,
    port: Option<u16>,
    serve_host: Option<String>,
    serve_port: Option<u16>,
    threads: Option<usize>,
    mcts_sims: Option<usize>,
    worker_id: Option<String>,
    model_cache_path: Option<String>,
}

// ============================================================================
// 评估器：内置启发式 / 神经网络 热切换（enum 保持 E: Sized，零侵入核心 MCTS）
// ============================================================================

enum WorkerEvaluator {
    Heuristic(HeuristicEvaluator),
    #[cfg(feature = "torch")]
    Torch(banqi_4x8::engine::mcts_dl::TchEvaluator<DarkChessEnv>),
}

impl Evaluator<DarkChessEnv> for WorkerEvaluator {
    fn evaluate(&self, envs: &[DarkChessEnv]) -> (Vec<Vec<f32>>, Vec<f32>) {
        match self {
            Self::Heuristic(h) => h.evaluate(envs),
            #[cfg(feature = "torch")]
            Self::Torch(t) => t.evaluate(envs),
        }
    }
}

// ============================================================================
// 共享状态
// ============================================================================

/// 一批自对弈样本：data_id 用于标识批次，episodes 为该批对局。
type SampleBatch = (String, Vec<GameEpisode>);

struct WorkerState {
    /// 当前评估器（启发式 / 神经网络之间热切换）
    evaluator: RwLock<WorkerEvaluator>,
    /// 已生成待训练端拉取的样本批次队列（FIFO）
    sample_queue: Mutex<VecDeque<SampleBatch>>,
    /// 动态控制参数（由 SyncControl 同步）
    current_sims: AtomicUsize,
    is_paused: AtomicBool,
    playout_cap_random: AtomicBool,
    completed_games: AtomicUsize,
    /// 当前已加载模型版本（用于跳过重复热更新）
    current_model_version: Mutex<String>,
}

impl WorkerState {
    fn new(initial_sims: usize) -> Self {
        Self {
            evaluator: RwLock::new(WorkerEvaluator::Heuristic(HeuristicEvaluator::new())),
            sample_queue: Mutex::new(VecDeque::new()),
            current_sims: AtomicUsize::new(initial_sims),
            is_paused: AtomicBool::new(false),
            playout_cap_random: AtomicBool::new(false),
            completed_games: AtomicUsize::new(0),
            current_model_version: Mutex::new(String::new()),
        }
    }

    /// 弹出下一个待拉取批次。`data_id` 为空时取队首；非空时按 ID 精确匹配。
    fn pop_batch(&self, data_id: &str) -> Option<SampleBatch> {
        let mut q = self.sample_queue.lock().unwrap();
        if data_id.is_empty() {
            return q.pop_front();
        }
        if let Some(pos) = q.iter().position(|(id, _)| id == data_id) {
            return q.remove(pos);
        }
        None
    }
}

// ============================================================================
// gRPC Server：向训练端提供 PullGameData
// ============================================================================

pub struct WorkerServicer {
    state: Arc<WorkerState>,
    chunk_size: usize,
}

impl WorkerServicer {
    fn new(state: Arc<WorkerState>) -> Self {
        Self {
            state,
            chunk_size: 64 * 1024,
        }
    }
}

#[tonic::async_trait]
impl SelfPlayService for WorkerServicer {
    type PullGameDataStream = Pin<Box<dyn Stream<Item = Result<GameDataChunk, Status>> + Send>>;
    type FetchLatestModelStream = Pin<Box<dyn Stream<Item = Result<banqi_proto::ModelChunk, Status>> + Send>>;

    // 以下 RPC 由训练端（Python）作为 server 提供，worker 侧不实现，返回 UNIMPLEMENTED。
    async fn report_game_meta(
        &self,
        _request: Request<GameMeta>,
    ) -> Result<Response<banqi_proto::ReportMetaResponse>, Status> {
        Err(Status::unimplemented("report_game_meta not served by worker"))
    }

    async fn fetch_latest_model(
        &self,
        _request: Request<ModelRequest>,
    ) -> Result<Response<Self::FetchLatestModelStream>, Status> {
        Err(Status::unimplemented("fetch_latest_model not served by worker"))
    }

    async fn sync_control(
        &self,
        _request: Request<WorkerStatus>,
    ) -> Result<Response<banqi_proto::ControlCommand>, Status> {
        Err(Status::unimplemented("sync_control not served by worker"))
    }

    async fn pull_game_data(
        &self,
        request: Request<PullDataRequest>,
    ) -> Result<Response<Self::PullGameDataStream>, Status> {
        let req = request.into_inner();
        let (data_id, episodes) = match self.state.pop_batch(&req.data_id) {
            Some(batch) => batch,
            None => {
                // 暂无待消费样本：返回空分片，训练端据此感知无数据。
                let empty = vec![Ok(GameDataChunk {
                    data_id: req.data_id.clone(),
                    payload: Vec::new(),
                    is_last: true,
                })];
                return Ok(Response::new(Box::pin(futures::stream::iter(empty))));
            }
        };

        // 序列化本批所有对局为 episode dict 列表（与 PyO3 episode_to_dict 契约一致）
        let jsons: Vec<Value> = episodes.iter().map(episode_to_dict_json).collect();
        let payload =
            serde_json::to_vec(&jsons).map_err(|e| Status::internal(format!("序列化失败: {e}")))?;

        let chunks = split_payload_chunks(data_id, payload, self.chunk_size);
        Ok(Response::new(Box::pin(futures::stream::iter(chunks))))
    }
}

/// 将 payload 按固定大小切分为 `GameDataChunk` 流。
fn split_payload_chunks(
    data_id: String,
    payload: Vec<u8>,
    chunk_size: usize,
) -> Vec<Result<GameDataChunk, Status>> {
    let mut out = Vec::new();
    if payload.is_empty() {
        out.push(Ok(GameDataChunk {
            data_id,
            payload: Vec::new(),
            is_last: true,
        }));
        return out;
    }
    let total = payload.len();
    let mut offset = 0usize;
    while offset < total {
        let end = (offset + chunk_size).min(total);
        let is_last = end == total;
        out.push(Ok(GameDataChunk {
            data_id: data_id.clone(),
            payload: payload[offset..end].to_vec(),
            is_last,
        }));
        offset = end;
    }
    out
}

// ============================================================================
// gRPC Client：ReportGameMeta + SyncControl + FetchLatestModel
// ============================================================================

/// 周期性上报心跳 / 同步控制参数（模拟次数、暂停、算力随机化）。
async fn sync_control_loop(
    client: SelfPlayServiceClient<Channel>,
    state: Arc<WorkerState>,
    worker_id: String,
    threads: usize,
) {
    let mut client = client;
    let mut interval = tokio::time::interval(Duration::from_secs(5));
    loop {
        interval.tick().await;
        let status = WorkerStatus {
            worker_id: worker_id.clone(),
            current_threads: threads as i32,
            completed_games: state.completed_games.load(Ordering::Relaxed) as i32,
        };
        match client.sync_control(status).await {
            Ok(resp) => {
                let cmd = resp.into_inner();
                if cmd.mcts_sims > 0 {
                    state.current_sims.store(cmd.mcts_sims as usize, Ordering::Relaxed);
                }
                state.is_paused.store(cmd.pause_self_play, Ordering::Relaxed);
                state.playout_cap_random.store(cmd.playout_cap_random, Ordering::Relaxed);
            }
            Err(e) => eprintln!("   ⚠️ 控制指令同步失败: {e}"),
        }
    }
}

/// 周期性拉取最新模型文件并热更新共享评估器。
#[cfg(feature = "torch")]
async fn fetch_model_loop(
    client: SelfPlayServiceClient<Channel>,
    state: Arc<WorkerState>,
    model_cache_path: String,
) {
    use tokio::fs::File as TokioFile;
    use tokio::io::AsyncWriteExt;

    let mut client = client;
    let mut interval = tokio::time::interval(Duration::from_secs(10));
    let mut local_version = String::new();
    loop {
        interval.tick().await;
        let req = ModelRequest {
            current_version: local_version.clone(),
        };
        let mut stream = match client.fetch_latest_model(req).await {
            Ok(s) => s.into_inner(),
            Err(e) => {
                eprintln!("   ⚠️ 拉取模型元信息失败: {e}");
                continue;
            }
        };

        let mut buf = Vec::new();
        let mut version = String::new();
        while let Ok(Some(chunk)) = stream.message().await {
            if version.is_empty() && !chunk.version.is_empty() {
                version = chunk.version.clone();
            }
            buf.extend_from_slice(&chunk.chunk_data);
            if chunk.is_last {
                break;
            }
        }

        if version.is_empty() || version == local_version {
            continue; // 无新版本，跳过
        }
        if buf.is_empty() {
            eprintln!("   ⚠️ 模型流为空，跳过");
            continue;
        }

        // 写入临时文件
        let mut f = match TokioFile::create(&model_cache_path).await {
            Ok(f) => f,
            Err(e) => {
                eprintln!("   ⚠️ 创建模型缓存文件失败: {e}");
                continue;
            }
        };
        if let Err(e) = f.write_all(&buf).await {
            eprintln!("   ⚠️ 写入模型缓存失败: {e}");
            continue;
        }
        drop(f);

        // 加载并热更新评估器
        match load_tch_evaluator(&model_cache_path) {
            Ok(eval) => {
                if let Ok(mut guard) = state.evaluator.write() {
                    *guard = WorkerEvaluator::Torch(eval);
                    *state.current_model_version.lock().unwrap() = version.clone();
                    println!("   ✅ 模型热更新成功: version={}", version);
                }
                local_version = version;
            }
            Err(e) => eprintln!("   ⚠️ 模型加载失败: {e}"),
        }
    }
}

#[cfg(feature = "torch")]
fn load_tch_evaluator(
    path: &str,
) -> Result<banqi_4x8::engine::mcts_dl::TchEvaluator<DarkChessEnv>> {
    use banqi_4x8::engine::mcts_dl::{ModelWrapper, TchEvaluator};
    let wrapper = ModelWrapper::load_from_file(path).map_err(|e| anyhow::anyhow!("{e}"))?;
    Ok(TchEvaluator::<DarkChessEnv>::new(Arc::new(wrapper)))
}

// ============================================================================
// 主程序入口
// ============================================================================

#[tokio::main]
async fn main() -> Result<()> {
    let args = CliArgs::parse();

    // 1. 读取配置文件（若指定）
    let mut host = args.host;
    let mut port = args.port;
    let mut serve_host = args.serve_host;
    let mut serve_port = args.serve_port;
    let mut threads = args.threads;
    let mut mcts_sims = args.sims;
    let mut worker_id = args.worker_id;
    let mut model_cache_path = args.model_cache_path;

    if let Some(config_path) = args.config {
        println!("📄 正在读取配置文件: {}", config_path);
        let mut file = File::open(&config_path)
            .with_context(|| format!("无法打开配置文件: {}", config_path))?;
        let mut contents = String::new();
        file.read_to_string(&mut contents)?;
        let cfg: WorkerConfigFile = serde_yaml::from_str(&contents)
            .with_context(|| "解析 YAML 配置文件失败")?;

        if let Some(h) = cfg.host { host = h; }
        if let Some(p) = cfg.port { port = p; }
        if let Some(h) = cfg.serve_host { serve_host = h; }
        if let Some(p) = cfg.serve_port { serve_port = p; }
        if let Some(t) = cfg.threads { threads = t; }
        if let Some(s) = cfg.mcts_sims { mcts_sims = s; }
        if let Some(w) = cfg.worker_id { worker_id = w; }
        if let Some(p) = cfg.model_cache_path { model_cache_path = p; }
    }

    println!("============================================================");
    println!("🚀 Banqi 自对弈 Worker CLI 启动");
    println!("   - Worker ID        : {}", worker_id);
    println!("   - Trainer (client) : gRPC://{}:{}", host, port);
    println!("   - Serve  (server)  : {}:{}", serve_host, serve_port);
    println!("   - Parallel Threads : {}", threads);
    println!("   - MCTS Sims        : {}", mcts_sims);
    println!("   - NN 推理          : {}", if cfg!(feature = "torch") { "已启用" } else { "未启用(启发式)" });
    println!("============================================================");

    // 2. 作为 client 连接 Python 训练服务端
    let server_addr = format!("http://{}:{}", host, port);
    let channel = Channel::from_shared(server_addr.clone())?
        .connect_timeout(Duration::from_secs(5))
        .connect()
        .await
        .with_context(|| format!("无法连接训练服务端: {}", server_addr))?;
    let grpc_client = SelfPlayServiceClient::new(channel);
    println!("✅ 已连接训练服务端 {}", server_addr);

    // 3. 共享状态
    let state = Arc::new(WorkerState::new(mcts_sims));

    // 4. 启动 gRPC Server（供训练端拉取样本）
    let serve_addr: std::net::SocketAddr = format!("{}:{}", serve_host, serve_port).parse()?;
    let servicer = WorkerServicer::new(state.clone());
    tokio::spawn(async move {
        match Server::builder()
            .add_service(SelfPlayServiceServer::new(servicer))
            .serve(serve_addr)
            .await
        {
            Ok(_) => println!("🛑 Worker gRPC Server 已退出"),
            Err(e) => eprintln!("⚠️ Worker gRPC Server 启动/运行失败: {e}"),
        }
    });
    println!("✅ Worker gRPC Server 已在 {}:{} 启动", serve_host, serve_port);

    // 5. 启动 client 后台任务
    tokio::spawn(sync_control_loop(
        grpc_client.clone(),
        state.clone(),
        worker_id.clone(),
        threads,
    ));
    #[cfg(feature = "torch")]
    tokio::spawn(fetch_model_loop(
        grpc_client.clone(),
        state.clone(),
        model_cache_path,
    ));
    #[cfg(not(feature = "torch"))]
    {
        let _ = &model_cache_path;
        println!(
            "   (NN 推理未启用，Worker 使用内置启发式；如需神经网络热更新请以 --features torch 构建)"
        );
    }

    // 6. 主线程多线程自对弈数据收集循环
    let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build()?;
    let mut batch_index = 0usize;

    loop {
        if state.is_paused.load(Ordering::Relaxed) {
            tokio::time::sleep(Duration::from_secs(1)).await;
            continue;
        }

        let sims = state.current_sims.load(Ordering::Relaxed);
        let playout_cap_random = state.playout_cap_random.load(Ordering::Relaxed);
        let config = SelfPlayConfig {
            mcts_sims: sims,
            max_considered_actions: 16,
            scenario: ScenarioType::Standard,
            c_scale: 1.0,
            gumbel_scale: 1.0,
            playout_cap_random_enabled: playout_cap_random,
            fast_mcts_sims: 16,
            full_search_prob: 0.25,
        };

        batch_index += 1;
        let start_time = Instant::now();

        // 每次并行生成 `threads` 局游戏
        let episodes = {
            let eval_guard = state.evaluator.read().unwrap();
            pool.install(|| {
                (0..threads)
                    .into_par_iter()
                    .map(|_| run_self_play(&*eval_guard, &config, DarkChessEnv::new))
                    .collect::<Vec<_>>()
            })
        };

        let duration = start_time.elapsed();
        let total_steps: usize = episodes.iter().map(|ep| ep.game_length).sum();
        state.completed_games.fetch_add(episodes.len(), Ordering::Relaxed);

        println!(
            "[{}] Batch #{}: 生成 {} 局游戏 | 总步数: {} | 耗时: {:.2}s ({:.1} steps/s)",
            worker_id,
            batch_index,
            episodes.len(),
            total_steps,
            duration.as_secs_f64(),
            total_steps as f64 / duration.as_secs_f64()
        );

        // 入队待训练端拉取
        let game_count = episodes.len();
        let data_id = format!("{}-{}-{}", worker_id, batch_index, get_timestamp());
        {
            let mut q = state.sample_queue.lock().unwrap();
            q.push_back((data_id.clone(), episodes));
            // 限制队列积压，防止训练端消费不及时导致内存膨胀
            while q.len() > 8 {
                q.pop_front();
            }
        }

        // 上报元信息，通知训练端可拉取该批数据
        let meta = GameMeta {
            worker_id: worker_id.clone(),
            data_id: data_id.clone(),
            game_count: game_count as i32,
            total_steps: total_steps as i32,
            winner: 0,
            timestamp: get_timestamp() as i64,
            model_version: {
                let v = state.current_model_version.lock().unwrap();
                if v.is_empty() {
                    "heuristic-v1".to_string()
                } else {
                    v.clone()
                }
            },
        };
        {
            let mut client = grpc_client.clone();
            match client.report_game_meta(meta).await {
                Ok(resp) => {
                    if resp.into_inner().accepted {
                        // 静默成功，训练端会主动 PullGameData 拉取
                    }
                }
                Err(e) => eprintln!("   ⚠️ 元信息上报失败: {e}"),
            }
        }

        // 短暂休眠避免过快循环
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}

fn get_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}
