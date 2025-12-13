// src/data_collector.rs
//
// 分布式数据收集 Worker
// 功能：
// 1. 通过 gRPC 连接到推理服务器获取策略评估
// 2. 运行 MCTS 自对弈生成游戏数据
// 3. 将数据保存到 MongoDB

use anyhow::Result;
use banqi_4x8::game_env::{DarkChessEnv, ACTION_SPACE_SIZE};
use banqi_4x8::mcts::Evaluator;
use banqi_4x8::mongodb_storage::MongoStorage;
use banqi_4x8::self_play::{SelfPlayWorker, ScenarioType};

// 引入生成的 gRPC 代码
use banqi_4x8::rpc::inference::inference_service_client::InferenceServiceClient;
use banqi_4x8::rpc::inference::PredictRequest;
use tonic::transport::Channel;

use std::env;
use std::sync::Arc;
use std::time::Duration;

// ============================================================================
// gRPC Evaluator 实现
// ============================================================================

/// 基于 gRPC 的评估器
/// 实现了 Evaluator trait (Async)
pub struct GrpcEvaluator {
    /// gRPC 客户端 (Clone 开销很小，内部共享连接池)
    client: InferenceServiceClient<Channel>,
}

impl GrpcEvaluator {
    /// 连接到推理服务器
    pub async fn connect(endpoint: String) -> Result<Self> {
        println!("正在连接推理服务: {} ...", endpoint);
        // 使用 tonic 的 connect 连接 (Lazy connection)
        let client = InferenceServiceClient::connect(endpoint).await?;
        println!("✅ 已连接到推理服务");
        Ok(Self { client })
    }
}

#[async_trait::async_trait]
impl Evaluator for GrpcEvaluator {
    async fn evaluate(&self, env: &DarkChessEnv) -> (Vec<f32>, f32) {
        // 1. 准备数据
        let obs = env.get_state();
        let board_flat: Vec<f32> = obs.board.as_slice().unwrap().to_vec();
        let scalars_flat: Vec<f32> = obs.scalars.as_slice().unwrap().to_vec();
        let masks = env.action_masks();

        // 2. 构造 gRPC 请求
        let request = tonic::Request::new(PredictRequest {
            board: board_flat,
            scalars: scalars_flat,
            masks: masks,
        });

        // 3. 发送请求
        // 必须 clone client，因为 generated method 需要 mut self 或者 ownership
        // Channel 是轻量级的，clone 是廉价的
        match self.client.clone().predict(request).await {
            Ok(response) => {
                let resp = response.into_inner();
                // 确保 policy 长度正确
                let policy = if resp.policy.len() == ACTION_SPACE_SIZE {
                    resp.policy
                } else {
                    eprintln!("⚠️ 服务器返回的 Policy 长度不匹配: {}, 预期 {}", resp.policy.len(), ACTION_SPACE_SIZE);
                    vec![0.0; ACTION_SPACE_SIZE]
                };
                (policy, resp.value)
            }
            Err(e) => {
                eprintln!("❌ gRPC 请求失败: {}", e);
                // 发生错误时返回默认值或 panic，这里选择返回均匀分布以避免 crash
                (vec![0.0; ACTION_SPACE_SIZE], 0.0)
            }
        }
    }
}

// ============================================================================
// 主程序
// ============================================================================

#[tokio::main]
async fn main() -> Result<()> {
    // 1. 配置
    let mongo_uri = env::var("MONGODB_URI").unwrap_or_else(|_| "mongodb://localhost:27017".to_string());
    let grpc_endpoint = env::var("INFERENCE_SERVER").unwrap_or_else(|_| "http://127.0.0.1:50051".to_string());
    
    // 获取 Worker ID (用于区分日志)
    let args: Vec<String> = env::args().collect();
    let worker_id = if args.len() > 1 {
        args[1].parse::<usize>().unwrap_or(0)
    } else {
        0
    };

    let mcts_sims = 200; // 搜索次数

    println!("=== 数据收集 Worker-{} 启动 ===", worker_id);
    println!("MongoDB: {}", mongo_uri);
    println!("Inference Server: {}", grpc_endpoint);
    println!("MCTS Sims: {}", mcts_sims);

    // 2. 初始化组件
    // 连接 MongoDB (异步版本)
    let mongo_storage = match MongoStorage::new(&mongo_uri, "banqi_training", "games").await {
        Ok(s) => Arc::new(s),
        Err(e) => {
            eprintln!("无法连接到 MongoDB: {}", e);
            return Ok(());
        }
    };
    
    // 连接 gRPC Server (异步)
    let evaluator = match GrpcEvaluator::connect(grpc_endpoint.clone()).await {
        Ok(e) => Arc::new(e),
        Err(e) => {
            eprintln!("无法连接到推理服务器: {}", e);
            return Ok(());
        }
    };

    // 3. 创建 Worker
    // 使用泛型支持后的 SelfPlayWorker
    let worker = SelfPlayWorker::with_scenario_dirichlet_and_temperature(
        worker_id,
        evaluator.clone(), // Arc<GrpcEvaluator>
        mcts_sims,
        ScenarioType::Standard,
        0.3,  // Alpha
        0.25, // Epsilon
        12,   // Temperature steps
    );

    // 4. 循环收集
    let mut game_count = 0;
    loop {
        let start_time = std::time::Instant::now();
        
        // 执行一局游戏 (异步)
        let episode = worker.play_episode(game_count).await;
        
        let duration = start_time.elapsed();

        if episode.samples.is_empty() {
            eprintln!("⚠️ 生成了空游戏数据，跳过上传");
            continue;
        }

        // 打印简报
        let winner_str = match episode.winner {
            Some(1) => "红胜",
            Some(-1) => "黑胜",
            _ => "平局",
        };
        println!(
            "[Worker-{}] Game #{}: 步数={}, 结果={}, 耗时={:.1}s ({:.1} steps/s)",
            worker_id,
            game_count + 1,
            episode.game_length,
            winner_str,
            duration.as_secs_f64(),
            episode.game_length as f64 / duration.as_secs_f64()
        );

        // 上传到 MongoDB (异步调用)
        match mongo_storage.save_games(0, &[episode]).await {
            Ok(()) => {
                // 上传成功，静默处理
            }
            Err(e) => {
                eprintln!("❌ MongoDB 上传失败: {}", e);
                tokio::time::sleep(Duration::from_secs(5)).await;
            }
        }

        game_count += 1;
    }
}