// src/data_collector.rs

use crate::game_env::{DarkChessEnv, ACTION_SPACE_SIZE};
use crate::mcts::Evaluator;
use crate::mongodb_storage::MongoStorage;
use crate::self_play::{SelfPlayWorker, ScenarioType};
use anyhow::Result;
use std::env;
use std::sync::Arc;
use std::time::Duration;
use tokio::runtime::Runtime;

// ============================================================================
// gRPC 模块引入 (模拟)
// 实际项目中，这部分通常由 tonic-build 在 build.rs 中生成
// 这里假设生成的代码位于 crate::rpc 模块下
// ============================================================================
pub mod rpc_mock {
    // 这里使用 tonic 的宏或手动引入，为了代码可编译，这里做示意
    // use crate::rpc::inference_service_client::InferenceServiceClient;
    // use crate::rpc::PredictRequest;
    
    // ⚠️ 实际使用时请替换为真实的生成代码引用
    pub type InferenceServiceClient<T> = (); 
    pub struct PredictRequest {
        pub board: Vec<f32>,
        pub scalars: Vec<f32>,
        pub masks: Vec<i32>,
    }
}
// ----------------------------------------------------------------------------

// 引入真实的 tonic 依赖 (需要在 Cargo.toml 中添加)
use tonic::transport::Channel;
// 假设你的 proto package 是 banqi.inference
// use crate::rpc::inference::inference_service_client::InferenceServiceClient;
// use crate::rpc::inference::PredictRequest;

// 临时占位，为了让上面的代码逻辑通顺，请根据你的 proto 生成路径修改
// 假设生成的 client 名字是 BanqiInferenceClient
type BanqiInferenceClient = tonic::codegen::InterceptedService<Channel, ()>; 
// 这里为了演示逻辑，我们需要自定义一个 Evaluator

// ============================================================================
// gRPC Evaluator 实现
// ============================================================================

/// 基于 gRPC 的评估器
/// 将 MCTS 的同步 evaluate 调用转换为异步 gRPC 请求
pub struct GrpcEvaluator {
    // 使用 Option 是为了在 Drop 时处理 Runtime，或者直接持有 Runtime
    rt: Arc<Runtime>,
    // 实际的 gRPC Client (需要是从 proto 生成的类型)
    // client: InferenceServiceClient<Channel>, 
    // 这里使用一个占位符 URL 来模拟，实际开发中请替换为生成的 Client 类型
    endpoint: String,
}

impl GrpcEvaluator {
    pub async fn connect(endpoint: String) -> Result<Self> {
        // 创建 Tokio Runtime 用于在同步上下文中执行异步 gRPC
        let rt = Arc::new(Runtime::new()?);

        // 实际连接代码示例:
        // let client = InferenceServiceClient::connect(endpoint.clone()).await?;
        
        println!("已连接到推理服务: {}", endpoint);

        Ok(Self {
            rt,
            endpoint,
        })
    }
    
    // 模拟 gRPC 请求逻辑
    async fn call_remote_inference(&self, env: &DarkChessEnv) -> (Vec<f32>, f32) {
        // 1. 准备数据
        let obs = env.get_state();
        let board_flat: Vec<f32> = obs.board.as_slice().unwrap().to_vec();
        let scalars_flat: Vec<f32> = obs.scalars.as_slice().unwrap().to_vec();
        let masks = env.action_masks();

        /* // 2. 构造 gRPC 请求 (示例)
        let request = tonic::Request::new(PredictRequest {
            board: board_flat,
            scalars: scalars_flat,
            masks: masks.clone(),
        });

        // 3. 发送请求
        // let mut client = self.client.clone();
        // let response = client.predict(request).await.unwrap().into_inner();
        
        // 4. 解析响应
        // return (response.policy, response.value);
        */

        // ⚠️ 模拟返回 (为了代码能跑通，实际必须实现上面的 gRPC 调用)
        // 在没有真实 Server 时，这里返回随机值防止 Panic
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let mut policy = vec![0.0; ACTION_SPACE_SIZE];
        let valid_count = masks.iter().sum::<i32>() as f32;
        if valid_count > 0.0 {
            for (i, &m) in masks.iter().enumerate() {
                if m == 1 { policy[i] = 1.0 / valid_count; }
            }
        }
        let value = rng.gen_range(-0.5..0.5);
        
        // 模拟网络延迟
        tokio::time::sleep(Duration::from_millis(5)).await;

        (policy, value)
    }
}

impl Evaluator for GrpcEvaluator {
    fn evaluate(&self, env: &DarkChessEnv) -> (Vec<f32>, f32) {
        // 使用 Runtime 阻塞执行异步请求
        // MCTS 是 CPU 密集型的，通常运行在专用线程中，所以 block_on 是安全的
        self.rt.block_on(self.call_remote_inference(env))
    }
}

// ============================================================================
// 主程序
// ============================================================================

fn main() -> Result<()> {
    // 1. 配置
    let mongo_uri = env::var("MONGODB_URI").unwrap_or_else(|_| "mongodb://localhost:27017".to_string());
    let grpc_endpoint = env::var("INFERENCE_SERVER").unwrap_or_else(|_| "http://127.0.0.1:50051".to_string());
    let worker_id = 0; // 单 Worker 模式
    let mcts_sims = 400; // 数据收集通常可以用更高的搜索次数以获得更高质量数据

    println!("=== 数据收集 Worker 启动 ===");
    println!("MongoDB: {}", mongo_uri);
    println!("Inference Server: {}", grpc_endpoint);
    println!("MCTS Sims: {}", mcts_sims);

    // 2. 初始化组件
    // 这里需要先创建一个临时的 Runtime 来 await 连接过程
    let evaluator = {
        let rt = Runtime::new()?;
        rt.block_on(GrpcEvaluator::connect(grpc_endpoint))?
    };
    let evaluator = Arc::new(evaluator);

    let mongo_storage = MongoStorage::new(&mongo_uri, "banqi_training", "games")?;

    // 3. 创建 Worker
    // 注意：这里使用了泛型支持后的 SelfPlayWorker
    let worker = SelfPlayWorker::with_scenario_dirichlet_and_temperature(
        worker_id,
        evaluator.clone(),
        mcts_sims,
        ScenarioType::Standard,
        0.3,  // Alpha
        0.25, // Epsilon
        12,   // Temperature steps (收集数据时可以稍微增加探索步数)
    );

    // 4. 循环收集
    let mut game_count = 0;
    loop {
        let start_time = std::time::Instant::now();
        
        // 执行一局游戏
        let episode = worker.play_episode(game_count);
        
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
            "Game #{}: 步数={}, 结果={}, 耗时={:.1}s",
            game_count + 1,
            episode.game_length,
            winner_str,
            duration.as_secs_f64()
        );

        // 上传到 MongoDB
        // save_games 接受 Slice，所以我们包装成 Vec
        if let Err(e) = mongo_storage.save_games(0, &[episode]) {
            eprintln!("❌ MongoDB 上传失败: {}", e);
            // 这里可以选择是否重试或暂停
            std::thread::sleep(Duration::from_secs(5));
        } else {
            println!("  ✅ 数据已上传");
        }

        game_count += 1;
    }
}