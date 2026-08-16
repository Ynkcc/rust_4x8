// src/data_collector.rs
//
// 数据收集器 - 本地模型版本
// 功能：
// 1. 直接加载 tch-rs 模型进行推理
// 2. 运行 Gumbel MCTS 自对弈生成游戏数据
// 3. 将数据保存到 MongoDB

use anyhow::Result;
use banqi_4x8::local_evaluator::LocalEvaluator;
use banqi_4x8::mongodb_storage::MongoStorage;
use banqi_4x8::self_play::{ScenarioType, SelfPlayConfig, run_self_play};

use std::env;
use std::time::Instant;
use tch::Device;

// ============================================================================
// 主程序
// ============================================================================

fn main() -> Result<()> {
    // 1. 配置
    let mongo_uri =
        env::var("MONGODB_URI").unwrap_or_else(|_| "mongodb://localhost:27017".to_string());
    let model_path = env::var("MODEL_PATH").unwrap_or_else(|_| "banqi_model_latest.pt".to_string());

    // 获取 Worker ID
    let args: Vec<String> = env::args().collect();
    let worker_id = if args.len() > 1 {
        args[1].parse::<usize>().unwrap_or(0)
    } else {
        0
    };

    let mcts_sims = 64;

    println!("=== 数据收集器-{} 启动 ===", worker_id);
    println!("MongoDB: {}", mongo_uri);
    println!("Model: {}", model_path);
    println!("MCTS Sims: {}", mcts_sims);

    // 2. 设备配置
    let device = if tch::Cuda::is_available() {
        println!("Using CUDA");
        Device::Cuda(0)
    } else {
        println!("Using CPU");
        Device::Cpu
    };

    // 3. 加载 TorchScript 模型
    println!("Loading model from {}", model_path);
    let mut evaluator = match LocalEvaluator::new(&model_path, device) {
        Ok(eval) => {
            println!("✅ TorchScript 模型加载成功");
            eval
        }
        Err(e) => {
            eprintln!("❌ 模型加载失败: {}", e);
            return Ok(());
        }
    };

    let mut last_modified = std::fs::metadata(&model_path)
        .and_then(|m| m.modified())
        .unwrap_or(std::time::SystemTime::now());

    // 4. 连接 MongoDB
    let mongo_storage = match MongoStorage::new(&mongo_uri, "banqi_training", "games") {
        Ok(s) => s,
        Err(e) => {
            eprintln!("无法连接到 MongoDB: {}", e);
            return Ok(());
        }
    };

    // 5. 配置自对弈
    let config = SelfPlayConfig {
        mcts_sims,
        max_considered_actions: 16,
        temperature_steps: 12,
        scenario: ScenarioType::Standard,
        c_scale: 1.0,
        gumbel_scale: 1.0,
    };

    // 6. 循环收集
    let mut game_count = 0;
    let mut iteration: usize = 0;
    // 每 GAMES_PER_ITERATION 局游戏迭代号 +1，便于按迭代清理/统计
    const GAMES_PER_ITERATION: usize = 100;
    loop {
        // 检查模型更新
        if let Ok(metadata) = std::fs::metadata(&model_path) {
            if let Ok(modified) = metadata.modified() {
                if modified > last_modified {
                    println!("🔄 检测到模型更新，正在重载...");
                    match LocalEvaluator::new(&model_path, device) {
                        Ok(new_eval) => {
                            evaluator = new_eval;
                            last_modified = modified;
                            println!("✅ 模型重载成功");
                        }
                        Err(e) => {
                            eprintln!("⚠️ 模型重载失败 (保持旧模型): {}", e);
                        }
                    }
                }
            }
        }

        let start_time = Instant::now();

        // 执行一局游戏
        let episode = run_self_play(&evaluator, &config, DarkChessEnv::new);

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
            "[Worker-{}] Game #{} (iter={}): 步数={}, 结果={}, 耗时={:.1}s ({:.1} steps/s)",
            worker_id,
            game_count + 1,
            iteration,
            episode.game_length,
            winner_str,
            duration.as_secs_f64(),
            episode.game_length as f64 / duration.as_secs_f64()
        );

        // 上传到 MongoDB（迭代号递增）
        match mongo_storage.save_games(iteration, vec![episode]) {
            Ok(()) => {}
            Err(e) => {
                eprintln!("❌ MongoDB 上传失败: {}", e);
                std::thread::sleep(std::time::Duration::from_secs(5));
            }
        }

        game_count += 1;
        if game_count % GAMES_PER_ITERATION == 0 {
            iteration += 1;
            println!(
                "[Worker-{}] 📍 完成 {} 局 → 进入迭代 {}",
                worker_id, game_count, iteration
            );
        }
    }
}
