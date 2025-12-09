// run_lr_finder.rs - 学习率扫描器执行程序
//
// 使用方法:
//   cargo run --bin banqi-lr-finder [model_path] [iteration_start] [iteration_end] [max_games]
//
// 功能:
// 1. 从MongoDB数据库加载训练样本
// 2. 加载现有模型（可选）
// 3. 执行学习率扫描
// 4. 生成 lr_finder_results.csv 文件
// 5. 提供学习率选择建议
//
// 示例:
//   cargo run --bin banqi-lr-finder                           # 使用默认参数
//   cargo run --bin banqi-lr-finder model.ot                  # 加载模型
//   cargo run --bin banqi-lr-finder model.ot 0 100            # 加载迭代0-100的游戏
//   cargo run --bin banqi-lr-finder model.ot 0 100 500        # 最多加载500局游戏

use anyhow::Result;
use banqi_4x8::lr_finder::{find_learning_rate_from_episodes, LRFinderConfig};
use banqi_4x8::mongodb_storage::{GameDocument, MongoStorage};
use banqi_4x8::nn_model::BanqiNet;
use banqi_4x8::self_play::GameEpisode;
use bson::doc;
use ndarray::{Array1, Array4};
use std::env;
use tch::{nn, Device};

/// 从MongoDB加载游戏数据并转换为GameEpisode
fn load_episodes_from_db(
    storage: &MongoStorage,
    iteration_start: Option<usize>,
    iteration_end: Option<usize>,
    max_games: Option<usize>,
) -> Result<Vec<GameEpisode>> {
    println!("\n=== 从MongoDB数据库加载游戏数据 ===");

    let collection = storage.get_collection();

    // 构建查询过滤器
    let filter = match (iteration_start, iteration_end) {
        (Some(start), Some(end)) => {
            println!("  加载范围: iteration {} 到 {}", start, end);
            doc! {
                "iteration": {
                    "$gte": start as i64,
                    "$lte": end as i64
                }
            }
        }
        (Some(start), None) => {
            println!("  加载范围: iteration >= {}", start);
            doc! { "iteration": { "$gte": start as i64 } }
        }
        (None, Some(end)) => {
            println!("  加载范围: iteration <= {}", end);
            doc! { "iteration": { "$lte": end as i64 } }
        }
        (None, None) => {
            println!("  加载所有游戏");
            doc! {}
        }
    };

    // 统计符合条件的游戏数
    let total_count = collection.count_documents(filter.clone()).run()?;
    println!("  数据库中符合条件的游戏数: {}", total_count);

    // 查询游戏数据（按时间戳降序，最新的优先）
    let find_options = mongodb::options::FindOptions::builder()
        .sort(doc! { "timestamp": -1 })
        .limit(max_games.map(|n| n as i64))
        .build();

    let mut cursor = collection.find(filter).with_options(find_options).run()?;

    // 转换为 GameEpisode
    let mut episodes = Vec::new();
    let mut load_count = 0;
    let mut total_samples = 0;

    println!("  正在加载游戏数据...");
    while cursor.advance()? {
        let raw_doc = cursor.current();
        if let Ok(game_doc) = bson::from_slice::<GameDocument>(raw_doc.as_bytes()) {
            let mut samples = Vec::new();

            for sample in game_doc.samples {
                // 重建 Observation
                let board_array = Array4::from_shape_vec((1, 16, 4, 8), sample.board_state)
                    .expect("Invalid board shape");
                let scalar_array = Array1::from_vec(sample.scalar_state);

                let obs = banqi_4x8::game_env::Observation {
                    board: board_array,
                    scalars: scalar_array,
                };

                samples.push((
                    obs,
                    sample.policy_probs,
                    sample.mcts_value,
                    sample.game_result_value,
                    sample.action_mask,
                ));
            }

            total_samples += samples.len();

            episodes.push(GameEpisode {
                samples,
                game_length: game_doc.game_length,
                winner: game_doc.winner,
            });

            load_count += 1;
            if load_count % 100 == 0 {
                print!("\r  已加载 {} 局游戏, {} 个样本", load_count, total_samples);
            }
        }
    }

    println!(
        "\r  ✓ 成功加载 {} 局游戏, {} 个样本",
        load_count, total_samples
    );
    Ok(episodes)
}

fn main() -> Result<()> {
    println!("========================================");
    println!("   暗棋 4x8 学习率扫描器");
    println!("========================================\n");

    // 解析命令行参数
    let args: Vec<String> = env::args().collect();

    // 设备配置
    let cuda_available = tch::Cuda::is_available();
    println!("CUDA available: {}", cuda_available);

    let device = if cuda_available {
        println!("Using CUDA device 0");
        Device::Cuda(0)
    } else {
        println!("Using CPU");
        Device::Cpu
    };

    // 解析参数
    let model_path = if args.len() > 1 {
        Some(args[1].clone())
    } else {
        None
    };

    let iteration_start = if args.len() > 2 {
        args[2].parse::<usize>().ok()
    } else {
        None
    };

    let iteration_end = if args.len() > 3 {
        args[3].parse::<usize>().ok()
    } else {
        None
    };

    let max_games = if args.len() > 4 {
        args[4].parse::<usize>().ok()
    } else {
        Some(1000) // 默认最多加载1000局
    };

    // MongoDB连接配置
    let mongo_uri =
        env::var("MONGODB_URI").unwrap_or_else(|_| "mongodb://localhost:27017".to_string());
    let storage = MongoStorage::new(&mongo_uri, "banqi_training", "games")?;

    println!("\n=== 数据加载配置 ===");
    if let Some(ref path) = model_path {
        println!("加载模型: {}", path);
    } else {
        println!("使用随机初始化的模型");
    }
    if let Some(max) = max_games {
        println!("最大游戏数: {}", max);
    }

    // 从MongoDB加载游戏数据
    let episodes = load_episodes_from_db(&storage, iteration_start, iteration_end, max_games)?;

    if episodes.is_empty() {
        eprintln!("❌ 数据库中没有游戏数据");
        eprintln!("提示: 请先运行并行训练程序生成游戏数据");
        anyhow::bail!("游戏数据集为空");
    }

    // 统计信息
    let total_samples: usize = episodes.iter().map(|e| e.samples.len()).sum();
    println!("\n=== 数据集统计 ===");
    println!("总游戏数: {}", episodes.len());
    println!("总样本数: {}", total_samples);

    // 限制样本数量以加快扫描速度
    let max_sample_games = 500; // 最多使用500局游戏
    let episodes_to_use = if episodes.len() > max_sample_games {
        println!("  (使用最新的 {} 局游戏以加快扫描)", max_sample_games);
        episodes[..max_sample_games].to_vec()
    } else {
        episodes
    };

    let samples_count: usize = episodes_to_use.iter().map(|e| e.samples.len()).sum();
    println!("实际使用: {} 局游戏, {} 个样本", episodes_to_use.len(), samples_count);

    // 创建模型
    let mut vs = nn::VarStore::new(device);
    let net = BanqiNet::new(&vs.root());

    // 加载现有模型（如果提供）
    if let Some(ref path) = model_path {
        println!("\n正在加载模型: {}", path);
        match vs.load(path) {
            Ok(_) => println!("✓ 模型加载成功"),
            Err(e) => {
                eprintln!("⚠️ 加载模型失败: {}", e);
                println!("  继续使用随机初始化的模型进行扫描");
            }
        }
    } else {
        println!("\n未指定模型路径，使用随机初始化的模型");
        println!("提示: 可以使用命令 'cargo run --bin banqi-lr-finder <model.ot>' 加载现有模型");
    }

    // 配置学习率扫描器
    let config = LRFinderConfig {
        start_lr: 1e-7,            // 起始学习率
        end_lr: 1.0,               // 结束学习率
        num_steps: 100,            // 扫描步数
        num_batches_per_step: 2,   // 每步训练批次数
        batch_size: 64,            // 批量大小
        smooth_window: 5,          // 平滑窗口
        divergence_threshold: 4.0, // 发散阈值
    };

    println!("\n准备开始学习率扫描...");
    println!("此过程可能需要几分钟，请耐心等待。\n");

    // 执行学习率扫描
    let results = find_learning_rate_from_episodes(&net, &episodes_to_use, device, &config)?;

    // 输出统计信息
    println!("\n========================================");
    println!("扫描完成！");
    println!("========================================");
    println!("数据点数量: {}", results.len());
    println!("输出文件: lr_finder_results.csv");

    println!("\n下一步:");
    println!("1. 查看 lr_finder_results.csv 文件");
    println!("2. 使用 Python 或其他工具绘制学习率-损失曲线:");
    println!("   ```python");
    println!("   import pandas as pd");
    println!("   import matplotlib.pyplot as plt");
    println!("   ");
    println!("   df = pd.read_csv('lr_finder_results.csv')");
    println!("   plt.figure(figsize=(10, 6))");
    println!("   plt.plot(df['learning_rate'], df['loss'])");
    println!("   plt.xscale('log')");
    println!("   plt.xlabel('Learning Rate')");
    println!("   plt.ylabel('Loss')");
    println!("   plt.title('Learning Rate Finder - Banqi 4x8')");
    println!("   plt.grid(True)");
    println!("   plt.show()");
    println!("   ```");
    println!("3. 根据曲线和建议选择合适的学习率");
    println!("4. 在训练代码中应用新的学习率");

    Ok(())
}
