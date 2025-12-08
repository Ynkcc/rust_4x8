// train_from_db.rs - 从MongoDB数据库加载样本进行训练
//
// 架构设计:
// - 从MongoDB加载指定迭代范围或最近N个迭代的游戏数据
// - 转换为GameEpisode格式供训练使用
// - 支持增量训练（加载新模型继续训练）
// - 支持数据过滤（只使用特定结果的游戏）

use anyhow::Result;
use banqi_4x8::game_env::Observation;
use banqi_4x8::mongodb_storage::{GameDocument, MongoStorage};
use banqi_4x8::nn_model::BanqiNet;
use banqi_4x8::self_play::GameEpisode;
use banqi_4x8::training::{get_loss_weights, train_step};
use banqi_4x8::training_log::TrainingLog;
use bson::doc;
use ndarray::{Array1, Array4};
use std::env;
use std::time::Instant;
use tch::{nn, nn::OptimizerConfig, Device};

// ================ 数据加载 ================

/// 从MongoDB加载游戏数据
pub fn load_games_from_db(
    storage: &MongoStorage,
    iteration_start: Option<usize>,
    iteration_end: Option<usize>,
    max_games: Option<usize>,
) -> Result<Vec<GameEpisode>> {
    println!("\n=== 从数据库加载游戏数据 ===");

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
                // board_state: Vec<f32> -> Array4<f32> with shape [1, 16, 4, 8]
                let board_array = Array4::from_shape_vec((1, 16, 4, 8), sample.board_state)
                    .expect("Invalid board shape");

                // scalar_state: Vec<f32> -> Array1<f32>
                let scalar_array =
                    Array1::from_vec(sample.scalar_state);

                let obs = Observation {
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

    println!("\r  ✓ 成功加载 {} 局游戏, {} 个样本", load_count, total_samples);
    Ok(episodes)
}

/// 过滤游戏数据（例如只保留有明确胜负的游戏）
pub fn filter_episodes(
    episodes: Vec<GameEpisode>,
    include_draws: bool,
    min_game_length: Option<usize>,
    max_game_length: Option<usize>,
) -> Vec<GameEpisode> {
    let original_count = episodes.len();
    let original_samples: usize = episodes.iter().map(|e| e.samples.len()).sum();

    let filtered: Vec<GameEpisode> = episodes
        .into_iter()
        .filter(|ep| {
            // 过滤平局
            if !include_draws && (ep.winner.is_none() || ep.winner == Some(0)) {
                return false;
            }

            // 过滤游戏长度
            if let Some(min_len) = min_game_length {
                if ep.game_length < min_len {
                    return false;
                }
            }
            if let Some(max_len) = max_game_length {
                if ep.game_length > max_len {
                    return false;
                }
            }

            true
        })
        .collect();

    let filtered_count = filtered.len();
    let filtered_samples: usize = filtered.iter().map(|e| e.samples.len()).sum();

    if filtered_count < original_count {
        println!(
            "  过滤后: {} 局游戏 ({} 样本) -> {} 局 ({} 样本)",
            original_count, original_samples, filtered_count, filtered_samples
        );
    }

    filtered
}

/// 只保留每局游戏中距离结束最近的N步样本
pub fn keep_final_steps(episodes: Vec<GameEpisode>, final_steps: usize) -> Vec<GameEpisode> {
    let original_samples: usize = episodes.iter().map(|e| e.samples.len()).sum();

    let filtered: Vec<GameEpisode> = episodes
        .into_iter()
        .map(|mut ep| {
            let total_samples = ep.samples.len();
            if total_samples > final_steps {
                // 只保留最后 final_steps 个样本
                ep.samples = ep.samples.split_off(total_samples - final_steps);
            }
            ep
        })
        .collect();

    let filtered_samples: usize = filtered.iter().map(|e| e.samples.len()).sum();

    println!(
        "  保留最后{}步: {} 个样本 -> {} 个样本",
        final_steps, original_samples, filtered_samples
    );

    filtered
}

// ================ 主训练函数 ================

fn main() -> Result<()> {
    if let Err(e) = train_from_database() {
        eprintln!("训练失败: {}", e);
        std::process::exit(1);
    }
    Ok(())
}

pub fn train_from_database() -> Result<()> {
    // ========== 配置参数 ==========
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

    // 训练参数
    let batch_size = 512;
    let learning_rate = 1e-4;
    let num_epochs = 10; // 每次训练的epoch数
    let save_interval = 5; // 每N个epoch保存一次模型

    println!("\n=== 从数据库训练配置 ===");
    println!("批量大小: {}", batch_size);
    println!("学习率: {}", learning_rate);
    println!("训练epoch数: {}", num_epochs);

    // MongoDB连接
    let mongo_uri =
        env::var("MONGODB_URI").unwrap_or_else(|_| "mongodb://localhost:27017".to_string());
    let storage = MongoStorage::new(&mongo_uri, "banqi_training", "games")?;

    // ========== 数据加载配置 ==========
    // 可以通过命令行参数控制：
    // train_from_db [model_path] [iteration_start] [iteration_end] [max_games]
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

    println!("\n=== 数据加载配置 ===");
    if let Some(ref path) = model_path {
        println!("加载模型: {}", path);
    } else {
        println!("从头开始训练（随机初始化）");
    }
    if let Some(max) = max_games {
        println!("最大游戏数: {}", max);
    }

    // 加载游戏数据
    let mut episodes =
        load_games_from_db(&storage, iteration_start, iteration_end, max_games)?;

    if episodes.is_empty() {
        eprintln!("错误: 没有加载到任何游戏数据！");
        return Ok(());
    }

    // 数据过滤
    println!("\n=== 数据过滤 ===");
    let include_draws = true; // 是否包含平局
    let min_game_length = Some(10); // 最短游戏长度
    let max_game_length = None; // 最长游戏长度

    println!("  包含平局: {}", include_draws);
    if let Some(min_len) = min_game_length {
        println!("  最短游戏长度: {}", min_len);
    }
    if let Some(max_len) = max_game_length {
        println!("  最长游戏长度: {}", max_len);
    }

    episodes = filter_episodes(episodes, include_draws, min_game_length, max_game_length);

    if episodes.is_empty() {
        eprintln!("错误: 过滤后没有剩余任何游戏数据！");
        return Ok(());
    }

    // 只保留距离游戏结束最近的6步样本
    println!("\n=== 样本截取 ===");
    episodes = keep_final_steps(episodes, 6);

    if episodes.is_empty() {
        eprintln!("错误: 截取后没有剩余任何游戏数据！");
        return Ok(());
    }

    // 统计游戏信息
    let total_games = episodes.len();
    let total_samples: usize = episodes.iter().map(|e| e.samples.len()).sum();
    let red_wins = episodes.iter().filter(|e| e.winner == Some(1)).count();
    let black_wins = episodes.iter().filter(|e| e.winner == Some(-1)).count();
    let draws = episodes
        .iter()
        .filter(|e| e.winner.is_none() || e.winner == Some(0))
        .count();

    println!("\n=== 数据集统计 ===");
    println!("总游戏数: {}", total_games);
    println!("总样本数: {}", total_samples);
    println!(
        "红方胜: {} ({:.1}%)",
        red_wins,
        red_wins as f32 / total_games as f32 * 100.0
    );
    println!(
        "黑方胜: {} ({:.1}%)",
        black_wins,
        black_wins as f32 / total_games as f32 * 100.0
    );
    println!(
        "平局: {} ({:.1}%)",
        draws,
        draws as f32 / total_games as f32 * 100.0
    );

    // ========== 创建模型和优化器 ==========
    let mut vs = nn::VarStore::new(device);
    let net = BanqiNet::new(&vs.root());
    let mut opt = nn::Adam::default().build(&vs, learning_rate)?;

    // 加载模型（如果提供）
    if let Some(ref path) = model_path {
        println!("\n正在加载模型: {}", path);
        if let Err(e) = vs.load(path) {
            eprintln!("⚠️ 加载模型失败: {}", e);
            println!("将使用随机初始化继续训练");
        } else {
            println!("✓ 模型加载成功！");
        }
    }

    // ========== 训练循环 ==========
    println!("\n=== 开始训练 ===");

    for epoch in 0..num_epochs {
        println!("\n---------- Epoch {}/{} ----------", epoch + 1, num_epochs);

        let train_start = Instant::now();
        let (loss, p_loss, v_loss) = train_step(&mut opt, &net, &episodes, batch_size, device, epoch);
        let train_elapsed = train_start.elapsed();

        println!(
            "  Loss: {:.4} (Policy={:.4}, Value={:.4}) | 耗时: {:.1}s",
            loss,
            p_loss,
            v_loss,
            train_elapsed.as_secs_f64()
        );

        // 记录训练日志
        let (plw, vlw) = get_loss_weights(epoch);
        let log_record = TrainingLog {
            iteration: epoch,
            avg_total_loss: loss,
            avg_policy_loss: p_loss,
            avg_value_loss: v_loss,
            policy_loss_weight: plw,
            value_loss_weight: vlw,
            scenario1_value: 0.0,
            scenario1_unmasked_a38: 0.0,
            scenario1_unmasked_a39: 0.0,
            scenario1_unmasked_a40: 0.0,
            scenario1_masked_a38: 0.0,
            scenario1_masked_a39: 0.0,
            scenario1_masked_a40: 0.0,
            scenario2_value: 0.0,
            scenario2_unmasked_a3: 0.0,
            scenario2_unmasked_a5: 0.0,
            scenario2_masked_a3: 0.0,
            scenario2_masked_a5: 0.0,
            new_samples_count: total_samples,
            replay_buffer_size: total_samples,
            avg_game_steps: episodes.iter().map(|e| e.game_length as f32).sum::<f32>()
                / total_games as f32,
            red_win_ratio: red_wins as f32 / total_games as f32,
            draw_ratio: draws as f32 / total_games as f32,
            black_win_ratio: black_wins as f32 / total_games as f32,
            avg_policy_entropy: 0.0,
            high_confidence_ratio: 0.0,
        };

        // 写入CSV日志
        let csv_path = "training_from_db_log.csv";
        if let Err(e) = TrainingLog::write_header(csv_path) {
            eprintln!("  ⚠️ 初始化训练日志表头失败: {}", e);
        }
        if let Err(e) = log_record.append_to_csv(csv_path) {
            eprintln!("  ⚠️ 写入训练日志失败: {}", e);
        }

        // 定期保存模型
        if (epoch + 1) % save_interval == 0 || epoch == num_epochs - 1 {
            let save_path = format!("banqi_model_from_db_epoch_{}.ot", epoch + 1);
            vs.save(&save_path)?;
            println!("  ✓ 已保存模型: {}", save_path);
        }

        // 清理缓存
        std::thread::sleep(std::time::Duration::from_millis(50));
    }

    // 保存最终模型
    let final_path = "banqi_model_from_db_latest.ot";
    vs.save(final_path)?;
    println!("\n✓ 训练完成！已保存最终模型: {}", final_path);
    println!("\n使用以下命令测试模型:");
    println!("  cargo run --bin banqi-verify-trained -- {}", final_path);

    Ok(())
}
