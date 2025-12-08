// code_files/src/parallel_train.rs

// parallel_train.rs - 并行自对弈训练系统主控制器
//
// 本模块实现了基于 MCTS (蒙特卡洛树搜索) 和神经网络的 AlphaZero 风格训练流程。
// 采用了并行架构以提高数据生成效率。
//
// 架构设计:
// 1. 推理服务器 (Inference Server): 
//    - 在主循环中启动一个专用线程运行神经网络推理。
//    - 负责收集来自所有自对弈 Worker 的推理请求，进行组批 (Batching)，然后一次性送入 GPU 计算。
//    - 这种方式能显著提高 GPU 利用率，避免单次小批量推理造成的开销。
//
// 2. 自对弈工作者 (Self-Play Workers):
//    - 启动多个并行线程，每个线程运行一个独立的 MCTS 搜索实例进行自我对弈。
//    - Worker 通过 channel (通道) 将局面评估请求发送给推理服务器。
//
// 3. 数据管理:
//    - 内存缓冲区 (Replay Buffer): 保存最近 N 局游戏数据，用于训练。
//    - MongoDB: 持久化存储所有生成的游戏数据，支持后续离线分析或从特定检查点恢复训练。
//
// 4. 训练循环:
//    - 生成数据 -> 存储数据 -> 训练网络 -> 更新模型 -> 下一轮迭代。

use anyhow::Result;
use banqi_4x8::inference::{ChannelEvaluator, InferenceServer};
use banqi_4x8::mongodb_storage::MongoStorage;
use banqi_4x8::nn_model::BanqiNet;
use banqi_4x8::self_play::{GameEpisode, ScenarioType, SelfPlayWorker};
use banqi_4x8::training::{get_loss_weights, train_step};
use banqi_4x8::training_log::TrainingLog;
use std::env;
use std::sync::{mpsc, Arc};
use std::thread;
use std::time::Instant;
use tch::{nn, nn::OptimizerConfig, Device};

// ================ 程序入口 ================

fn main() -> Result<()> {
    // 捕获并打印训练过程中的任何错误
    if let Err(e) = parallel_train_loop() {
        eprintln!("训练失败: {}", e);
    }
    Ok(())
}

// ================ 并行训练主逻辑 ================

pub fn parallel_train_loop() -> Result<()> {
    // --- 1. 设备配置 ---
    // 检查并优先使用 CUDA (GPU) 设备，否则回退到 CPU
    let cuda_available = tch::Cuda::is_available();
    println!("CUDA available: {}", cuda_available);

    let device = if cuda_available {
        println!("Using CUDA device 0");
        Device::Cuda(0)
    } else {
        println!("Using CPU");
        Device::Cpu
    };

    // --- 2. 训练超参数配置 ---
    // 这些参数控制训练的规模、速度和探索程度
    
    // 并行度配置
    let num_workers = 32;          // 并行工作线程数，建议根据 CPU 核心数调整
    
    // MCTS 配置
    let mcts_sims = 800;           // 每次决策进行的模拟次数，越高越强但越慢
    
    // 训练流程配置
    let num_iterations = 2000;         // 总迭代轮数
    let num_episodes_per_iteration = 4; // 每轮每个 Worker 生成的对局数 (总数 = workers * episodes_per_worker)
    
    // 推理服务配置
    let inference_batch_size = 28;     // 推理服务器最大组批大小
    let inference_timeout_ms = 4;      // 组批等待超时时间 (毫秒)，平衡延迟和吞吐量
    
    // 神经网络训练配置
    let batch_size = 256;              // 训练时的批量大小
    let epochs_per_iteration = 10;     // 每轮迭代在 ReplayBuffer 上训练的 Epoch 数
    let learning_rate = 1e-4;          // 学习率
    
    // 经验回放缓冲区配置
    let max_buffer_games = 1000;       // 缓冲区保留的最大游戏局数 (滑动窗口)

    // 探索性配置 (Dirichlet 噪声)
    // 用于在 MCTS 根节点引入随机性，增加探索多样性
    let dirichlet_alpha = 0.3;     // Alpha 参数：越小噪声越集中在少数动作，越大越分散
    let dirichlet_epsilon = 0.25;  // 噪声混合比例：(1-ε)*P + ε*Noise
    
    // 温度采样配置 (Temperature Sampling)
    // 前 N 步使用 τ=1 按概率采样，保证开局多样性；之后使用 τ=0 贪婪选择，保证棋力
    let temperature_steps = 10;   

    println!("\n=== 场景自对弈训练配置 ===");
    println!("工作线程数: {} (全标准环境)", num_workers);
    println!("每轮每场景游戏数: {}", num_episodes_per_iteration);
    println!("MCTS模拟次数: {}", mcts_sims);
    println!("训练迭代次数: {}", num_iterations);
    println!("推理批量大小: {}", inference_batch_size);
    println!("游戏缓冲区: 最近 {} 局", max_buffer_games);
    println!("Dirichlet噪声: alpha={}, epsilon={}", dirichlet_alpha, dirichlet_epsilon);
    println!("温度采样: 前 {} 步 τ=1 (探索), 之后 τ=0 (贪心)", temperature_steps);
    println!("场景: Standard"); // 目前使用标准开局场景

    // --- 3. 初始化存储 ---
    // 连接 MongoDB 用于持久化存储生成的游戏数据
    let mongo_uri = env::var("MONGODB_URI").unwrap_or_else(|_| "mongodb://localhost:27017".to_string());
    let mongo_storage = MongoStorage::new(&mongo_uri, "banqi_training", "games")?;
    println!("MongoDB连接成功");

    // --- 4. 初始化模型与优化器 ---
    let mut vs = nn::VarStore::new(device);
    let net = BanqiNet::new(&vs.root());
    let mut opt = nn::Adam::default().build(&vs, learning_rate)?;

    // --- 5. 加载模型 (可选) ---
    // 如果命令行提供了模型路径，则加载该模型继续训练
    let args: Vec<String> = env::args().collect();
    if args.len() > 1 {
        let model_path = &args[1];
        println!("正在加载模型: {}", model_path);
        match vs.load(model_path) {
            Ok(_) => println!("模型加载成功！"),
            Err(e) => {
                eprintln!("❌ 加载模型失败: {}", e);
                eprintln!("提示: 如果您希望从随机模型开始训练，请不要指定模型路径参数。");
                eprintln!("      如果您希望微调现有模型，请确保路径正确且模型文件完整。");
                panic!("模型加载失败，程序终止。如需从头训练请移除命令行参数。");
            }
        }
    }

    // --- 6. 经验回放缓冲区 ---
    // 内存中保留最近的游戏数据。由于 GameEpisode 结构包含整局数据，
    // 这里直接存储 GameEpisode 对象，训练时再展开为样本。
    let mut game_buffer: Vec<GameEpisode> = Vec::new();

    // --- 7. 主循环 (Iterations) ---
    for iteration in 0..num_iterations {
        println!(
            "\n========== Iteration {}/{} ==========",
            iteration + 1,
            num_iterations
        );

        // 7.1 保存当前模型权重的临时副本
        // InferenceServer 将加载此文件来初始化其内部的网络模型。
        // 这确保了 Worker 使用的是当前最新的模型进行评估。
        let temp_model_path = format!("banqi_model_iter_{}_temp.ot", iteration);
        vs.save(&temp_model_path)?;

        // 7.2 创建通信通道
        // req_tx: 用于 Worker 发送推理请求
        // req_rx: 用于 InferenceServer 接收请求
        let (req_tx, req_rx) = mpsc::channel();

        // 7.3 启动推理服务器线程
        let temp_model_path_clone = temp_model_path.clone();
        let inference_handle = thread::spawn(move || {
            // InferenceServer 会阻塞运行，直到所有发送端断开连接
            match InferenceServer::new(
                &temp_model_path_clone,
                device,
                req_rx,
                inference_batch_size,
                inference_timeout_ms,
            ) {
                Ok(server) => server.run(),
                Err(e) => {
                    eprintln!("[InferenceServer] 初始化失败: {}", e);
                }
            }
        });

        // 7.4 启动自对弈 Worker 线程
        let scenarios = vec![ScenarioType::Standard; num_workers];
        let mut worker_handles = Vec::new();
        let mut result_rxs = Vec::new(); // 用于接收每个 Worker 完成后的游戏数据

        for (worker_id, scenario) in scenarios.iter().enumerate() {
            let req_tx_clone = req_tx.clone();
            let (result_tx, result_rx) = mpsc::channel();
            result_rxs.push(result_rx);
            
            let scenario_copy = *scenario;
            let alpha = dirichlet_alpha;
            let epsilon = dirichlet_epsilon;
            let temp_steps = temperature_steps;

            let handle = thread::spawn(move || {
                // ChannelEvaluator 实现了 Evaluator trait，
                // 它将评估请求通过 channel 发送给 InferenceServer。
                let evaluator = Arc::new(ChannelEvaluator::new(req_tx_clone));
                
                // 配置 Worker
                let worker = SelfPlayWorker::with_scenario_dirichlet_and_temperature(
                    worker_id,
                    evaluator,
                    mcts_sims,
                    scenario_copy,
                    alpha,
                    epsilon,
                    temp_steps,
                );

                // 执行指定数量的对局
                let mut all_episodes = Vec::new();
                for ep in 0..num_episodes_per_iteration {
                    let episode = worker.play_episode(ep);
                    all_episodes.push(episode);
                }

                println!(
                    "  [Worker-{}] 完成 {} 局 {} 游戏",
                    worker_id,
                    num_episodes_per_iteration,
                    scenario_copy.name()
                );
                
                // 将结果发回主线程
                result_tx.send(all_episodes).expect("无法发送结果");
            });

            worker_handles.push(handle);
        }

        // 7.5 关闭主线程持有的请求发送端
        // 这一点至关重要：InferenceServer 只有在所有 req_tx 都被 drop 后才会退出循环。
        // 主线程如果不 drop，Server 会一直等待。
        drop(req_tx);

        // 7.6 收集结果
        let mut all_episodes = Vec::new();
        for (i, result_rx) in result_rxs.into_iter().enumerate() {
            match result_rx.recv() {
                Ok(episodes) => all_episodes.extend(episodes),
                Err(e) => eprintln!("⚠️ [Main] 无法从 Worker-{} 接收数据 (线程可能已崩溃): {}", i, e),
            }
        }

        // 7.7 等待所有线程结束
        for handle in worker_handles {
            handle.join().expect("工作线程异常");
        }
        inference_handle.join().expect("推理服务器异常");

        // 7.8 清理临时模型文件
        if let Err(e) = std::fs::remove_file(&temp_model_path) {
            eprintln!("  ⚠️ 清理临时模型失败: {} - {}", temp_model_path, e);
        }

        // 7.9 数据统计与验证
        let total_games = all_episodes.len();
        // 过滤掉可能产生的空数据 (异常对局)
        let valid_games = all_episodes.iter().filter(|ep| !ep.samples.is_empty()).count();
        let error_games = total_games - valid_games;
        
        println!("  收集了 {} 局游戏 (有效: {}, 错误: {})", 
            total_games, valid_games, error_games);
        
        if error_games > 0 {
            eprintln!("  ⚠️ 发现 {} 局错误游戏（samples为空），将被跳过保存", error_games);
        }

        // 7.10 持久化存储 (MongoDB)
        println!("  正在保存数据到MongoDB...");
        mongo_storage.save_games(iteration, &all_episodes)?;
        
        // 打印存储统计
        match mongo_storage.get_iteration_stats(iteration) {
            Ok(stats) => stats.print(),
            Err(e) => eprintln!("⚠️ 获取 Iteration 统计信息失败: {}", e),
        }

        // 7.11 更新内存缓冲区 (Replay Buffer)
        // 将新数据加入缓冲区
        game_buffer.extend(all_episodes);
        
        // 维持缓冲区大小限制 (FIFO)
        if game_buffer.len() > max_buffer_games {
            let remove_count = game_buffer.len() - max_buffer_games;
            game_buffer.drain(0..remove_count);
        }
        
        let total_samples: usize = game_buffer.iter().map(|ep| ep.samples.len()).sum();
        println!("  游戏缓冲区: {} 局游戏, {} 个样本", game_buffer.len(), total_samples);

        if total_samples == 0 {
            println!("  ⚠️ 本轮没有收集到有效样本，跳过训练");
            continue;
        }

        // 7.12 训练神经网络
        println!("  开始训练...");
        let train_start = Instant::now();
        let mut loss_sum = 0.0_f64;
        let mut p_loss_sum = 0.0_f64;
        let mut v_loss_sum = 0.0_f64;
        
        for epoch in 0..epochs_per_iteration {
            // train_step 负责将 GameEpisode 展开为 Batch 并进行反向传播
            let (loss, p_loss, v_loss) =
                train_step(&mut opt, &net, &game_buffer, batch_size, device, epoch);
            
            println!(
                "    Epoch {}/{}: Loss={:.4} (Policy={:.4}, Value={:.4})",
                epoch + 1,
                epochs_per_iteration,
                loss,
                p_loss,
                v_loss
            );
            loss_sum += loss;
            p_loss_sum += p_loss;
            v_loss_sum += v_loss;
        }
        let train_elapsed = train_start.elapsed();
        println!("  训练完成，耗时 {:.1}s", train_elapsed.as_secs_f64());

        // 内存清理提示 (PyTorch 缓存)
        std::thread::sleep(std::time::Duration::from_millis(50));

        // 7.13 记录训练日志 (CSV)
        // 选取最近生成的数据进行胜率统计 (用于监控当前模型强度)
        let recent_episodes_count = (num_workers * num_episodes_per_iteration).min(game_buffer.len());
        
        if game_buffer.is_empty() || recent_episodes_count == 0 {
            println!("  ⚠️ 缓冲区为空，跳过统计和日志记录");
            continue;
        }
        
        let recent_episodes = &game_buffer[(game_buffer.len() - recent_episodes_count)..];
        let total_rec_games = recent_episodes.len();
        
        // 统计胜负平
        let red_wins = recent_episodes.iter().filter(|ep| ep.winner == Some(1)).count();
        let black_wins = recent_episodes.iter().filter(|ep| ep.winner == Some(-1)).count();
        let draws = recent_episodes.iter().filter(|ep| ep.winner.is_none() || ep.winner == Some(0)).count();
        
        let avg_steps: f32 = recent_episodes
            .iter()
            .map(|ep| ep.game_length as f32)
            .sum::<f32>()
            / total_rec_games as f32;

        // 计算策略熵和高置信度比例 (监控策略网络的确定性)
        let (avg_entropy, high_conf_ratio) = if !recent_episodes.is_empty() {
            let mut ent_sum = 0.0_f32;
            let mut count = 0usize;
            let mut high_conf = 0usize;
            for ep in recent_episodes {
                for (_, probs, _, _, _) in &ep.samples {
                    let mut e = 0.0_f32;
                    let mut maxp = 0.0_f32;
                    for &p in probs {
                        if p > 1e-8 {
                            e += -p * p.ln();
                        }
                        if p > maxp {
                            maxp = p;
                        }
                    }
                    ent_sum += e;
                    count += 1;
                    if maxp >= 0.90 {
                        high_conf += 1;
                    }
                }
            }
            if count > 0 {
                (ent_sum / count as f32, high_conf as f32 / count as f32)
            } else {
                eprintln!("  ⚠️ 本轮所有游戏样本为空，熵统计无效");
                (0.0, 0.0)
            }
        } else {
            (0.0, 0.0)
        };

        // 计算平均 Loss
        let avg_total_loss = loss_sum / epochs_per_iteration as f64;
        let avg_policy_loss = p_loss_sum / epochs_per_iteration as f64;
        let avg_value_loss = v_loss_sum / epochs_per_iteration as f64;
        let (plw, vlw) = get_loss_weights(epochs_per_iteration.saturating_sub(1));

        // 构造日志记录
        let log_record = TrainingLog {
            iteration,
            avg_total_loss,
            avg_policy_loss,
            avg_value_loss,
            policy_loss_weight: plw,
            value_loss_weight: vlw,
            // 场景验证字段预留 (当前未使用)
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
            // 统计数据
            new_samples_count: recent_episodes.iter().map(|ep| ep.samples.len()).sum::<usize>(),
            replay_buffer_size: total_samples,
            avg_game_steps: avg_steps,
            red_win_ratio: red_wins as f32 / total_rec_games as f32,
            draw_ratio: draws as f32 / total_rec_games as f32,
            black_win_ratio: black_wins as f32 / total_rec_games as f32,
            avg_policy_entropy: avg_entropy,
            high_confidence_ratio: high_conf_ratio,
        };

        // 写入文件
        let csv_path = "training_log.csv";
        if let Err(e) = TrainingLog::write_header(csv_path) {
            eprintln!("  ⚠️ 初始化训练日志表头失败: {}", e);
        }
        if let Err(e) = log_record.append_to_csv(csv_path) {
            eprintln!("  ⚠️ 写入训练日志失败: {}", e);
        } else {
            println!("  已记录训练日志到 {}", csv_path);
        }

        // 7.14 定期保存模型检查点 (Checkpoint)
        if (iteration + 1) % 5 == 0 || iteration == num_iterations - 1 {
            let model_path = format!("banqi_model_{}.ot", iteration + 1);
            vs.save(&model_path)?;
            println!("  已保存模型: {}", model_path);
        }
        
        println!("  ======== Iteration {} 完成 ========\n", iteration + 1);
    }

    // --- 8. 训练结束 ---
    // 保存最终模型并提示测试
    vs.save("banqi_model_latest.ot")?;
    println!("\n训练完成！已保存模型: banqi_model_latest.ot");
    println!("\n请使用以下命令测试模型:");
    println!("  cargo run --bin banqi-verify-trained -- banqi_model_latest.ot");

    Ok(())
}