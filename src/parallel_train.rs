// parallel_train.rs - 并行自对弈训练系统主控制器
//
// 架构设计:
// - 主线程: 运行模型推理服务 (InferenceServer)
// - 工作线程池: 每个线程运行独立的自对弈游戏
// - 通信: 通过 channel 发送推理请求和接收结果
// - 批量推理: 收集多个请求后批量处理，提高GPU利用率

use banqi_3x4::nn_model::BanqiNet;
use banqi_3x4::inference::{InferenceServer, ChannelEvaluator};
use banqi_3x4::self_play::{SelfPlayWorker, ScenarioType};
use banqi_3x4::scenario_validation::validate_model_on_scenarios_with_net;
use banqi_3x4::training::{train_step, get_loss_weights};
use banqi_3x4::game_env::Observation;
use banqi_3x4::training_log::TrainingLog;
use anyhow::Result;
use std::sync::{Arc, mpsc};
use std::time::Instant;
use tch::{nn, nn::OptimizerConfig, Device};
use std::thread;
use std::env;

// ================ 主训练循环 ================

pub fn parallel_train_loop() -> Result<()> {
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
    
    // 训练配置
    let num_workers = 32; // 每个场景一个工作线程
    let mcts_sims = 800; // MCTS模拟次数
    let num_iterations = 2000; // 训练迭代次数
    let num_episodes_per_iteration = 4; // 每轮每个场景的游戏数
    let inference_batch_size = num_workers/2;
    let inference_timeout_ms = 5;
    let batch_size = 128;
    let epochs_per_iteration = 5;
    let max_buffer_size = 20000;
    let learning_rate = 1e-4;
    
    println!("\n=== 场景自对弈训练配置 ===");
    println!("工作线程数: {} (全标准环境)", num_workers);
    println!("每轮每场景游戏数: {}", num_episodes_per_iteration);
    println!("MCTS模拟次数: {}", mcts_sims);
    println!("训练迭代次数: {}", num_iterations);
    println!("推理批量大小: {}", inference_batch_size);
    println!("经验回放缓冲区: {}", max_buffer_size);
    println!("场景: Standard");
    
    // 创建模型和优化器
    let mut vs = nn::VarStore::new(device);
    let net = BanqiNet::new(&vs.root());
    let mut opt = nn::Adam::default().build(&vs, learning_rate)?;

    // 加载模型 (如果提供了参数)
    let args: Vec<String> = env::args().collect();
    if args.len() > 1 {
        let model_path = &args[1];
        println!("正在加载模型: {}", model_path);
        if let Err(e) = vs.load(model_path) {
             eprintln!("加载模型失败: {}", e);
        } else {
             println!("模型加载成功！");
        }
    }
    
    // 经验回放缓冲区
    let mut replay_buffer: Vec<(Observation, Vec<f32>, f32, Vec<i32>)> = Vec::new();
    
    // 主训练循环
    for iteration in 0..num_iterations {
        println!("\n========== Iteration {}/{} ==========", iteration + 1, num_iterations);
        
        // 保存临时模型供推理服务器使用
        let temp_model_path = format!("banqi_model_iter_{}_temp.ot", iteration);
        vs.save(&temp_model_path)?;
        
        // 创建推理通道
        let (req_tx, req_rx) = mpsc::channel();
        
        // 启动推理服务器线程
        let temp_model_path_clone = temp_model_path.clone();
        let inference_handle = thread::spawn(move || {
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
        
        // 启动工作线程 - 每个场景一个
        let scenarios = vec![ScenarioType::Standard; num_workers];
        let mut worker_handles = Vec::new();
        let mut result_rxs = Vec::new();
        
        for (worker_id, scenario) in scenarios.iter().enumerate() {
            let req_tx_clone = req_tx.clone();
            let (result_tx, result_rx) = mpsc::channel();
            result_rxs.push(result_rx);
            let scenario_copy = *scenario;
            
            let handle = thread::spawn(move || {
                let evaluator = Arc::new(ChannelEvaluator::new(req_tx_clone));
                let worker = SelfPlayWorker::with_scenario(worker_id, evaluator, mcts_sims, scenario_copy);
                
                let mut all_episodes = Vec::new();
                for ep in 0..num_episodes_per_iteration {
                    let episode = worker.play_episode(ep);
                    all_episodes.push(episode);
                }
                
                println!("  [Worker-{}] 完成 {} 局 {} 游戏", 
                    worker_id, num_episodes_per_iteration, scenario_copy.name());
                result_tx.send(all_episodes).expect("无法发送结果");
            });
            
            worker_handles.push(handle);
        }
        
        // 关闭主请求发送端
        drop(req_tx);
        
        // 收集所有工作线程的结果
        let mut all_episodes = Vec::new();
        for result_rx in result_rxs {
            if let Ok(episodes) = result_rx.recv() {
                all_episodes.extend(episodes);
            }
        }
        
        // 等待所有工作线程完成
        for handle in worker_handles {
            handle.join().expect("工作线程异常");
        }
        
        // 等待推理服务器退出
        inference_handle.join().expect("推理服务器异常");
        
        // 清理临时模型文件
        let _ = std::fs::remove_file(&temp_model_path);
        
        // // 过滤掉平局的游戏 (已注释，使用所有游戏)
        // let filtered_episodes: Vec<_> = all_episodes.iter()
        //     .filter(|ep| ep.winner.is_some() && ep.winner.unwrap() != 0)
        //     .cloned()
        //     .collect();
        
        // 使用所有游戏（包括平局）
        let filtered_episodes = all_episodes.clone();
        
        println!("  收集了 {} 局游戏", filtered_episodes.len());
        
        // 提取样本
        let mut new_samples = Vec::new();
        for episode in &filtered_episodes {
            new_samples.extend(episode.samples.clone());
        }
        
        if new_samples.is_empty() {
            println!("  ⚠️ 本轮没有收集到有效样本，跳过训练");
            continue;
        }
        
        println!("  收集了 {} 个训练样本", new_samples.len());
        
        // 更新经验回放缓冲区
        replay_buffer.extend(new_samples);
        if replay_buffer.len() > max_buffer_size {
            let remove_count = replay_buffer.len() - max_buffer_size;
            replay_buffer.drain(0..remove_count);
        }
        println!("  经验回放缓冲区: {} 个样本", replay_buffer.len());
        
        // 训练
        println!("  开始训练...");
        let train_start = Instant::now();
        let mut loss_sum = 0.0_f64;
        let mut p_loss_sum = 0.0_f64;
        let mut v_loss_sum = 0.0_f64;
        for epoch in 0..epochs_per_iteration {
            let (loss, p_loss, v_loss) = train_step(&mut opt, &net, &replay_buffer, batch_size, device, epoch);
            println!("    Epoch {}/{}: Loss={:.4} (Policy={:.4}, Value={:.4})", 
                epoch + 1, epochs_per_iteration, loss, p_loss, v_loss);
            loss_sum += loss;
            p_loss_sum += p_loss;
            v_loss_sum += v_loss;
        }
        let train_elapsed = train_start.elapsed();
        println!("  训练完成，耗时 {:.1}s", train_elapsed.as_secs_f64());
        
        // 验证模型
        println!("\n  ========== 模型验证 ==========");
        let (scenario1, scenario2) = validate_model_on_scenarios_with_net(&net, device, iteration);
        println!("    场景1 (TwoAdvisors): a38={:.1}%, value={:.3}", 
            scenario1.masked_probs[38] * 100.0, scenario1.value);
        println!("    场景2 (HiddenThreats): a3={:.1}%, value={:.3}", 
            scenario2.masked_probs[3] * 100.0, scenario2.value);

        // ========== 生成训练日志 ==========
        // 统计对弈整体数据（包含平局）
        let total_games = all_episodes.len().max(1);
        let red_wins = all_episodes.iter().filter(|ep| ep.winner == Some(1)).count();
        let black_wins = all_episodes.iter().filter(|ep| ep.winner == Some(-1)).count();
        let draws = all_episodes.iter().filter(|ep| ep.winner.is_none() || ep.winner == Some(0)).count();
        let avg_steps: f32 = all_episodes.iter().map(|ep| ep.game_length as f32).sum::<f32>() / total_games as f32;

        // 针对本轮新样本的策略熵与高置信度比率
        let (avg_entropy, high_conf_ratio) = if !filtered_episodes.is_empty() {
            let mut ent_sum = 0.0_f32;
            let mut count = 0usize;
            let mut high_conf = 0usize;
            for ep in &filtered_episodes {
                for (_, probs, _, _) in &ep.samples {
                    // 避免ln(0)
                    let mut e = 0.0_f32;
                    let mut maxp = 0.0_f32;
                    for &p in probs {
                        if p > 1e-8 {
                            e += -p * p.ln();
                        }
                        if p > maxp { maxp = p; }
                    }
                    ent_sum += e;
                    count += 1;
                    if maxp >= 0.90 { high_conf += 1; }
                }
            }
            if count > 0 {
                (ent_sum / count as f32, high_conf as f32 / count as f32)
            } else { (0.0, 0.0) }
        } else { (0.0, 0.0) };

        let avg_total_loss = loss_sum / epochs_per_iteration as f64;
        let avg_policy_loss = p_loss_sum / epochs_per_iteration as f64;
        let avg_value_loss = v_loss_sum / epochs_per_iteration as f64;
        let (plw, vlw) = get_loss_weights(epochs_per_iteration.saturating_sub(1));

        // 从场景验证结果中提取指标
        let log_record = TrainingLog {
            iteration,
            avg_total_loss,
            avg_policy_loss,
            avg_value_loss,
            policy_loss_weight: plw,
            value_loss_weight: vlw,
            // 场景1 (TwoAdvisors)
            scenario1_value: scenario1.value,
            scenario1_unmasked_a38: scenario1.unmasked_probs[38],
            scenario1_unmasked_a39: scenario1.unmasked_probs[39],
            scenario1_unmasked_a40: scenario1.unmasked_probs[40],
            scenario1_masked_a38: scenario1.masked_probs[38],
            scenario1_masked_a39: scenario1.masked_probs[39],
            scenario1_masked_a40: scenario1.masked_probs[40],
            // 场景2 (Hidden Threat)
            scenario2_value: scenario2.value,
            scenario2_unmasked_a3: scenario2.unmasked_probs[3],
            scenario2_unmasked_a5: scenario2.unmasked_probs[5],
            scenario2_masked_a3: scenario2.masked_probs[3],
            scenario2_masked_a5: scenario2.masked_probs[5],
            // 样本统计
            new_samples_count: filtered_episodes.iter().map(|ep| ep.samples.len()).sum::<usize>(),
            replay_buffer_size: replay_buffer.len(),
            avg_game_steps: avg_steps,
            red_win_ratio: red_wins as f32 / total_games as f32,
            draw_ratio: draws as f32 / total_games as f32,
            black_win_ratio: black_wins as f32 / total_games as f32,
            avg_policy_entropy: avg_entropy,
            high_confidence_ratio: high_conf_ratio,
        };

        // 写入CSV
        let csv_path = "training_log.csv";
        let _ = TrainingLog::write_header(csv_path);
        if let Err(e) = log_record.append_to_csv(csv_path) {
            eprintln!("  ⚠️ 写入训练日志失败: {}", e);
        } else {
            println!("  已记录训练日志到 {}", csv_path);
        }
        
        // 保存模型
        if (iteration + 1) % 5 == 0 || iteration == num_iterations - 1 {
            let model_path = format!("banqi_model_{}.ot", iteration + 1);
            vs.save(&model_path)?;
            println!("  已保存模型: {}", model_path);
        }
    }
    
    // 保存最终模型
    vs.save("banqi_model_scenario_latest.ot")?;
    println!("\n训练完成！已保存模型: banqi_model_scenario_latest.ot");
    println!("\n请使用以下命令测试模型:");
    println!("  cargo run --bin banqi-verify-trained -- banqi_model_scenario_latest.ot");
    
    Ok(())
}

fn main() {
    if let Err(e) = parallel_train_loop() {
        eprintln!("训练失败: {}", e);
    }
}
