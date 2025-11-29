// code_files/src/train.rs
use banqi_3x4::game_env::{DarkChessEnv, Observation};
use banqi_3x4::mcts::{Evaluator, MCTS, MCTSConfig};
use banqi_3x4::nn_model::BanqiNet;
use anyhow::Result;
use std::sync::Arc;
use tch::{nn, nn::OptimizerConfig, Device, Tensor, Kind};
use rusqlite::{Connection, params};

// 初始化SQLite数据库
fn init_database(db_path: &str) -> Result<Connection> {
    let conn = Connection::open(db_path)?;
    
    conn.execute(
        "CREATE TABLE IF NOT EXISTS training_samples (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            iteration INTEGER NOT NULL,
            episode_type TEXT NOT NULL,
            board_state BLOB NOT NULL,
            scalar_state BLOB NOT NULL,
            policy_probs BLOB NOT NULL,
            value_target REAL NOT NULL,
            action_mask BLOB NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )",
        [],
    )?;
    
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_iteration ON training_samples(iteration)",
        [],
    )?;
    
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_episode_type ON training_samples(episode_type)",
        [],
    )?;
    
    println!("数据库初始化完成: {}", db_path);
    Ok(conn)
}

// 保存训练样本到数据库
fn save_samples_to_db(
    conn: &mut Connection,
    iteration: usize,
    episode_type: &str,
    samples: &[(Observation, Vec<f32>, f32, Vec<i32>)]
) -> Result<()> {
    // 使用事务将多条 INSERT 合并提交，显著降低 I/O 次数
    let tx = conn.transaction()?;
    {
        let mut stmt = tx.prepare(
            "INSERT INTO training_samples 
             (iteration, episode_type, board_state, scalar_state, policy_probs, value_target, action_mask) 
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)"
        )?;
        
        for (obs, probs, value, mask) in samples {
            // 将ndarray转换为字节
            let board_bytes: Vec<u8> = obs.board.as_slice().unwrap()
                .iter()
                .flat_map(|&x| x.to_le_bytes())
                .collect();
            
            let scalar_bytes: Vec<u8> = obs.scalars.as_slice().unwrap()
                .iter()
                .flat_map(|&x| x.to_le_bytes())
                .collect();
            
            let probs_bytes: Vec<u8> = probs.iter()
                .flat_map(|&x| x.to_le_bytes())
                .collect();
            
            let mask_bytes: Vec<u8> = mask.iter()
                .flat_map(|&x| x.to_le_bytes())
                .collect();
            
            stmt.execute(params![
                iteration as i64,
                episode_type,
                board_bytes,
                scalar_bytes,
                probs_bytes,
                value,
                mask_bytes,
            ])?;
        }
    }
    // 先释放 stmt 的借用，再提交事务
    tx.commit()?;
    Ok(())
}

// 查询数据库统计信息
fn print_db_stats(conn: &Connection) -> Result<()> {
    let total: i64 = conn.query_row(
        "SELECT COUNT(*) FROM training_samples",
        [],
        |row| row.get(0)
    )?;
    
    let self_play: i64 = conn.query_row(
        "SELECT COUNT(*) FROM training_samples WHERE episode_type = 'self_play'",
        [],
        |row| row.get(0)
    )?;
    
    let scenario: i64 = conn.query_row(
        "SELECT COUNT(*) FROM training_samples WHERE episode_type = 'scenario'",
        [],
        |row| row.get(0)
    )?;
    
    println!("\n=== 数据库统计 ===");
    println!("总样本数: {}", total);
    println!("自对弈样本: {} ({:.1}%)", self_play, self_play as f64 / total as f64 * 100.0);
    println!("场景样本: {} ({:.1}%)", scenario, scenario as f64 / total as f64 * 100.0);
    
    Ok(())
}

// 从数据库加载训练样本
fn load_samples_from_db(conn: &Connection) -> Result<Vec<(Observation, Vec<f32>, f32, Vec<i32>)>> {
    let mut stmt = conn.prepare(
        "SELECT board_state, scalar_state, policy_probs, value_target, action_mask 
         FROM training_samples"
    )?;
    
    let samples = stmt.query_map([], |row| {
        let board_bytes: Vec<u8> = row.get(0)?;
        let scalar_bytes: Vec<u8> = row.get(1)?;
        let probs_bytes: Vec<u8> = row.get(2)?;
        let value: f32 = row.get(3)?;
        let mask_bytes: Vec<u8> = row.get(4)?;
        
        // 将字节转换回f32数组
        let board_data: Vec<f32> = board_bytes.chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();
        
        let scalar_data: Vec<f32> = scalar_bytes.chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();
        
        let probs: Vec<f32> = probs_bytes.chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();
        
        let mask: Vec<i32> = mask_bytes.chunks_exact(4)
            .map(|chunk| i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();
        
        // 重构Observation
        use ndarray::Array;
        let board = Array::from_shape_vec((2, 8, 3, 4), board_data)
            .expect("Failed to reshape board data");
        let scalars = Array::from_vec(scalar_data);
        
        let obs = Observation { board, scalars };
        
        Ok((obs, probs, value, mask))
    })?;
    
    let mut result = Vec::new();
    for sample in samples {
        result.push(sample?);
    }
    
    Ok(result)
}

// Wrapper for NN to implement Evaluator trait
struct NNEvaluator {
    net: BanqiNet,
    device: Device,
}

impl NNEvaluator {
    fn new(vs: &nn::Path, device: Device) -> Self {
        Self {
            net: BanqiNet::new(vs),
            device,
        }
    }
}

impl Evaluator for NNEvaluator {
    fn evaluate(&self, env: &DarkChessEnv) -> (Vec<f32>, f32) {
        // Convert Observation to Tensors
        let obs = env.get_state(); // Assuming get_state creates Observation
        
        // Board: [1, C, H, W]
        let board_tensor = Tensor::from_slice(&obs.board.into_raw_vec())
            .view([1, 16, 3, 4])
            .to(self.device);
            
        // Scalars: [1, F]
        let scalar_tensor = Tensor::from_slice(&obs.scalars.to_vec())
            .view([1, 112])
            .to(self.device);
            
        let (logits, value) = self.net.forward_inference(&board_tensor, &scalar_tensor);
        
        // --- 修复：应用动作掩码 ---
        // 获取有效动作掩码
        let masks: Vec<f32> = env.action_masks().iter().map(|&m| m as f32).collect();
        let mask_tensor = Tensor::from_slice(&masks).to(self.device).view([1, 46]);
        
        // 将掩码为 0 的位置的 logits 设为极小值 (-1e9)
        // 公式: masked_logits = logits + (mask - 1.0) * 1e9
        let masked_logits = &logits + (&mask_tensor - 1.0) * 1e9;
        
        let probs = masked_logits.softmax(-1, Kind::Float);
        // --- 修复结束 ---
        
        // Extract to Vec via data_ptr/try_data_ptr or use .shallow_clone() + as_slice
        let probs_flat = probs.view([-1]);
        let probs_vec: Vec<f32> = (0..probs_flat.size()[0])
            .map(|i| probs_flat.double_value(&[i]) as f32)
            .collect();
        let value_scalar: f32 = value.squeeze().double_value(&[]) as f32;
        
        (probs_vec, value_scalar)
    }
}

// 从验证场景收集训练样本（让MCTS自己玩）
#[allow(dead_code)]
fn collect_scenario_samples(
    evaluator: Arc<NNEvaluator>,
    mcts_sims: usize,
) -> Vec<(Observation, Vec<f32>, f32, Vec<i32>)> {
    use banqi_3x4::game_env::Player;
    
    let mut all_samples = Vec::new();
    
    println!("  [Debug] === 场景1: R_A vs B_A ===");
    // 场景1: R_A vs B_A
    {
        let mut env = DarkChessEnv::new();
        env.setup_two_advisors(Player::Black);
        let mut mcts = MCTS::new(&env, evaluator.clone(), MCTSConfig { num_simulations: mcts_sims, cpuct: 1.0 });
        
        let mut episode_data = Vec::new();
        let mut step = 0;
        
        loop {
            mcts.run();
            let probs = mcts.get_root_probabilities();
            let masks = env.action_masks();
            
            // 打印MCTS策略
            if step == 0 {
                let mut prob_actions: Vec<(usize, f32)> = probs.iter().enumerate()
                    .filter(|(_, &p)| p > 0.01)
                    .map(|(i, &p)| (i, p))
                    .collect();
                prob_actions.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                println!("    Step {}, MCTS策略: {}", 
                    step,
                    prob_actions.iter().take(5)
                        .map(|(a, p)| format!("a{}({:.1}%)", a, p*100.0))
                        .collect::<Vec<_>>().join(", ")
                );
                println!("    目标: action38应该主导 (期望>90%)");
            }
            
            episode_data.push((env.get_state(), probs.clone(), env.get_current_player(), masks));
            
            // 选择概率最高的动作（贪婪策略，用于验证场景）
            let action = probs.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx)
                .unwrap_or(0);
            
            match env.step(action, None) {
                Ok((_, _, terminated, truncated, winner)) => {
                    mcts.step_next(&env, action);
                    
                    if terminated || truncated {
                        let reward_red = match winner {
                            Some(1) => 1.0,
                            Some(-1) => -1.0,
                            _ => 0.0,
                        };
                        
                        println!("    场景1结束: 步数={}, 胜者={:?}", step, winner);
                        
                        // Backfill value
                        for (obs, p, player, mask) in episode_data {
                            let val = if player.val() == 1 { reward_red } else { -reward_red };
                            all_samples.push((obs, p, val, mask));
                        }
                        break;
                    }
                },
                Err(e) => {
                    eprintln!("    场景1错误: {}", e);
                    break;
                }
            }
            
            step += 1;
            if step > 20 { break; }
        }
    }
    
    println!("  [Debug] === 场景2: Hidden Threat ===");
    // 场景2: Hidden Threat
    {
        let mut env = DarkChessEnv::new();
        env.setup_hidden_threats();
        let mut mcts = MCTS::new(&env, evaluator.clone(), MCTSConfig { num_simulations: mcts_sims, cpuct: 1.0 });
        
        let mut episode_data = Vec::new();
        let mut step = 0;
        
        loop {
            mcts.run();
            let probs = mcts.get_root_probabilities();
            let masks = env.action_masks();
            
            // 打印MCTS策略
            if step == 0 {
                let mut prob_actions: Vec<(usize, f32)> = probs.iter().enumerate()
                    .filter(|(_, &p)| p > 0.01)
                    .map(|(i, &p)| (i, p))
                    .collect();
                prob_actions.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                println!("    Step {}, MCTS策略: {}", 
                    step,
                    prob_actions.iter().take(5)
                        .map(|(a, p)| format!("a{}({:.1}%)", a, p*100.0))
                        .collect::<Vec<_>>().join(", ")
                );
                println!("    目标: action3应该主导 (期望>90%)");
            }
            
            episode_data.push((env.get_state(), probs.clone(), env.get_current_player(), masks));
            
            // 选择概率最高的动作
            let action = probs.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx)
                .unwrap_or(0);
            
            match env.step(action, None) {
                Ok((_, _, terminated, truncated, winner)) => {
                    mcts.step_next(&env, action);
                    
                    if terminated || truncated {
                        let reward_red = match winner {
                            Some(1) => 1.0,
                            Some(-1) => -1.0,
                            _ => 0.0,
                        };
                        
                        println!("    场景2结束: 步数={}, 胜者={:?}", step, winner);
                        
                        // Backfill value
                        for (obs, p, player, mask) in episode_data {
                            let val = if player.val() == 1 { reward_red } else { -reward_red };
                            all_samples.push((obs, p, val, mask));
                        }
                        break;
                    }
                },
                Err(e) => {
                    eprintln!("    场景2错误: {}", e);
                    break;
                }
            }
            
            step += 1;
            if step > 20 { break; }
        }
    }
    
    all_samples
}

// 验证函数：在两个标准场景上测试模型
fn validate_scenarios(net: &BanqiNet, device: Device, iteration: usize) {
    use banqi_3x4::game_env::Player;
    
    // 场景1: 仅剩 R_A 与 B_A
    {
        let mut env = DarkChessEnv::new();
        env.setup_two_advisors(Player::Black);
        
        let obs = env.get_state();
        let board_tensor = Tensor::from_slice(obs.board.as_slice().unwrap())
            .view([1, 16, 3, 4])
            .to(device);
        let scalar_tensor = Tensor::from_slice(obs.scalars.as_slice().unwrap())
            .view([1, 112])
            .to(device);
        
        let masks: Vec<f32> = env.action_masks().iter().map(|&m| m as f32).collect();
        let mask_tensor = Tensor::from_slice(&masks).to(device).view([1, 46]);
        
        let (logits, value) = net.forward_inference(&board_tensor, &scalar_tensor);
        
        // 调试：打印原始logits (应用mask前)
        if iteration <= 2 {
            let logits_vec: Vec<f32> = (0..46).map(|i| logits.double_value(&[0, i]) as f32).collect();
            let top_logits: Vec<(usize, f32)> = logits_vec.iter().enumerate()
                .map(|(i, &l)| (i, l))
                .collect::<Vec<_>>();
            let mut sorted_logits = top_logits.clone();
            sorted_logits.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            println!("    [Debug] 原始Logits Top-5: {}", 
                sorted_logits.iter().take(5)
                    .map(|(a, l)| format!("a{}({:.2})", a, l))
                    .collect::<Vec<_>>().join(", "));
            println!("    [Debug] action38 logit={:.2}, action39 logit={:.2}", 
                logits_vec[38], logits_vec[39]);
        }
        
        let masked_logits = &logits + (&mask_tensor - 1.0) * 1e9;
        let probs = masked_logits.softmax(-1, Kind::Float);
        
        let value_pred: f32 = value.squeeze().double_value(&[]) as f32;
        let probs_vec: Vec<f32> = (0..46).map(|i| probs.double_value(&[0, i]) as f32).collect();
        
        let mut indexed: Vec<(usize, f32)> = probs_vec.iter().enumerate()
            .filter(|(_, &p)| p > 0.001)
            .map(|(i, &p)| (i, p))
            .collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        println!("  场景1 (R_A vs B_A):");
        println!("    Value: {:.4}", value_pred);
        println!("    Policy Top-3: {}", 
            indexed.iter().take(3)
                .map(|(a, p)| format!("action{}({:.1}%)", a, p*100.0))
                .collect::<Vec<_>>().join(", "));
        println!("    目标: action38(9->5) 应该概率最高 (MCTS结果: 99.34%)");
        println!("    实际: action38={:.1}%, action39={:.1}%", 
            probs_vec[38]*100.0, probs_vec[39]*100.0);
    }
    
    // 场景2: 隐藏的威胁
    {
        let mut env = DarkChessEnv::new();
        env.setup_hidden_threats();
        
        let obs = env.get_state();
        let board_tensor = Tensor::from_slice(obs.board.as_slice().unwrap())
            .view([1, 16, 3, 4])
            .to(device);
        let scalar_tensor = Tensor::from_slice(obs.scalars.as_slice().unwrap())
            .view([1, 112])
            .to(device);
        
        let masks: Vec<f32> = env.action_masks().iter().map(|&m| m as f32).collect();
        let mask_tensor = Tensor::from_slice(&masks).to(device).view([1, 46]);
        
        let (logits, value) = net.forward_inference(&board_tensor, &scalar_tensor);
        
        // 调试：打印原始logits
        if iteration <= 2 {
            let logits_vec: Vec<f32> = (0..46).map(|i| logits.double_value(&[0, i]) as f32).collect();
            println!("    [Debug] action3 logit={:.2}, action5 logit={:.2}", 
                logits_vec[3], logits_vec[5]);
        }
        
        let masked_logits = &logits + (&mask_tensor - 1.0) * 1e9;
        let probs = masked_logits.softmax(-1, Kind::Float);
        
        let value_pred: f32 = value.squeeze().double_value(&[]) as f32;
        let probs_vec: Vec<f32> = (0..46).map(|i| probs.double_value(&[0, i]) as f32).collect();
        
        let mut indexed: Vec<(usize, f32)> = probs_vec.iter().enumerate()
            .filter(|(_, &p)| p > 0.001)
            .map(|(i, &p)| (i, p))
            .collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        println!("  场景2 (Hidden Threat):");
        println!("    Value: {:.4}", value_pred);
        println!("    Policy Top-2: {}", 
            indexed.iter().take(2)
                .map(|(a, p)| format!("action{}({:.1}%)", a, p*100.0))
                .collect::<Vec<_>>().join(", "));
        println!("    目标: action3(reveal@3) 应该概率最高 (MCTS结果: 98.02%)");
        println!("    实际: action3={:.1}%, action5={:.1}%", 
            probs_vec[3]*100.0, probs_vec[5]*100.0);
    }
}

pub fn train_loop() -> Result<()> {
    // Check CUDA availability
    let cuda_available = tch::Cuda::is_available();
    let cuda_device_count = tch::Cuda::device_count();
    println!("CUDA available: {}", cuda_available);
    println!("CUDA device count: {}", cuda_device_count);
    
    let device = if cuda_available {
        println!("Using CUDA device 0");
        Device::Cuda(0)
    } else {
        println!("CUDA not available, using CPU");
        Device::Cpu
    };
    println!("Training using device: {:?}", device);
    
    // 初始化数据库
    let db_path = "training_samples.db";
    let mut conn = init_database(db_path)?;
    println!("样本将保存到: {}", db_path);
    
    let vs = nn::VarStore::new(device);
    let evaluator = Arc::new(NNEvaluator::new(&vs.root(), device));
    
    // 降低学习率以提高训练稳定性
    let learning_rate = 1e-4; // 从1e-3降到1e-4
    let mut opt = nn::Adam::default().build(&vs, learning_rate)?;
    
    println!("优化器配置: Adam, 学习率 = {}", learning_rate);
    
    // Training Hyperparameters - 正式训练配置
    let num_iterations = 200;      // 增加到200轮正式训练
    let num_episodes = 50;         // 每轮50局自对弈
    let mcts_sims = 800;           // 增加MCTS模拟次数以提升策略质量
    let batch_size = 256;          // 增大batch size 以提高训练效率
    let epochs_per_iteration = 10; // 每轮10个epoch
    let validation_interval = 10;  // 每10轮进行一次验证
    
    println!("训练配置: iterations={}, episodes={}, mcts_sims={}, batch_size={}, epochs={}", 
        num_iterations, num_episodes, mcts_sims, batch_size, epochs_per_iteration);
    
    // ========== 第一阶段：从数据库加载已有数据进行训练 ==========
    println!("\n============================================================");
    println!("第一阶段：从数据库加载已有样本进行训练");
    println!("============================================================");
    
    let existing_samples = load_samples_from_db(&conn)?;
    if !existing_samples.is_empty() {
        println!("成功加载 {} 个已有训练样本", existing_samples.len());
        
        // 使用已有样本进行初始训练
        let initial_training_epochs = 20; // 对已有数据训练更多轮
        println!("开始使用已有样本进行 {} 轮训练...", initial_training_epochs);
        
        let mut total_losses = Vec::new();
        let mut policy_losses = Vec::new();
        let mut value_losses = Vec::new();
        
        for epoch in 0..initial_training_epochs {
            let (loss, p_loss, v_loss) = train_step(&mut opt, &evaluator.net, &existing_samples, batch_size, device, epoch);
            total_losses.push(loss);
            policy_losses.push(p_loss);
            value_losses.push(v_loss);
            
            if (epoch + 1) % 5 == 0 || epoch == initial_training_epochs - 1 {
                println!("  Epoch {}/{}, Loss={:.4} (Policy={:.4}, Value={:.4})", 
                    epoch + 1, initial_training_epochs, loss, p_loss, v_loss);
            }
        }
        
        let avg_loss: f64 = total_losses.iter().sum::<f64>() / total_losses.len() as f64;
        let avg_p_loss: f64 = policy_losses.iter().sum::<f64>() / policy_losses.len() as f64;
        let avg_v_loss: f64 = value_losses.iter().sum::<f64>() / value_losses.len() as f64;
        println!("  初始训练平均Loss: {:.4} (Policy={:.4}, Value={:.4})", avg_loss, avg_p_loss, avg_v_loss);
        
        // 验证模型
        println!("\n  ========== 初始训练后模型验证 ==========");
        validate_scenarios(&evaluator.net, device, 0);
        
        // 保存初始训练后的模型
        vs.save("banqi_model_pretrained.ot")?;
        println!("已保存初始训练模型: banqi_model_pretrained.ot");
    } else {
        println!("数据库中没有已有样本，跳过初始训练阶段");
    }
    
    // ========== 第二阶段：开始自对弈收集数据并训练 ==========
    println!("\n============================================================");
    println!("第二阶段：开始自对弈收集新数据并训练");
    println!("============================================================");
    
    for iteration in 0..num_iterations {
        println!("\n============================================================");
        println!("Iteration {}/{}", iteration, num_iterations);
        println!("============================================================");
        
        let mut examples = Vec::new();
        
        // 1. Self-Play
        for eps in 0..num_episodes {
            let mut env = DarkChessEnv::new();
            let mut mcts = MCTS::new(&env, evaluator.clone(), MCTSConfig { num_simulations: mcts_sims, cpuct: 1.0 });
            
            let mut episode_step = 0;
            let mut episode_data = Vec::new(); // (Observation, PolicyProbs, Player, ActionMasks)
            
            loop {
                // Run MCTS
                mcts.run();
                let probs = mcts.get_root_probabilities();
                
                // 调试：打印第一个episode的前几步MCTS策略
                if iteration == 0 && eps == 0 && episode_step < 3 {
                    let mut prob_actions: Vec<(usize, f32)> = probs.iter().enumerate()
                        .filter(|(_, &p)| p > 0.01)
                        .map(|(i, &p)| (i, p))
                        .collect();
                    prob_actions.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                    println!("    [Debug] Episode 0, Step {}, MCTS策略 Top-3: {}", 
                        episode_step,
                        prob_actions.iter().take(3)
                            .map(|(a, p)| format!("a{}({:.1}%)", a, p*100.0))
                            .collect::<Vec<_>>().join(", ")
                    );
                }
                
                // Store data with action masks
                let masks = env.action_masks();
                episode_data.push((env.get_state(), probs.clone(), env.get_current_player(), masks));
                
                // Select action with temperature (前30步用高温度增加探索)
                let temperature = if episode_step < 30 { 1.2 } else { 0.8 };
                let action = sample_action(&probs, &env, temperature);
                
                // Step Env
                match env.step(action, None) {
                    Ok((_, _, terminated, truncated, winner)) => {
                        // Advance MCTS root for reuse
                        mcts.step_next(&env, action);
                        
                        if terminated || truncated {
                            // Assign rewards
                            let reward_red = match winner {
                                Some(1) => 1.0,
                                Some(-1) => -1.0,
                                _ => 0.0,
                            };
                            
                            // 调试：打印游戏结果
                            if iteration == 0 && eps == 0 {
                                println!("    [Debug] Episode 0 结束: 步数={}, 胜者={:?}, reward_red={}", 
                                    episode_step, winner, reward_red);
                            }
                            
                            // Backfill value to examples
                            for (obs, p, player, mask) in episode_data {
                                let val = if player.val() == 1 { reward_red } else { -reward_red };
                                examples.push((obs, p, val, mask));
                            }
                            break;
                        }
                    },
                    Err(e) => panic!("Error: {}", e),
                }
                
                episode_step += 1;
                if episode_step > 200 { 
                    break; 
                }
            }
            
            // 每完成一局游戏打印进度
            if (eps + 1) % 5 == 0 || eps == num_episodes - 1 {
                println!("  Self-play progress: {}/{} episodes", eps + 1, num_episodes);
            }
        }
        
        println!("  收集了 {} 个训练样本", examples.len());
        
        // 保存自对弈样本到数据库
    save_samples_to_db(&mut conn, iteration, "self_play", &examples)?;
        println!("  [Debug] 已保存 {} 个自对弈样本到数据库", examples.len());
        
        // 移除在特定场景上的训练，仅保留自对弈样本
        // 注意：仍会在验证阶段使用固定场景进行评估（见 validate_scenarios）
        
        // 调试：分析训练样本的价值分布和策略分布
        if iteration <= 2 {
            let mut value_counts = [0, 0, 0]; // [负值, 0, 正值]
            for (_, _, val, _) in examples.iter() {
                if *val < -0.1 { value_counts[0] += 1; }
                else if *val > 0.1 { value_counts[2] += 1; }
                else { value_counts[1] += 1; }
            }
            println!("  [Debug] 价值分布: 负值={}, 零值={}, 正值={}", 
                value_counts[0], value_counts[1], value_counts[2]);
            
            // 检查前3个样本的策略熵和最大概率动作
            for i in 0..3.min(examples.len()) {
                let (_, probs, val, _) = &examples[i];
                let entropy: f32 = probs.iter()
                    .filter(|&&p| p > 1e-8)
                    .map(|&p| -p * p.ln())
                    .sum();
                let max_prob = probs.iter().cloned().fold(0.0f32, f32::max);
                let max_action = probs.iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(idx, _)| idx)
                    .unwrap_or(0);
                println!("  [Debug] 样本{}: value={:.3}, max_action=a{}, max_prob={:.3}, entropy={:.3}", 
                    i, val, max_action, max_prob, entropy);
            }
            
            // 统计关键动作出现频率
            let mut action38_count = 0;
            let mut action3_count = 0;
            for (_, probs, _, _) in examples.iter() {
                if probs[38] > 0.5 { action38_count += 1; }
                if probs[3] > 0.5 { action3_count += 1; }
            }
            println!("  [Debug] 训练集中: action38主导的样本={}, action3主导的样本={}", 
                action38_count, action3_count);
        }
        
        // 2. Training - 多个epoch遍历所有数据
        let mut total_losses = Vec::new();
        let mut policy_losses = Vec::new();
        let mut value_losses = Vec::new();
        
        for epoch in 0..epochs_per_iteration {
            let (loss, p_loss, v_loss) = train_step(&mut opt, &evaluator.net, &examples, batch_size, device, epoch);
            total_losses.push(loss);
            policy_losses.push(p_loss);
            value_losses.push(v_loss);
            
            // 每2个epoch或最后一个epoch打印进度
            if (epoch + 1) % 2 == 0 || epoch == epochs_per_iteration - 1 {
                println!("  Training: Epoch {}/{}, Loss={:.4} (Policy={:.4}, Value={:.4})", 
                    epoch + 1, epochs_per_iteration, loss, p_loss, v_loss);
            }
        }
        
        let avg_loss: f64 = total_losses.iter().sum::<f64>() / total_losses.len() as f64;
        let avg_p_loss: f64 = policy_losses.iter().sum::<f64>() / policy_losses.len() as f64;
        let avg_v_loss: f64 = value_losses.iter().sum::<f64>() / value_losses.len() as f64;
        println!("  平均Loss: {:.4} (Policy={:.4}, Value={:.4})", avg_loss, avg_p_loss, avg_v_loss);
        
        // 3. 定期验证 - 使用两个标准场景测试模型
        if iteration % validation_interval == 0 || iteration == num_iterations - 1 {
            println!("\n  ========== 模型验证 (Iteration {}) ==========", iteration);
            validate_scenarios(&evaluator.net, device, iteration);
        }
        
        println!();
        
        // Save model every iteration for quick verification
        vs.save(format!("banqi_model_{}.ot", iteration))?;
        
        // Also save as "latest" for easy loading
        if iteration == num_iterations - 1 {
            vs.save("banqi_model_latest.ot")?;
        }
    }
    
    // 打印数据库统计信息
    print_db_stats(&conn)?;
    println!("\n训练完成！所有样本已保存到: {}", db_path);
    
    Ok(())
}

fn sample_action(probs: &[f32], env: &DarkChessEnv, temperature: f32) -> usize {
    use rand::distributions::WeightedIndex;
    use rand::prelude::*;
    
    // Filter out zero probabilities and check if we have any valid actions
    let non_zero_sum: f32 = probs.iter().sum();
    
    if non_zero_sum == 0.0 {
        // Fallback: choose uniformly from valid actions according to action mask
        let masks = env.action_masks();
        let valid_actions: Vec<usize> = masks.iter()
            .enumerate()
            .filter_map(|(i, &m)| if m == 1 { Some(i) } else { None })
            .collect();
        
        let mut rng = thread_rng();
        *valid_actions.choose(&mut rng).expect("No valid actions available")
    } else {
        // 应用温度参数：temperature > 1.0 增加探索，< 1.0 增加利用
        let adjusted_probs: Vec<f32> = if temperature != 1.0 {
            let sum: f32 = probs.iter()
                .map(|&p| p.powf(1.0 / temperature))
                .sum();
            probs.iter()
                .map(|&p| p.powf(1.0 / temperature) / sum)
                .collect()
        } else {
            probs.to_vec()
        };
        
        let dist = WeightedIndex::new(&adjusted_probs).unwrap();
        let mut rng = thread_rng();
        dist.sample(&mut rng)
    }
}

fn train_step(
    opt: &mut nn::Optimizer,
    net: &BanqiNet,
    examples: &[(Observation, Vec<f32>, f32, Vec<i32>)],
    batch_size: usize,
    device: Device,
    epoch: usize,
) -> (f64, f64, f64) {  // 返回 (total_loss, policy_loss, value_loss)
    if examples.is_empty() { return (0.0, 0.0, 0.0); }
    
    use rand::seq::SliceRandom;
    use rand::thread_rng;
    
    let mut shuffled_examples = examples.to_vec();
    shuffled_examples.shuffle(&mut thread_rng());
    
    let mut total_loss_sum = 0.0;
    let mut policy_loss_sum = 0.0;
    let mut value_loss_sum = 0.0;
    let mut num_samples = 0;
    
    // 策略损失权重：逐渐增加策略的重要性
    let policy_weight = 1.0 + (epoch as f32 * 0.2).min(2.0);
    
    // 遍历所有样本，每次处理batch_size个
    for batch_start in (0..shuffled_examples.len()).step_by(batch_size) {
        let batch_end = (batch_start + batch_size).min(shuffled_examples.len());
        let batch = &shuffled_examples[batch_start..batch_end];
        
        // 对batch中的每个样本进行训练
        for (obs, target_probs, target_val, masks) in batch.iter() {
            // Prepare tensors
            let board_tensor = Tensor::from_slice(obs.board.as_slice().unwrap()).view([1, 16, 3, 4]).to(device);
            let scalar_tensor = Tensor::from_slice(obs.scalars.as_slice().unwrap()).view([1, 112]).to(device);
            let target_p = Tensor::from_slice(target_probs).view([1, 46]).to(device);
            let target_v = Tensor::from_slice(&[*target_val]).view([1, 1]).to(device);
            
            // 将掩码转换为 Tensor [1, 46]
            let mask_vec: Vec<f32> = masks.iter().map(|&m| m as f32).collect();
            let mask_tensor = Tensor::from_slice(&mask_vec).view([1, 46]).to(device);
            
            let (logits, value) = net.forward(&board_tensor, &scalar_tensor);
            
            // 应用动作掩码到训练 Loss 计算
            let masked_logits = &logits + (&mask_tensor - 1.0) * 1e9;
            let log_probs = masked_logits.log_softmax(-1, Kind::Float);
            
            // 交叉熵损失：-sum(target * log(pred))
            let p_loss = (&target_p * &log_probs).sum(Kind::Float).neg() * (policy_weight as f64);
            let v_loss = value.mse_loss(&target_v, tch::Reduction::Mean);
            
            let total_loss = &p_loss + &v_loss;
            
            opt.backward_step(&total_loss);
            
            total_loss_sum += total_loss.double_value(&[]);
            policy_loss_sum += p_loss.double_value(&[]) / policy_weight as f64;
            value_loss_sum += v_loss.double_value(&[]);
            num_samples += 1;
        }
    }
    
    if num_samples > 0 { 
        (total_loss_sum / num_samples as f64,
         policy_loss_sum / num_samples as f64,
         value_loss_sum / num_samples as f64)
    } else { 
        (0.0, 0.0, 0.0)
    }
}

fn main() {
    // Entry point for training binary
    if let Err(e) = train_loop() {
        eprintln!("Training failed: {}", e);
    }
}