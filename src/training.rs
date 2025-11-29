// training.rs - 训练步骤模块
//
// 提供神经网络训练的核心逻辑，包括批量训练、损失计算等

use crate::game_env::Observation;
use crate::nn_model::BanqiNet;
use tch::{nn, Device, Tensor, Kind};
use rand::seq::SliceRandom;
use rand::thread_rng;

// ================ 训练步骤 ================

/// 执行一个训练步骤（一个epoch）
/// 
/// # 参数
/// - `opt`: 优化器
/// - `net`: 神经网络模型
/// - `examples`: 训练样本 (观察, 策略概率, 价值目标, 动作掩码)
/// - `batch_size`: 批量大小
/// - `device`: 训练设备 (CPU/GPU)
/// - `epoch`: 当前epoch编号 (用于动态调整损失权重)
/// 
/// # 返回
/// (总损失, 策略损失, 价值损失) - 每个样本的平均值
pub fn train_step(
    opt: &mut nn::Optimizer,
    net: &BanqiNet,
    examples: &[(Observation, Vec<f32>, f32, Vec<i32>)],
    batch_size: usize,
    device: Device,
    epoch: usize,
) -> (f64, f64, f64) {
    if examples.is_empty() { 
        return (0.0, 0.0, 0.0); 
    }
    
    // 打乱训练样本
    let mut shuffled_examples = examples.to_vec();
    shuffled_examples.shuffle(&mut thread_rng());
    
    let mut total_loss_sum = 0.0;
    let mut policy_loss_sum = 0.0;
    let mut value_loss_sum = 0.0;
    let mut num_samples = 0;
    
    // 动态调整策略权重: 早期更注重策略学习,后期平衡
    let policy_weight = 1.5 + (epoch as f32 * 0.1).min(1.0); // 从1.5逐渐增加到2.5
    let value_weight = 1.0; 

    // 🐛 DEBUG: 检查样本统计
    let mut value_stats = Vec::new();
    let mut entropy_stats = Vec::new();
    
    for batch_start in (0..shuffled_examples.len()).step_by(batch_size) {
        let batch_end = (batch_start + batch_size).min(shuffled_examples.len());
        let batch = &shuffled_examples[batch_start..batch_end];
        let bsz = batch.len();
        if bsz == 0 { continue; }

        let mut board_buf = Vec::with_capacity(bsz * 8 * 3 * 4);
        let mut scalar_buf = Vec::with_capacity(bsz * 56);
        let mut target_prob_buf = Vec::with_capacity(bsz * 46);
        let mut target_val_buf = Vec::with_capacity(bsz);
        let mut mask_buf = Vec::with_capacity(bsz * 46);
        
        for (obs, target_probs, target_val, masks) in batch.iter() {
            let board_slice = obs.board.as_slice().expect("board slice");
            board_buf.extend_from_slice(board_slice);
            let scalar_slice = obs.scalars.as_slice().expect("scalar slice");
            scalar_buf.extend_from_slice(scalar_slice);
            target_prob_buf.extend_from_slice(target_probs);
            target_val_buf.push(*target_val);
            mask_buf.extend(masks.iter().map(|&m| m as f32));
            
            // 🐛 DEBUG: 收集统计数据
            value_stats.push(*target_val);
            let entropy: f32 = target_probs.iter()
                .filter(|&&p| p > 1e-8)
                .map(|&p| -p * p.ln())
                .sum();
            entropy_stats.push(entropy);
        }

        let board_tensor = Tensor::from_slice(&board_buf)
            .view([bsz as i64, 8, 3, 4])
            .to(device);
        let scalar_tensor = Tensor::from_slice(&scalar_buf)
            .view([bsz as i64, 56])
            .to(device);
        let target_p = Tensor::from_slice(&target_prob_buf)
            .view([bsz as i64, 46])
            .to(device);
        let target_v = Tensor::from_slice(&target_val_buf)
            .view([bsz as i64, 1])
            .to(device);
        let mask_tensor = Tensor::from_slice(&mask_buf)
            .view([bsz as i64, 46])
            .to(device);

        let (logits, value) = net.forward(&board_tensor, &scalar_tensor);
        let masked_logits = &logits + (&mask_tensor - 1.0) * 1e9;
        let log_probs = masked_logits.log_softmax(-1, Kind::Float);

        // 策略损失: 交叉熵 (按样本平均)
        let reduce_dim = [-1i64];
        let p_loss = (&target_p * &log_probs)
            .sum_dim_intlist(&reduce_dim[..], false, Kind::Float)
            .mean(Kind::Float)
            .neg() * (policy_weight as f64);
        // 价值损失: MSE
        let v_loss = value.mse_loss(&target_v, tch::Reduction::Mean) * (value_weight as f64);

        let total_loss = &p_loss + &v_loss;
        opt.backward_step(&total_loss);

        // 获取batch平均损失值
        let batch_loss_val = total_loss.double_value(&[]);
        let batch_p_loss_val = p_loss.double_value(&[]) / policy_weight as f64;
        let batch_v_loss_val = v_loss.double_value(&[]) / value_weight as f64;

        // 还原为总和 (乘以 bsz) - 修复统计bug
        // 因为损失已经是Reduction::Mean的结果,需要乘以batch_size还原为总和
        total_loss_sum += batch_loss_val * bsz as f64;
        policy_loss_sum += batch_p_loss_val * bsz as f64;
        value_loss_sum += batch_v_loss_val * bsz as f64;
        num_samples += bsz;
    }
    
    // 🐛 DEBUG: 输出样本质量统计
    if epoch == 0 && !value_stats.is_empty() {
        let avg_value: f32 = value_stats.iter().sum::<f32>() / value_stats.len() as f32;
        let std_value: f32 = (value_stats.iter().map(|v| (v - avg_value).powi(2)).sum::<f32>() / value_stats.len() as f32).sqrt();
        let avg_entropy: f32 = entropy_stats.iter().sum::<f32>() / entropy_stats.len() as f32;
        
        let positive_values = value_stats.iter().filter(|&&v| v > 0.0).count();
        let negative_values = value_stats.iter().filter(|&&v| v < 0.0).count();
        let zero_values = value_stats.iter().filter(|&&v| v == 0.0).count();
        
        println!("    🐛 样本统计: 总数={}, 价值[avg={:.3}, std={:.3}], 熵[avg={:.3}]", 
            value_stats.len(), avg_value, std_value, avg_entropy);
        println!("    🐛 价值分布: 正={} ({:.1}%), 零={} ({:.1}%), 负={} ({:.1}%)",
            positive_values, positive_values as f32 / value_stats.len() as f32 * 100.0,
            zero_values, zero_values as f32 / value_stats.len() as f32 * 100.0,
            negative_values, negative_values as f32 / value_stats.len() as f32 * 100.0);
    }
    
    if num_samples > 0 { 
        (total_loss_sum / num_samples as f64,
         policy_loss_sum / num_samples as f64,
         value_loss_sum / num_samples as f64)
    } else { 
        (0.0, 0.0, 0.0)
    }
}

/// 获取当前epoch的损失权重
/// 
/// # 参数
/// - `epoch`: 当前epoch编号
/// 
/// # 返回
/// (策略权重, 价值权重)
pub fn get_loss_weights(epoch: usize) -> (f64, f64) {
    let policy_weight = 1.0;
    let value_weight = 1.0;
    (policy_weight as f64, value_weight as f64)
}
