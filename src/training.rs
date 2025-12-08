// src/training.rs - 训练步骤模块
//
// 本模块实现了神经网络训练的核心循环逻辑。
// 主要功能：
// 1. 数据准备：将 GameEpisode 列表打平为单个样本列表并进行随机打乱。
// 2. 批量处理：将样本分批转换为 Tensor 格式（棋盘、标量、目标概率、目标价值、动作掩码）。
// 3. 前向传播：通过 BanqiNet 计算预测策略和价值。
// 4. 损失计算：
//    - 策略损失：带掩码的交叉熵损失 (Cross Entropy with Action Masks)。
//    - 价值损失：均方误差损失 (MSE)。
// 5. 反向传播与优化：执行梯度下降更新模型参数。
// 6. 内存管理：显式释放中间 Tensor 以防止在训练循环中通过 tch-rs 产生内存泄漏。

use crate::game_env::{Observation, ACTION_SPACE_SIZE, BOARD_CHANNELS, BOARD_COLS, BOARD_ROWS, SCALAR_FEATURE_COUNT};
use crate::nn_model::BanqiNet;
use crate::self_play::GameEpisode;
use rand::seq::SliceRandom;
use rand::thread_rng;
use tch::{nn, Device, Kind, Tensor};

// ================ 训练步骤 ================

/// 执行一个训练步骤（通常对应一个 epoch）
///
/// # 参数
/// - `opt`: 优化器 (Optimizer)，用于更新模型参数。
/// - `net`: 神经网络模型 (BanqiNet)。
/// - `game_episodes`: 包含训练数据的游戏回合列表。函数内部会将其打平为样本。
/// - `batch_size`: 批量大小。
/// - `device`: 训练设备 (CPU 或 CUDA)。
/// - `epoch`: 当前 epoch 索引，可用于调整日志输出或动态权重（目前权重固定）。
///
/// # 返回
/// 返回元组 (总损失, 策略损失, 价值损失)，均为该 epoch 内所有样本的平均值。
pub fn train_step(
    opt: &mut nn::Optimizer,
    net: &BanqiNet,
    game_episodes: &[GameEpisode],
    batch_size: usize,
    device: Device,
    epoch: usize,
) -> (f64, f64, f64) {
    // 1. 数据打平与收集
    // 将结构化的 GameEpisode 数据转换为扁平的样本引用列表，以便进行 Shuffle 和 Batching
    let mut sample_refs: Vec<&(Observation, Vec<f32>, f32, f32, Vec<i32>)> = Vec::new();
    for episode in game_episodes {
        for sample in &episode.samples {
            sample_refs.push(sample);
        }
    }
    
    if sample_refs.is_empty() {
        return (0.0, 0.0, 0.0);
    }

    // 2. 随机打乱样本
    // 打破样本间的时间相关性，这对 SGD 的稳定性至关重要
    sample_refs.shuffle(&mut thread_rng());

    let mut total_loss_sum = 0.0;
    let mut policy_loss_sum = 0.0;
    let mut value_loss_sum = 0.0;
    let mut num_samples = 0;

    // 损失权重配置 (目前设为 1.0，可视需要调整)
    let policy_weight = 1.0;
    let value_weight = 1.0;

    // 调试统计变量
    let mut value_stats = Vec::new();
    let mut entropy_stats = Vec::new();

    // 3. 批量训练循环
    // 使用 no_grad 包裹非计算图操作（如数据拷贝），虽然这里主要是在循环外层，
    // 但注意 Tensor 的创建通常不需要梯度。
    for batch_start in (0..sample_refs.len()).step_by(batch_size) {
        let batch_end = (batch_start + batch_size).min(sample_refs.len());
        let batch = &sample_refs[batch_start..batch_end];
        let bsz = batch.len();
        if bsz == 0 {
            continue;
        }

        // 预分配缓冲区，减少内存碎片
        // 棋盘数据: [Batch, Channels, Height, Width]
        let mut board_buf = Vec::with_capacity(bsz * BOARD_CHANNELS * BOARD_ROWS * BOARD_COLS);
        // 标量特征: [Batch, Features]
        let mut scalar_buf = Vec::with_capacity(bsz * SCALAR_FEATURE_COUNT);
        // 目标策略概率: [Batch, ActionSize]
        let mut target_prob_buf = Vec::with_capacity(bsz * ACTION_SPACE_SIZE);
        // 目标价值: [Batch]
        let mut target_val_buf = Vec::with_capacity(bsz);
        // 动作掩码: [Batch, ActionSize]
        let mut mask_buf = Vec::with_capacity(bsz * ACTION_SPACE_SIZE);

        // 填充缓冲区
        for &(obs, target_probs, mcts_val, _game_result_val, masks) in batch.iter() {
            let board_slice = obs.board.as_slice().expect("board slice");
            board_buf.extend_from_slice(board_slice);
            
            let scalar_slice = obs.scalars.as_slice().expect("scalar slice");
            scalar_buf.extend_from_slice(scalar_slice);
            
            target_prob_buf.extend_from_slice(target_probs);
            
            // 这里使用 MCTS 搜索得到的根节点价值 (mcts_val) 作为训练目标，
            // 也可以尝试使用游戏最终结果 (_game_result_val) 或两者的混合。
            target_val_buf.push(*mcts_val);  
            
            mask_buf.extend(masks.iter().map(|&m| m as f32));

            // 收集调试统计信息
            if epoch == 0 { // 仅在第一个 epoch 收集，减少开销
                value_stats.push(*mcts_val);
                let entropy: f32 = target_probs
                    .iter()
                    .filter(|&&p| p > 1e-8)
                    .map(|&p| -p * p.ln())
                    .sum();
                entropy_stats.push(entropy);
            }
        }

        // 4. 构建 Tensor
        // 将 Rust Vec 转换为 PyTorch Tensor 并上传到计算设备 (GPU/CPU)
        let board_tensor = Tensor::from_slice(&board_buf)
            .view([bsz as i64, BOARD_CHANNELS as i64, BOARD_ROWS as i64, BOARD_COLS as i64])
            .to(device);
        let scalar_tensor = Tensor::from_slice(&scalar_buf)
            .view([bsz as i64, SCALAR_FEATURE_COUNT as i64])
            .to(device);
        let target_p = Tensor::from_slice(&target_prob_buf)
            .view([bsz as i64, ACTION_SPACE_SIZE as i64])
            .to(device);
        let target_v = Tensor::from_slice(&target_val_buf)
            .view([bsz as i64, 1])
            .to(device);
        let mask_tensor = Tensor::from_slice(&mask_buf)
            .view([bsz as i64, 352]) // 4x8棋盘: 352个动作
            .to(device);

        // 5. 前向传播
        let (logits, value) = net.forward(&board_tensor, &scalar_tensor);

        // 6. 计算损失
        // 应用掩码：将非法动作的 logits 设为负无穷 (-1e9)，使其 softmax 后概率为 0
        let masked_logits = &logits + (&mask_tensor - 1.0) * 1e9;
        let log_probs = masked_logits.log_softmax(-1, Kind::Float);

        // 策略损失: KL 散度 / 交叉熵
        // Loss = - sum(target_p * log_probs)
        let reduce_dim = [-1i64];
        let p_loss = (&target_p * &log_probs)
            .sum_dim_intlist(&reduce_dim[..], false, Kind::Float)
            .mean(Kind::Float) // 对 Batch 取平均
            .neg()
            * (policy_weight as f64);
        
        // 价值损失: 均方误差 (MSE)
        let v_loss = value.mse_loss(&target_v, tch::Reduction::Mean) * (value_weight as f64);

        let total_loss = &p_loss + &v_loss;
        
        // 7. 反向传播与优化
        // 在 backward 之前提取 loss 的标量值用于日志记录，
        // 这样做可以断开计算图引用，防止内存泄漏。
        let batch_loss_val = total_loss.double_value(&[]);
        let batch_p_loss_val = p_loss.double_value(&[]) / policy_weight as f64;
        let batch_v_loss_val = v_loss.double_value(&[]) / value_weight as f64;
        
        opt.backward_step(&total_loss);

        // 累加损失 (注意：backward_step 里的 mean 是对 batch 的平均，所以还原总和需乘以 bsz)
        total_loss_sum += batch_loss_val * bsz as f64;
        policy_loss_sum += batch_p_loss_val * bsz as f64;
        value_loss_sum += batch_v_loss_val * bsz as f64;
        num_samples += bsz;
        
        // 8. 资源释放 (重要)
        // 显式释放 Tensor，因为 tch-rs 的自动释放可能滞后，
        // 在密集循环中手动释放是防止 GPU 显存溢出的最佳实践。
        drop(board_tensor);
        drop(scalar_tensor);
        drop(target_p);
        drop(target_v);
        drop(mask_tensor);
        drop(logits);
        drop(value);
        drop(masked_logits);
        drop(log_probs);
        drop(p_loss);
        drop(v_loss);
        drop(total_loss);
    }

    // 9. 调试输出 (仅在 Epoch 0)
    // 输出样本的价值分布和策略熵，帮助判断数据质量。
    if epoch == 0 && !value_stats.is_empty() {
        let avg_value: f32 = value_stats.iter().sum::<f32>() / value_stats.len() as f32;
        let std_value: f32 = (value_stats
            .iter()
            .map(|v| (v - avg_value).powi(2))
            .sum::<f32>()
            / value_stats.len() as f32)
            .sqrt();
        let avg_entropy: f32 = entropy_stats.iter().sum::<f32>() / entropy_stats.len() as f32;

        let positive_values = value_stats.iter().filter(|&&v| v > 0.0).count();
        let negative_values = value_stats.iter().filter(|&&v| v < 0.0).count();
        let zero_values = value_stats.iter().filter(|&&v| v == 0.0).count();

        println!(
            "    🐛 样本统计: 总数={}, 价值[avg={:.3}, std={:.3}], 熵[avg={:.3}]",
            value_stats.len(),
            avg_value,
            std_value,
            avg_entropy
        );
        println!(
            "    🐛 价值分布: 正={} ({:.1}%), 零={} ({:.1}%), 负={} ({:.1}%)",
            positive_values,
            positive_values as f32 / value_stats.len() as f32 * 100.0,
            zero_values,
            zero_values as f32 / value_stats.len() as f32 * 100.0,
            negative_values,
            negative_values as f32 / value_stats.len() as f32 * 100.0
        );
    }

    // 返回平均损失
    if num_samples > 0 {
        (
            total_loss_sum / num_samples as f64,
            policy_loss_sum / num_samples as f64,
            value_loss_sum / num_samples as f64,
        )
    } else {
        (0.0, 0.0, 0.0)
    }
}

/// 获取当前 epoch 的损失权重
///
/// 可以在此实现动态权重调整策略，例如随着训练进行逐渐增加策略损失的权重。
/// 目前返回固定值 (1.0, 1.0)。
///
/// # 参数
/// - `_epoch`: 当前 epoch 索引 (未使用)
///
/// # 返回
/// (策略损失权重, 价值损失权重)
pub fn get_loss_weights(_epoch: usize) -> (f64, f64) {
    let policy_weight = 1.0;
    let value_weight = 1.0;
    (policy_weight as f64, value_weight as f64)
}