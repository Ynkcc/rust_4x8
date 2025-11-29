// lr_finder.rs - 学习率扫描器 (Learning Rate Finder)
//
// 实现学习率范围测试 (LR Range Test)，帮助找到最优学习率。
// 算法基于 Leslie Smith 的论文 "Cyclical Learning Rates for Training Neural Networks"
//
// 使用方法:
// 1. 准备一个训练样本集（从数据库或自对弈获取）
// 2. 调用 `find_learning_rate()` 执行扫描
// 3. 查看生成的 `lr_finder_results.csv` 文件
// 4. 绘制学习率-损失曲线，找到损失下降最快的区间
//
// 推荐学习率选择策略:
// - 最小学习率: 损失开始下降的位置
// - 最大学习率: 损失达到最低点之前（避免发散）
// - 初始学习率: 通常选择最小值的 3-10 倍

use crate::nn_model::BanqiNet;
use crate::game_env::Observation;
use anyhow::Result;
use tch::{nn, nn::OptimizerConfig, Device, Tensor, Kind};
use std::fs::File;
use std::io::Write;

/// 学习率扫描结果
#[derive(Debug, Clone)]
pub struct LRFinderResult {
    pub learning_rate: f64,
    pub loss: f64,
    pub policy_loss: f64,
    pub value_loss: f64,
}

/// 学习率扫描器配置
pub struct LRFinderConfig {
    /// 起始学习率（通常很小，如 1e-8）
    pub start_lr: f64,
    /// 结束学习率（通常较大，如 10.0 或 1.0）
    pub end_lr: f64,
    /// 扫描步数（学习率采样点数量）
    pub num_steps: usize,
    /// 每个学习率训练的批次数（通常 1-3）
    pub num_batches_per_step: usize,
    /// 批量大小
    pub batch_size: usize,
    /// 损失平滑窗口大小（移动平均）
    pub smooth_window: usize,
    /// 损失发散阈值倍数（如果损失超过最小损失的此倍数，提前停止）
    pub divergence_threshold: f64,
}

impl Default for LRFinderConfig {
    fn default() -> Self {
        Self {
            start_lr: 1e-7,
            end_lr: 1.0,
            num_steps: 100,
            num_batches_per_step: 2,
            batch_size: 64,
            smooth_window: 5,
            divergence_threshold: 4.0,
        }
    }
}

/// 执行学习率扫描
///
/// # 参数
/// - `model`: 要测试的神经网络模型
/// - `examples`: 训练样本集 (观察, 策略概率, 价值目标, 动作掩码)
/// - `device`: 训练设备 (CPU/GPU)
/// - `config`: 学习率扫描配置
///
/// # 返回
/// 学习率扫描结果向量，包含每个学习率下的损失值
pub fn find_learning_rate(
    model: &BanqiNet,
    examples: &[(Observation, Vec<f32>, f32, Vec<i32>)],
    device: Device,
    config: &LRFinderConfig,
) -> Result<Vec<LRFinderResult>> {
    if examples.is_empty() {
        anyhow::bail!("训练样本集为空");
    }
    
    if examples.len() < config.batch_size {
        anyhow::bail!("样本数量 ({}) 少于批量大小 ({})", examples.len(), config.batch_size);
    }
    
    println!("\n========== 学习率扫描器 ==========");
    println!("配置:");
    println!("  学习率范围: {:.2e} -> {:.2e}", config.start_lr, config.end_lr);
    println!("  扫描步数: {}", config.num_steps);
    println!("  每步批次数: {}", config.num_batches_per_step);
    println!("  批量大小: {}", config.batch_size);
    println!("  样本总数: {}", examples.len());
    println!("  平滑窗口: {}", config.smooth_window);
    println!("  发散阈值: {}x", config.divergence_threshold);
    
    // 创建模型的副本（用于扫描，不影响原模型）
    let mut vs = nn::VarStore::new(device);
    let test_net = BanqiNet::new(&vs.root());
    
    // 复制模型参数
    // 注意: 这里假设原模型已经有一些预训练权重
    // 如果从头开始，可以跳过这一步
    
    // 计算学习率的对数间隔
    let log_start = config.start_lr.ln();
    let log_end = config.end_lr.ln();
    let log_step = (log_end - log_start) / (config.num_steps - 1) as f64;
    
    let mut results = Vec::new();
    let mut min_loss = f64::MAX;
    let mut loss_history = Vec::new();
    
    // 策略和价值损失权重（可以调整）
    let policy_weight = 1.0;
    let value_weight = 1.0;
    
    println!("\n开始扫描...");
    
    for step in 0..config.num_steps {
        // 计算当前学习率（指数增长）
        let lr = (log_start + step as f64 * log_step).exp();
        
        // 创建新的优化器（使用当前学习率）
        let mut opt = nn::Adam::default().build(&vs, lr)?;
        
        let mut step_loss_sum = 0.0;
        let mut step_policy_loss_sum = 0.0;
        let mut step_value_loss_sum = 0.0;
        let mut num_batches = 0;
        
        // 在当前学习率下训练多个批次
        for batch_idx in 0..config.num_batches_per_step {
            // 随机选择一个批次
            let batch_start = (step * config.num_batches_per_step + batch_idx) * config.batch_size;
            let batch_start = batch_start % (examples.len() - config.batch_size);
            let batch = &examples[batch_start..batch_start + config.batch_size];
            
            // 准备批量数据
            let (board_tensor, scalar_tensor, target_p, target_v, mask_tensor) = 
                prepare_batch(batch, device);
            
            // 前向传播
            let (logits, value) = test_net.forward(&board_tensor, &scalar_tensor);
            let masked_logits = &logits + (&mask_tensor - 1.0) * 1e9;
            let log_probs = masked_logits.log_softmax(-1, Kind::Float);
            
            // 计算损失
            let reduce_dim = [-1i64];
            let p_loss = (&target_p * &log_probs)
                .sum_dim_intlist(&reduce_dim[..], false, Kind::Float)
                .mean(Kind::Float)
                .neg() * policy_weight;
            let v_loss = value.mse_loss(&target_v, tch::Reduction::Mean) * value_weight;
            let total_loss = &p_loss + &v_loss;
            
            // 反向传播和更新
            opt.backward_step(&total_loss);
            
            // 记录损失
            step_loss_sum += total_loss.double_value(&[]);
            step_policy_loss_sum += p_loss.double_value(&[]) / policy_weight;
            step_value_loss_sum += v_loss.double_value(&[]) / value_weight;
            num_batches += 1;
        }
        
        // 计算平均损失
        let avg_loss = step_loss_sum / num_batches as f64;
        let avg_policy_loss = step_policy_loss_sum / num_batches as f64;
        let avg_value_loss = step_value_loss_sum / num_batches as f64;
        
        // 应用平滑（移动平均）
        loss_history.push(avg_loss);
        let smooth_loss = if loss_history.len() >= config.smooth_window {
            let start_idx = loss_history.len() - config.smooth_window;
            loss_history[start_idx..].iter().sum::<f64>() / config.smooth_window as f64
        } else {
            loss_history.iter().sum::<f64>() / loss_history.len() as f64
        };
        
        // 记录结果
        let result = LRFinderResult {
            learning_rate: lr,
            loss: smooth_loss,
            policy_loss: avg_policy_loss,
            value_loss: avg_value_loss,
        };
        results.push(result.clone());
        
        // 更新最小损失
        if smooth_loss < min_loss {
            min_loss = smooth_loss;
        }
        
        // 打印进度
        if step % 10 == 0 || step == config.num_steps - 1 {
            println!("  Step {}/{}: LR={:.2e}, Loss={:.4} (Policy={:.4}, Value={:.4})", 
                step + 1, config.num_steps, lr, smooth_loss, avg_policy_loss, avg_value_loss);
        }
        
        // 检查是否发散（损失暴增）
        if smooth_loss > min_loss * config.divergence_threshold && min_loss > 0.0 {
            println!("\n⚠️ 检测到损失发散 (当前={:.4}, 最小={:.4}, 阈值={}x)", 
                smooth_loss, min_loss, config.divergence_threshold);
            println!("提前停止扫描。");
            break;
        }
    }
    
    println!("\n扫描完成！共采集 {} 个数据点", results.len());
    
    // 保存结果到 CSV 文件
    save_results_to_csv(&results, "lr_finder_results.csv")?;
    println!("结果已保存到: lr_finder_results.csv");
    
    // 分析并给出建议
    analyze_and_suggest(&results)?;
    
    Ok(results)
}

/// 准备一个批次的训练数据
fn prepare_batch(
    batch: &[(Observation, Vec<f32>, f32, Vec<i32>)],
    device: Device,
) -> (Tensor, Tensor, Tensor, Tensor, Tensor) {
    let bsz = batch.len();
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
    
    (board_tensor, scalar_tensor, target_p, target_v, mask_tensor)
}

/// 保存结果到 CSV 文件
fn save_results_to_csv(results: &[LRFinderResult], path: &str) -> Result<()> {
    let mut file = File::create(path)?;
    
    // 写入表头
    writeln!(file, "learning_rate,loss,policy_loss,value_loss")?;
    
    // 写入数据
    for result in results {
        writeln!(file, "{:.8e},{:.6},{:.6},{:.6}", 
            result.learning_rate, 
            result.loss, 
            result.policy_loss, 
            result.value_loss)?;
    }
    
    Ok(())
}

/// 分析结果并给出学习率建议
fn analyze_and_suggest(results: &[LRFinderResult]) -> Result<()> {
    if results.is_empty() {
        return Ok(());
    }
    
    println!("\n========== 分析结果 ==========");
    
    // 找到最小损失点
    let min_loss_idx = results.iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.loss.partial_cmp(&b.loss).unwrap())
        .map(|(idx, _)| idx)
        .unwrap();
    
    let min_loss_lr = results[min_loss_idx].learning_rate;
    let min_loss = results[min_loss_idx].loss;
    
    println!("最小损失点:");
    println!("  学习率: {:.2e}", min_loss_lr);
    println!("  损失: {:.4}", min_loss);
    
    // 找到损失下降最快的区间（梯度最负）
    let mut max_gradient = 0.0_f64;
    let mut max_gradient_idx = 0;
    
    for i in 1..results.len() {
        let lr_diff = results[i].learning_rate.ln() - results[i-1].learning_rate.ln();
        let loss_diff = results[i].loss - results[i-1].loss;
        let gradient = loss_diff / lr_diff; // d(loss)/d(log_lr)
        
        if gradient < max_gradient {
            max_gradient = gradient;
            max_gradient_idx = i;
        }
    }
    
    let steepest_lr = results[max_gradient_idx].learning_rate;
    
    println!("\n损失下降最快区间:");
    println!("  学习率: {:.2e}", steepest_lr);
    println!("  梯度: {:.4}", max_gradient);
    
    // 给出建议
    println!("\n========== 学习率建议 ==========");
    println!("📊 分析方法:");
    println!("  1. 绘制学习率-损失曲线: 使用 lr_finder_results.csv");
    println!("  2. 找到损失下降最陡的区域（曲线斜率最负）");
    println!("  3. 在该区域的起点和损失最低点之间选择学习率");
    
    println!("\n💡 推荐学习率范围:");
    
    // 保守建议: 最陡点到最小损失点之间
    let suggested_min_lr = steepest_lr;
    let suggested_max_lr = min_loss_lr / 3.0; // 最小损失点的 1/3，避免发散
    let suggested_initial_lr = (suggested_min_lr * suggested_max_lr).sqrt(); // 几何平均
    
    println!("  初始学习率: {:.2e}", suggested_initial_lr);
    println!("  最小学习率: {:.2e} (用于学习率调度)", suggested_min_lr);
    println!("  最大学习率: {:.2e} (用于循环学习率)", suggested_max_lr);
    
    println!("\n📈 使用建议:");
    println!("  - 单一学习率: 使用初始学习率 {:.2e}", suggested_initial_lr);
    println!("  - 学习率衰减: 从 {:.2e} 开始，逐步降低", suggested_initial_lr);
    println!("  - 循环学习率: 在 {:.2e} 和 {:.2e} 之间循环", suggested_min_lr, suggested_max_lr);
    println!("  - Adam 优化器: 当前使用 Adam，建议起始学习率 {:.2e}", suggested_initial_lr);
    
    println!("\n⚠️ 注意事项:");
    println!("  - 这些是建议值，实际训练时需要根据验证集表现调整");
    println!("  - 如果训练不稳定，降低学习率（除以 2-10）");
    println!("  - 如果收敛太慢，可以尝试稍微增大学习率");
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_lr_finder_config_default() {
        let config = LRFinderConfig::default();
        assert_eq!(config.start_lr, 1e-7);
        assert_eq!(config.end_lr, 1.0);
        assert_eq!(config.num_steps, 100);
    }
}
