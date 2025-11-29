// training_log.rs - 训练日志记录模块
//
// 提供CSV格式的训练日志记录功能，用于跟踪训练进度和指标

use anyhow::Result;
use std::fs::OpenOptions;
use std::io::Write;

// ================ CSV日志记录 ================

/// 训练日志记录结构
#[derive(Debug, Clone)]
pub struct TrainingLog {
    pub iteration: usize,
    // 损失指标（epoch平均）
    pub avg_total_loss: f64,
    pub avg_policy_loss: f64,
    pub avg_value_loss: f64,
    pub policy_loss_weight: f64,
    pub value_loss_weight: f64,
    
    // 场景1: R_A vs B_A
    pub scenario1_value: f32,
    pub scenario1_unmasked_a38: f32,
    pub scenario1_unmasked_a39: f32,
    pub scenario1_unmasked_a40: f32,
    pub scenario1_masked_a38: f32,
    pub scenario1_masked_a39: f32,
    pub scenario1_masked_a40: f32,
    
    // 场景2: Hidden Threat
    pub scenario2_value: f32,
    pub scenario2_unmasked_a3: f32,
    pub scenario2_unmasked_a5: f32,
    pub scenario2_masked_a3: f32,
    pub scenario2_masked_a5: f32,
    
    // 样本统计
    pub new_samples_count: usize,
    pub replay_buffer_size: usize,
    pub avg_game_steps: f32,
    pub red_win_ratio: f32,
    pub draw_ratio: f32,
    pub black_win_ratio: f32,
    pub avg_policy_entropy: f32,
    pub high_confidence_ratio: f32,
}

impl TrainingLog {
    pub fn write_header(csv_path: &str) -> Result<()> {
        let mut file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(false)
            .open(csv_path)?;
        
        // 检查文件是否为空（新文件需要写入表头）
        let metadata = std::fs::metadata(csv_path)?;
        if metadata.len() == 0 {
            writeln!(file, "iteration,avg_total_loss,avg_policy_loss,avg_value_loss,policy_loss_weight,value_loss_weight,\
                scenario1_value,scenario1_unmasked_a38,scenario1_unmasked_a39,scenario1_unmasked_a40,\
                scenario1_masked_a38,scenario1_masked_a39,scenario1_masked_a40,\
                scenario2_value,scenario2_unmasked_a3,scenario2_unmasked_a5,scenario2_masked_a3,scenario2_masked_a5,\
                new_samples_count,replay_buffer_size,avg_game_steps,red_win_ratio,draw_ratio,black_win_ratio,\
                avg_policy_entropy,high_confidence_ratio")?;
        }
        
        Ok(())
    }
    
    pub fn append_to_csv(&self, csv_path: &str) -> Result<()> {
        let mut file = OpenOptions::new()
            .write(true)
            .create(true)
            .append(true)
            .open(csv_path)?;
        
        writeln!(file, "{},{:.6},{:.6},{:.6},{:.3},{:.3},\
            {:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},\
            {:.4},{:.4},{:.4},{:.4},{:.4},\
            {},{},{:.2},{:.4},{:.4},{:.4},{:.4},{:.4}",
            self.iteration,
            self.avg_total_loss, self.avg_policy_loss, self.avg_value_loss,
            self.policy_loss_weight, self.value_loss_weight,
            self.scenario1_value, self.scenario1_unmasked_a38, self.scenario1_unmasked_a39, self.scenario1_unmasked_a40,
            self.scenario1_masked_a38, self.scenario1_masked_a39, self.scenario1_masked_a40,
            self.scenario2_value, self.scenario2_unmasked_a3, self.scenario2_unmasked_a5,
            self.scenario2_masked_a3, self.scenario2_masked_a5,
            self.new_samples_count, self.replay_buffer_size, self.avg_game_steps,
            self.red_win_ratio, self.draw_ratio, self.black_win_ratio,
            self.avg_policy_entropy, self.high_confidence_ratio
        )?;
        
        Ok(())
    }
}