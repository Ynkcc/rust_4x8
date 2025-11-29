// run_lr_finder.rs - 学习率扫描器执行程序
//
// 使用方法:
//   cargo run --bin banqi-lr-finder [model_path]
//
// 功能:
// 1. 从数据库加载训练样本
// 2. 加载现有模型（可选）
// 3. 执行学习率扫描
// 4. 生成 lr_finder_results.csv 文件
// 5. 提供学习率选择建议

use banqi_3x4::nn_model::BanqiNet;
use banqi_3x4::lr_finder::{find_learning_rate, LRFinderConfig};
use banqi_3x4::database::load_samples_from_db;
use anyhow::Result;
use tch::{nn, Device};
use std::env;

fn main() -> Result<()> {
    println!("========================================");
    println!("   暗棋 3x4 学习率扫描器");
    println!("========================================\n");
    
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
    
    // 加载训练样本
    println!("\n正在从数据库加载训练样本...");
    let samples = match load_samples_from_db("training_samples.db", None) {
        Ok(samples) => samples,
        Err(e) => {
            eprintln!("❌ 加载样本失败: {}", e);
            eprintln!("提示: 请先运行训练程序生成训练样本");
            return Err(e);
        }
    };
    
    if samples.is_empty() {
        eprintln!("❌ 数据库中没有训练样本");
        eprintln!("提示: 请先运行训练程序生成训练样本");
        anyhow::bail!("样本集为空");
    }
    
    println!("✓ 成功加载 {} 个训练样本", samples.len());
    
    // 限制样本数量以加快扫描速度（使用最近的样本）
    let max_samples = 10000;
    let samples_to_use = if samples.len() > max_samples {
        println!("  (使用最新的 {} 个样本以加快扫描)", max_samples);
        samples[samples.len() - max_samples..].to_vec()
    } else {
        samples
    };
    
    // 创建模型
    let mut vs = nn::VarStore::new(device);
    let net = BanqiNet::new(&vs.root());
    
    // 加载现有模型（如果提供）
    let args: Vec<String> = env::args().collect();
    if args.len() > 1 {
        let model_path = &args[1];
        println!("\n正在加载模型: {}", model_path);
        match vs.load(model_path) {
            Ok(_) => println!("✓ 模型加载成功"),
            Err(e) => {
                eprintln!("⚠️ 加载模型失败: {}", e);
                println!("  继续使用随机初始化的模型进行扫描");
            }
        }
    } else {
        println!("\n未指定模型路径，使用随机初始化的模型");
        println!("提示: 可以使用 'cargo run --bin banqi-lr-finder <model.ot>' 加载现有模型");
    }
    
    // 配置学习率扫描器
    let config = LRFinderConfig {
        start_lr: 1e-7,        // 起始学习率
        end_lr: 1.0,           // 结束学习率
        num_steps: 100,        // 扫描步数
        num_batches_per_step: 2, // 每步训练批次数
        batch_size: 64,        // 批量大小
        smooth_window: 5,      // 平滑窗口
        divergence_threshold: 4.0, // 发散阈值
    };
    
    println!("\n准备开始学习率扫描...");
    println!("此过程可能需要几分钟，请耐心等待。\n");
    
    // 执行学习率扫描
    let results = find_learning_rate(&net, &samples_to_use, device, &config)?;
    
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
    println!("   plt.title('Learning Rate Finder')");
    println!("   plt.grid(True)");
    println!("   plt.show()");
    println!("   ```");
    println!("3. 根据曲线和建议选择合适的学习率");
    println!("4. 在训练代码中应用新的学习率");
    
    Ok(())
}
