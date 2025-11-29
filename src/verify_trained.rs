// 验证训练后的模型在特定场景下的先验概率输出
use banqi_3x4::game_env::{DarkChessEnv, Player};
use banqi_3x4::nn_model::BanqiNet;
use anyhow::Result;
use tch::{nn, Device, Tensor, Kind};

struct NNEvaluator {
    net: BanqiNet,
    device: Device,
}

impl NNEvaluator {
    fn new(vs: &nn::Path, device: Device) -> Self {
        Self { net: BanqiNet::new(vs), device }
    }
    
    fn evaluate(&self, env: &DarkChessEnv) -> (Vec<f32>, f32) {
        let obs = env.get_state();
        let board_tensor = Tensor::from_slice(obs.board.as_slice().unwrap())
            .view([1, 8, 3, 4])
            .to(self.device);
        let scalar_tensor = Tensor::from_slice(obs.scalars.as_slice().unwrap())
            .view([1, 56])
            .to(self.device);
        let (logits, value) = self.net.forward_inference(&board_tensor, &scalar_tensor);

        // Apply action mask
        let masks: Vec<f32> = env.action_masks().iter().map(|&m| m as f32).collect();
        let mask_tensor = Tensor::from_slice(&masks).to(self.device).view([1, 46]);
        let masked_logits = logits + (mask_tensor - 1.0) * 1e9;
        let probs = masked_logits.softmax(-1, Kind::Float);

        let probs_flat = probs.view([-1]);
        let probs_vec: Vec<f32> = (0..probs_flat.size()[0])
            .map(|i| probs_flat.double_value(&[i]) as f32)
            .collect();
        let value_scalar = value.squeeze().double_value(&[]) as f32;
        (probs_vec, value_scalar)
    }
}

fn print_top_priors(env: &DarkChessEnv, probs: &[f32], top_k: usize) {
    let mut indexed_probs: Vec<(usize, f32)> = probs.iter()
        .enumerate()
        .filter(|(_, &p)| p > 0.0)
        .map(|(i, &p)| (i, p))
        .collect();
    
    indexed_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    
    println!("\n先验概率 Top {} 动作:", top_k);
    for (i, (action, prob)) in indexed_probs.iter().take(top_k).enumerate() {
        let coord_str = if let Some(coords) = env.get_coords_for_action(*action) {
            if coords.len() == 1 {
                format!("reveal@{}", coords[0])
            } else {
                format!("move {}->{}", coords[0], coords[1])
            }
        } else {
            "?".to_string()
        };
        println!("  #{}: action {:02} ({:<15}) prob={:.6}", i+1, action, coord_str, prob);
    }
}

fn verify_scenario_1(evaluator: &NNEvaluator) {
    println!("\n{}", "=".repeat(60));
    println!("场景 1: 仅剩 R_A 与 B_A (verify.rs 场景)");
    println!("{}", "=".repeat(60));
    
    let mut env = DarkChessEnv::new();
    env.setup_two_advisors(Player::Black);
    env.print_board();
    
    let (probs, value) = evaluator.evaluate(&env);
    println!("\n模型输出价值估计: {:.4}", value);
    print_top_priors(&env, &probs, 20);
}

fn verify_scenario_2(evaluator: &NNEvaluator) {
    println!("\n{}", "=".repeat(60));
    println!("场景 2: 隐藏的威胁 (verify_2.rs 场景)");
    println!("{}", "=".repeat(60));
    
    let mut env = DarkChessEnv::new();
    env.setup_hidden_threats();
    env.print_board();
    println!("Hidden Pieces Pool: {:?}", env.hidden_pieces);
    println!("Reveal Probabilities: {:?}", env.get_reveal_probabilities());
    
    let (probs, value) = evaluator.evaluate(&env);
    println!("\n模型输出价值估计: {:.4}", value);
    print_top_priors(&env, &probs, 20);
}

fn main() -> Result<()> {
    // 检查命令行参数
    let args: Vec<String> = std::env::args().collect();
    let model_path = if args.len() > 1 {
        &args[1]
    } else {
        "banqi_model_latest.ot"
    };
    
    println!("\n加载模型: {}", model_path);
    
    // 设备选择
    let device = if tch::Cuda::is_available() {
        println!("使用 CUDA 设备");
        Device::Cuda(0)
    } else {
        println!("使用 CPU");
        Device::Cpu
    };
    
    // 加载模型
    let mut vs = nn::VarStore::new(device);
    let evaluator = NNEvaluator::new(&vs.root(), device);
    
    match vs.load(model_path) {
        Ok(_) => println!("模型加载成功！\n"),
        Err(e) => {
            eprintln!("错误: 无法加载模型 '{}': {}", model_path, e);
            eprintln!("请先运行训练: cargo run --bin banqi-train");
            return Err(e.into());
        }
    }
    
    // 验证两个场景
    verify_scenario_1(&evaluator);
    verify_scenario_2(&evaluator);
    
    println!("\n{}", "=".repeat(60));
    println!("验证完成！");
    println!("{}\n", "=".repeat(60));
    
    Ok(())
}
