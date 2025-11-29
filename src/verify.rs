// Verification binary: Construct a deterministic scenario with only two Advisors (R_A, B_A)
// Run MCTS with an untrained neural network evaluator for 100 simulations and
// output the highest PUCT path, final leaf value, and visit counts per step.

use banqi_3x4::game_env::{DarkChessEnv, Player};
use banqi_3x4::mcts::{Evaluator, MCTS, MCTSConfig, MctsNode};
use banqi_3x4::nn_model::BanqiNet;
use std::sync::Arc;
use tch::{nn, Device, Tensor, Kind};

// Simple NN evaluator identical structure to training evaluator but untrained.
struct NNEvaluator {
    net: BanqiNet,
    device: Device,
}

impl NNEvaluator {
    fn new(vs: &nn::Path, device: Device) -> Self {
        Self { net: BanqiNet::new(vs), device }
    }
}

impl Evaluator for NNEvaluator {
    fn evaluate(&self, env: &DarkChessEnv) -> (Vec<f32>, f32) {
        let obs = env.get_state();
        let board_tensor = Tensor::from_slice(obs.board.as_slice().unwrap())
            .view([1, 8, 3, 4]) // (stack=1, channels=8)
            .to(self.device);
        let scalar_tensor = Tensor::from_slice(obs.scalars.as_slice().unwrap())
            .view([1, 56]) // (stack=1, features=56)
            .to(self.device);
        let (logits, value) = self.net.forward_inference(&board_tensor, &scalar_tensor);

        // Apply action mask to logits
        let masks: Vec<f32> = env.action_masks().iter().map(|&m| m as f32).collect();
        let mask_tensor = Tensor::from_slice(&masks).to(self.device).view([1, 46]);
        let masked_logits = logits + (mask_tensor - 1.0) * 1e9; // invalid -> -1e9
        let probs = masked_logits.softmax(-1, Kind::Float);

        let probs_flat = probs.view([-1]);
        let probs_vec: Vec<f32> = (0..probs_flat.size()[0])
            .map(|i| probs_flat.double_value(&[i]) as f32)
            .collect();
        let value_scalar = value.squeeze().double_value(&[]) as f32;
        (probs_vec, value_scalar)
    }
}

// Compute the best PUCT path starting from root.
// Returns vector of (action, q, prior, puct, visit_count)
fn best_puct_path(root: &MctsNode, cpuct: f32, depth_limit: usize) -> Vec<(usize, f32, f32, f32, u32)> {
    let mut path = Vec::new();
    let mut current = root;
    let mut depth = 0;
    while depth < depth_limit && !current.children.is_empty() {
        let sqrt_total = (current.visit_count as f32).sqrt();
        let mut best: Option<(usize, f32, f32, f32, u32)> = None; // (action,q,prior,puct,visits)
        for (&action, child) in &current.children {
            let q = child.q_value();
            let prior = child.prior;
            let puct = q + cpuct * prior * sqrt_total / (1.0 + child.visit_count as f32);
            let vc = child.visit_count;
            if let Some((_ba,_bq,_bp,_bpuct,_bvc)) = best {
                if puct > _bpuct { best = Some((action, q, prior, puct, vc)); }
            } else {
                best = Some((action, q, prior, puct, vc));
            }
        }
        if let Some(record) = best {
            let action = record.0;
            path.push(record);
            // move deeper
            current = current.children.get(&action).unwrap();
        } else {
            break;
        }
        depth += 1;
    }
    path
}

fn main() {
    // Device selection (CPU is fine for verification)
    let device = if tch::Cuda::is_available() { Device::Cuda(0) } else { Device::Cpu };

    // Build untrained network
    let vs = nn::VarStore::new(device);
    let evaluator = Arc::new(NNEvaluator::new(&vs.root(), device));
    // (No optimizer or training; weights are random initialization.)

    // Construct environment and custom scenario
    let mut env = DarkChessEnv::new();
    env.setup_two_advisors(Player::Black); // 当前轮到黑方

    println!("===== 自定义场景: 仅剩 R_A 与 B_A =====");
    env.print_board();

    // Run MCTS with detailed logging
    let cpuct_val = 1.0; // 保存 cpuct 方便后续计算路径
    let num_sims = 10000;
    let config = MCTSConfig { cpuct: cpuct_val, num_simulations: num_sims }; // 10000 次模拟
    let mut mcts = MCTS::new(&env, evaluator.clone(), config);
    
    println!("\n===== 开始 MCTS 搜索 (配置 {} 次评估) =====", num_sims);
    mcts.run();

    println!("\n===== MCTS 搜索完成 =====");
    println!("配置的 num_simulations (评估预算): {}", num_sims);
    println!("根节点访问次数 (总模拟次数): {}", mcts.root.visit_count);


    // Compute best PUCT path
    let path = best_puct_path(&mcts.root, cpuct_val, 20); // depth limit 20

    println!("\nPUCT 最高路径 (action, coords, Q, prior, PUCT, visits):");
    for (step_idx, (action, q, prior, puct, visits)) in path.iter().enumerate() {
        let coord_str = if let Some(coords) = env.get_coords_for_action(*action) {
            if coords.len() == 1 { format!("reveal@{}", coords[0]) } else { format!("move {}->{}", coords[0], coords[1]) }
        } else { "?".to_string() };
        println!(" Step {:02}: action {:02} ({}) | Q={:.4} prior={:.4} PUCT={:.4} visits={}" , step_idx, action, coord_str, q, prior, puct, visits);
    }

    // Final leaf value (if any)
    if let Some((last_action, _q, _prior, _puct, _visits)) = path.last() {
        if let Some(leaf) = mcts.root.children.get(last_action) {
            println!("\n最终叶节点平均价值 Q: {:.4}", leaf.q_value());
        }
    }

    // Also dump root policy probabilities for reference
    let probs = mcts.get_root_probabilities();
    println!("\n根节点策略概率 (前 20 个非零项):");
    for (i, p) in probs.iter().enumerate().filter(|(_,p)| **p > 0.0).take(20) {
        let coord_str = if let Some(coords) = env.get_coords_for_action(i) {
            if coords.len() == 1 { format!("reveal@{}", coords[0]) } else { format!("{}->{}", coords[0], coords[1]) }
        } else { "?".to_string() };
        println!("  action {:02} ({:<10}) prob={:.4}", i, coord_str, p);
    }

    println!("\n===== 验证结束 =====");
}
