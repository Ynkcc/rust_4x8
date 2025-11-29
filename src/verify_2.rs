// code_files/src/verify_scenario_2.rs

use banqi_3x4::game_env::DarkChessEnv;
use banqi_3x4::mcts::{Evaluator, MCTS, MCTSConfig, MctsNode};
use std::sync::Arc;

// --- 简单策略评估器 ---
struct SimpleEvaluator;

impl SimpleEvaluator {
    fn new() -> Self {
        Self
    }
}

impl Evaluator for SimpleEvaluator {
    fn evaluate(&self, env: &DarkChessEnv) -> (Vec<f32>, f32) {
        let action_masks = env.action_masks();
        
        // 计算有效动作数量
        let valid_actions: Vec<usize> = action_masks
            .iter()
            .enumerate()
            .filter(|(_, &is_valid)| is_valid != 0)
            .map(|(i, _)| i)
            .collect();
        
        let num_valid_actions = valid_actions.len() as f32;
        
        // 为所有动作分配概率
        let mut probs = vec![0.0; action_masks.len()];
        if num_valid_actions > 0.0 {
            let uniform_prob = 1.0 / num_valid_actions;
            for &action_idx in &valid_actions {
                probs[action_idx] = uniform_prob;
            }
        }
        
        // 固定价值估计为 0.1
        let value = 0.1;
        
        (probs, value)
    }
}

// --- 提取 PUCT 最佳路径 ---
fn best_puct_path(root: &MctsNode, cpuct: f32, depth_limit: usize) -> Vec<(usize, f32, f32, f32, u32)> {
    let mut path = Vec::new();
    let mut current = root;
    let mut depth = 0;
    while depth < depth_limit && !current.children.is_empty() {
        let sqrt_total = (current.visit_count as f32).sqrt();
        let mut best: Option<(usize, f32, f32, f32, u32)> = None;
        for (&action, child) in &current.children {
            let q = child.q_value();
            let prior = child.prior;
            let puct = q + cpuct * prior * sqrt_total / (1.0 + child.visit_count as f32);
            let vc = child.visit_count;
            if let Some((_, _, _, _bpuct, _)) = best {
                if puct > _bpuct { best = Some((action, q, prior, puct, vc)); }
            } else {
                best = Some((action, q, prior, puct, vc));
            }
        }
        if let Some(record) = best {
            let action = record.0;
            path.push(record);
            current = current.children.get(&action).unwrap();
        } else {
            break;
        }
        depth += 1;
    }
    path
}

fn main() {
    // 1. 初始化简单策略评估器
    let evaluator = Arc::new(SimpleEvaluator::new());

    // 2. 初始化环境并设置特定场景
    let mut env = DarkChessEnv::new();
    // 调用我们在 game_env.rs 中新添加的方法
    env.setup_hidden_threats(); 

    println!("===== 自定义场景: 隐藏的威胁 =====");
    println!();
    env.print_board();
    println!("Hidden Pieces Pool: {:?}", env.hidden_pieces);
    println!("Reveal Probabilities: {:?}", env.get_reveal_probabilities());

    // 3. 运行 MCTS (10000 次模拟) 
    let cpuct_val = 1.0;
    let num_simulations = 10000;
    let config = MCTSConfig { cpuct: cpuct_val, num_simulations };
    let mut mcts = MCTS::new(&env, evaluator.clone(), config);
    
    println!("\n===== 开始 MCTS 搜索 (共 {} 次模拟) =====", num_simulations);
    
    // 运行 MCTS
    mcts.run();
    
    println!("===== MCTS 搜索完成 =====");
    println!("最终根节点访问次数: {}", mcts.root.visit_count);

    // 4. 输出最佳 PUCT 路径
    let path = best_puct_path(&mcts.root, cpuct_val, 20);
    println!("\nPUCT 最高路径 (action, coords, Q, prior, PUCT, visits):");
    for (step_idx, (action, q, prior, puct, visits)) in path.iter().enumerate() {
        let coord_str = if let Some(coords) = env.get_coords_for_action(*action) {
            if coords.len() == 1 { format!("reveal@{}", coords[0]) } else { format!("move {}->{}", coords[0], coords[1]) }
        } else { "?".to_string() };
        println!(" Step {:02}: action {:02} ({}) | Q={:.4} prior={:.4} PUCT={:.4} visits={}" , step_idx, action, coord_str, q, prior, puct, visits);
    }
    
    // 最终叶节点价值
    if let Some((last_action, _, _, _, _)) = path.last() {
        if let Some(leaf) = mcts.root.children.get(last_action) {
            println!("\n最终叶节点平均价值 Q: {:.4}", leaf.q_value());
        }
    }

    // 根节点策略概率
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