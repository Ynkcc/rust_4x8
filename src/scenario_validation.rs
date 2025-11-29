// scenario_validation.rs - 场景验证模块
//
// 提供模型在标准场景上的验证功能，用于监控训练进度

use crate::game_env::{DarkChessEnv, Player, ACTION_SPACE_SIZE};
use crate::nn_model::BanqiNet;
use crate::self_play::get_top_k_actions;
use tch::{Device, Tensor, Kind};

// ================ 场景验证结果 ================

/// 场景验证结果
#[derive(Debug, Clone)]
pub struct ScenarioResult {
    pub value: f32,
    pub unmasked_probs: Vec<f32>,  // 原始softmax概率
    pub masked_probs: Vec<f32>,    // 应用mask后的概率
}

#[derive(Debug, Clone)]
pub struct ScenarioMetric {
    pub name: &'static str,
    pub target_action: usize,
    pub target_prob: f32,
    pub value: f32,
    pub best_action: usize,
    pub best_prob: f32,
}

// ================ 场景验证函数 ================

fn evaluate_env_metric<F>(name: &'static str, target_action: usize, setup_fn: F, net: &BanqiNet, device: Device) -> ScenarioMetric
where
    F: FnOnce(&mut DarkChessEnv),
{
    let mut env = DarkChessEnv::new();
    setup_fn(&mut env);

    let obs = env.get_state();
    let board_tensor = Tensor::from_slice(obs.board.as_slice().unwrap())
        .view([1, 8, 3, 4])
        .to(device);
    let scalar_tensor = Tensor::from_slice(obs.scalars.as_slice().unwrap())
        .view([1, 56])
        .to(device);
    let masks: Vec<f32> = env.action_masks().iter().map(|&m| m as f32).collect();
    let mask_tensor = Tensor::from_slice(&masks).view([1, ACTION_SPACE_SIZE as i64]).to(device);

    let (logits, value) = tch::no_grad(|| net.forward_inference(&board_tensor, &scalar_tensor));
    let masked_logits = &logits + (&mask_tensor - 1.0) * 1e9;
    let probs = masked_logits.softmax(-1, Kind::Float);
    let prob_vec: Vec<f32> = (0..ACTION_SPACE_SIZE)
        .map(|i| probs.double_value(&[0, i as i64]) as f32)
        .collect();
    let (best_action, best_prob) = prob_vec
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap();

    ScenarioMetric {
        name,
        target_action,
        target_prob: prob_vec[target_action],
        value: value.squeeze().double_value(&[]) as f32,
        best_action,
        best_prob: *best_prob,
    }
}

pub fn evaluate_training_scenarios(net: &BanqiNet, device: Device) -> [ScenarioMetric; 2] {
    let scenario1 = evaluate_env_metric(
        "R_A vs B_A",
        38,
        |env| env.setup_two_advisors(Player::Black),
        net,
        device,
    );
    let scenario2 = evaluate_env_metric(
        "Hidden Threat",
        3,
        |env| env.setup_hidden_threats(),
        net,
        device,
    );
    [scenario1, scenario2]
}

/// 验证模型在标准场景上的表现，返回详细数据
/// 注意：必须传入训练时使用的同一个 BanqiNet 实例，
/// 否则在同一个 VarStore 中创建新网络会导致变量命名冲突
pub fn validate_model_on_scenarios_with_net(net: &BanqiNet, device: Device, _iteration: usize) -> (ScenarioResult, ScenarioResult) {
    // 场景1: R_A vs B_A
    let scenario1_result = {
        let mut env = DarkChessEnv::new();
        env.setup_two_advisors(Player::Black);
        
        let obs = env.get_state();
        let board_tensor = Tensor::from_slice(obs.board.as_slice().unwrap())
            .view([1, 8, 3, 4])
            .to(device);
        let scalar_tensor = Tensor::from_slice(obs.scalars.as_slice().unwrap())
            .view([1, 56])
            .to(device);
        
        let masks: Vec<f32> = env.action_masks().iter().map(|&m| m as f32).collect();
        let mask_tensor = Tensor::from_slice(&masks).to(device).view([1, 46]);
        
        let (logits, value) = tch::no_grad(|| net.forward_inference(&board_tensor, &scalar_tensor));
        
        // 🐛 DEBUG: 打印原始logits
        let logits_vec: Vec<f32> = (0..46).map(|i| logits.double_value(&[0, i]) as f32).collect();
        let top_logits = get_top_k_actions(&logits_vec, 5);
        println!("      🐛 原始logits (top-5): {:?}", top_logits);
        
        // 未应用mask的概率分布
        let unmasked_probs_tensor = logits.softmax(-1, Kind::Float);
        let unmasked_probs: Vec<f32> = (0..46).map(|i| unmasked_probs_tensor.double_value(&[0, i]) as f32).collect();
        
        // 应用mask后的概率分布
        let masked_logits = &logits + (&mask_tensor - 1.0) * 1e9;
        let masked_probs_tensor = masked_logits.softmax(-1, Kind::Float);
        let masked_probs: Vec<f32> = (0..46).map(|i| masked_probs_tensor.double_value(&[0, i]) as f32).collect();
        
        let value_pred: f32 = value.squeeze().double_value(&[]) as f32;
        
        // 🐛 DEBUG: 检查有效动作
        let valid_actions: Vec<usize> = masks.iter()
            .enumerate()
            .filter_map(|(i, &m)| if m == 1.0 { Some(i) } else { None })
            .collect();
        println!("      🐛 有效动作数: {}, 包括: {:?}", valid_actions.len(), &valid_actions[..valid_actions.len().min(10)]);
        
        println!("    场景1 (R_A vs B_A): value={:.3}", value_pred);
        println!("      未应用mask: a38={:.1}%, a39={:.1}%, a40={:.1}%", 
            unmasked_probs[38]*100.0, unmasked_probs[39]*100.0, unmasked_probs[40]*100.0);
        println!("      应用mask后: a38={:.1}%, a39={:.1}%, a40={:.1}%", 
            masked_probs[38]*100.0, masked_probs[39]*100.0, masked_probs[40]*100.0);
        println!("      期望: action38主导(>90%), value应偏向当前玩家(黑方)略优或平局");
        
        ScenarioResult {
            value: value_pred,
            unmasked_probs,
            masked_probs,
        }
    };
    
    // 场景2: Hidden Threat
    let scenario2_result = {
        let mut env = DarkChessEnv::new();
        env.setup_hidden_threats();
        
        let obs = env.get_state();
        let board_tensor = Tensor::from_slice(obs.board.as_slice().unwrap())
            .view([1, 8, 3, 4])
            .to(device);
        let scalar_tensor = Tensor::from_slice(obs.scalars.as_slice().unwrap())
            .view([1, 56])
            .to(device);
        
        let masks: Vec<f32> = env.action_masks().iter().map(|&m| m as f32).collect();
        let mask_tensor = Tensor::from_slice(&masks).to(device).view([1, 46]);
        
        let (logits, value) = tch::no_grad(|| net.forward_inference(&board_tensor, &scalar_tensor));
        
        // 🐛 DEBUG: 打印原始logits
        let logits_vec: Vec<f32> = (0..46).map(|i| logits.double_value(&[0, i]) as f32).collect();
        let top_logits = get_top_k_actions(&logits_vec, 5);
        println!("      🐛 原始logits (top-5): {:?}", top_logits);
        
        // 未应用mask的概率分布
        let unmasked_probs_tensor = logits.softmax(-1, Kind::Float);
        let unmasked_probs: Vec<f32> = (0..46).map(|i| unmasked_probs_tensor.double_value(&[0, i]) as f32).collect();
        
        // 应用mask后的概率分布
        let masked_logits = &logits + (&mask_tensor - 1.0) * 1e9;
        let masked_probs_tensor = masked_logits.softmax(-1, Kind::Float);
        let masked_probs: Vec<f32> = (0..46).map(|i| masked_probs_tensor.double_value(&[0, i]) as f32).collect();
        
        let value_pred: f32 = value.squeeze().double_value(&[]) as f32;
        
        // 🐛 DEBUG: 检查有效动作
        let valid_actions: Vec<usize> = masks.iter()
            .enumerate()
            .filter_map(|(i, &m)| if m == 1.0 { Some(i) } else { None })
            .collect();
        println!("      🐛 有效动作数: {}, 包括: {:?}", valid_actions.len(), &valid_actions[..valid_actions.len().min(10)]);
        
        println!("    场景2 (Hidden Threat): value={:.3}", value_pred);
        println!("      未应用mask: a3={:.1}%, a5={:.1}%", 
            unmasked_probs[3]*100.0, unmasked_probs[5]*100.0);
        println!("      应用mask后: a3={:.1}%, a5={:.1}%", 
            masked_probs[3]*100.0, masked_probs[5]*100.0);
        println!("      期望: action3主导(>90%), value应能反映位置优势");
        
        ScenarioResult {
            value: value_pred,
            unmasked_probs,
            masked_probs,
        }
    };
    
    (scenario1_result, scenario2_result)
}