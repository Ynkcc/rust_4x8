//! MCTS + 深度学习策略（支持搜索树复用）
//!
//! 提供 `MctsDlPolicy`：
//! - 维持一个持久的 `MCTS` 实例
//! - 可动态调整模拟次数
//! - 基于已加载的 `BanqiNet` 模型做策略+价值评估
//! - 在每个外部动作后调用 `advance(action, env)` 来复用搜索树
//!
//! 使用流程：
//! 1. 加载模型 -> `ModelWrapper::load_from_file`
//! 2. 创建策略 -> `MctsDlPolicy::new(model, &env, sims)`
//! 3. 每当玩家或 AI 执行动作后调用 `advance`
//! 4. 需要选择动作时调用 `choose_action(&env)`

use std::sync::{Arc, Mutex};
use crate::{DarkChessEnv, ACTION_SPACE_SIZE};
use crate::mcts::{Evaluator, MCTS, MCTSConfig};
use crate::nn_model::BanqiNet;
use tch::{nn, Kind, Tensor, Device};

// ---------------- Model 封装 ----------------

pub struct ModelWrapper {
    _vs: nn::VarStore,  // 加下划线避免未使用警告，但需要持有以保持模型权重在内存中
    net: BanqiNet,
    device: Device,
    gate: Mutex<()>, // 串行化前向以保线程安全
}

impl ModelWrapper {
    pub fn load_from_file(path: &str) -> Result<Self, String> {
        let device = Device::Cpu;
        let mut vs = nn::VarStore::new(device);
        let net = BanqiNet::new(&vs.root());
        vs.load(path).map_err(|e| format!("模型加载失败: {}", e))?;
        Ok(Self { _vs: vs, net, device, gate: Mutex::new(()) })
    }
}

// 由于内部有互斥锁保护，允许跨线程共享
unsafe impl Send for ModelWrapper {}
unsafe impl Sync for ModelWrapper {}

// ---------------- Evaluator ----------------

pub struct TchEvaluator {
    pub model: Arc<ModelWrapper>,
}

impl Evaluator for TchEvaluator {
    fn evaluate(&self, env: &DarkChessEnv) -> (Vec<f32>, f32) {
        let _guard = self.model.gate.lock().unwrap();
        let obs = env.get_state();

        // board: [STATE_STACK, 8, 3, 4] -> 展开为 [1, (STATE_STACK*8), 3, 4]
        let board_flat: Vec<f32> = obs.board.iter().cloned().collect();
        let board_t = Tensor::from_slice(&board_flat)
            .to_device(self.model.device)
            .view([1, (crate::STATE_STACK_SIZE * 8) as i64, 3, 4]);

        // scalars: [STATE_STACK * 56] -> [1, STATE_STACK*56]
        let scalars_flat: Vec<f32> = obs.scalars.iter().cloned().collect();
        let scalars_t = Tensor::from_slice(&scalars_flat)
            .to_device(self.model.device)
            .view([1, (56 * crate::STATE_STACK_SIZE) as i64]);

        let (policy_logits, value_t) = self.model.net.forward_inference(&board_t, &scalars_t);

        // 掩码
        let masks = env.action_masks();
        let mask_t = Tensor::from_slice(&masks)
            .to_device(self.model.device)
            .to_kind(Kind::Bool)
            .view([1, ACTION_SPACE_SIZE as i64]);
        let invalid = mask_t.logical_not().to_kind(Kind::Float);
        let masked_logits = &policy_logits + &invalid * -1e9;

        let probs_t = masked_logits.softmax(-1, Kind::Float);
        let probs_1d = probs_t.squeeze();
        let mut probs: Vec<f32> = vec![0.0; ACTION_SPACE_SIZE];
        probs_1d.to_device(Device::Cpu).copy_data(&mut probs, ACTION_SPACE_SIZE);

        let value = value_t.squeeze().to_device(Device::Cpu).double_value(&[]) as f32;
        (probs, value)
    }
}

// ---------------- 策略对象（持久化 MCTS） ----------------

pub struct MctsDlPolicy {
    evaluator: Arc<TchEvaluator>,
    mcts: MCTS<TchEvaluator>,
    num_simulations: usize,
    cpuct: f32,
}

impl MctsDlPolicy {
    pub fn new(model: Arc<ModelWrapper>, env: &DarkChessEnv, num_simulations: usize) -> Self {
        let evaluator = Arc::new(TchEvaluator { model });
        let config = MCTSConfig { cpuct: 1.0, num_simulations };
        let mcts = MCTS::new(env, evaluator.clone(), config);
        Self { evaluator, mcts, num_simulations, cpuct: 1.0 }
    }

    pub fn set_iterations(&mut self, sims: usize) {
        self.num_simulations = sims.max(1);
        // 由于 MCTSConfig 字段是私有的，这里通过重新构建根节点方式更新配置
        let root_env = self.mcts.root.env.as_ref().unwrap().clone();
        let new_cfg = MCTSConfig { cpuct: self.cpuct, num_simulations: self.num_simulations };
        self.mcts = MCTS::new(&root_env, self.evaluator.clone(), new_cfg);
    }

    /// 在外部环境执行 action 后调用，用于推进/复用搜索树
    pub fn advance(&mut self, env: &DarkChessEnv, action: usize) {
        self.mcts.step_next(env, action);
    }

    /// 选择动作：如发现根环境与传入 env 不一致（步数不匹配），重置搜索树
    pub fn choose_action(&mut self, env: &DarkChessEnv) -> Option<usize> {
        // 简单一致性判断：总步数不同 -> 重置
        if self.mcts.root.env.as_ref().map(|e| e.get_total_steps()) != Some(env.get_total_steps()) {
            self.mcts = MCTS::new(env, self.evaluator.clone(), MCTSConfig { cpuct: self.cpuct, num_simulations: self.num_simulations });
        }
        self.mcts.run()
    }
}
