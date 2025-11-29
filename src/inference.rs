// inference.rs - 推理服务器模块
//
// 提供批量神经网络推理服务，用于支持并行自对弈训练
// 通过通道收集多个推理请求，批量处理以提高GPU利用率

use crate::game_env::{DarkChessEnv, Observation};
use crate::mcts::Evaluator;
use crate::nn_model::BanqiNet;
use anyhow::Result;
use std::sync::mpsc;
use std::time::Duration;
use tch::{nn, Device, Tensor, Kind};

// ================ 推理请求和响应 ================

/// 推理请求
#[derive(Debug)]
pub struct InferenceRequest {
    pub observation: Observation,
    pub action_masks: Vec<i32>,
    pub response_tx: mpsc::Sender<InferenceResponse>, // 每个请求携带自己的响应通道
}

/// 推理响应
#[derive(Debug, Clone)]
pub struct InferenceResponse {
    pub policy: Vec<f32>,
    pub value: f32,
}

// ================ 批量推理服务器 ================

pub struct InferenceServer {
    _vs: nn::VarStore,    // 持有 VarStore（包含模型权重）- 加下划线避免警告
    net: BanqiNet,        // 网络结构
    device: Device,
    request_rx: mpsc::Receiver<InferenceRequest>,
    batch_size: usize,
    batch_timeout_ms: u64,
}

impl InferenceServer {
    pub fn new(
        model_path: &str,
        device: Device,
        request_rx: mpsc::Receiver<InferenceRequest>,
        batch_size: usize,
        batch_timeout_ms: u64,
    ) -> Result<Self> {
        let mut vs = nn::VarStore::new(device);
        let net = BanqiNet::new(&vs.root());
        
        // 加载模型权重
        vs.load(model_path)?;
        
        Ok(Self {
            _vs: vs,
            net,
            device,
            request_rx,
            batch_size,
            batch_timeout_ms,
        })
    }

    /// 运行推理服务（阻塞）
    pub fn run(&self) {
        println!("[InferenceServer] 启动，batch_size={}, timeout={}ms", 
            self.batch_size, self.batch_timeout_ms);
        
        let mut batch = Vec::new();
        let mut total_requests = 0;
        let mut total_batches = 0;
        let batch_timeout = Duration::from_millis(self.batch_timeout_ms);
        
        loop {
            // 尝试快速收集一批请求
            
            // 首先尝试非阻塞接收，快速收集可用的请求
            loop {
                match self.request_rx.try_recv() {
                    Ok(req) => {
                        batch.push(req);
                        total_requests += 1;
                        
                        // 如果达到批量大小，立即处理
                        if batch.len() >= self.batch_size {
                            break;
                        }
                    },
                    Err(mpsc::TryRecvError::Empty) => {
                        // 没有更多请求了
                        break;
                    },
                    Err(mpsc::TryRecvError::Disconnected) => {
                        // 所有发送者已断开
                        if !batch.is_empty() {
                            println!("[InferenceServer] 最终批次: {} 个请求", batch.len());
                            self.process_batch(&batch);
                            total_batches += 1;
                        }
                        println!("[InferenceServer] 所有客户端已断开，退出 (总计: {} 请求, {} 批次)", 
                            total_requests, total_batches);
                        return;
                    }
                }
            }
            
            // 如果收集到了请求，立即处理（不等待超时）
            if !batch.is_empty() {
                if total_batches % 4000 == 0 {
                    println!("[InferenceServer] 处理批次#{}: {} 个请求", total_batches + 1, batch.len());
                }
                self.process_batch(&batch);
                total_batches += 1;
                batch.clear();
                continue;
            }
            
            // 如果没有请求，阻塞等待新请求（带超时）
            match self.request_rx.recv_timeout(batch_timeout) {
                Ok(req) => {
                    batch.push(req);
                    total_requests += 1;
                },
                Err(mpsc::RecvTimeoutError::Timeout) => {
                    // 超时但没有请求，继续等待
                    continue;
                },
                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    println!("[InferenceServer] 所有客户端已断开，退出 (总计: {} 请求, {} 批次)", 
                        total_requests, total_batches);
                    return;
                }
            }
        }
    }

    /// 批量处理推理请求
    fn process_batch(&self, batch: &Vec<InferenceRequest>) {
        if batch.is_empty() { return; }
        
        // let start_time = Instant::now();
        let batch_len = batch.len();
        
        // 准备批量输入张量
        let mut board_data = Vec::new();
        let mut scalar_data = Vec::new();
        let mut mask_data = Vec::new();
        
        for req in batch {
            // Board: [STATE_STACK_SIZE, 8, 3, 4] -> flatten
            let board_flat: Vec<f32> = req.observation.board.as_slice().unwrap().to_vec();
            board_data.extend_from_slice(&board_flat);
            
            // Scalars: [STATE_STACK_SIZE * 56]
            let scalars_flat: Vec<f32> = req.observation.scalars.as_slice().unwrap().to_vec();
            scalar_data.extend_from_slice(&scalars_flat);
            
            // Masks: [46]
            let masks_f32: Vec<f32> = req.action_masks.iter().map(|&m| m as f32).collect();
            mask_data.extend_from_slice(&masks_f32);
        }
        
        // 构建张量: [batch, C, H, W]
        let board_tensor = Tensor::from_slice(&board_data)
            .view([batch_len as i64, 8, 3, 4])  // 禁用状态堆叠后: STATE_STACK_SIZE=1, 所以是8通道
            .to(self.device);
        
        let scalar_tensor = Tensor::from_slice(&scalar_data)
            .view([batch_len as i64, 56])  // 禁用状态堆叠后: 56个特征
            .to(self.device);
        
        let mask_tensor = Tensor::from_slice(&mask_data)
            .view([batch_len as i64, 46])
            .to(self.device);
        
        // 前向推理（推理模式）
        let (logits, values) = tch::no_grad(|| {
            self.net.forward_inference(&board_tensor, &scalar_tensor)
        });
        
        // 应用掩码并计算概率
        let masked_logits = &logits + (&mask_tensor - 1.0) * 1e9;
        let probs = masked_logits.softmax(-1, Kind::Float);
        
        // 提取结果并发送响应到各自的通道
        for (i, req) in batch.iter().enumerate() {
            let policy_slice = probs.get(i as i64);
            let mut policy = vec![0.0f32; 46];
            policy_slice.to_device(Device::Cpu).copy_data(&mut policy, 46);
            
            let value = values.get(i as i64).squeeze().double_value(&[]) as f32;
            
            let response = InferenceResponse {
                policy,
                value,
            };
            
            // 发送响应到请求者的专属通道（忽略发送失败）
            let _ = req.response_tx.send(response);
        }
        
        
        // let elapsed = start_time.elapsed();
        // if batch_len >= 4 {  // 只在批量较大时输出日志
        //     println!("[InferenceServer] 批次处理: {} 个请求耗时 {:.2}ms", 
        //         batch_len, elapsed.as_secs_f64() * 1000.0);
        // }
    }
}

// ================ Channel Evaluator（用于MCTS） ================

pub struct ChannelEvaluator {
    request_tx: mpsc::Sender<InferenceRequest>,
}

impl ChannelEvaluator {
    pub fn new(request_tx: mpsc::Sender<InferenceRequest>) -> Self {
        Self { request_tx }
    }
}

impl Evaluator for ChannelEvaluator {
    fn evaluate(&self, env: &DarkChessEnv) -> (Vec<f32>, f32) {
        // 为此次请求创建一次性响应通道
        let (response_tx, response_rx) = mpsc::channel();
        
        // 发送推理请求
        let req = InferenceRequest {
            observation: env.get_state(),
            action_masks: env.action_masks(),
            response_tx,
        };
        
        self.request_tx.send(req).expect("推理服务已断开");
        
        // 等待响应（阻塞）
        let resp = response_rx.recv().expect("推理服务无响应");
        
        (resp.policy, resp.value)
    }
}