// src/inference.rs
//
// 推理服务器模块 (Inference Server Module)
//
// 功能:
// 1. 提供基于 Channel 的异步推理服务
// 2. 支持将来自多个 Self-Play Worker 的请求合并为 Batch 进行推理 (提高 GPU 利用率)
// 3. 处理 PyTorch 张量的内存管理 (避免内存泄漏)

use crate::game_env::{
    DarkChessEnv, Observation, ACTION_SPACE_SIZE, BOARD_CHANNELS, BOARD_COLS, BOARD_ROWS,
    SCALAR_FEATURE_COUNT,
};
use crate::mcts::Evaluator;
use crate::nn_model::BanqiNet;
use anyhow::Result;
use std::sync::mpsc;
use std::time::Duration;
use tch::{nn, Device, Kind, Tensor};

// ================ 推理请求和响应 ================

/// 推理请求结构
/// 用于从 Worker 发送到 InferenceServer
#[derive(Debug)]
pub struct InferenceRequest {
    /// 游戏观测状态 (包含棋盘和标量特征)
    pub observation: Observation,
    /// 动作掩码 (用于屏蔽非法动作)
    pub action_masks: Vec<i32>,
    /// 响应通道发送端 (Server 处理完后通过此通道发回结果)
    pub response_tx: mpsc::Sender<InferenceResponse>,
}

/// 推理响应结构
/// 用于从 InferenceServer 发回给 Worker
#[derive(Debug, Clone)]
pub struct InferenceResponse {
    /// 策略概率分布 (已应用 Softmax 和 Mask)
    pub policy: Vec<f32>,
    /// 状态价值 ([-1, 1])
    pub value: f32,
}

// ================ 批量推理服务器 ================

/// 推理服务器
/// 负责收集请求、组装 Batch、执行模型推理并分发结果
pub struct InferenceServer {
    _vs: nn::VarStore, // 持有模型权重 (使用下划线前缀抑制未使用警告)
    net: BanqiNet,     // 神经网络模型
    device: Device,    // 推理设备 (CPU/CUDA)
    request_rx: mpsc::Receiver<InferenceRequest>, // 请求接收通道
    batch_size: usize,      // 最大批次大小
    batch_timeout_ms: u64,  // 批次收集超时时间 (毫秒)
}

impl InferenceServer {
    /// 创建新的推理服务器实例
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

    /// 运行推理服务 (阻塞式循环)
    pub fn run(&self) {
        println!(
            "[InferenceServer] 启动，batch_size={}, timeout={}ms",
            self.batch_size, self.batch_timeout_ms
        );

        let mut batch = Vec::new();
        let mut total_requests = 0;
        let mut total_batches = 0;
        let batch_timeout = Duration::from_millis(self.batch_timeout_ms);

        loop {
            // --- 阶段 1: 快速收集 ---
            // 尝试非阻塞接收，尽可能填满一个 Batch
            loop {
                match self.request_rx.try_recv() {
                    Ok(req) => {
                        batch.push(req);
                        total_requests += 1;

                        // 如果达到批量大小，立即处理
                        if batch.len() >= self.batch_size {
                            break;
                        }
                    }
                    Err(mpsc::TryRecvError::Empty) => {
                        // 队列暂时为空
                        break;
                    }
                    Err(mpsc::TryRecvError::Disconnected) => {
                        // 所有发送端已断开，处理剩余请求后退出
                        if !batch.is_empty() {
                            println!("[InferenceServer] 最终批次: {} 个请求", batch.len());
                            self.process_batch(&batch);
                            total_batches += 1;
                        }
                        println!(
                            "[InferenceServer] 所有客户端已断开，退出 (总计: {} 请求, {} 批次)",
                            total_requests, total_batches
                        );
                        return;
                    }
                }
            }

            // --- 阶段 2: 超时等待 ---
            // 如果 Batch 非空，立即处理；如果为空，则阻塞等待
            if !batch.is_empty() {
                if total_batches % 4000 == 0 {
                    println!(
                        "[InferenceServer] 处理批次#{}: {} 个请求",
                        total_batches + 1,
                        batch.len()
                    );
                }
                self.process_batch(&batch);
                total_batches += 1;
                batch.clear();
                continue;
            }

            // 队列为空，阻塞等待第一个请求
            match self.request_rx.recv_timeout(batch_timeout) {
                Ok(req) => {
                    batch.push(req);
                    total_requests += 1;
                }
                Err(mpsc::RecvTimeoutError::Timeout) => {
                    // 超时仍无请求，继续循环
                    continue;
                }
                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    println!(
                        "[InferenceServer] 所有客户端已断开，退出 (总计: {} 请求, {} 批次)",
                        total_requests, total_batches
                    );
                    return;
                }
            }
        }
    }

    /// 处理单个批次的推理请求
    fn process_batch(&self, batch: &Vec<InferenceRequest>) {
        if batch.is_empty() {
            return;
        }

        let batch_len = batch.len();

        // 关键: 使用 no_grad 块，避免构建计算图，节省内存并提高速度
        tch::no_grad(|| {
            // 1. 准备数据容器
            let mut board_data = Vec::new();
            let mut scalar_data = Vec::new();
            let mut mask_data = Vec::new();

            // 2. 扁平化收集 Batch 数据
            for req in batch {
                // Board: flatten [C, H, W]
                let board_flat: Vec<f32> = req.observation.board.as_slice().unwrap().to_vec();
                board_data.extend_from_slice(&board_flat);

                // Scalars: flatten [Features]
                let scalars_flat: Vec<f32> = req.observation.scalars.as_slice().unwrap().to_vec();
                scalar_data.extend_from_slice(&scalars_flat);

                // Masks: flatten [ActionSize]
                let masks_f32: Vec<f32> = req.action_masks.iter().map(|&m| m as f32).collect();
                mask_data.extend_from_slice(&masks_f32);
            }

            // 3. 构建 PyTorch 张量 (移动到指定 Device)
            let board_tensor = Tensor::from_slice(&board_data)
                .view([
                    batch_len as i64,
                    BOARD_CHANNELS as i64,
                    BOARD_ROWS as i64,
                    BOARD_COLS as i64,
                ])
                .to(self.device);

            let scalar_tensor = Tensor::from_slice(&scalar_data)
                .view([batch_len as i64, SCALAR_FEATURE_COUNT as i64])
                .to(self.device);

            let mask_tensor = Tensor::from_slice(&mask_data)
                .view([batch_len as i64, ACTION_SPACE_SIZE as i64])
                .to(self.device);

            // 4. 执行前向推理
            // forward_inference 内部处理了 eval 模式
            let (logits, values) = self.net.forward_inference(&board_tensor, &scalar_tensor);

            // 5. 应用动作掩码 (Invalid actions -> -inf)
            let masked_logits = &logits + (&mask_tensor - 1.0) * 1e9;
            let probs = masked_logits.softmax(-1, Kind::Float);

            // 6. 分发结果
            for (i, req) in batch.iter().enumerate() {
                // 提取策略概率
                let policy_slice = probs.get(i as i64);
                let mut policy = vec![0.0f32; ACTION_SPACE_SIZE];
                policy_slice
                    .to_device(Device::Cpu)
                    .copy_data(&mut policy, ACTION_SPACE_SIZE);

                // 提取价值
                let value = values.get(i as i64).squeeze().double_value(&[]) as f32;

                let response = InferenceResponse { policy, value };

                // 发送回请求者
                if let Err(e) = req.response_tx.send(response) {
                    eprintln!("  ⚠️ [InferenceServer] 发送响应失败: {}", e);
                }
                
                // 显式释放切片张量引用
                drop(policy_slice);
            }
            
            // 7. 显式释放中间张量 (防止某些环境下的内存积累)
            drop(board_tensor);
            drop(scalar_tensor);
            drop(mask_tensor);
            drop(logits);
            drop(values);
            drop(masked_logits);
            drop(probs);
        });
    }
}

// ================ 基于通道的评估器 (用于 Worker) ================

/// 通道评估器
/// 实现了 Evaluator trait，将评估请求转发给 InferenceServer
pub struct ChannelEvaluator {
    request_tx: mpsc::Sender<InferenceRequest>,
}

impl ChannelEvaluator {
    pub fn new(request_tx: mpsc::Sender<InferenceRequest>) -> Self {
        Self { request_tx }
    }
}

#[async_trait::async_trait]
impl Evaluator for ChannelEvaluator {
    async fn evaluate(&self, env: &DarkChessEnv) -> (Vec<f32>, f32) {
        // 创建一次性通道用于接收结果
        let (response_tx, response_rx) = mpsc::channel();

        // 准备请求数据
        let mut masks = vec![0; crate::game_env::ACTION_SPACE_SIZE];
        env.action_masks_into(&mut masks);
        
        let req = InferenceRequest {
            observation: env.get_state(),
            action_masks: masks,
            response_tx,
        };

        // 发送请求
        self.request_tx.send(req).expect("推理服务已断开");

        // 阻塞等待结果
        let resp = response_rx.recv().expect("推理服务无响应");

        (resp.policy, resp.value)
    }
}