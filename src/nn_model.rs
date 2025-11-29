// code_files/src/nn_model.rs
use tch::{nn, Tensor};

const BOARD_CHANNELS: i64 = 8;
const STATE_STACK: i64 = 1; // 禁用状态堆叠 (game_env.rs STATE_STACK_SIZE = 1)
const TOTAL_CHANNELS: i64 = BOARD_CHANNELS * STATE_STACK; // 8
const BOARD_H: i64 = 3;
const BOARD_W: i64 = 4;
const SCALAR_FEATURES: i64 = 56 * STATE_STACK; // 56
const ACTION_SIZE: i64 = 46;

// 网络通道数：针对3x4小棋盘，64通道足够
const HIDDEN_CHANNELS: i64 = 64;

// 标准残差块：保持通道数不变，加强特征提取
struct BasicBlock {
    conv1: nn::Conv2D,
    bn1: nn::BatchNorm,
    conv2: nn::Conv2D,
    bn2: nn::BatchNorm,
}

impl BasicBlock {
    fn new(vs: &nn::Path, channels: i64) -> Self {
        let conv_cfg = nn::ConvConfig { padding: 1, ..Default::default() };
        Self {
            conv1: nn::conv2d(vs / "conv1", channels, channels, 3, conv_cfg),
            bn1: nn::batch_norm2d(vs / "bn1", channels, Default::default()),
            conv2: nn::conv2d(vs / "conv2", channels, channels, 3, conv_cfg),
            bn2: nn::batch_norm2d(vs / "bn2", channels, Default::default()),
        }
    }

    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        let residual = xs;
        let out = xs
            .apply(&self.conv1)
            .apply_t(&self.bn1, train)
            .relu()
            .apply(&self.conv2)
            .apply_t(&self.bn2, train);
        
        // 残差连接 + ReLU
        (out + residual).relu()
    }
}

pub struct BanqiNet {
    // 初始卷积层：将输入通道转换为隐藏通道
    conv_input: nn::Conv2D,
    bn_input: nn::BatchNorm,
    
    // 残差塔：3个残差块用于深层特征提取
    res_block1: BasicBlock,
    res_block2: BasicBlock,
    res_block3: BasicBlock,
    
    // 全连接层：适度大小
    fc1: nn::Linear,
    fc2: nn::Linear,
    
    // 输出头
    policy_head: nn::Linear,
    value_head: nn::Linear,
}

impl BanqiNet {
    pub fn new(vs: &nn::Path) -> Self {
        let conv_cfg = nn::ConvConfig { padding: 1, ..Default::default() };
        
        // 1. Input Conv: [Batch, 8, 3, 4] -> [Batch, 64, 3, 4]
        let conv_input = nn::conv2d(vs / "conv_input", TOTAL_CHANNELS, HIDDEN_CHANNELS, 3, conv_cfg);
        let bn_input = nn::batch_norm2d(vs / "bn_input", HIDDEN_CHANNELS, Default::default());
        
        // 2. 残差塔：3个残差块，保持 [Batch, 64, 3, 4]
        let res_block1 = BasicBlock::new(&(vs / "res1"), HIDDEN_CHANNELS);
        let res_block2 = BasicBlock::new(&(vs / "res2"), HIDDEN_CHANNELS);
        let res_block3 = BasicBlock::new(&(vs / "res3"), HIDDEN_CHANNELS);
        
        // 3. 计算 Flatten 后的尺寸
        let flat_size = HIDDEN_CHANNELS * BOARD_H * BOARD_W; // 64 * 3 * 4 = 768
        let total_fc_input = flat_size + SCALAR_FEATURES;    // 768 + 56 = 824
        
        // 4. 全连接层：824 -> 256 -> 128
        // 相比原先的 3128 -> 1024 -> 512 -> 256，参数量大幅减少但保持足够容量
        let fc1 = nn::linear(vs / "fc1", total_fc_input, 256, Default::default());
        let fc2 = nn::linear(vs / "fc2", 256, 128, Default::default());
        
        // 5. 输出头
        let policy_head = nn::linear(vs / "policy", 128, ACTION_SIZE, Default::default());
        let value_head = nn::linear(vs / "value", 128, 1, Default::default());
        
        Self {
            conv_input,
            bn_input,
            res_block1,
            res_block2,
            res_block3,
            fc1,
            fc2,
            policy_head,
            value_head,
        }
    }
    
    fn forward_t(&self, board: &Tensor, scalars: &Tensor, train: bool) -> (Tensor, Tensor) {
        // 1. 输入卷积：提取初始特征
        let x = board
            .apply(&self.conv_input)
            .apply_t(&self.bn_input, train)
            .relu();
        
        // 2. 通过残差塔：逐步深化特征
        let x = self.res_block1.forward_t(&x, train);
        let x = self.res_block2.forward_t(&x, train);
        let x = self.res_block3.forward_t(&x, train);
        
        // 3. 展平卷积特征
        let x = x.flatten(1, -1); // [Batch, 768]
        
        // 4. 拼接标量特征 (当前玩家、存活情况等)
        let combined = Tensor::cat(&[&x, scalars], 1); // [Batch, 824]
        
        // 5. 全连接层：融合所有特征
        let shared = combined
            .apply(&self.fc1).relu()
            .apply(&self.fc2).relu();
        
        // 6. 输出头
        let policy_logits = shared.apply(&self.policy_head);
        let value = shared.apply(&self.value_head).tanh();
        
        (policy_logits, value)
    }

    /// 训练模式的前向传播（默认）
    /// 等价于 forward_t(board, scalars, true)
    pub fn forward(&self, board: &Tensor, scalars: &Tensor) -> (Tensor, Tensor) {
        self.forward_t(board, scalars, true)
    }
    
    /// 推理模式的前向传播（用于验证/推理）
    /// 等价于 forward_t(board, scalars, false)
    pub fn forward_inference(&self, board: &Tensor, scalars: &Tensor) -> (Tensor, Tensor) {
        self.forward_t(board, scalars, false)
    }
}