// code_files/src/nn_model.rs
use tch::{nn, Tensor};
use crate::game_env::{
    ACTION_SPACE_SIZE, BOARD_CHANNELS, BOARD_COLS, BOARD_ROWS,
    SCALAR_FEATURE_COUNT, STATE_STACK_SIZE,
};

// ========== 4×8 棋盘配置 (使用 game_env.rs 的常量) ==========
// 这里的常量定义用于适配 tch 的 tensor 维度需求
// 数值直接衍生自 game_env.rs 中定义的物理环境参数
const TOTAL_CHANNELS: i64 = (BOARD_CHANNELS * STATE_STACK_SIZE) as i64;
const BOARD_H: i64 = BOARD_ROWS as i64;
const BOARD_W: i64 = BOARD_COLS as i64;
const SCALAR_FEATURES: i64 = SCALAR_FEATURE_COUNT as i64;
const ACTION_SIZE: i64 = ACTION_SPACE_SIZE as i64;

// 网络隐藏层通道数
// 针对暗棋的复杂性（尤其是炮的跳吃和翻棋的不确定性），
// 使用较宽的通道数以捕捉更丰富的空间特征
const HIDDEN_CHANNELS: i64 = 128;

// 标准残差块 (Residual Block)
// 结构: Conv -> BN -> ReLU -> Conv -> BN -> (+Input) -> ReLU
// 保持输入输出的空间尺寸和通道数不变，用于构建深层网络而不丢失梯度
struct BasicBlock {
    conv1: nn::Conv2D,
    bn1: nn::BatchNorm,
    conv2: nn::Conv2D,
    bn2: nn::BatchNorm,
}

impl BasicBlock {
    fn new(vs: &nn::Path, channels: i64) -> Self {
        // 使用 padding=1 配合 kernel_size=3 以保持空间尺寸不变
        let conv_cfg = nn::ConvConfig {
            padding: 1,
            ..Default::default()
        };
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

        // 残差连接：将输入直接加到输出上，然后进行激活
        (out + residual).relu()
    }
}

// BanqiNet: 暗棋策略价值网络
// 架构:
// 1. 输入卷积层 (Input Conv)
// 2. 残差塔 (Residual Tower): 堆叠多个残差块
// 3. 策略头 (Policy Head): 输出所有合法动作的概率 logits
// 4. 价值头 (Value Head): 输出当前局面的胜率评估 [-1, 1]
pub struct BanqiNet {
    // 初始卷积层：将输入状态映射到隐藏特征空间
    conv_input: nn::Conv2D,
    bn_input: nn::BatchNorm,

    // 残差塔：用于提取深层空间特征
    res_block1: BasicBlock,
    res_block2: BasicBlock,
    res_block3: BasicBlock,

    // 全连接层：处理扁平化后的空间特征与全局标量特征
    fc1: nn::Linear,
    fc2: nn::Linear,

    // 输出头
    policy_head: nn::Linear,
    value_head: nn::Linear,
}

impl BanqiNet {
    pub fn new(vs: &nn::Path) -> Self {
        let conv_cfg = nn::ConvConfig {
            padding: 1,
            ..Default::default()
        };

        // 1. Input Conv
        // 将棋盘状态张量转换为隐藏层特征
        let conv_input = nn::conv2d(
            vs / "conv_input",
            TOTAL_CHANNELS,
            HIDDEN_CHANNELS,
            3,
            conv_cfg,
        );
        let bn_input = nn::batch_norm2d(vs / "bn_input", HIDDEN_CHANNELS, Default::default());

        // 2. 残差塔
        // 连续经过3个残差块，深化特征表示
        let res_block1 = BasicBlock::new(&(vs / "res1"), HIDDEN_CHANNELS);
        let res_block2 = BasicBlock::new(&(vs / "res2"), HIDDEN_CHANNELS);
        let res_block3 = BasicBlock::new(&(vs / "res3"), HIDDEN_CHANNELS);

        // 3. 计算全连接层输入维度
        // 空间特征展平大小
        let flat_size = HIDDEN_CHANNELS * BOARD_H * BOARD_W;
        
        // 总输入 = 空间特征 + 全局标量特征
        let total_fc_input = flat_size + SCALAR_FEATURES;

        // 4. 全连接层 (Shared Dense Layers)
        // 融合空间信息和全局信息（如步数、血量、存活棋子统计等）
        let fc1 = nn::linear(vs / "fc1", total_fc_input, 512, Default::default());
        let fc2 = nn::linear(vs / "fc2", 512, 256, Default::default());

        // 5. 输出头
        // Policy Head: 映射到动作空间大小
        // Value Head: 映射到标量价值
        let policy_head = nn::linear(vs / "policy", 256, ACTION_SIZE, Default::default());
        let value_head = nn::linear(vs / "value", 256, 1, Default::default());

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
        // 1. 输入卷积处理
        // 提取初步的空间特征
        let x = board
            .apply(&self.conv_input)
            .apply_t(&self.bn_input, train)
            .relu();

        // 2. 通过残差塔
        // 深层特征提取
        let x = self.res_block1.forward_t(&x, train);
        let x = self.res_block2.forward_t(&x, train);
        let x = self.res_block3.forward_t(&x, train);

        // 3. 特征展平 (Flatten)
        // 将 [Batch, Channels, H, W] 展平为 [Batch, Features]
        let x = x.flatten(1, -1);

        // 4. 特征拼接 (Concatenate)
        // 将卷积提取的图像特征与全局标量特征拼接
        let combined = Tensor::cat(&[&x, scalars], 1);

        // 5. 全连接层处理
        // 共享的稠密层，用于后续分头
        let shared = combined.apply(&self.fc1).relu().apply(&self.fc2).relu();

        // 6. 输出计算
        // Policy: 输出 Logits (未经过 Softmax)
        // Value: 输出 Tanh 激活的价值 [-1, 1]
        let policy_logits = shared.apply(&self.policy_head);
        let value = shared.apply(&self.value_head).tanh();

        (policy_logits, value)
    }

    /// 训练模式的前向传播
    /// 启用 BatchNorm 的统计更新和 Dropout (如果有)
    pub fn forward(&self, board: &Tensor, scalars: &Tensor) -> (Tensor, Tensor) {
        self.forward_t(board, scalars, true)
    }

    /// 推理模式的前向传播
    /// 使用固定的 BatchNorm 统计量，禁用 Dropout
    pub fn forward_inference(&self, board: &Tensor, scalars: &Tensor) -> (Tensor, Tensor) {
        self.forward_t(board, scalars, false)
    }
}