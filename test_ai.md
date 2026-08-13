# AI 对手功能测试指南

## 修复内容总结

### 1. 修复了 `ai/mcts_dl.rs` 的生命周期问题
- 将 `MctsDlPolicy` 改为每次调用 `choose_action` 时创建新的 MCTS 实例
- 移除了复杂的自引用结构，避免生命周期编译错误
- 保留了 `choose_action_once` 函数用于一次性 MCTS 搜索

### 2. 启用了 `tauri_main.rs` 中的 MctsDL 功能
- 添加了 `model: Mutex<Option<Arc<ModelWrapper>>>` 存储加载的模型
- 添加了 `mcts_policy: Mutex<Option<MctsDlPolicy>>` 存储策略实例
- 添加了 `mcts_num_simulations: Mutex<usize>` 配置搜索次数
- 在 `bot_move` 命令中实现了 MctsDL 对手逻辑

### 3. 添加了模型管理命令
- `list_models()`: 列出当前目录下的 `.pt` 模型文件
- `load_model(path)`: 加载 TorchScript 模型（`.pt` 格式）
- `set_mcts_iterations(iters)`: 设置 MCTS 搜索次数

## 测试步骤

### 前提条件
1. 确保有已训练的模型文件（例如 `banqi_model_latest.pt`）
2. 使用 `--features="torch"` 启用 torch 特性编译

### 编译
```bash
cargo build --bin banqi-tauri --release --features="torch"
```

### 运行
```bash
cargo run --bin banqi-tauri --release --features="torch"
```

### 前端测试流程
1. 启动应用后，调用 `list_models()` 查看可用模型
2. 调用 `load_model("banqi_model_latest.pt")` 加载模型
3. 调用 `reset_game("MctsDL")` 开始 AI 对战
4. 人类走子后，调用 `bot_move()` 让 AI 行动

### 配置调优
```javascript
// 设置 MCTS 搜索次数（默认 200）
await invoke('set_mcts_iterations', { iters: 100 });  // 快速但棋力较弱
await invoke('set_mcts_iterations', { iters: 400 });  // 较强但速度慢
```

## 训练流程回顾

### 1. 数据生成（Rust）
```bash
cargo run --release --bin banqi-data-collector
```
- 使用 Gumbel AlphaZero MCTS 进行自我对弈
- 生成训练样本存储到 MongoDB

### 2. 模型训练（Python）
```bash
python python/training_service.py
```
- 从 MongoDB 加载游戏数据
- 训练神经网络
- 保存 `.pth`（训练用）和 `.pt`（推理用）

### 3. 与 AI 对战（Tauri）
```bash
cargo run --bin banqi-tauri --release --features="torch"
```
- 加载 `.pt` 模型
- 使用 MCTS + 神经网络策略

## 架构说明

```
┌─────────────────────────────────────────────────────┐
│                   Hybrid Architecture                │
├─────────────────────────────────────────────────────┤
│                                                       │
│  ┌─────────────────┐      ┌──────────────────────┐  │
│  │   Rust (Self-   │      │  Python (Training)   │  │
│  │   Play & MCTS)  │      │  Service             │  │
│  │                 │      │                      │  │
│  │  • game_env.rs  │      │  • training_service  │  │
│  │  • mcts.rs      │◄─────┤  • nn_model.py       │  │
│  │  • self_play.rs │ .pt  │  • constant.py       │  │
│  │                 │      │                      │  │
│  │  ┌───────────┐  │      │  ┌────────────────┐  │  │
│  │  │ MongoDB   │◄─┼──────┼─►│ MongoDB        │  │  │
│  │  │ (Games)   │  │      │  │ (Fetch Games)  │  │  │
│  │  └───────────┘  │      │  └────────────────┘  │  │
│  └─────────────────┘      └──────────────────────┘  │
│           │                         │                │
│           │                         │                │
│           ▼                         ▼                │
│  ┌─────────────────────────────────────────────┐    │
│  │        banqi_model_traced.pt (TorchScript)  │    │
│  └─────────────────────────────────────────────┘    │
│           │                                          │
│           ▼                                          │
│  ┌─────────────────┐                                │
│  │  Tauri (UI)     │                                │
│  │                 │                                │
│  │  • tauri_main   │                                │
│  │  • MctsDlPolicy │                                │
│  │  • Frontend     │                                │
│  └─────────────────┘                                │
│                                                       │
└─────────────────────────────────────────────────────┘
```

## 常见问题

### Q: 编译时提示找不到 libtorch
A: 请确保安装了 PyTorch 并设置了 `LIBTORCH` 或 `TORCH_CUDA_VERSION` 环境变量

### Q: 模型加载失败
A: 检查：
1. 模型文件是否存在
2. 模型是否为 TorchScript 格式（`.pt`）
3. 模型的输入输出维度是否正确

### Q: AI 思考时间过长
A: 调用 `set_mcts_iterations` 减少搜索次数：
```javascript
await invoke('set_mcts_iterations', { iters: 50 });
```

### Q: AI 棋力不够强
A: 需要更多训练数据和更长时间的训练：
1. 运行更多自我对弈游戏
2. 增加训练轮次
3. 增加 MCTS 搜索次数
