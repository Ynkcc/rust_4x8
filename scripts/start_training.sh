#!/bin/bash
# 一键启动训练系统

set -e

echo "🚀 Banqi 4x8 训练系统启动脚本"
echo "================================"

# 配置参数
INFERENCE_PORT=50051
TRAINING_PORT=50052
MODEL_PATH="banqi_model_latest.pt"
WORKERS=32
EPISODES=10
MCTS_SIMS=200
BATCH_SIZE=64
BUFFER_SIZE=1000
LEARNING_RATE=0.0001

# 检查 Python 虚拟环境
if [ ! -d "venv" ]; then
    echo "❌ 未找到 venv 虚拟环境"
    echo "   请运行: python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

# 检查 gRPC 代码是否生成
if [ ! -f "src/banqi_pb2.py" ]; then
    echo "⚙️  生成 Python gRPC 代码..."
    source venv/bin/activate
    python -m grpc_tools.protoc \
        -I proto \
        --python_out=src \
        --grpc_python_out=src \
        proto/banqi.proto
    echo "✅ gRPC 代码生成完成"
fi

# 检查 Rust 是否编译
if [ ! -f "target/release/banqi-data-collector" ]; then
    echo "⚙️  编译 Rust 项目..."
    cargo build --release --bin banqi-data-collector
    echo "✅ Rust 编译完成"
fi

# 创建日志目录
mkdir -p logs

# 清理旧进程
echo "🧹 清理旧进程..."
pkill -f "inference_service.py" || true
pkill -f "training_service.py" || true
pkill -f "banqi-data-collector" || true
sleep 2

# 启动推理服务
echo ""
echo "📊 启动推理服务 (端口 $INFERENCE_PORT)..."
source venv/bin/activate
nohup python src/inference_service.py \
    --model "$MODEL_PATH" \
    --host "[::]:$INFERENCE_PORT" \
    --batch-size $BATCH_SIZE \
    --timeout 5 \
    > logs/inference.log 2>&1 &
INFERENCE_PID=$!
echo "   PID: $INFERENCE_PID"

# 等待推理服务启动
sleep 3

# 启动训练服务
echo ""
echo "🎓 启动训练服务 (端口 $TRAINING_PORT)..."
nohup python src/training_service.py \
    --model "$MODEL_PATH" \
    --host "[::]:$TRAINING_PORT" \
    --buffer $BUFFER_SIZE \
    --lr $LEARNING_RATE \
    > logs/training.log 2>&1 &
TRAINING_PID=$!
echo "   PID: $TRAINING_PID"

# 等待训练服务启动
sleep 3

# 启动数据收集
echo ""
echo "♟️  启动数据收集器 ($WORKERS workers)..."
echo "   MCTS 模拟次数: $MCTS_SIMS"
echo "   每轮游戏数: $EPISODES"
echo ""

# 记录 PID
echo $INFERENCE_PID > logs/inference.pid
echo $TRAINING_PID > logs/training.pid

./target/release/banqi-data-collector \
    --inference-addr "http://127.0.0.1:$INFERENCE_PORT" \
    --training-addr "http://127.0.0.1:$TRAINING_PORT" \
    --workers $WORKERS \
    --episodes $EPISODES \
    --mcts-sims $MCTS_SIMS \
    2>&1 | tee logs/data_collector.log

# 数据收集结束后的清理
echo ""
echo "✅ 数据收集完成"
echo ""
echo "📌 服务仍在运行:"
echo "   推理服务 PID: $INFERENCE_PID"
echo "   训练服务 PID: $TRAINING_PID"
echo ""
echo "💡 提示:"
echo "   - 查看推理日志: tail -f logs/inference.log"
echo "   - 查看训练日志: tail -f logs/training.log"
echo "   - 停止所有服务: ./scripts/stop_training.sh"
echo "   - 查看训练指标: cat training_log.csv"
