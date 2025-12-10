#!/bin/bash
# 停止所有训练服务

echo "🛑 停止训练系统..."

# 从 PID 文件停止
if [ -f "logs/inference.pid" ]; then
    INFERENCE_PID=$(cat logs/inference.pid)
    echo "停止推理服务 (PID: $INFERENCE_PID)..."
    kill $INFERENCE_PID 2>/dev/null || true
    rm logs/inference.pid
fi

if [ -f "logs/training.pid" ]; then
    TRAINING_PID=$(cat logs/training.pid)
    echo "停止训练服务 (PID: $TRAINING_PID)..."
    kill $TRAINING_PID 2>/dev/null || true
    rm logs/training.pid
fi

# 强制清理
pkill -f "inference_service.py" || true
pkill -f "training_service.py" || true
pkill -f "banqi-data-collector" || true

echo "✅ 所有服务已停止"
