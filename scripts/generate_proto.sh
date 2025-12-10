#!/bin/bash
# 生成 gRPC 代码（Python 和 Rust）

set -e

echo "🔧 生成 gRPC 代码..."

# Python
echo "📦 生成 Python gRPC 代码..."
if [ ! -d "venv" ]; then
    echo "❌ 未找到虚拟环境，请先运行: python3 -m venv venv"
    exit 1
fi

source venv/bin/activate
python -m grpc_tools.protoc \
    -I proto \
    --python_out=python/generated \
    --grpc_python_out=python/generated \
    proto/banqi.proto

echo "✅ Python 代码生成: python/generated/banqi_pb2.py, python/generated/banqi_pb2_grpc.py"
# Rust (通过 cargo build 自动生成)
echo ""
echo "🦀 生成 Rust gRPC 代码..."
cargo build

echo "✅ Rust 代码生成: target/debug/build/.../banqi.rs"
echo ""
echo "✨ 完成！可以开始使用 gRPC 服务了"
