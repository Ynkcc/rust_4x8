#!/usr/bin/env bash
# =============================================================================
# Python gRPC 代码生成脚本
# =============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

PROTO_SRC="${PROJECT_ROOT}/proto/banqi_service.proto"
PYTHON_BIN="/home/ynk/miniconda3/bin/python"

if [ ! -f "${PROTO_SRC}" ]; then
    PROTO_SRC="${SCRIPT_DIR}/banqi_service.proto"
fi

echo "🔄 Generating Python gRPC code from ${PROTO_SRC}..."

"${PYTHON_BIN}" -m grpc_tools.protoc \
    -I"$(dirname "${PROTO_SRC}")" \
    --python_out="${SCRIPT_DIR}" \
    --grpc_python_out="${SCRIPT_DIR}" \
    "${PROTO_SRC}"

# 保持本地一份 .proto 备份
if [ "${PROTO_SRC}" != "${SCRIPT_DIR}/banqi_service.proto" ] && [ -f "${PROJECT_ROOT}/proto/banqi_service.proto" ]; then
    cp "${PROJECT_ROOT}/proto/banqi_service.proto" "${SCRIPT_DIR}/banqi_service.proto"
fi

# 修复 banqi_service_pb2_grpc.py 中的绝对导入问题
sed -i 's/import banqi_service_pb2 as banqi__service__pb2/from . import banqi_service_pb2 as banqi__service__pb2/g' "${SCRIPT_DIR}/banqi_service_pb2_grpc.py"

echo "✅ Generated successfully in ${SCRIPT_DIR}"
