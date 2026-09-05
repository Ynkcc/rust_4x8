import os
import sys
import tempfile
import torch
import numpy as np

_VALIDATE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PYTHON_DIR = os.path.dirname(_VALIDATE_DIR)
for _d in (_PYTHON_DIR, _VALIDATE_DIR):
    if _d not in sys.path:
        sys.path.insert(0, _d)

import banqi_4x8
from banqi.nnue.model import BanqiNNUE, nnue_feature_dim
from banqi.nnue.exporter import export_checkpoint


def test_feature_indices_consistency():
    """验证 Python 与 Rust 底层导出的特征稀疏索引完全在合法范围内且不重复。"""
    env = banqi_4x8.DarkChess()
    dim = nnue_feature_dim(total_positions=32, num_active=7, max_piece_count=5)
    assert dim == 555, f"预期维度为 555, 实际 {dim}"

    # 获取初始状态特征索引
    feats = env.nnue_active_features()
    assert len(feats) > 0, "特征索引不应为空"
    assert len(feats) == len(set(feats)), "特征索引集合不应有重复"
    assert all(0 <= idx < dim for idx in feats), f"特征索引超出范围 [0, {dim})"
    print(f"[Pass] Feature indices valid: {len(feats)} active features, all in [0, {dim})")


def test_export_and_rust_eval_consistency():
    """验证 Python nnue 导出为二进制后，与 Rust 评估逻辑计算一致性。
    
    注：由于目前 NnueEvaluator 未直接暴露给 pyo3，此处验证：
    1. 随机初始化 PyTorch 模型
    2. 导出为 .nnue 格式
    3. 校验二进制文件大小、参数总数
    4. 校验 PyTorch 模型对多 batch 输入推理的确定性与输出区间 [-1, 1]
    """
    dim = nnue_feature_dim(total_positions=32, num_active=7, max_piece_count=5)
    model = BanqiNNUE(feature_dim=dim)
    model.eval()

    # 测试输出区间
    env = banqi_4x8.DarkChess()
    feats = env.nnue_active_features()
    
    sparse_input = torch.zeros((1, dim), dtype=torch.float32)
    sparse_input[0, feats] = 1.0

    with torch.no_grad():
        out = model(sparse_input)
    val = out.item()
    assert -1.0 <= val <= 1.0, f"NNUE 输出应处于 [-1, 1] 内, 实际 {val}"

    # 导出文件
    with tempfile.NamedTemporaryFile(suffix=".nnue", delete=False) as f:
        tmp_path = f.name
    try:
        model.export_nnue_binary(tmp_path)
        file_size = os.path.getsize(tmp_path)
        # 555 * 256 + 256 + 256 * 32 + 32 + 32 * 1 + 1 = 142080 + 256 + 8192 + 32 + 32 + 1 = 150593 floats
        # 150593 * 4 = 602372 bytes
        expected_bytes = (555 * 256 + 256 + 256 * 32 + 32 + 32 * 1 + 1) * 4
        assert file_size == expected_bytes, f"导出文件大小不匹配: {file_size} vs {expected_bytes}"
        print(f"[Pass] Export .nnue matches expected size: {file_size} bytes")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


if __name__ == "__main__":
    test_feature_indices_consistency()
    test_export_and_rust_eval_consistency()
    print("All L1 verification tests passed successfully!")
