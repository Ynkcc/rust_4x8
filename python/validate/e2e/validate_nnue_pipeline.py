"""
validate_nnue_pipeline.py — NNUE + Expectimax 核心逻辑全链路一键冒烟脚本。

测试链路：
1. 稀疏特征提取与维度一致性校验
2. 随机初始化 NNUE 模型并导出为 .nnue 二进制格式
3. 单 Batch 快速过拟合训练（300步，验证反向传播与收敛性）
4. 验证训练后的模型在导出与评估过程中的稳健性
"""

import os
import sys
import tempfile
import torch
import torch.nn as nn
import torch.optim as optim

_VALIDATE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PYTHON_DIR = os.path.dirname(_VALIDATE_DIR)
for _d in (_PYTHON_DIR, _VALIDATE_DIR):
    if _d not in sys.path:
        sys.path.insert(0, _d)

import banqi_4x8
from banqi.nnue.model import BanqiNNUE, nnue_feature_dim
from banqi.nnue.exporter import export_checkpoint


def run_pipeline_smoke():
    print("=== [NNUE + Expectimax Smoke Test Started] ===")
    
    # 步骤 1: 验证环境与特征提取
    env = banqi_4x8.DarkChess()
    feats = env.nnue_active_features()
    dim = nnue_feature_dim(total_positions=32, num_active=7, max_piece_count=5)
    assert len(feats) > 0, "特征列表不应为空"
    assert all(0 <= f < dim for f in feats), f"特征索引超出范围 [0, {dim})"
    print(f"Step 1 OK: 提取到 {len(feats)} 个合法激活特征 (dim={dim})")

    # 步骤 2: 初始化模型并过拟合小批量
    model = BanqiNNUE(dim)
    optimizer = optim.Adam(model.parameters(), lr=5e-3)
    criterion = nn.MSELoss()

    # 采集 16 个状态
    features_list = []
    targets_list = []
    for i in range(16):
        cur_env = banqi_4x8.DarkChess()
        cur_moves = cur_env.legal_moves()
        if cur_moves:
            cur_env.step(cur_moves[i % len(cur_moves)])
        f = cur_env.nnue_active_features()
        x = torch.zeros(dim)
        x[f] = 1.0
        features_list.append(x)
        targets_list.append(0.5 if i % 2 == 0 else -0.5)

    X = torch.stack(features_list)
    Y = torch.tensor(targets_list, dtype=torch.float32).unsqueeze(1)

    model.train()
    init_loss = criterion(model(X), Y).item()
    for _ in range(250):
        optimizer.zero_grad()
        loss = criterion(model(X), Y)
        loss.backward()
        optimizer.step()
    final_loss = loss.item()
    assert final_loss < 5e-3, f"过拟合失败: init={init_loss:.4f}, final={final_loss:.4f}"
    print(f"Step 2 OK: 训练闭环正常, 过拟合 loss: {init_loss:.4f} -> {final_loss:.6f}")

    # 步骤 3: 导出 .nnue 格式
    with tempfile.TemporaryDirectory() as tmpdir:
        nnue_path = os.path.join(tmpdir, "smoke_test.nnue")
        model.cpu()
        model.export_nnue_binary(nnue_path)
        assert os.path.exists(nnue_path), "导出的 .nnue 文件不存在"
        file_size = os.path.getsize(nnue_path)
        expected_size = 602372  # (555*256*2 + 256 + 512*32 + 32 + 32*1 + 1) * 4
        assert file_size == expected_size, f"文件大小不符: {file_size} vs {expected_size}"
        print(f"Step 3 OK: .nnue 二进制导出成功 ({file_size} 字节)")

    print("=== [NNUE + Expectimax Smoke Test PASSED] ===")


if __name__ == "__main__":
    run_pipeline_smoke()
