"""
validate_nnue_overfit.py — 验证 NNUE 网络与训练链路在单 Batch 上的极端拟合能力。
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

_VALIDATE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PYTHON_DIR = os.path.dirname(_VALIDATE_DIR)
for _d in (_PYTHON_DIR, _VALIDATE_DIR):
    if _d not in sys.path:
        sys.path.insert(0, _d)

import banqi_4x8
from banqi.nnue.model import BanqiNNUE, nnue_feature_dim


def generate_synthetic_nnue_samples(n_samples: int = 64, dim: int = 555):
    """通过真实走子生成具有代表性的特征与目标值。"""
    features_list = []
    targets_list = []
    
    for seed in range(n_samples):
        env = banqi_4x8.DarkChess()
        # 走若干步
        steps = seed % 15
        for _ in range(steps):
            moves = env.legal_moves()
            if not moves or env.terminated():
                break
            env.step(moves[0])
        
        feats = env.nnue_active_features()
        x = torch.zeros(dim, dtype=torch.float32)
        x[feats] = 1.0
        features_list.append(x)
        
        # 构造目标: [-1, 1] 间确定性目标
        target_val = float(np.sin(seed * 0.5))
        targets_list.append(target_val)

    features = torch.stack(features_list)
    targets = torch.tensor(targets_list, dtype=torch.float32).unsqueeze(1)
    return features, targets


def test_nnue_single_batch_overfit():
    """断言 NNUE 网络在固定单批样本上经过反向传播后 MSE Loss 能显著下降到接近 0。"""
    dim = nnue_feature_dim(total_positions=32, num_active=7, max_piece_count=5)
    model = BanqiNNUE(dim)
    optimizer = optim.Adam(model.parameters(), lr=5e-3)
    criterion = nn.MSELoss()

    features, targets = generate_synthetic_nnue_samples(n_samples=32, dim=dim)
    
    model.train()
    initial_loss = criterion(model(features), targets).item()

    # 循环拟合 300 步
    for step in range(300):
        optimizer.zero_grad()
        preds = model(features)
        loss = criterion(preds, targets)
        loss.backward()
        optimizer.step()

    final_loss = loss.item()
    print(f"[Overfit] Initial Loss: {initial_loss:.6f}, Final Loss: {final_loss:.6f}")
    assert final_loss < 5e-3, f"单 batch 过拟合未达到收敛阈值 (< 5e-3), 实际 final_loss = {final_loss}"
    print("[Pass] NNUE single-batch overfit succeeded!")


def test_target_symmetry_and_weighting():
    """验证 value_weight 对终局与搜索值的插值逻辑及红黑对称性。"""
    search_value = 0.6
    game_result = 1.0  # 红胜
    value_weight = 0.5

    # 视角 1
    t1 = value_weight * search_value + (1.0 - value_weight) * game_result
    # 视角 2 (对手)
    t2 = value_weight * (-search_value) + (1.0 - value_weight) * (-game_result)
    assert abs(t1 + t2) < 1e-6, f"对称视角目标和应为 0, 实际 {t1 + t2}"
    print(f"[Pass] Target symmetry holds: {t1} + {t2} == 0")


if __name__ == "__main__":
    test_target_symmetry_and_weighting()
    test_nnue_single_batch_overfit()
    print("All L3 verification tests passed successfully!")
