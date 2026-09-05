import os
import sys
import json
import time
import random
import tempfile
import torch
import numpy as np

_VALIDATE_DIR = os.path.dirname(os.path.abspath(__file__))
_PYTHON_DIR = os.path.dirname(_VALIDATE_DIR)
for _d in (_PYTHON_DIR, _VALIDATE_DIR):
    if _d not in sys.path:
        sys.path.insert(0, _d)

import banqi_4x8
from banqi.nnue.model import BanqiNNUE, nnue_feature_dim
from banqi.nnue.train import NnueSampleDataset, train_nnue


def main():
    print("=== [NNUE + Expectimax 端到端冒烟测试启动] ===")
    t_start = time.time()
    feature_dim = nnue_feature_dim(total_positions=32, num_active=7, max_piece_count=5)
    workdir = tempfile.mkdtemp(prefix="nnue_smoke_")

    # 1. 极小自对弈 (N=8, sims=16, collect_nnue_features=True)
    print("\n[Step 1/5] 执行极小自对弈 (N=8, sims=16, collect_nnue_features=True)...")
    def dummy_predict(obs, scalar):
        batch_size = len(obs)
        return np.ones((batch_size, 352), dtype=np.float32) / 352.0, np.zeros((batch_size, 1), dtype=np.float32)

    cfg = banqi_4x8.SelfPlayConfig(mcts_sims=16, collect_nnue_features=True)
    episodes = banqi_4x8.run_python_match(dummy_predict, cfg, num_games=8, variant_id="4x8")

    jsonl_path = os.path.join(workdir, "smoke_selfplay.jsonl")
    total_steps = 0
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for ep in episodes:
            d = ep.to_dict()
            f.write(json.dumps(d) + "\n")
            total_steps += len(d.get("boards", []))
    print(f"  自对弈完成: 生成 {len(episodes)} 个对局, 收集到 {total_steps} 步数据")

    # 2. nnue/train.py 过拟合这批数据
    print("\n[Step 2/5] 使用 train_nnue 过拟合这批数据...")
    ds = NnueSampleDataset([jsonl_path])
    print(f"  有效 NNUE 样本数: {len(ds)}")
    assert len(ds) > 0, "必须收集到有效样本"

    nnue_path = os.path.join(workdir, "model.nnue")
    # 3. export .nnue (train_nnue 内部训练完后会自动导出到 output_nnue)
    print("\n[Step 2-3/5] 训练并自动导出 .nnue 二进制模型...")
    model = train_nnue(
        dataset=ds,
        epochs=5,
        batch_size=32,
        lr=0.01,
        output_nnue=nnue_path,
    )
    assert os.path.exists(nnue_path), ".nnue 文件必须生成"
    print(f"  训练并导出完成: {nnue_path}, 尺寸: {os.path.getsize(nnue_path)} 字节")

    # 4. Rust 加载 → 断言 Python / Rust 评估误差 < 1e-5
    print("\n[Step 4/5] Rust 加载并验证 Python/Rust 评估一致性 (误差 < 1e-5)...")
    env = banqi_4x8.DarkChess()
    rust_val = env.nnue_evaluate(nnue_path)

    # 提取特征转为 Python 稀疏张量并 forward
    feat_indices = env.nnue_active_features()
    feat_tensor = torch.zeros(1, feature_dim, dtype=torch.float32)
    feat_tensor[0, feat_indices] = 1.0
    with torch.no_grad():
        py_val = model(feat_tensor).item()

    err = abs(rust_val - py_val)
    print(f"  Rust 评估值: {rust_val:.6f}, Python 评估值: {py_val:.6f}, 误差: {err:.2e}")
    assert err < 1e-5, f"Python/Rust 推理误差超标: {err} >= 1e-5"
    print("  评估一致性断言通过!")

    # 5. expectimax vs random 对局 20 局 → 断言胜率 > 70%
    print("\n[Step 5/5] Expectimax vs Random 对局 20 局 (限制节点预算保障速度)...")
    num_games = 20
    exp_wins = 0
    draws = 0

    for i in range(num_games):
        game_env = banqi_4x8.DarkChess()
        # 轮流执红
        exp_player = 1 if i % 2 == 0 else -1
        step_cnt = 0
        while not game_env.terminated() and step_cnt < 200:
            cur = game_env.current_player()
            legal = game_env.legal_moves()
            if not legal:
                break
            if cur == exp_player:
                # Expectimax: depth 2, node_budget 600
                a = game_env.expectimax_action(nnue_path, max_depth=2, node_budget=600)
                if a is None or a not in legal:
                    a = random.choice(legal)
            else:
                # 随机走子
                a = random.choice(legal)
            game_env.step(a)
            step_cnt += 1

        w = game_env.winner()
        if w == exp_player:
            exp_wins += 1
        elif w == 0:
            draws += 1
        print(f"  第 {i+1:02d}/20 局结束: 胜者={w}, Expectimax执方={exp_player}, 总胜场={exp_wins}")

    win_rate = exp_wins / num_games
    print(f"\n对局统计: 胜场={exp_wins}, 平局={draws}, 胜率={win_rate:.1%}")
    assert win_rate >= 0.70, f"Expectimax 对随机走子胜率需 >= 70%, 实际: {win_rate:.1%}"

    # 6. Expectimax + NNUE 自对弈回环：用训练好的 .nnue 生成 NNUE 专属 JSONL
    #    并验证可被 NnueSampleDataset 直接消费（契约一致性）
    print("\n[Step 6/6] Expectimax + NNUE 自对弈 (N=4, workers=2) → NNUE JSONL...")
    exp_jsonl = os.path.join(workdir, "smoke_expectimax_selfplay.jsonl")
    stats = banqi_4x8.run_expectimax_self_play(
        nnue_path,
        n_games=4,
        num_workers=2,
        node_budget=2000,
        max_depth=2,
        seed=42,
        out_jsonl=exp_jsonl,
    )
    print(f"  自对弈完成: {stats['games']} 局, 共 {stats['steps']} 步 "
          f"(A胜={stats['a_wins']}, B胜={stats['b_wins']}, 平={stats['draws']})")
    assert stats["games"] == 4, "Expectimax 自对弈局数不符"

    exp_ds = NnueSampleDataset([exp_jsonl])
    print(f"  NNUE JSONL 有效样本数: {len(exp_ds)}")
    assert len(exp_ds) > 0, "Expectimax 自对弈必须产出可消费的 NNUE 样本"
    x0, y0 = exp_ds[0]
    assert x0.shape[0] == feature_dim, "NNUE 样本特征维度与模型布局不符"
    print("  NNUE 数据契约验证通过!")

    print(f"\n=== [全部验证通过! 总耗时: {time.time() - t_start:.1f}s] ===")


if __name__ == "__main__":
    main()
