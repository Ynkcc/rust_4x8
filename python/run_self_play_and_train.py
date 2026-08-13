"""
run_self_play_and_train.py

使用 maturin 编译出来的 PyO3 扩展 (banqi_4x8)，从 Python 侧驱动自对弈，
并把整局 GameEpisode 拿回来直接喂给训练循环。

两个主要步骤：
    Step 1. 创建模型 + 预测包装函数 (predictor)
    Step 2. 调用 banqi_4x8.run_self_play_with_predictor(predictor, ...)
            -> 每局 MCTS 叶子评估都回调 predictor(board, scalars)
            -> 返回 GameEpisode 列表 (Python 对象，可 to_dict() / get_samples())
    Step 3. 在 Python 侧保存训练数据，迭代训练模型

使用方法（首次需要先构建 PyO3 扩展）：
    1)  pip install maturin
    2)  maturin develop --features pyo3    # 把 banqi_4x8 装到当前 venv
    3)  python python/run_self_play_and_train.py
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple

import numpy as np

try:
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except ImportError:  # pragma: no cover
    HAS_TORCH = False

# maturin develop --features pyo3 后即可 import
import banqi_4x8

from constant import (
    ACTION_SPACE_SIZE,
    BOARD_CHANNELS,
    BOARD_COLS,
    BOARD_ROWS,
    SCALAR_FEATURE_COUNT,
)
from nn_model import BanqiNet
from predictor_entry import (
    estimate_memory_bytes,
    estimate_single_game_suspended,
    estimate_mcts_tree,
    estimate_game_state_suspended,
    estimate_episode_storage,
    print_memory_estimate_report,
    _sizeof_fmt,
)


# ============================================================================
# 1. PyTorch 模型 + 预测包装函数 (传给 Rust 做 MCTS 评估)
# ============================================================================

class Predictor:
    """
    薄包装：
        - 确保模型在 eval / no_grad
        - 输入/输出都是 numpy (Rust 侧转成 numpy 后传进来)
        - 简易模型热重载 (检查 MODEL_PATH 文件 mtime)
    """

    def __init__(self, model: "BanqiNet", device: "torch.device") -> None:
        self.model = model.to(device)
        self.device = device
        self.model_path: str | None = os.environ.get("MODEL_PATH")
        self._mtime: float = 0.0
        self.model.eval()
        self._maybe_reload_weights(force=True)

    def _maybe_reload_weights(self, force: bool = False) -> None:
        if not self.model_path or not os.path.exists(self.model_path):
            return
        mtime = os.path.getmtime(self.model_path)
        if force or mtime > self._mtime:
            try:
                st = torch.load(self.model_path, map_location=self.device, weights_only=True)
                self.model.load_state_dict(st)
                self.model.eval()
                self._mtime = mtime
                print(f"[Predictor] reloaded weights from {self.model_path}")
            except Exception as exc:  # pragma: no cover
                print(f"[Predictor] load weights FAILED: {exc}")

    def __call__(self, board: np.ndarray, scalars: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        self._maybe_reload_weights()

        with torch.no_grad():
            b = torch.from_numpy(np.ascontiguousarray(board)).to(self.device)
            s = torch.from_numpy(np.ascontiguousarray(scalars)).to(self.device)
            logits, value = self.model(b, s)
            return (
                logits.detach().cpu().numpy().astype(np.float32),
                value.detach().cpu().numpy().reshape(-1).astype(np.float32),
            )


# ============================================================================
# 2. 训练数据结构 & 存储
# ============================================================================

@dataclass
class ReplayBuffer:
    max_episodes: int = 5000
    episodes: List[Dict] = field(default_factory=list)

    def add(self, episode_dict: Dict) -> None:
        self.episodes.append(episode_dict)
        if len(self.episodes) > self.max_episodes:
            self.episodes = self.episodes[-self.max_episodes :]

    def __len__(self) -> int:
        return len(self.episodes)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "wb") as fp:
            pickle.dump(self.episodes, fp, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)
        print(f"[ReplayBuffer] saved {len(self)} episodes -> {path}")

    def load(self, path: str) -> None:
        if not os.path.exists(path):
            return
        with open(path, "rb") as fp:
            self.episodes = pickle.load(fp)
        print(f"[ReplayBuffer] loaded {len(self)} episodes from {path}")


def flatten_episodes_to_samples(
    episodes: List[Dict],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    把若干 episode_dict 展平成训练样本张量 (numpy)。
    返回:
        boards   (N, 16, 4, 8) float32
        scalars  (N, 35)      float32
        policies (N, 352)     float32  (improved_policy 作为 soft label)
        values   (N,)         float32  (MCTS value)
        game_outcome (N,)     float32  (整局结果，价值头训练目标)
        masks    (N, 352)     float32  (动作掩码)
    """
    all_boards: List[np.ndarray] = []
    all_scalars: List[np.ndarray] = []
    all_policies: List[np.ndarray] = []
    all_values: List[np.ndarray] = []
    all_outcomes: List[np.ndarray] = []
    all_masks: List[np.ndarray] = []

    for ep in episodes:
        boards = np.asarray(ep["boards"], dtype=np.float32)
        scalars = np.asarray(ep["scalars"], dtype=np.float32)
        policies = np.asarray(ep["policies"], dtype=np.float32)
        values = np.asarray(ep["mcts_values"], dtype=np.float32)
        outcomes = np.asarray(ep["game_results"], dtype=np.float32)
        masks = np.asarray(ep["action_masks"], dtype=np.float32)

        all_boards.append(boards)
        all_scalars.append(scalars)
        all_policies.append(policies)
        all_values.append(values)
        all_outcomes.append(outcomes)
        all_masks.append(masks)

    if not all_boards:
        empty = lambda *s: np.zeros(s, dtype=np.float32)
        return (
            empty(0, BOARD_CHANNELS, BOARD_ROWS, BOARD_COLS),
            empty(0, SCALAR_FEATURE_COUNT),
            empty(0, ACTION_SPACE_SIZE),
            empty(0),
            empty(0),
            empty(0, ACTION_SPACE_SIZE),
        )

    return (
        np.concatenate(all_boards, axis=0),
        np.concatenate(all_scalars, axis=0),
        np.concatenate(all_policies, axis=0),
        np.concatenate(all_values, axis=0),
        np.concatenate(all_outcomes, axis=0),
        np.concatenate(all_masks, axis=0),
    )


# ============================================================================
# 3. 训练循环 (纯 Python / PyTorch)
# ============================================================================

def train_one_epoch(
    model: "BanqiNet",
    optimizer: "torch.optim.Optimizer",
    data: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    batch_size: int,
    device: "torch.device",
    value_weight: float = 1.0,
    lr_scheduler: "torch.optim.lr_scheduler._LRScheduler | None" = None,
) -> Dict[str, float]:
    boards, scalars, policies, values, outcomes, masks = data

    if boards.shape[0] == 0:
        return {"policy_loss": 0.0, "value_loss": 0.0, "loss": 0.0}

    ds = TensorDataset(
        torch.from_numpy(boards),
        torch.from_numpy(scalars),
        torch.from_numpy(policies),
        torch.from_numpy(values),
        torch.from_numpy(outcomes),
        torch.from_numpy(masks),
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

    model.train()
    total_p = 0.0
    total_v = 0.0
    total = 0.0
    n_batches = 0

    for b_b, b_s, b_p, b_v, b_o, b_m in loader:
        b_b = b_b.to(device)
        b_s = b_s.to(device)
        b_p = b_p.to(device)
        b_o = b_o.to(device)
        b_m = b_m.to(device)

        logits, value_pred = model(b_b, b_s)

        # --- Policy: mask then cross-entropy against MCTS improved policy ---
        logits_masked = logits * b_m + (b_m - 1.0) * 1e9
        log_probs = F.log_softmax(logits_masked, dim=1)
        policy_loss = -(b_p * log_probs).sum(dim=1).mean()

        # --- Value: MSE against game outcome (with MCTS value as soft auxiliary) ---
        value_pred_flat = value_pred.reshape(-1)
        value_loss = F.mse_loss(value_pred_flat, b_o)
        _ = b_v  # reserved for future composite targets

        loss = policy_loss + value_weight * value_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_p += float(policy_loss.detach().cpu().item())
        total_v += float(value_loss.detach().cpu().item())
        total += float(loss.detach().cpu().item())
        n_batches += 1

    if lr_scheduler is not None:
        lr_scheduler.step()

    return {
        "policy_loss": total_p / max(n_batches, 1),
        "value_loss": total_v / max(n_batches, 1),
        "loss": total / max(n_batches, 1),
    }


# ============================================================================
# 4. 主循环：自对弈生成数据 + 训练
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--games-per-iter", type=int, default=10,
                        help="每次迭代自对弈多少局 (默认 10，示例偏小)")
    parser.add_argument("--mcts-sims", type=int, default=32,
                        help="MCTS 每次决策模拟次数 (默认 32)")
    parser.add_argument("--train-epochs", type=int, default=3,
                        help="每次迭代训练 epoch 数")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--iters", type=int, default=5,
                        help="总迭代数 (自对弈+训练 循环次数)")
    parser.add_argument("--output", type=str, default="./training_data/loop/")
    parser.add_argument("--weights", type=str, default=None,
                        help="模型权重 .pt 路径，同时也用于 Rust 回调热重载")
    parser.add_argument("--replay", type=str, default=None,
                        help="ReplayBuffer pickle 路径，用于增量累积")
    parser.add_argument("--worker-id", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    if not HAS_TORCH:
        raise SystemExit("PyTorch 未安装。此示例需要 PyTorch 才能训练模型。pip install torch")

    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )
    print(f"[Main] device = {device}")

    os.makedirs(args.output, exist_ok=True)
    weights_path = args.weights or os.path.join(args.output, "model_latest.pt")
    replay_path = args.replay or os.path.join(args.output, "replay_buffer.pkl")
    if args.weights is None:
        os.environ["MODEL_PATH"] = weights_path
    else:
        os.environ.setdefault("MODEL_PATH", weights_path)

    model = BanqiNet()
    if os.path.exists(weights_path):
        model.load_state_dict(
            torch.load(weights_path, map_location=device, weights_only=True)
        )
        print(f"[Main] 加载权重: {weights_path}")
    model = model.to(device)

    predictor = Predictor(model, device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.iters * args.train_epochs)
    )

    buffer = ReplayBuffer(max_episodes=5000)
    buffer.load(replay_path)

    # --- 准备 SelfPlayConfig (也可以直接用默认 None) ---
    sp_cfg = banqi_4x8.SelfPlayConfig(
        mcts_sims=args.mcts_sims,
        max_considered_actions=16,
        dirichlet_alpha=0.3,
        dirichlet_epsilon=0.25,
        temperature_steps=12,
    )
    print(
        f"[Main] SelfPlayConfig: mcts_sims={sp_cfg.mcts_sims}, "
        f"games/iter={args.games_per_iter}, iters={args.iters}"
    )
    print(f"[Main] Weights path: {weights_path}")
    print(f"[Main] Replay buffer: {replay_path}")

    # Banqi_4x8 常量确认
    print(
        f"[Main] banqi_4x8 constants: BOARD=({banqi_4x8.BOARD_CHANNELS},"
        f"{banqi_4x8.BOARD_ROWS},{banqi_4x8.BOARD_COLS}), "
        f"SCALAR={banqi_4x8.SCALAR_FEATURE_COUNT}, "
        f"ACTION={banqi_4x8.ACTION_SPACE_SIZE}"
    )

    # ----------- 内存估算: 挂起单局游戏 (游戏状态 + MCTS) -----------
    print_memory_estimate_report(
        mcts_sims=args.mcts_sims,
        games_per_iter=args.games_per_iter,
        num_workers=1,
    )

    # 程序化 API: 可用字节数估算（例如传给资源管理器）
    suggested_bytes_per_game = estimate_memory_bytes(
        mcts_sims=args.mcts_sims,
        expected_game_length=100,
        include_episode_storage=True,
        safety_factor=1.5,
    )
    suggested_total_bytes = suggested_bytes_per_game * args.games_per_iter
    print(
        f"[Memory Estimate] 单局挂起建议预留: {_sizeof_fmt(suggested_bytes_per_game)}"
        f"  ({_sizeof_fmt(suggested_total_bytes)}  for {args.games_per_iter} games/iter)"
    )

    # 估算 ReplayBuffer 的最大内存占用
    est_ep = estimate_episode_storage(100)
    buffer_max_bytes = est_ep.total_bytes * 5000
    print(
        f"[Memory Estimate] ReplayBuffer (5000 episodes max) ≈ "
        f"{_sizeof_fmt(int(buffer_max_bytes * 1.2))}  (×1.2 Python object overhead)"
    )
    # ----------------------------------------------------------------

    for it in range(args.iters):
        t0 = time.time()

        # ---------------- Step A: Self-play via Rust ----------------
        print(f"\n=== Iter {it}/{args.iters} - Self-Play ===")
        episodes: List[banqi_4x8.GameEpisode] = banqi_4x8.run_self_play_with_predictor(
            predict_fn=predictor,
            config=sp_cfg,
            num_games=args.games_per_iter,
            worker_id=args.worker_id,
        )

        # 保存原始 JSON (按 iter 归档) + 送入 ReplayBuffer
        iter_dir = os.path.join(args.output, f"iter_{it:04d}")
        os.makedirs(iter_dir, exist_ok=True)
        for idx, ep in enumerate(episodes):
            d = ep.to_dict()
            buffer.add(d)
            with open(os.path.join(iter_dir, f"game_{idx:04d}.json"), "w") as fp:
                json.dump(d, fp, ensure_ascii=False)
        buffer.save(replay_path)

        # ---------------- Step B: Train ----------------
        print(f"\n=== Iter {it}/{args.iters} - Train ===")
        data_tensors = flatten_episodes_to_samples(buffer.episodes)
        sample_count = data_tensors[0].shape[0]
        print(f"[Train] total samples in buffer = {sample_count}")

        if sample_count > 0:
            for epoch in range(args.train_epochs):
                metrics = train_one_epoch(
                    model, optimizer, data_tensors,
                    batch_size=args.batch_size, device=device,
                    value_weight=1.0,
                    lr_scheduler=(scheduler if epoch == args.train_epochs - 1 else None),
                )
                print(
                    f"[Train] epoch {epoch + 1}/{args.train_epochs}: "
                    f"policy_loss={metrics['policy_loss']:.4f}, "
                    f"value_loss={metrics['value_loss']:.4f}, "
                    f"total_loss={metrics['loss']:.4f}"
                )

            torch.save(model.state_dict(), weights_path + ".tmp")
            os.replace(weights_path + ".tmp", weights_path)
            print(f"[Train] 权重已保存: {weights_path}")
        else:
            print("[Train] 跳过训练：样本为空")

        print(f"[Main] Iter {it} 总耗时 {time.time() - t0:.1f}s")

    print("\n[Main] 全部迭代完成")


if __name__ == "__main__":
    main()
