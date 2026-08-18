"""backtest_health.py — 用冷存储数据回测「带血量差异头」新网络 vs 现有 BanqiNet

公平对比设计（唯一变量 = 是否有血量差异头）：
  - 两个模型都从**随机初始化**开始（现有 RL 模型训练不足，直接舍弃）
  - 相同数据：training_data/archive_4x4 冷存储（注意：数据由未充分训练的
    模型生成，质量有限，但作为回测对照是公平的——两模型用同一份）
  - 相同划分：按局随机 95/5 分 train/val（固定种子，保证两模型同训练/验证集）
  - 相同超参：优化器 / lr / batch / epochs / weight_decay / 随机种子
  - BanqiNetHealth 额外多一个 health loss：MSE(预测终局血量差, 冷存储标签)

输出：
  - 每 epoch train/val 的 policy / value / health loss 对比
  - 验证集上 health 头预测质量：MSE / MAE / Pearson 相关系数（回测核心指标）
  - 保存两个模型 checkpoint（.pth + .pt TorchScript）
  - 可选（--eval-games>0）：统一协议棋力评估 vs minimax3 / heuristic64 / 对头

用法：
    python python/game_4x4/backtest_health.py --epochs 6 --batch 256 \
        --eval-games 40
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch

torch.set_num_threads(int(os.getenv("G4X4_TORCH_THREADS", "4")))

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.append(os.path.dirname(_HERE))  # python/

from banqi.constants import build_constants  # noqa: E402
from banqi.nn_model import BanqiNet  # noqa: E402
from banqi.nn_model_health import BanqiNetHealth, count_params  # noqa: E402
from banqi.storage import iter_jsonl_episodes, episode_dict_to_samples  # noqa: E402
from banqi.variant import get_variant  # noqa: E402

VARIANT = get_variant("4x4")
C = build_constants(VARIANT)

# 默认冷存储目录（项目根下 training_data/archive_4x4）
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_HERE))
DEFAULT_ARCHIVE = os.path.normpath(
    os.path.join(_PROJECT_ROOT, "training_data", "archive_4x4")
)


# ============================================================================
# 数据加载
# ============================================================================

def load_dataset(archive_dir: str, val_frac: float = 0.05, seed: int = 42):
    """流式加载冷存储 episode → 样本，按局划分 train/val（固定种子）。

    返回 (train_dict, val_dict)，dict 含 numpy 数组：
      boards (N,16,4,4), scalars (N,19), probs (N,112), values (N,),
      masks (N,112), healths (N,)
    """
    episodes = []
    t0 = time.time()
    for ep in iter_jsonl_episodes(archive_dir):
        episodes.append(ep)
    n_games = len(episodes)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_games)
    n_val = max(1, int(round(n_games * val_frac)))

    def _collect(indices):
        boards, scalars, probs, values, masks, healths = [], [], [], [], [], []
        for gi in indices:
            for s in episode_dict_to_samples(episodes[gi]):
                boards.append(np.array(s["board_state"], dtype=np.float32).reshape(
                    C.TOTAL_INPUT_CHANNELS, C.BOARD_ROWS, C.BOARD_COLS))
                scalars.append(np.array(s["scalar_state"], dtype=np.float32)[:C.SCALAR_FEATURE_COUNT])
                probs.append(np.array(s["policy_probs"], dtype=np.float32))
                values.append(float(s["mcts_value"]))
                masks.append(np.array(s["action_mask"], dtype=np.float32))
                healths.append(float(s.get("health_diff", 0.0)))
        return {
            "boards": np.stack(boards), "scalars": np.stack(scalars),
            "probs": np.stack(probs), "values": np.asarray(values, dtype=np.float32),
            "masks": np.stack(masks), "healths": np.asarray(healths, dtype=np.float32),
        }

    train = _collect(perm[n_val:])
    val = _collect(perm[:n_val])
    print(f"[Data] 冷存储 {n_games} 局 → 训练 {len(train['boards'])} 样本 / "
          f"验证 {len(val['boards'])} 样本（耗时 {time.time()-t0:.1f}s）", flush=True)
    print(f"[Data] health 标签: train mean={train['healths'].mean():.3f} std={train['healths'].std():.3f} "
          f"| val mean={val['healths'].mean():.3f} std={val['healths'].std():.3f}", flush=True)
    return train, val


def to_tensors(d, device):
    return (
        torch.from_numpy(d["boards"]).to(device),
        torch.from_numpy(d["scalars"]).to(device),
        torch.from_numpy(d["probs"]).to(device),
        torch.from_numpy(d["values"]).view(-1, 1).to(device),
        torch.from_numpy(d["masks"]).to(device),
        torch.from_numpy(d["healths"]).view(-1, 1).to(device),
    )


# ============================================================================
# 训练
# ============================================================================

def forward_loss(model, boards, scalars, probs, values, masks, healths, health_weight=1.0):
    """统一前向 + 损失。兼容 BanqiNet(2 输出) / BanqiNetHealth(3 输出)。"""
    out = model(boards, scalars)
    if len(out) == 3:
        logits, value, health = out
    else:
        logits, value = out
        health = None

    masked = logits + (masks - 1.0) * 1e9
    log_probs = torch.log_softmax(masked, dim=1)
    policy_loss = -(probs * log_probs).sum(dim=1).mean()
    value_loss = torch.nn.functional.mse_loss(value, values)
    health_loss = torch.tensor(0.0, device=boards.device)
    if health is not None:
        health_loss = torch.nn.functional.mse_loss(health, healths)
    total = policy_loss + value_loss + health_weight * health_loss
    return total, policy_loss, value_loss, health_loss


def evaluate(model, tensors, device, health_weight=1.0):
    """在给定数据集上评估（不更新梯度）。返回 (total, pol, val, health)。"""
    boards, scalars, probs, values, masks, healths = tensors
    model.eval()
    with torch.inference_mode():
        total, pol, val, h = forward_loss(
            model, boards, scalars, probs, values, masks, healths, health_weight)
    model.train()
    return float(total), float(pol), float(val), float(h)


def health_metrics(model, tensors):
    """health 头预测质量：MSE / MAE / Pearson 相关系数（验证集）。"""
    boards, scalars, _, _, _, healths = tensors
    model.eval()
    with torch.inference_mode():
        out = model(boards, scalars)
        pred = out[2].cpu().numpy().reshape(-1) if len(out) == 3 else None
    model.train()
    if pred is None:
        return None
    gt = healths.cpu().numpy().reshape(-1)
    mse = float(np.mean((pred - gt) ** 2))
    mae = float(np.mean(np.abs(pred - gt)))
    corr = float(np.corrcoef(pred, gt)[0, 1]) if np.std(pred) > 1e-9 and np.std(gt) > 1e-9 else 0.0
    return mse, mae, corr


def train_one(model, name, train_t, val_t, epochs, batch, lr, wd, device,
              health_weight, seed):
    """训练单个模型，返回每 epoch 记录。"""
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    boards, scalars, probs, values, masks, healths = train_t
    n = len(boards)
    rng = np.random.default_rng(seed)
    history = []
    for epoch in range(epochs):
        idx = rng.permutation(n)
        tl = pl = vl = hl = 0.0
        n_batch = 0
        t0 = time.time()
        for i in range(0, n, batch):
            sel = idx[i:i + batch]
            opt.zero_grad()
            total, pol, val, h = forward_loss(
                model, boards[sel], scalars[sel], probs[sel], values[sel],
                masks[sel], healths[sel], health_weight)
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            tl += float(total); pl += float(pol)
            vl += float(val); hl += float(h)
            n_batch += 1
        tl /= n_batch; pl /= n_batch; vl /= n_batch; hl /= n_batch
        vt, vp, vv, vh = evaluate(model, val_t, device, health_weight)
        hm = health_metrics(model, val_t)
        rec = dict(epoch=epoch, train_loss=tl, train_pol=pl, train_val=vl, train_health=hl,
                   val_loss=vt, val_pol=vp, val_val=vv, val_health=vh, hm=hm,
                   secs=time.time() - t0)
        history.append(rec)
        hm_s = f" | health(MSE/MAE/corr)={hm[0]:.4f}/{hm[1]:.4f}/{hm[2]:.4f}" if hm else ""
        print(f"[{name}] epoch{epoch}: train L={tl:.4f}(pol {pl:.4f}, val {vl:.4f}, "
              f"hp {hl:.4f}) | val L={vt:.4f}(pol {vp:.4f}, val {vv:.4f}, hp {vh:.4f})"
              f"{hm_s} | {time.time()-t0:.0f}s", flush=True)
    return history


def save_model(model, tag: str, out_dir: str):
    """保存 .pth（state_dict + 元数据）+ .pt（TorchScript 供推理）。"""
    os.makedirs(out_dir, exist_ok=True)
    pth = os.path.join(out_dir, f"banqi4x4_{tag}_backtest.pth")
    pt = os.path.join(out_dir, f"banqi4x4_{tag}_backtest.pt")
    model.eval()
    torch.save({
        "model_state_dict": model.state_dict(),
        "variant": "4x4",
        "arch": model.__class__.__name__,
    }, pth)
    with torch.inference_mode():
        dev = next(model.parameters()).device
        b = torch.randn(1, C.TOTAL_INPUT_CHANNELS, C.BOARD_ROWS, C.BOARD_COLS, device=dev)
        s = torch.randn(1, C.SCALAR_FEATURE_COUNT, device=dev)
        traced = torch.jit.trace(model, (b, s))
        traced.save(pt)
    print(f"[Save] {model.__class__.__name__} → {pth} + {pt}", flush=True)
    return pth, pt


def load_model_any(path: str, device):
    """加载模型（兼容 .pth state_dict / .pt TorchScript），返回 (model, arch)。"""
    if path.endswith(".pth"):
        state = torch.load(path, map_location=device)
        arch = state.get("arch", "BanqiNet")
        model = (BanqiNetHealth if arch == "BanqiNetHealth" else BanqiNet)(VARIANT).to(device)
        model.load_state_dict(state["model_state_dict"])
    else:
        jit = torch.jit.load(path, map_location=device)
        # 通过 state_dict 键判断架构
        keys = set(jit.state_dict().keys())
        arch = "BanqiNetHealth" if any("health" in k for k in keys) else "BanqiNet"
        model = (BanqiNetHealth if arch == "BanqiNetHealth" else BanqiNet)(VARIANT).to(device)
        model.load_state_dict(jit.state_dict())
    return model, arch


# ============================================================================
# 棋力评估（复用 eval_common 统一协议）
# ============================================================================

class PredictorCompat:
    """兼容 eval_common 的 predictor：从 2/3 输出模型中取 (logits, value)。"""

    def __init__(self, model, device):
        self.model = model.to(device).eval()
        self.device = device

    def __call__(self, boards, scalars):
        with torch.inference_mode():
            b = torch.from_numpy(np.ascontiguousarray(boards)).to(self.device)
            s = torch.from_numpy(np.ascontiguousarray(scalars)).to(self.device)
            out = self.model(b, s)
            logits, value = out[0], out[1]
            return (logits.cpu().numpy().astype(np.float32),
                    value.cpu().numpy().reshape(-1).astype(np.float32))


def eval_strength(model_path, tag, games, device, sims=64):
    from banqi.eval import report, report_vs
    import banqi_4x8  # noqa: F401
    model, arch = load_model_any(model_path, device)
    pred = PredictorCompat(model, device)
    print(f"\n[Eval] {tag} ({arch}): vs minimax3 n={games}", flush=True)
    report(pred, f"{tag}_mm3", n=games, model_sims=sims, opponent="minimax3")
    print(f"[Eval] {tag}: vs heuristic64 n={games}", flush=True)
    report(pred, f"{tag}_h64", n=games, model_sims=sims, opponent="heuristic64")
    return pred


def eval_head_to_head(base_model_path, health_model_path, games, device, sims=64):
    from banqi.eval import report_vs
    import banqi_4x8  # noqa: F401
    m_base, _ = load_model_any(base_model_path, device)
    m_health, _ = load_model_any(health_model_path, device)
    p_base = PredictorCompat(m_base, device)
    p_health = PredictorCompat(m_health, device)
    print(f"\n[Eval] 对头: Health(新) vs Base(对照) n={games}", flush=True)
    report_vs(p_health, p_base, "health_vs_base", n=games, model_sims=sims)


# ============================================================================
# main
# ============================================================================

def main():
    ap = argparse.ArgumentParser(description="冷存储回测：血量差异头 vs 现有结构")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--health-weight", type=float, default=1.0,
                    help="health loss 权重（仅新网络）")
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data-dir", default=DEFAULT_ARCHIVE)
    ap.add_argument("--out-dir", default=_HERE, help="模型保存目录（默认 game_4x4 目录）")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--eval-games", type=int, default=0,
                    help=">0 时训练后跑统一协议棋力评估（每项局数）")
    args = ap.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[Backtest] device={device} epochs={args.epochs} batch={args.batch} "
          f"lr={args.lr} health_weight={args.health_weight} seed={args.seed}", flush=True)

    train, val = load_dataset(args.data_dir, args.val_frac, args.seed)
    train_t = to_tensors(train, device)
    val_t = to_tensors(val, device)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ---- 同时训练两个模型（同超参）----
    models = [
        ("base", BanqiNet(VARIANT).to(device)),
        ("health", BanqiNetHealth(VARIANT).to(device)),
    ]
    print(f"\n[Backtest] 参数量: BanqiNet={count_params(models[0][1])} | "
          f"BanqiNetHealth={count_params(models[1][1])}", flush=True)

    histories = {}
    for tag, model in models:
        hist = train_one(model, f"{tag.upper()}", train_t, val_t,
                         args.epochs, args.batch, args.lr, args.wd, device,
                         args.health_weight, args.seed + (1 if tag == "health" else 0))
        histories[tag] = hist
        save_model(model, tag, args.out_dir)

    # ---- 回测结论摘要 ----
    print("\n" + "=" * 70)
    print("  回测摘要（验证集）")
    print("=" * 70)
    for tag, hist in histories.items():
        last = hist[-1]
        print(f"  [{tag.upper()}] val_loss={last['val_loss']:.4f} "
              f"(pol={last['val_pol']:.4f}, val={last['val_val']:.4f})", flush=True)
    hm = health_metrics(models[1][1], val_t)
    print(f"\n  [HEALTH 新头回测] 验证集终局血量差预测: MSE={hm[0]:.4f} "
          f"MAE={hm[1]:.4f} Pearson r={hm[2]:.4f}", flush=True)
    # 对比：用 ±1 胜负标签做 baseline 的 MSE（若输出恒 0）
    val_health = val["healths"]
    zero_mse = float(np.mean(val_health ** 2))
    mean_mse = float(np.mean((val_health - val_health.mean()) ** 2))
    print(f"  [baseline] 恒预测0 的 MSE={zero_mse:.4f} | 恒预测均值 MSE={mean_mse:.4f} "
          f"(health 头需显著低于该值才有信息量)", flush=True)
    print("=" * 70, flush=True)

    # ---- 可选棋力评估 ----
    if args.eval_games > 0:
        base_path = os.path.join(args.out_dir, "banqi4x4_base_backtest.pth")
        health_path = os.path.join(args.out_dir, "banqi4x4_health_backtest.pth")
        eval_strength(base_path, "BASE", args.eval_games, device)
        eval_strength(health_path, "HEALTH", args.eval_games, device)
        eval_head_to_head(base_path, health_path, args.eval_games, device)


if __name__ == "__main__":
    main()
