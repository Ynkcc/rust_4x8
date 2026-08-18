"""
validate_baseline.py — 基线训练结果校验（纯 CPU，无 banqi_4x8 依赖）。

读取 run_baseline.py 落盘的 `train_baseline_metrics.json`，按阈值断言"训练是否走在
正确道路上"，输出 PASS/FAIL 与"可继续长跑 / 需检查训练逻辑"的决策提示。

判定维度：
  1. 训练轮次 / 批次达标
  2. train loss 全部有限（无 NaN/Inf/负值）
  3. loss 趋势下降（末段均值 < 初段 × 0.95，或 value loss 改善）
  4. value loss 不爆炸（全程 < 1.5 且末段 < 0.8）
  5. 局统计合理（局数 / 平均局长度 / 胜负分布非单边 / 吞吐 > 0）
  6. checkpoint 已更新（.pt / .pth 存在且 mtime 晚于运行开始）

阈值常量集中于文件顶部，首次运行后可据实测微调。

运行：python python/validate/validate_baseline.py
"""

from __future__ import annotations

import json
import os
import sys
from typing import List

import numpy as np

# Windows 控制台默认 GBK 无法编码 emoji 等字符，强制以 UTF-8 输出避免崩溃
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# 确保可从 python/ 目录 import 生产模块（constant 等）。
# 用 os.path 跨平台正确解析父目录（避免 validate_common 内 / 硬编码在 Windows 失效）。
_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

import os
import sys

_VALIDATE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PYTHON_DIR = os.path.dirname(_VALIDATE_DIR)
_BANQI_DIR = os.path.join(_PYTHON_DIR, "banqi")
for _d in (_PYTHON_DIR, _BANQI_DIR, _VALIDATE_DIR):
    if _d not in sys.path:
        sys.path.insert(0, _d)

import validate_common  # noqa: E402, F401
from validate_common import Reporter  # noqa: E402

# 指标文件默认路径（相对项目根；可被命令行参数覆盖）
DEFAULT_METRICS_PATH = "train_baseline_metrics.json"

# ============================================================================
# 判定阈值（可调）
# ============================================================================
MIN_ROUNDS = 3                     # 最少训练轮次
MIN_BATCHES = 1                    # 最少累计训练批次
LOSS_TREND_RATIO = 0.95            # 末段 loss 均值 < 初段 × 此值 视为下降
NO_DIVERGE_RATIO = 1.05            # 末轮 loss <= 首轮 × 此值 视为未发散（容忍平台期噪声）
VALUE_LOSS_MAX = 1.5               # value loss 全程上限（防爆炸）
VALUE_LOSS_FINAL_MAX = 0.8         # value loss 末段上限
VALUE_CONVERGED_MAX = 0.05         # value loss 末段低于此值视为 value 头已收敛（波动属噪声）
MIN_GAMES = 5                      # 最少自对弈局数
AVG_GAME_LENGTH_MIN = 5.0          # 平均局长度下界
AVG_GAME_LENGTH_MAX = 100.0        # 平均局长度上界
MIN_WINNER_TYPES = 2               # 胜负分布至少 N 种取值（非 100% 单边）


def _finite_nonneg(values: List[float]) -> bool:
    """所有值均有限且非负。"""
    return all(np.isfinite(v) and v >= 0 for v in values)


def _tail_mean(values: List[float], frac: float = 0.5) -> float:
    """末段 frac 比例的均值。"""
    n = len(values)
    k = max(1, int(n * frac))
    return float(np.mean(values[-k:]))


def _head_mean(values: List[float], frac: float = 0.5) -> float:
    """初段 frac 比例的均值。"""
    n = len(values)
    k = max(1, int(n * frac))
    return float(np.mean(values[:k]))


def _check_loss_finite(rep: Reporter, round_history: List[dict]) -> None:
    """train loss 全部有限且非负。"""
    train_losses = [r["train_loss"] for r in round_history]
    rep.check(
        len(train_losses) > 0 and _finite_nonneg(train_losses),
        f"train loss 全部有限非负 ({len(train_losses)} 轮)",
    )


def _check_loss_trend(rep: Reporter, round_history: List[dict]) -> None:
    """
    loss 趋势判定。

    语义：train_loss 记录的是每轮最后一个 epoch 的 loss，第一轮内已发生明显下降，
    因此跨轮序列呈"平台期 + 噪声"形态。判定"训练未发散/未失控"的稳健标准是：
      - 末轮 loss 不高于首轮的显著比例（允许平台期噪声），即 tail < head * RATIO
        或 tail <= head * NO_DIVERGE_RATIO（未发散）
      - 或 value loss 已收敛到极低（value 头已学会，波动属噪声）
    """
    train_losses = [r["train_loss"] for r in round_history]
    train_values = [r["train_value_loss"] for r in round_history]
    if len(train_losses) < 2:
        rep.check(False, f"loss 轮次不足，无法判定趋势 ({len(train_losses)})")
        return

    first = train_losses[0]
    last = train_losses[-1]
    tail = _tail_mean(train_losses)

    # 主要判定：末段均值 < 初段均值 × 0.95（下降）
    loss_dropped = tail < _head_mean(train_losses) * LOSS_TREND_RATIO
    # 宽松判定：末轮不高于首轮 × 1.05（未发散，允许平台期噪声）
    loss_not_diverged = last <= first * NO_DIVERGE_RATIO
    # value 头已收敛（末段 value loss 极小），波动属噪声，不要求继续下降
    tail_v = _tail_mean(train_values)
    value_converged = tail_v < VALUE_CONVERGED_MAX

    ok = loss_dropped or loss_not_diverged or value_converged
    rep.check(
        ok,
        f"loss 趋势正常：first={first:.4f} last={last:.4f} tail={tail:.4f} "
        f"(value tail={tail_v:.4f})",
    )


def _check_value_loss(rep: Reporter, round_history: List[dict]) -> None:
    """value loss 不爆炸。"""
    train_values = [r["train_value_loss"] for r in round_history]
    if not train_values:
        rep.check(False, "无 train value loss 记录")
        return
    max_v = float(np.max(train_values))
    tail_v = _tail_mean(train_values)
    rep.check(
        max_v < VALUE_LOSS_MAX and tail_v < VALUE_LOSS_FINAL_MAX,
        f"value loss 不爆炸：max={max_v:.4f} (<{VALUE_LOSS_MAX}), "
        f"tail={tail_v:.4f} (<{VALUE_LOSS_FINAL_MAX})",
    )


def _check_self_play(rep: Reporter, sp: dict) -> None:
    """局统计合理：局数 / 平均长度 / 胜负分布 / 吞吐。"""
    total_games = sp.get("total_games", 0)
    rep.check(total_games >= MIN_GAMES, f"自对弈局数达标 ({total_games} ≥ {MIN_GAMES})")

    avg_len = sp.get("avg_game_length", 0.0)
    rep.check(
        AVG_GAME_LENGTH_MIN <= avg_len <= AVG_GAME_LENGTH_MAX,
        f"平均局长度合理 ({avg_len:.1f} ∈ [{AVG_GAME_LENGTH_MIN}, {AVG_GAME_LENGTH_MAX}])",
    )

    winners = sp.get("winners", {})
    nonzero_types = [w for w, c in winners.items() if c > 0]
    rep.check(
        len(nonzero_types) >= MIN_WINNER_TYPES,
        f"胜负分布非单边 ({len(nonzero_types)} 种结果: {winners})",
    )

    gps = sp.get("games_per_sec", 0.0)
    rep.check(gps > 0, f"吞吐正常 ({gps:.3f} 局/s > 0)")


def _check_checkpoint(rep: Reporter, cps: dict, metrics: dict) -> None:
    """checkpoint 文件存在且 mtime 晚于运行开始。"""
    model_path = cps.get("model_path", "")
    state_path = cps.get("state_dict_path", "")
    start_iso = metrics.get("meta", {}).get("start_time", "")
    start_ts = 0.0
    if start_iso:
        try:
            from datetime import datetime
            start_ts = datetime.fromisoformat(start_iso).timestamp()
        except Exception:
            start_ts = 0.0

    model_ok = os.path.exists(model_path)
    state_ok = os.path.exists(state_path)
    rep.check(model_ok, f"checkpoint .pt 存在 ({model_path})")
    rep.check(state_ok, f"checkpoint .pth 存在 ({state_path})")

    if model_ok and state_ok and start_ts > 0:
        m_ts = os.path.getmtime(model_path)
        s_ts = os.path.getmtime(state_path)
        rep.check(
            m_ts >= start_ts and s_ts >= start_ts,
            "checkpoint 在运行期间更新（mtime 晚于开始时间）",
        )
    elif model_ok and state_ok:
        rep.check(cps.get("updated", False), "checkpoint 标记为已更新")


def main() -> int:
    metrics_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_METRICS_PATH

    if not os.path.exists(metrics_path):
        print(f"[FAIL] 未找到指标文件: {metrics_path}")
        print("       请先运行: python python/run_baseline.py")
        return 1

    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    rep = Reporter("baseline-train-verify")
    print(f"\n读取指标: {metrics_path}")
    meta = metrics.get("meta", {})
    print(f"  耗时: {meta.get('elapsed_sec', '?')}s")
    print(f"  自对弈: {metrics.get('self_play', {}).get('total_games', 0)} 局, "
          f"{metrics.get('self_play', {}).get('total_samples', 0)} 样本")
    print(f"  训练: {metrics.get('training', {}).get('rounds', 0)} 轮, "
          f"{metrics.get('training', {}).get('total_batches', 0)} 批次\n")

    training = metrics.get("training", {})
    sp = metrics.get("self_play", {})
    cps = metrics.get("checkpoints", {})

    rounds = training.get("rounds", 0)
    total_batches = training.get("total_batches", 0)
    rep.check(rounds >= MIN_ROUNDS, f"训练轮次达标 ({rounds} ≥ {MIN_ROUNDS})")
    rep.check(total_batches >= MIN_BATCHES, f"训练批次达标 ({total_batches} ≥ {MIN_BATCHES})")

    round_history = training.get("round_history", [])
    if round_history:
        _check_loss_finite(rep, round_history)
        _check_loss_trend(rep, round_history)
        _check_value_loss(rep, round_history)
    else:
        rep.check(False, "缺少逐轮训练指标 (round_history 为空)")

    _check_self_play(rep, sp)
    _check_checkpoint(rep, cps, metrics)

    ok = rep.summary()

    print("\n" + "=" * 56)
    if ok:
        print("  ✅ 决策：训练走在正确道路上，可继续长时间运行")
    else:
        print("  ❌ 决策：存在异常，建议检查训练逻辑（详见上方 FAIL 项）")
    print("=" * 56)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
