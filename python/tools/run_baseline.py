"""
run_baseline.py — 快速基线训练验证入口（无 CLI 参数）

以紧凑配置短时间跑通真实闭环（自对弈 → replay buffer → 训练 → checkpoint），
达到"完成指定轮次 或 运行达到时长"后优雅退出，并把逐轮/逐局指标汇总写入
`train_baseline_metrics.json`，供 `validate_baseline.py` 判定训练是否走在正确道路上。

与生产 `run_training.py` 的关键区别：
  - 运行时覆盖 config 单例为紧凑参数（不改 config.py 默认值）
  - 使用独立 checkpoint 路径（banqi_model_baseline.*），绝不污染真实模型
  - 归档强制走本地 JSONL（mongo_uri=""），不依赖外部 Mongo 服务
  - 有限运行（默认 ≥4 轮 或 300 秒），而非无限运行

运行方式（需先 maturin develop --features pyo3）：
    python python/run_baseline.py
"""

from __future__ import annotations
import os
import sys

_TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
_PYTHON_DIR = os.path.dirname(_TOOLS_DIR)
_BANQI_DIR = os.path.join(_PYTHON_DIR, "banqi")
for _d in (_PYTHON_DIR, _BANQI_DIR):
    if _d not in sys.path:
        sys.path.insert(0, _d)


import json
import os
import queue
import sys
import time
from datetime import datetime, timezone
from typing import Dict, List

# Windows 控制台默认 GBK 无法编码 emoji 等字符，强制以 UTF-8 输出避免启动崩溃
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

try:
    import banqi_4x8
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "无法导入 banqi_4x8。请先执行: maturin develop --features pyo3"
    ) from exc

from config import config
from archiver import ArchiverWorker
from self_play import (
    SelfPlayWorker,
    build_predictor,
    build_self_play_config,
)
from training_service import TrainWorker


# ============================================================================
# 基线配置（运行时覆盖 config 单例；独立 checkpoint 路径）
# ============================================================================

BASELINE_MODEL_PATH = "banqi_model_baseline.pt"
BASELINE_STATE_DICT_PATH = "banqi_model_baseline.pth"
METRICS_PATH = "train_baseline_metrics.json"

# 停止条件：满足其一即优雅退出
MIN_ROUNDS = 4      # 至少完成 N 个训练轮次
MAX_SECONDS = 300.0  # 或最长运行时长（秒）

# 轮询间隔（秒）
POLL_INTERVAL = 1.0


def _override_baseline_config() -> Dict:
    """运行时把 config 单例实例属性收紧为紧凑基线参数（仅影响本进程）。"""
    overrides = {
        # 自对弈
        "MCTS_SIMS": 24,
        "MAX_CONSIDERED_ACTIONS": 16,
        "TEMPERATURE_STEPS": 8,
        "GAMES_PER_ITER": 10,
        "NUM_WORKERS": 2,
        "GAMES_PER_WORKER": 5,  # 总对局数 = 2 × 5 = 10 = GAMES_PER_ITER
        # 训练
        "TRAIN_BATCH": 16,
        "LEARNING_RATE": 1e-3,
        "MIN_LR": 5e-6,
        "LR_DECAY_STEPS": 600,
        "TRAIN_EPOCHS_PER_ROUND": 2,
        "MAX_SAMPLE_BUFFER_SIZE": 50000,
        "MIN_SAMPLES_TO_START": 128,
        "QUEUE_FETCH_BATCH": 4,
        # 独立 checkpoint 路径（隔离生产模型）
        "MODEL_PATH": BASELINE_MODEL_PATH,
        "STATE_DICT_PATH": BASELINE_STATE_DICT_PATH,
        # 队列 / 线程
        "DATA_QUEUE_MAXSIZE": 128,
        "ARCHIVE_QUEUE_MAXSIZE": 256,
        "CHECKPOINT_EVERY_N_ROUNDS": 1,
        # 归档
        "ARCHIVE_BATCH": 8,
    }
    for key, val in overrides.items():
        setattr(config, key, val)
    return overrides


def _cleanup_baseline_artifacts() -> None:
    """清理残留的基线模型与指标文件，保证每次运行从干净状态开始。"""
    for p in (BASELINE_MODEL_PATH, BASELINE_STATE_DICT_PATH, METRICS_PATH):
        if os.path.exists(p):
            try:
                os.remove(p)
                print(f"[Baseline] 清理旧产物: {p}")
            except OSError as exc:
                print(f"[Baseline] ⚠️ 清理失败 {p}: {exc}")


def _is_stop_reached(train_worker: TrainWorker, start_ts: float) -> bool:
    """判断是否满足停止条件（轮次或时长达标）。"""
    if train_worker.stats()["round_num"] >= MIN_ROUNDS:
        return True
    if (time.time() - start_ts) >= MAX_SECONDS:
        return True
    return False


def _collect_metrics(
    self_play_worker: SelfPlayWorker,
    train_worker: TrainWorker,
    archiver_worker: ArchiverWorker,
    start_ts: float,
    baseline_config: Dict,
) -> Dict:
    """汇总逐轮/逐局指标 + 吞吐 + checkpoint 信息，返回 metrics dict。"""
    end_ts = time.time()
    elapsed = end_ts - start_ts

    sp_stats = self_play_worker.stats()
    tr_stats = train_worker.stats()
    ar_stats = archiver_worker.stats()

    game_records = self_play_worker.game_records_snapshot()
    round_history = train_worker.round_history_snapshot()

    total_games = sp_stats["total_games"]
    total_samples = sp_stats["total_samples"]
    games_per_sec = total_games / elapsed if elapsed > 0 else 0.0
    samples_per_sec = total_samples / elapsed if elapsed > 0 else 0.0

    # 胜负分布
    winners: Dict[str, int] = {"1": 0, "-1": 0, "0": 0}
    for g in game_records:
        w = str(g.get("winner", 0))
        winners[w] = winners.get(w, 0) + 1

    # 平均局长度
    avg_game_length = 0.0
    if game_records:
        avg_game_length = float(
            sum(g.get("game_length", 0) for g in game_records) / len(game_records)
        )

    # checkpoint 是否更新（mtime 晚于运行开始）
    cp_updated = False
    model_mtime = 0.0
    state_mtime = 0.0
    if os.path.exists(config.MODEL_PATH):
        model_mtime = os.path.getmtime(config.MODEL_PATH)
    if os.path.exists(config.STATE_DICT_PATH):
        state_mtime = os.path.getmtime(config.STATE_DICT_PATH)
    cp_updated = (
        os.path.exists(config.MODEL_PATH)
        and os.path.exists(config.STATE_DICT_PATH)
        and model_mtime >= start_ts
        and state_mtime >= start_ts
    )

    return {
        "meta": {
            "start_time": datetime.fromtimestamp(start_ts, tz=timezone.utc).isoformat(),
            "end_time": datetime.fromtimestamp(end_ts, tz=timezone.utc).isoformat(),
            "elapsed_sec": round(elapsed, 2),
            "baseline_config": baseline_config,
        },
        "self_play": {
            "total_games": total_games,
            "total_samples": total_samples,
            "iteration": sp_stats["iteration"],
            "games_per_sec": round(games_per_sec, 4),
            "samples_per_sec": round(samples_per_sec, 4),
            "avg_game_length": round(avg_game_length, 2),
            "winners": winners,
            "games": game_records,
        },
        "training": {
            "rounds": tr_stats["round_num"],
            "total_batches": tr_stats["total_batches"],
            "round_history": round_history,
        },
        "archiver": {
            "archived_games": ar_stats["archived_games"],
        },
        "checkpoints": {
            "model_path": config.MODEL_PATH,
            "state_dict_path": config.STATE_DICT_PATH,
            "updated": cp_updated,
        },
    }


def main() -> None:
    print("=" * 56)
    print("  🔬 快速基线训练验证（真实闭环，紧凑配置）")
    print("=" * 56)

    baseline_config = _override_baseline_config()
    _cleanup_baseline_artifacts()

    print(f"  MCTS_SIMS        = {config.MCTS_SIMS}")
    print(f"  GAMES_PER_ITER   = {config.GAMES_PER_ITER}")
    print(f"  TRAIN_BATCH      = {config.TRAIN_BATCH}")
    print(f"  MIN_SAMPLES      = {config.MIN_SAMPLES_TO_START}")
    print(f"  LR               = {config.LEARNING_RATE}")
    print(f"  MODEL_PATH       = {config.MODEL_PATH}")
    print(f"  STATE_DICT_PATH  = {config.STATE_DICT_PATH}")
    print(f"  停止条件          = ≥{MIN_ROUNDS} 轮 或 {MAX_SECONDS:.0f} 秒")
    print("=" * 56)

    stop_flag: List[bool] = [False]

    # ---- 队列 ----
    data_q: "queue.Queue" = queue.Queue(maxsize=config.DATA_QUEUE_MAXSIZE)
    archive_q: "queue.Queue" = queue.Queue(maxsize=config.ARCHIVE_QUEUE_MAXSIZE)

    # ---- 构建 Predictor + SelfPlayConfig ----
    # 推理用 CPU（config.INFER_DEVICE），不占 GPU
    predictor, _device = build_predictor(config.MODEL_PATH, device_str=config.INFER_DEVICE)
    sp_cfg = build_self_play_config()

    # ---- 三线程（归档强制本地 JSONL，不依赖 Mongo） ----
    workers = [
        SelfPlayWorker(predictor, sp_cfg, data_q, archive_q, stop_flag, worker_id=0),
        TrainWorker(data_q, stop_flag),
        ArchiverWorker(archive_q, stop_flag, mongo_uri=""),
    ]
    for w in workers:
        w.start()

    self_play_worker: SelfPlayWorker = workers[0]
    train_worker: TrainWorker = workers[1]
    archiver_worker: ArchiverWorker = workers[2]

    start_ts = time.time()
    print("[Baseline] 三线程已启动，运行中...\n")

    # ---- 主循环：轮询停止条件 ----
    try:
        while not stop_flag[0]:
            if _is_stop_reached(train_worker, start_ts):
                break
            # 监控线程是否意外退出
            if not all(w.is_alive() for w in workers):
                print("[Baseline] ⚠️ 有线程意外退出，提前停止")
                break
            time.sleep(POLL_INTERVAL)
    except KeyboardInterrupt:
        print("\n[Baseline] 收到 Ctrl-C，优雅退出...")

    stop_flag[0] = True

    # ---- 优雅关闭（先停生产者 → 训练 finalize → 归档排空） ----
    print("\n[Baseline] 正在优雅关闭各线程...")
    if self_play_worker.is_alive():
        self_play_worker.join(timeout=15)
    if train_worker.is_alive():
        train_worker.join(timeout=30)
    if train_worker.is_alive():
        train_worker.join(timeout=10)
    train_worker.finalize()
    if archiver_worker.is_alive():
        archiver_worker.join(timeout=15)

    # ---- 汇总指标并落盘 ----
    metrics = _collect_metrics(
        self_play_worker, train_worker, archiver_worker, start_ts, baseline_config
    )
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"\n[Baseline] ✅ 指标已写入 {METRICS_PATH}")

    # ---- 结束统计 ----
    sep = "=" * 56
    sp = metrics["self_play"]
    tr = metrics["training"]
    print(f"\n{sep}")
    print("  基线训练结束")
    print(f"{sep}")
    print(f"  耗时:              {metrics['meta']['elapsed_sec']:.1f}s")
    print(f"  累计自对弈局数:    {sp['total_games']}")
    print(f"  累计样本数:        {sp['total_samples']}")
    print(f"  吞吐:              {sp['games_per_sec']:.2f} 局/s, "
          f"{sp['samples_per_sec']:.2f} 样本/s")
    print(f"  平均局长度:        {sp['avg_game_length']}")
    print(f"  胜负分布:          红={sp['winners'].get('1', 0)}, "
          f"黑={sp['winners'].get('-1', 0)}, 平={sp['winners'].get('0', 0)}")
    print(f"  训练轮次:          {tr['rounds']}")
    print(f"  累计训练批次:      {tr['total_batches']}")
    print(f"  checkpoint 更新:   {metrics['checkpoints']['updated']}")
    print(f"{sep}")


if __name__ == "__main__":
    main()
