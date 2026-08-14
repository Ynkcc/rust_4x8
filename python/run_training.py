"""
run_training.py — 自对弈 + 训练闭环入口（无 CLI 参数）

单进程双线程模型，通过共享内存队列通信：
  - SelfPlayWorker（生产者）：调用 Rust PyO3 绑定生成 episode，压入数据队列与归档队列
  - TrainWorker（消费者）：从数据队列消费，填充 replay buffer 并持续训练
  - ArchiverWorker（归档）：从归档队列批量写 MongoDB（冷存储），失败降级本地 JSONL

运行方式（需先 maturin develop --features pyo3）：
    python python/run_training.py
"""

from __future__ import annotations

import queue
import signal
import sys
from typing import List

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

# 系统资源监控（psutil + pynvml），缺失依赖时静默禁用
try:
    from system_monitor import SystemMonitor
    HAS_MONITOR = True
except ImportError:  # pragma: no cover
    HAS_MONITOR = False


def main() -> None:
    print("=" * 56)
    print("  🚀 自对弈 + 训练闭环启动（单进程双线程 + Mongo 冷归档）")
    print("=" * 56)
    print(f"  PREDICT_BATCH   = {config.PREDICT_BATCH}")
    print(f"  TRAIN_BATCH     = {config.TRAIN_BATCH}")
    print(f"  MCTS_SIMS       = {config.MCTS_SIMS}")
    print(f"  GAMES_PER_ITER  = {config.GAMES_PER_ITER}")
    print(f"  MODEL_PATH      = {config.MODEL_PATH}")
    print(f"  STATE_DICT_PATH = {config.STATE_DICT_PATH}")
    print(f"  MONGO_URI       = {config.MONGO_URI}")
    print(f"  COLLECTION      = {config.DB_NAME}.{config.COLLECTION}")
    if config.MONITOR_ENABLED and HAS_MONITOR:
        print(f"  MONITOR         = 每 {config.MONITOR_INTERVAL:.0f}s 采样一次"
              f"（CSV: {config.MONITOR_CSV_PATH or '关闭'}）")
    print("=" * 56)

    # ---- 优雅退出标志 ----
    stop_flag: List[bool] = [False]

    def _handler(signum, frame):
        if stop_flag[0]:
            print("\n[Main] 再次收到 Ctrl-C，强制退出")
            sys.exit(1)
        stop_flag[0] = True
        print("\n[Main] 收到 Ctrl-C，将在当前批结束后优雅退出...")

    signal.signal(signal.SIGINT, _handler)

    # ---- 队列 ----
    data_q: "queue.Queue" = queue.Queue(maxsize=config.DATA_QUEUE_MAXSIZE)
    archive_q: "queue.Queue" = queue.Queue(maxsize=config.ARCHIVE_QUEUE_MAXSIZE)

    # ---- 构建 Predictor + SelfPlayConfig ----
    predictor, _device = build_predictor(config.MODEL_PATH, device_str="auto")
    sp_cfg = build_self_play_config()
    print(
        f"[Main] banqi_4x8: BOARD=({banqi_4x8.BOARD_CHANNELS},"
        f"{banqi_4x8.BOARD_ROWS},{banqi_4x8.BOARD_COLS}), "
        f"SCALAR={banqi_4x8.SCALAR_FEATURE_COUNT}, ACTION={banqi_4x8.ACTION_SPACE_SIZE}"
    )

    # ---- 三线程 ----
    workers = [
        SelfPlayWorker(predictor, sp_cfg, data_q, archive_q, stop_flag, worker_id=0),
        TrainWorker(data_q, stop_flag),
        ArchiverWorker(archive_q, stop_flag),
    ]

    for w in workers:
        w.start()

    # ---- 系统资源监控线程（psutil + pynvml）----
    monitor = None
    if config.MONITOR_ENABLED and HAS_MONITOR:
        monitor = SystemMonitor(
            interval=config.MONITOR_INTERVAL,
            show_per_core=config.MONITOR_PER_CORE,
            csv_path=config.MONITOR_CSV_PATH,
            stop_flag=stop_flag,
        )
        monitor.start()
        print(f"[Main] 📊 系统资源监控已启动（每 {config.MONITOR_INTERVAL:.0f}s 采样）")
    elif not HAS_MONITOR:
        print("[Main] ⚠️ 未安装 psutil/nvidia-ml-py，系统资源监控已跳过"
              "（pip install psutil nvidia-ml-py）")

    print("[Main] 三线程已启动，正在运行（Ctrl-C 优雅退出）...\n")

    # ---- 主线程：等待或优雅退出 ----
    try:
        while not stop_flag[0]:
            # 每 2 秒打印一次汇总（可选，低频避免刷屏）
            threads_alive = [w.name for w in workers if w.is_alive()]
            if len(threads_alive) < len(workers):
                print(f"[Main] 有线程退出: {threads_alive}")
                break
            signal.pause()  # 由信号处理器唤醒，避免 busy-loop
    except KeyboardInterrupt:
        stop_flag[0] = True
    except AttributeError:
        # 平台不支持 signal.pause 时退化为轮询
        import time

        while not stop_flag[0]:
            time.sleep(0.5)

    # ---- 等待各线程退出 ----
    print("\n[Main] 正在优雅关闭各线程...")
    # 先让生产者停止（不再产新数据）
    self_play_worker = workers[0]
    train_worker: TrainWorker = workers[1]  # type: ignore[assignment]
    if self_play_worker.is_alive():
        self_play_worker.join(timeout=15)
    # 训练线程处理完队列后退出并落盘 checkpoint
    if train_worker.is_alive():
        train_worker.join(timeout=30)
    if train_worker.is_alive():
        # 强制停止训练线程，落盘最终 checkpoint
        stop_flag[0] = True
        train_worker.join(timeout=10)
    train_worker.finalize()
    # 归档线程排空残余
    archiver_worker: ArchiverWorker = workers[2]  # type: ignore[assignment]
    if archiver_worker.is_alive():
        archiver_worker.join(timeout=15)

    # 停止监控线程（共享 stop_flag，run 循环很快自行退出）
    if monitor is not None and monitor.is_alive():
        monitor.join(timeout=3)

    # ---- 结束统计 ----
    sp_stats = self_play_worker.stats()
    tr_stats = train_worker.stats()
    ar_stats = archiver_worker.stats()
    sep = "=" * 56
    print(f"\n{sep}")
    print("  数据收集 / 训练结束")
    print(f"{sep}")
    print(f"  最终迭代:          {sp_stats['iteration']}")
    print(f"  累计自对弈局数:    {sp_stats['total_games']}")
    print(f"  累计样本数:        {sp_stats['total_samples']}")
    print(f"  训练轮次:          {tr_stats['round_num']}")
    print(f"  累计训练批次:      {tr_stats['total_batches']}")
    print(f"  平均 Loss:         {tr_stats['avg_loss']:.4f} "
          f"(Pol: {tr_stats['avg_policy_loss']:.4f}, Val: {tr_stats['avg_value_loss']:.4f})")
    print(f"  归档局数:          {ar_stats['archived_games']}")
    print(f"{sep}")


if __name__ == "__main__":
    main()
