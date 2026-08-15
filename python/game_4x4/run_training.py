"""
run_training.py — 4x4 暗棋训练闭环入口

启动 SelfPlayWorker4x4（生产者）+ TrainWorker（消费者），
训练到 config.MAX_RUNTIME_SECONDS 后优雅停止并落盘 checkpoint。

用法：
    python python/game_4x4/run_training.py
    # 环境变量：
    #   G4X4_MAX_RUNTIME  训练时长（秒，默认 3600）
    #   G4X4_MODEL_PATH / G4X4_STATE_DICT_PATH  模型路径覆盖
"""
from __future__ import annotations

import os
import queue
import signal
import sys
import time
from typing import List

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from config import config
from archiver import ArchiverWorker
from self_play import SelfPlayWorker4x4, build_predictor4x4, build_self_play_config
from training_service import TrainWorker

# 系统资源监控（psutil + pynvml），缺失依赖时静默禁用
try:
    from system_monitor import SystemMonitor
    HAS_MONITOR = True
except ImportError:  # pragma: no cover
    HAS_MONITOR = False

# TensorBoard 训练日志（tb_logger 内部处理依赖缺失，可安全导入）
try:
    import tb_logger
    HAS_TB_LOGGER = True
except ImportError:  # pragma: no cover
    HAS_TB_LOGGER = False

START = time.time()


def main() -> int:
    # ---- TensorBoard 初始化（每次运行独立时间戳子目录，便于对比多次训练）----
    tb_log_dir = ""
    tb_ok = False
    if HAS_TB_LOGGER:
        tb_log_dir = os.path.join(
            config.TENSORBOARD_LOG_DIR,
            time.strftime("%Y%m%d-%H%M%S"),
        )
        tb_ok = tb_logger.init_summary_writer(
            log_dir=tb_log_dir,
            enabled=config.TENSORBOARD_ENABLED,
        )

    print("=" * 64)
    print("  4x4 暗棋训练启动（CPU）")
    print("=" * 64)
    print(f"  MCTS sims         = {config.MCTS_SIMS}")
    print(f"  GAMES_PER_ITER    = {config.GAMES_PER_ITER}（{config.NUM_WORKERS} workers）")
    print(f"  训练时限          = {config.MAX_RUNTIME_SECONDS}s")
    print(f"  模型输出          = {config.MODEL_PATH}")
    print(f"  增强              = {'✅ 开' if config.DATA_AUGMENT_ENABLED else '❌ 关'}"
          f"（对称变换: {config.DATA_AUGMENT_TRANSFORMS}，"
          f"保留原始: {config.DATA_AUGMENT_KEEP_ORIGINAL}；仅训练侧，冷存储存原始数据）")
    print(f"  归档              = {'✅ 开' if config.ARCHIVE_ENABLED else '❌ 关'}"
          f"（Mongo: {config.DB_NAME}.{config.COLLECTION}，失败降级本地 JSONL）")
    if config.MONITOR_ENABLED and HAS_MONITOR:
        print(f"  监控              = 每 {config.MONITOR_INTERVAL:.0f}s 采样一次"
              f"（CSV: {config.MONITOR_CSV_PATH or '关闭'}）")
    if config.TENSORBOARD_ENABLED and tb_ok:
        print(f"  TensorBoard       = {tb_log_dir}"
              f"（tensorboard --logdir {config.TENSORBOARD_LOG_DIR} 查看）")
    print("=" * 64)

    stop_flag: List[bool] = [False]

    def _handler(signum, frame):
        if stop_flag[0]:
            print("\n[Main] 再次收到信号，强制退出")
            sys.exit(1)
        stop_flag[0] = True
        print("\n[Main] 收到信号，将在当前批结束后优雅退出...")

    signal.signal(signal.SIGINT, _handler)

    data_q: "queue.Queue" = queue.Queue(maxsize=config.DATA_QUEUE_MAXSIZE)
    archive_q: "queue.Queue" = queue.Queue(maxsize=config.ARCHIVE_QUEUE_MAXSIZE)
    predictor, _ = build_predictor4x4(config.MODEL_PATH, device_str=config.INFER_DEVICE)
    sp_cfg = build_self_play_config()

    workers = [
        SelfPlayWorker4x4(predictor, sp_cfg, data_q, archive_q, stop_flag, worker_id=0),
        TrainWorker(data_q, stop_flag),
    ]
    if config.ARCHIVE_ENABLED:
        workers.append(ArchiverWorker(archive_q, stop_flag))
    for w in workers:
        w.start()

    train_worker: TrainWorker = workers[1]
    sp_worker: SelfPlayWorker4x4 = workers[0]

    # ---- 系统资源监控线程（psutil + pynvml）----
    monitor = None
    if config.MONITOR_ENABLED and HAS_MONITOR:
        monitor = SystemMonitor(
            interval=config.MONITOR_INTERVAL,
            show_per_core=config.MONITOR_PER_CORE,
            csv_path=config.MONITOR_CSV_PATH,
            log_to_tb=bool(config.TENSORBOARD_LOG_SYS and tb_ok),
            stop_flag=stop_flag,
        )
        monitor.start()
        print(f"[Main] 📊 系统资源监控已启动（每 {config.MONITOR_INTERVAL:.0f}s 采样）")
    elif config.MONITOR_ENABLED and not HAS_MONITOR:
        print("[Main] ⚠️ 未安装 psutil/nvidia-ml-py，系统资源监控已跳过"
              "（pip install psutil nvidia-ml-py）")

    try:
        while not stop_flag[0]:
            elapsed = time.time() - START
            if elapsed >= config.MAX_RUNTIME_SECONDS:
                print(f"\n[Main] 达到训练时限 {config.MAX_RUNTIME_SECONDS}s，停止")
                stop_flag[0] = True
                break
            if not all(w.is_alive() for w in workers):
                print("[Main] 有线程退出")
                break
            time.sleep(5)
    except KeyboardInterrupt:
        stop_flag[0] = True

    print("\n[Main] 正在优雅关闭各线程...")
    if sp_worker.is_alive():
        sp_worker.join(timeout=10)
    if train_worker.is_alive():
        train_worker.join(timeout=30)
    stop_flag[0] = True
    if train_worker.is_alive():
        train_worker.join(timeout=10)
    train_worker.finalize()
    # 归档线程排空残余
    if config.ARCHIVE_ENABLED:
        archiver_worker: ArchiverWorker = workers[2]  # type: ignore[assignment]
        if archiver_worker.is_alive():
            archiver_worker.join(timeout=15)
    # 停止监控线程（共享 stop_flag，run 循环很快自行退出）
    if monitor is not None and monitor.is_alive():
        monitor.join(timeout=3)
    # 关闭 TensorBoard writer（flush 落盘）
    if HAS_TB_LOGGER:
        tb_logger.close_summary_writer()

    tr = train_worker.stats()
    history = train_worker.round_history_snapshot()
    sp_stats = sp_worker.stats()
    ar_stats = archiver_worker.stats() if config.ARCHIVE_ENABLED else {"archived_games": 0}
    print("\n" + "=" * 64)
    print("  4x4 训练结束")
    print("=" * 64)
    print(f"  自对弈局数:   {sp_stats['total_games']}（样本 {sp_stats['total_samples']}）")
    print(f"  训练轮次:     {tr['round_num']}")
    print(f"  累计批次:     {tr['total_batches']:.0f}")
    print(f"  平均 Loss:    {tr['avg_loss']:.4f}"
          f"（Pol: {tr['avg_policy_loss']:.4f}, Val: {tr['avg_value_loss']:.4f}）")
    if history:
        print(f"  Loss 变化:    {history[0]['train_loss']:.4f} → {history[-1]['train_loss']:.4f}")
    print(f"  归档局数:     {ar_stats['archived_games']}")
    print(f"  总耗时:       {time.time() - START:.0f}s")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    sys.exit(main())
