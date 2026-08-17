"""banqi/train.py — 自对弈 + 训练闭环统一入口（共享实现，4x2 / 4x4 / 4x8 通用）

单进程多线程模型，通过共享内存队列通信：
  - SelfPlayWorker（生产者）：按变体分派 Rust 绑定生成 episode，压入数据队列（+ 归档队列）
  - TrainWorker（消费者）：从数据队列消费，填充 replay buffer 并持续训练
  - ArchiverWorker（归档，可选）：从归档队列批量写 MongoDB（冷存储），失败降级本地 JSONL

变体薄壳入口（variants/*/run_training.py）调用本模块：
    from banqi.train import main
    main("4x4")

运行方式（需先 maturin develop --features pyo3）：
    python python/variants/4x8/run_training.py
"""

from __future__ import annotations

import os
import queue
import signal
import sys
import time
from typing import List, Optional

# Windows 控制台默认 GBK 无法编码 emoji 等字符，强制以 UTF-8 输出避免启动崩溃
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

try:
    import torch
    HAS_TORCH = True
except ImportError:  # pragma: no cover
    HAS_TORCH = False

try:
    import banqi_4x8
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "无法导入 banqi_4x8。请先执行: maturin develop --features pyo3"
    ) from exc

from banqi.archiver import ArchiverWorker
from banqi.config import Config, make_config
from banqi.memory_guard import start_memory_guard
from banqi.self_play import (
    SelfPlayWorker,
    build_mixed_predictor,
    build_predictor,
    build_self_play_config,
)
from banqi.tb_logger import close_summary_writer, init_summary_writer
from banqi.training_service import TrainWorker
from banqi.variant import get_variant

# 系统资源监控（psutil + pynvml），缺失依赖时静默禁用
try:
    from banqi.system_monitor import SystemMonitor
    HAS_MONITOR = True
except ImportError:  # pragma: no cover
    HAS_MONITOR = False
    SystemMonitor = None  # type: ignore[assignment,misc]


def main(variant_id: str) -> None:
    variant = get_variant(variant_id)
    config: Config = make_config(variant_id)
    tag = f"[{variant.id}]"

    # ---- 进程级线程数（可选）：{prefix}TORCH_THREADS 设置时限制 torch 线程 ----
    if HAS_TORCH:
        threads_env = os.getenv(variant.env_prefix + "TORCH_THREADS")
        if threads_env:
            torch.set_num_threads(int(threads_env))
            print(f"{tag} torch.set_num_threads = {threads_env}")

    # ---- TensorBoard 初始化（每次运行独立时间戳子目录）----
    tb_log_dir = ""
    tb_ok = False
    if config.TENSORBOARD_ENABLED:
        tb_log_dir = os.path.join(
            config.TENSORBOARD_LOG_DIR,
            time.strftime("%Y%m%d-%H%M%S"),
        )
        tb_ok = init_summary_writer(log_dir=tb_log_dir, enabled=True)

    print("=" * 56)
    print(f"  🚀 自对弈 + 训练闭环启动（变体 {variant_id}，单进程多线程）")
    print("=" * 56)
    print(f"  PREDICT_BATCH   = {config.PREDICT_BATCH}")
    print(f"  TRAIN_BATCH     = {config.TRAIN_BATCH}")
    print(f"  MCTS_SIMS       = {config.MCTS_SIMS}")
    print(f"  GAMES_PER_ITER  = {config.GAMES_PER_ITER} (workers={config.NUM_WORKERS})")
    print(f"  MODEL_PATH      = {config.MODEL_PATH}")
    print(f"  STATE_DICT_PATH = {config.STATE_DICT_PATH}")
    print(f"  MONGO_URI       = {config.MONGO_URI if config.ARCHIVE_ENABLED else '（归档关闭）'}")
    print(f"  COLLECTION      = {config.DB_NAME}.{config.COLLECTION}")
    print(f"  INFER_DEVICE    = {config.INFER_DEVICE}（自对弈 MCTS 推理）")
    print(f"  CPU_AUX_WORKERS = {config.INFER_CPU_AUX_WORKERS}（>0 启用 GPU+CPU 混合推理）")
    print(f"  TRAIN_DEVICE    = {config.TRAIN_DEVICE}（训练，auto 自动选择）")
    print(f"  VALUE_TARGET    = {config.VALUE_TARGET_MODE}（value 目标模式）")
    print(f"  AUGMENT         = {'✅ 开' if config.DATA_AUGMENT_ENABLED else '❌ 关'}"
          f"（对称变换: {config.DATA_AUGMENT_TRANSFORMS}，"
          f"保留原始: {config.DATA_AUGMENT_KEEP_ORIGINAL}；仅训练侧，冷存储存原始数据）")
    if config.MONITOR_ENABLED and HAS_MONITOR:
        print(f"  MONITOR         = 每 {config.MONITOR_INTERVAL:.0f}s 采样一次"
              f"（CSV: {config.MONITOR_CSV_PATH or '关闭'}）")
    if tb_ok:
        print(f"  TENSORBOARD     = {tb_log_dir}"
              f"（tensorboard --logdir {config.TENSORBOARD_LOG_DIR} 查看）")
    if config.MAX_RUNTIME_SECONDS > 0:
        print(f"  运行时限        = {config.MAX_RUNTIME_SECONDS}s")
    print("=" * 56)

    # ---- 内存看门守护线程（超限主动终止，防止长时间卡死 / 拖垮整机）----
    start_memory_guard()

    # ---- 优雅退出标志 ----
    stop_flag: List[bool] = [False]

    def _handler(signum, frame):
        if stop_flag[0]:
            print(f"\n{tag} 再次收到 Ctrl-C，强制退出")
            sys.exit(1)
        stop_flag[0] = True
        print(f"\n{tag} 收到 Ctrl-C，将在当前批结束后优雅退出...")

    signal.signal(signal.SIGINT, _handler)

    # ---- 队列 ----
    data_q: "queue.Queue" = queue.Queue(maxsize=config.DATA_QUEUE_MAXSIZE)
    use_archive = bool(config.ARCHIVE_ENABLED and variant.archive_dir is not None)
    archive_q: Optional["queue.Queue"] = (
        queue.Queue(maxsize=config.ARCHIVE_QUEUE_MAXSIZE) if use_archive else None
    )

    # ---- 构建 Predictor + SelfPlayConfig ----
    if config.INFER_CPU_AUX_WORKERS > 0:
        predictor, infer_device = build_mixed_predictor(
            variant, config.MODEL_PATH,
            device_str=config.INFER_DEVICE,
            cpu_workers=config.INFER_CPU_AUX_WORKERS,
            cpu_fraction=config.INFER_CPU_FRACTION,
            min_split_batch=config.INFER_MIN_SPLIT_BATCH,
        )
    else:
        predictor, infer_device = build_predictor(
            variant, config.MODEL_PATH, device_str=config.INFER_DEVICE
        )
    sp_cfg = build_self_play_config(variant)
    print(
        f"{tag} banqi_4x8 {variant.env_const_prefix or '标准'}常量: "
        f"BOARD=({build_const(variant, 'BOARD_CHANNELS')},"
        f"{build_const(variant, 'BOARD_ROWS')},{build_const(variant, 'BOARD_COLS')}), "
        f"SCALAR={build_const(variant, 'SCALAR_FEATURE_COUNT')}, "
        f"ACTION={build_const(variant, 'ACTION_SPACE_SIZE')}"
    )
    print(f"{tag} 推理设备 = {infer_device}（MCTS 自对弈）")
    if config.INFER_CPU_AUX_WORKERS > 0 and infer_device.type == "cuda":
        print(f"{tag} ✅ CPU 辅助推理已启用: {config.INFER_CPU_AUX_WORKERS} 个 CPU 线程, "
              f"每批 {config.INFER_CPU_FRACTION:.0%} 给 CPU")

    # ---- 线程组 ----
    workers = [
        SelfPlayWorker(predictor, sp_cfg, variant, data_q, archive_q, stop_flag, worker_id=0),
        TrainWorker(data_q, stop_flag, variant),
    ]
    if use_archive:
        workers.append(ArchiverWorker(archive_q, stop_flag, variant))

    for w in workers:
        w.start()

    # ---- 系统资源监控线程 ----
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
        print(f"{tag} 📊 系统资源监控已启动（每 {config.MONITOR_INTERVAL:.0f}s 采样）")
    elif not HAS_MONITOR:
        print(f"{tag} ⚠️ 未安装 psutil/nvidia-ml-py，系统资源监控已跳过"
              "（pip install psutil nvidia-ml-py）")

    print(f"{tag} 线程已启动，正在运行（Ctrl-C 优雅退出）...\n")

    # ---- 主线程：等待 / 达到时限 / 优雅退出 ----
    start_t = time.time()
    try:
        while not stop_flag[0]:
            elapsed = time.time() - start_t
            if config.MAX_RUNTIME_SECONDS > 0 and elapsed >= config.MAX_RUNTIME_SECONDS:
                print(f"\n{tag} 达到运行时限 {config.MAX_RUNTIME_SECONDS}s，优雅停止...")
                stop_flag[0] = True
                break
            threads_alive = [w.name for w in workers if w.is_alive()]
            if len(threads_alive) < len(workers):
                print(f"{tag} 有线程退出: {threads_alive}")
                break
            time.sleep(2)
    except KeyboardInterrupt:
        stop_flag[0] = True

    # ---- 优雅关闭 ----
    print(f"\n{tag} 正在优雅关闭各线程...")
    sp_worker: SelfPlayWorker = workers[0]
    train_worker: TrainWorker = workers[1]
    if sp_worker.is_alive():
        sp_worker.join(timeout=15)
    if train_worker.is_alive():
        train_worker.join(timeout=30)
    if train_worker.is_alive():
        stop_flag[0] = True
        train_worker.join(timeout=10)
    train_worker.finalize()
    if use_archive:
        archiver_worker: ArchiverWorker = workers[2]
        if archiver_worker.is_alive():
            archiver_worker.join(timeout=15)
    if monitor is not None and monitor.is_alive():
        monitor.join(timeout=3)
    close_summary_writer()

    # ---- 结束统计 ----
    sp_stats = sp_worker.stats()
    tr_stats = train_worker.stats()
    history = train_worker.round_history_snapshot()
    sep = "=" * 56
    print(f"\n{sep}")
    print(f"  {variant_id} 变体 数据收集 / 训练结束")
    print(f"{sep}")
    print(f"  最终迭代:          {sp_stats['iteration']}")
    print(f"  累计自对弈局数:    {sp_stats['total_games']}")
    print(f"  累计样本数:        {sp_stats['total_samples']}")
    print(f"  训练轮次:          {tr_stats['round_num']}")
    print(f"  累计训练批次:      {tr_stats['total_batches']}")
    print(f"  平均 Loss:         {tr_stats['avg_loss']:.4f} "
          f"(Pol: {tr_stats['avg_policy_loss']:.4f}, Val: {tr_stats['avg_value_loss']:.4f})")
    if use_archive:
        print(f"  归档局数:          {archiver_worker.stats()['archived_games']}")
    if history:
        first, last = history[0], history[-1]
        print(f"  Loss 变化:         {first['train_loss']:.4f} → {last['train_loss']:.4f} "
              f"(round {first['round']} → {last['round']})")
    print(f"{sep}")


def build_const(variant, name: str) -> int:
    """读取 banqi_4x8 上带变体前缀的维度常量。"""
    return int(getattr(banqi_4x8, variant.env_const_prefix + name))


if __name__ == "__main__":
    vid = sys.argv[1] if len(sys.argv) > 1 else "4x8"
    main(vid)
