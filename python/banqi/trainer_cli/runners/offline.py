"""banqi/trainer_cli/runners/offline.py — 离线训练路径。

`TRAIN_MODE="archive"`（冷存储归档数据训练）与 `TRAIN_MODE="rule_selfplay"`
（纯规则教师自对弈生成数据训练）共用此编排：数据生产者 → 队列 → TrainWorker。
"""

from __future__ import annotations

import multiprocessing
import os
import signal
import sys
import threading
import time
from typing import List

from banqi.archiver import ArchiverWorker
from banqi.config import Config, make_config
from banqi.memory_guard import start_memory_guard
from banqi.rule_teacher import RuleTeacherWorker, rule_teacher_worker_main
from banqi.tb_logger import close_summary_writer, init_summary_writer
from banqi.training import TrainWorker
from banqi.variant import get_variant

from .archive_feeder import ArchiveFeederWorker
from .context import (
    HAS_MONITOR,
    HAS_TORCH,
    SystemMonitor,
    setup_variant_logging,
    log_meta_tb,
)


def run_offline(variant_id: str, train_mode: str) -> None:
    variant = get_variant(variant_id)
    config: Config = make_config(variant_id)
    config._variant = variant
    tag = f"[{variant.id}]"

    log_file = setup_variant_logging(variant)
    print(f"{tag} 📝 运行日志记录至: {log_file}")

    if HAS_TORCH:
        threads_env = os.getenv("TORCH_THREADS")
        if threads_env:
            import torch
            torch.set_num_threads(int(threads_env))
            print(f"{tag} torch.set_num_threads = {threads_env}")

    tb_log_dir = ""
    tb_ok = False
    if config.TENSORBOARD_ENABLED:
        tb_log_dir = os.path.join(config.TENSORBOARD_LOG_DIR, time.strftime("%Y%m%d-%H%M%S"))
        tb_ok = init_summary_writer(log_dir=tb_log_dir, enabled=True)
        if tb_ok:
            log_meta_tb(config, variant_id, tb_log_dir)

    print("=" * 56)
    mode_label = "冷存储离线训练" if train_mode == "archive" else "纯规则自对弈训练"
    print(f"  🚀 {mode_label} 启动（变体 {variant_id}，单进程多线程）")
    print("=" * 56)
    print(f"  PREDICT_BATCH   = {config.PREDICT_BATCH}")
    print(f"  TRAIN_BATCH     = {config.TRAIN_BATCH}")
    print(f"  MONGO_URI       = {config.MONGO_URI if config.ARCHIVE_ENABLED else '（归档关闭）'}")
    print(f"  COLLECTION      = {config.DB_NAME}.{config.COLLECTION}")
    print(f"  INFER_DEVICE    = {config.INFER_DEVICE}")
    print(f"  TRAIN_DEVICE    = {config.TRAIN_DEVICE}（训练，auto 自动选择）")
    print(f"  VALUE_TARGET    = {config.VALUE_TARGET_MODE}（value 目标模式）")
    print(f"  INIT_FROM_CKPT  = {getattr(config, 'INIT_FROM_CHECKPOINT', None) or '（无）'}")
    if train_mode == "rule_selfplay":
        print(f"  RULE_BACKEND    = {config.RULE_SELFPLAY_BACKEND}（规则自对弈后端）")
        print(f"  RULE_CONCURRENCY= {config.RULE_SELFPLAY_CONCURRENCY}（规则自对弈并发数）")
        print(f"  RULE_DEPTH      = {config.RULE_SELFPLAY_DEPTH}（minimax 搜索深度）")
        print(f"  RULE_ROUNDS     = {config.RULE_SELFPLAY_ROUNDS}（纯规则自对弈训练总轮数）")
    print("=" * 56)

    start_memory_guard()
    # 线程停止信号用 threading.Event；多进程子进程的停止信号用 multiprocessing.Event。
    thread_stop = threading.Event()

    def _handler(signum, frame):
        if thread_stop.is_set():
            print(f"\n{tag} 再次收到 Ctrl-C，强制退出")
            sys.exit(1)
        thread_stop.set()
        print(f"\n{tag} 收到 Ctrl-C，将在当前批结束后优雅退出...")

    signal.signal(signal.SIGINT, _handler)

    ctx = multiprocessing.get_context("spawn")
    stop_event = ctx.Event()
    data_q = ctx.Queue(maxsize=config.DATA_QUEUE_MAXSIZE)

    use_archive = bool(config.ARCHIVE_ENABLED and variant.archive_dir is not None)
    archive_q = ctx.Queue(maxsize=config.ARCHIVE_QUEUE_MAXSIZE) if use_archive else None

    # ---- 数据生产者 ----
    producers: List[object] = []
    use_mp = config.RULE_SELFPLAY_BACKEND.strip().lower() == "process"
    if train_mode == "rule_selfplay":
        if use_mp:
            procs = [
                ctx.Process(
                    target=rule_teacher_worker_main,
                    args=(variant_id, wid, data_q, stop_event),
                    name=f"RuleT-{wid}", daemon=True,
                )
                for wid in range(config.RULE_SELFPLAY_CONCURRENCY)
            ]
            for p in procs:
                p.start()
            producers.extend(procs)
            print(f"{tag} 🚀 Rust 教师自对弈子进程 × {len(procs)} 已启动")
        else:
            # thread 后端：Python 侧只需 1 个 Producer 线程作为 Rust 接口调度器。
            # Rust 侧的 run_*_self_play 内部会按 RULE_SELFPLAY_CONCURRENCY 参数
            # 开启线程池并行计算，在释放 GIL 的情况下彻底吃满多核。
            producers.append(
                RuleTeacherWorker(variant, data_q, thread_stop, worker_id=0)
            )
            print(f"{tag} 🚀 Rust 教师自对弈调度线程已启动"
                  f"（RULE_SELFPLAY_BACKEND=thread，Rust 内部并发="
                  f"{config.RULE_SELFPLAY_CONCURRENCY}）")
    else:
        producers.append(ArchiveFeederWorker(variant, data_q, thread_stop))

    # ---- 训练线程 ----
    workers: List[threading.Thread] = [TrainWorker(variant, config, data_q, thread_stop)]

    for p in producers:
        p.start()
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
            stop_event=thread_stop,
        )
        monitor.start()
        print(f"{tag} 📊 系统资源监控已启动（每 {config.MONITOR_INTERVAL:.0f}s 采样）")
    elif not HAS_MONITOR:
        print(f"{tag} ⚠️ 未安装 psutil/nvidia-ml-py，系统资源监控已跳过")

    print(f"{tag} 线程已启动，正在运行（Ctrl-C 优雅退出）...\n")

    # ---- 主线程：等待 / 达到时限 / 优雅退出 ----
    start_t = time.time()

    def _request_stop(reason: str = "") -> None:
        if thread_stop.is_set():
            return
        thread_stop.set()
        if stop_event is not None:
            stop_event.set()

    try:
        while not thread_stop.is_set():
            elapsed = time.time() - start_t
            if config.MAX_RUNTIME_SECONDS > 0 and elapsed >= config.MAX_RUNTIME_SECONDS:
                print(f"\n{tag} 达到运行时限 {config.MAX_RUNTIME_SECONDS}s，优雅停止...")
                _request_stop()
                break
            if not all(w.is_alive() for w in workers):
                print(f"{tag} 有训练线程退出")
                _request_stop()
                break
            if not all(p.is_alive() for p in producers):
                while not data_q.empty() and workers[0].is_alive():
                    time.sleep(0.5)
                print(f"{tag} 有数据供给进程/线程退出")
                _request_stop()
                break
            time.sleep(2)
    except KeyboardInterrupt:
        _request_stop()

    # ---- 优雅关闭 ----
    print(f"\n{tag} 正在优雅关闭各供给进程/线程...")
    for p in producers:
        if p.is_alive():
            p.join(timeout=10)
    # 多进程模式：仍有存活子进程则强制终止（防 Rust 线程挂住退出）
    for p in producers:
        if p.is_alive():
            print(f"{tag} ⚠️ 供给进程/线程 {p.name} 未在超时内退出，强制终止")
            if hasattr(p, "terminate"):
                p.terminate()
                p.join(timeout=5)
            else:
                thread_stop.set()
                p.join(timeout=5)
    train_worker: TrainWorker = workers[0]  # type: ignore[assignment]
    if train_worker.is_alive():
        train_worker.join(timeout=30)
    if train_worker.is_alive():
        thread_stop.set()
        train_worker.join(timeout=10)
    train_worker.finalize()
    if monitor is not None and monitor.is_alive():
        monitor.join(timeout=3)
    close_summary_writer()

    for q in (data_q, archive_q):
        if q is not None:
            q.close()
            q.cancel_join_thread()

    # ---- 结束统计 ----
    tr_stats = train_worker.stats()
    history = train_worker.round_history_snapshot()
    sep = "=" * 56
    print(f"\n{sep}")
    print(f"  {variant_id} 变体 离线训练结束（mode={train_mode}）")
    print(f"{sep}")
    print(f"  训练轮次:          {tr_stats['round_num']}")
    print(f"  累计训练批次:      {tr_stats['total_batches']}")
    print(f"  平均 Loss:         {tr_stats['avg_loss']:.4f} "
          f"(Pol: {tr_stats['avg_policy_loss']:.4f}, Val: {tr_stats['avg_value_loss']:.4f})")
    if history:
        first, last = history[0], history[-1]
        print(f"  Loss 变化:         {first['train_loss']:.4f} → {last['train_loss']:.4f} "
              f"(round {first['round']} → {last['round']})")
    print(f"{sep}")
