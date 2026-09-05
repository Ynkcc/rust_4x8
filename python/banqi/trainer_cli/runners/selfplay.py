"""banqi/trainer_cli/runners/selfplay.py — 标准自对弈 + 训练闭环路径。

数据生产（Rust 侧推理或 Python spawn 子进程）→ 队列 → TrainWorker 训练，
可选 ArchiverWorker 冷存储归档与 SystemMonitor 监控。
"""

from __future__ import annotations

import multiprocessing
import os
import signal
import sys
import threading
import time
from typing import List, Optional

from banqi.archiver import ArchiverWorker
from banqi.config import Config, make_config
from banqi.memory_guard import start_memory_guard
from banqi.selfplay import sp_worker_main
from banqi.tb_logger import close_summary_writer, init_summary_writer
from banqi.training import TrainWorker
from banqi.variant import get_variant

from .context import (
    HAS_MONITOR,
    HAS_TORCH,
    SystemMonitor,
    CountingQueue,
    TeeQueue,
    build_const,
    log_meta_tb,
    setup_variant_logging,
)
from .expectimax_sidecar import ExpectimaxSidecar


def run_selfplay(variant_id: str) -> None:
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
    print(f"  🚀 自对弈 + 训练闭环启动（变体 {variant_id}，单进程多线程）")
    print("=" * 56)
    print(f"  PREDICT_BATCH   = {config.PREDICT_BATCH}")
    print(f"  TRAIN_BATCH     = {config.TRAIN_BATCH}")
    print(f"  MCTS_SIMS       = {config.MCTS_SIMS}")
    print(f"  GAMES_PER_ITER  = {config.GAMES_PER_ITER} (workers={config.NUM_WORKERS})")
    print(f"  SELF_PLAY_PROC  = {config.SELF_PLAY_PROCESSES}（spawn 子进程，独立 GIL/CUDA）")
    print(f"  MODEL_PATH      = {config.MODEL_PATH}")
    print(f"  STATE_DICT_PATH = {config.STATE_DICT_PATH}")
    print(f"  MONGO_URI       = {config.MONGO_URI if config.ARCHIVE_ENABLED else '（归档关闭）'}")
    print(f"  COLLECTION      = {config.DB_NAME}.{config.COLLECTION}")
    print(f"  INFER_DEVICE    = {config.INFER_DEVICE}（自对弈 MCTS 推理）")
    print(f"  CPU_AUX_WORKERS = {config.INFER_CPU_AUX_WORKERS}（>0 启用 GPU+CPU 混合推理）")
    print(f"  TRAIN_DEVICE    = {config.TRAIN_DEVICE}（训练，auto 自动选择）")
    print(f"  VALUE_TARGET    = {config.VALUE_TARGET_MODE}（value 目标模式）")
    if config.MONITOR_ENABLED and HAS_MONITOR:
        print(f"  MONITOR         = 每 {config.MONITOR_INTERVAL:.0f}s 采样一次"
              f"（CSV: {config.MONITOR_CSV_PATH or '关闭'}）")
    if tb_ok:
        print(f"  TENSORBOARD     = {tb_log_dir}"
              f"（tensorboard --logdir {config.TENSORBOARD_LOG_DIR} 查看）")
    if config.MAX_RUNTIME_SECONDS > 0:
        print(f"  运行时限        = {config.MAX_RUNTIME_SECONDS}s")
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

    counting_q = CountingQueue(data_q)

    # ---- NNUE 蒸馏 / Expectimax 旁路（可选，默认关闭）----
    # data_q 升级为 TeeQueue：put 同时分流到旁路 side_q（NnueDistillWorker 消费），
    # 主队列消费路径（TrainWorker）与计数逻辑完全不变。
    nnue_distill = None
    expectimax_sidecar = None
    ckpt_event_distill = threading.Event()
    ckpt_event_sidecar = threading.Event()
    ckpt_events: List[threading.Event] = []
    if bool(getattr(config, "NNUE_DISTILL_ENABLED", False)):
        from banqi.nnue.distill import NnueDistillWorker

        side_q = ctx.Queue(maxsize=max(config.DATA_QUEUE_MAXSIZE, 8))
        data_q = TeeQueue(data_q, side_q)
        counting_q = CountingQueue(data_q)
        ckpt_events.append(ckpt_event_distill)
        print(f"{tag} 🧠 NNUE 蒸馏已启用（数据旁路分流 + 周期蒸馏导出 .nnue）")
    if bool(getattr(config, "EXPECTIMAX_SIDECAR_ENABLED", False)):
        ckpt_events.append(ckpt_event_sidecar)
        print(f"{tag} ⚡ Expectimax 强自对弈旁路已启用（checkpoint 事件触发）")

    print(
        f"{tag} banqi_4x8 {variant.id} 常量: "
        f"BOARD=({build_const(variant, 'BOARD_CHANNELS')},"
        f"{build_const(variant, 'BOARD_ROWS')},{build_const(variant, 'BOARD_COLS')}), "
        f"SCALAR={build_const(variant, 'SCALAR_FEATURE_COUNT')}, "
        f"ACTION={build_const(variant, 'ACTION_SPACE_SIZE')}"
    )

    infer_side = (config.INFER_SIDE or "python").strip().lower()
    if infer_side == "rust":
        # Rust 侧推理：多线程由 Rust 内部管理（run_native_match 用 rayon 线程池，
        # num_threads 参数透传 = BATCH_CONCURRENCY）。Python 侧仅 1 个调度线程调用
        # _run_native_model_loop（只做调度 + 入队），不创建计算线程。
        # model 格式按 MODEL_BACKEND 选择：onnx -> ONNX_PATH(.onnx)，否则 -> MODEL_PATH(.pt)。
        from banqi.selfplay.config import build_self_play_config
        from banqi.selfplay.worker import _run_native_model_loop

        sp_cfg = build_self_play_config(variant)
        model_backend = (config.MODEL_BACKEND or "torchscript").strip().lower()
        # 血量差异头：自对弈侧读取独立的 HEALTH 模型路径（与 selfplay/worker.py 保持一致）
        health_enabled = bool(getattr(config, "HEALTH_VALUE_HEAD_ENABLED", False))
        health_onnx_path = (getattr(config, "HEALTH_ONNX_PATH", "") or config.ONNX_PATH) if health_enabled else config.ONNX_PATH
        health_model_path = (getattr(config, "HEALTH_MODEL_PATH", "") or config.MODEL_PATH) if health_enabled else config.MODEL_PATH
        model_path = health_onnx_path if model_backend == "onnx" else health_model_path
        procs: List[object] = [
            threading.Thread(
                target=_run_native_model_loop,
                args=(variant, model_path, config, sp_cfg, data_q, archive_q,
                      stop_event, max(1, config.GAMES_PER_ITER), 0, f"{tag}[Rust]"),
                name="SelfPlayRust-0", daemon=True,
            )
        ]
        for p in procs:
            p.start()
        print(f"{tag} 🚀 Rust 侧推理自对弈已启动（Rust 内部线程池 "
              f"num_threads={config.BATCH_CONCURRENCY}, 免 GIL, model={model_backend}）")
    else:
        # Python 侧推理：spawn 子进程（每子进程独立解释器/GIL，Python run_python_match 推理）
        procs = [
            ctx.Process(
                target=sp_worker_main,
                args=(variant_id, wid, data_q, archive_q, stop_event),
                name=f"SelfPlayProc-{wid}", daemon=True,
            )
            for wid in range(config.SELF_PLAY_PROCESSES)
        ]
        for p in procs:
            p.start()
        print(f"{tag} 🚀 自对弈子进程 × {len(procs)} 已启动 "
              f"(推理设备={config.INFER_DEVICE}, Python 推理, 独立 GIL/CUDA)")

    workers: List[threading.Thread] = [
        TrainWorker(variant, config, counting_q, thread_stop, ckpt_events=ckpt_events)
    ]
    if bool(getattr(config, "NNUE_DISTILL_ENABLED", False)):
        from banqi.nnue.distill import NnueDistillWorker

        nnue_distill = NnueDistillWorker(
            variant, config, side_q, thread_stop, ckpt_event_distill, tag=f"{tag}[NNUE]"
        )
        workers.append(nnue_distill)
    if bool(getattr(config, "EXPECTIMAX_SIDECAR_ENABLED", False)):
        expectimax_sidecar = ExpectimaxSidecar(
            variant, config, thread_stop, ckpt_event_sidecar, tag=f"{tag}[EXPMAX]"
        )
        workers.append(expectimax_sidecar)
    archiver_worker: Optional[ArchiverWorker] = None
    if use_archive:
        archiver_worker = ArchiverWorker(archive_q, thread_stop, variant)
        workers.append(archiver_worker)

    for w in workers:
        w.start()

    monitor = None
    if config.MONITOR_ENABLED and HAS_MONITOR:
        monitor = SystemMonitor(
            interval=config.MONITOR_INTERVAL, show_per_core=config.MONITOR_PER_CORE,
            csv_path=config.MONITOR_CSV_PATH, log_to_tb=bool(config.TENSORBOARD_LOG_SYS and tb_ok),
            stop_event=thread_stop,
        )
        monitor.start()
        print(f"{tag} 📊 系统资源监控已启动（每 {config.MONITOR_INTERVAL:.0f}s 采样）")
    elif not HAS_MONITOR:
        print(f"{tag} ⚠️ 未安装 psutil/nvidia-ml-py，系统资源监控已跳过"
              "（pip install psutil nvidia-ml-py）")

    print(f"{tag} 线程已启动，正在运行（Ctrl-C 优雅退出）...\n")

    start_t = time.time()
    try:
        while not thread_stop.is_set():
            elapsed = time.time() - start_t
            if config.MAX_RUNTIME_SECONDS > 0 and elapsed >= config.MAX_RUNTIME_SECONDS:
                print(f"\n{tag} 达到运行时限 {config.MAX_RUNTIME_SECONDS}s，优雅停止...")
                thread_stop.set()
                stop_event.set()
                break
            threads_alive = [w.name for w in workers if w.is_alive()]
            if len(threads_alive) < len(workers):
                print(f"{tag} 有主进程线程退出: {threads_alive}")
                thread_stop.set()
                stop_event.set()
                break
            dead_procs = [p.name for p in procs if not p.is_alive()]
            if dead_procs:
                print(f"{tag} 有自对弈子进程退出: {dead_procs}，停止整个闭环")
                thread_stop.set()
                stop_event.set()
                break
            time.sleep(2)
    except KeyboardInterrupt:
        thread_stop.set()
        stop_event.set()

    print(f"\n{tag} 正在优雅关闭各线程/子进程...")
    for p in procs:
        p.join(timeout=15)
    # 注意：infer_side == "rust" 时 procs 是 threading.Thread（无 terminate）；
    # 仅对真正的 multiprocessing.Process 做强制终止。
    for p in procs:
        if p.is_alive():
            if isinstance(p, multiprocessing.Process):
                print(f"{tag} ⚠️ 子进程 {p.name} 未在超时内退出，强制终止")
                p.terminate()
                p.join(timeout=5)
            else:
                # 线程无法强制终止，作为 daemon 线程会在主进程退出时一并结束。
                print(f"{tag} ⚠️ 线程 {p.name} 未在超时内退出（daemon 线程，随主进程退出）")
    train_worker: TrainWorker = workers[0]
    if train_worker.is_alive():
        train_worker.join(timeout=30)
    if train_worker.is_alive():
        thread_stop.set()
        train_worker.join(timeout=10)
    train_worker.finalize()
    if archiver_worker is not None:
        if archiver_worker.is_alive():
            archiver_worker.join(timeout=15)
    # 旁路 worker：蒸馏线程在 stop 前做最终蒸馏；sidecar 随 stop 退出
    for w in (nnue_distill, expectimax_sidecar):
        if w is not None and w.is_alive():
            w.join(timeout=15)
    if monitor is not None and monitor.is_alive():
        monitor.join(timeout=3)
    close_summary_writer()

    for q in (data_q, archive_q):
        if q is not None:
            q.close()
            q.cancel_join_thread()

    sp_stats = {
        "iteration": counting_q.consumed_games // max(1, config.GAMES_PER_ITER),
        "total_games": counting_q.consumed_games,
        "total_samples": counting_q.consumed_samples,
    }
    tr_stats = train_worker.stats()
    history = train_worker.round_history_snapshot()
    sep = "=" * 56
    print(f"\n{sep}")
    print(f"  {variant_id} 变体 数据收集 / 训练结束")
    print(f"{sep}")
    print(f"  累计消费局数:      {sp_stats['total_games']}")
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
