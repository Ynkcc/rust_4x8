"""banqi/trainer_cli/runners/distributed.py — 分布式形态 Trainer（统一训练架构 L5）。

编排：R2EpisodeStore（拉取 worker 直传的 episode 批次）
     + TrainWorker（经队列语义消费训练）
     + RegistryPublisher 线程（新导出 onnx → R2 上传 + RegisterNetwork 登记）。

与 local_loop 的差异：episode 来源为 R2 而非本地目录；新模型发布走
scheduler（gatekeeper 判停晋级）而非本地指针文件。本进程不含 Collector，
自对弈 worker（banqi-collector --backend scheduler）独立部署启动。
TRAIN_MODE=distributed 时启用。
"""

from __future__ import annotations

import os
import signal
import sys
import threading
import time

from banqi.config import Config, make_config
from banqi.infra import SchedulerEpisodeStore, SchedulerModelRegistry
from banqi.memory_guard import start_memory_guard
from banqi.tb_logger import close_summary_writer, init_summary_writer
from banqi.training import TrainWorker
from banqi.variant import get_variant

from .context import CountingQueue, log_meta_tb, setup_variant_logging
from .local_loop import RegistryPublisher, _default_onnx_of


def run_distributed(variant_id: str) -> None:
    variant = get_variant(variant_id)
    config: Config = make_config(variant_id)
    config._variant = variant
    tag = f"[{variant.id}][distributed]"

    log_file = setup_variant_logging(variant)
    print(f"{tag} 📝 运行日志记录至: {log_file}")

    tb_ok = False
    if config.TENSORBOARD_ENABLED:
        tb_log_dir = os.path.join(config.TENSORBOARD_LOG_DIR, time.strftime("%Y%m%d-%H%M%S"))
        tb_ok = init_summary_writer(log_dir=tb_log_dir, enabled=True)
        if tb_ok:
            log_meta_tb(config, variant_id, tb_log_dir)

    start_memory_guard()

    store = SchedulerEpisodeStore()
    registry = SchedulerModelRegistry()
    counting_q = CountingQueue(store)

    onnx_path = _default_onnx_of(config)
    sep = "=" * 56
    print(sep)
    print(f"  🚀 分布式 Trainer 启动（变体 {variant_id}，无 Collector）")
    print(f"  EPISODE_SOURCE = scheduler ListEpisodes（预签名 GET 拉取）")
    print(f"  SCHEDULER      = {registry.endpoint}（SignNetworkUpload + RegisterNetwork）")
    print(f"  WATCH ONNX     = {onnx_path}")
    print(sep)

    thread_stop = threading.Event()

    def _handler(signum, frame):
        if thread_stop.is_set():
            sys.exit(1)
        thread_stop.set()
        print(f"\n{tag} 收到 Ctrl-C，将在当前批结束后优雅退出...")

    signal.signal(signal.SIGINT, _handler)

    publisher = RegistryPublisher(registry, onnx_path, thread_stop, tag)
    publisher.start()

    train_worker = TrainWorker(variant, config, counting_q, thread_stop)
    train_worker.start()

    start_t = time.time()
    try:
        while not thread_stop.is_set():
            if config.MAX_RUNTIME_SECONDS > 0 and \
                    time.time() - start_t >= config.MAX_RUNTIME_SECONDS:
                print(f"{tag} 达到运行时限 {config.MAX_RUNTIME_SECONDS}s，优雅停止...")
                thread_stop.set()
                break
            if not train_worker.is_alive():
                print(f"{tag} ⚠️ TrainWorker 已退出，停止闭环")
                thread_stop.set()
                break
            thread_stop.wait(2.0)
    finally:
        thread_stop.set()
        train_worker.join(timeout=60)
        if train_worker.is_alive():
            train_worker.join(timeout=10)
        train_worker.finalize()
        publisher.join(timeout=3)
        close_summary_writer()

    tr_stats = train_worker.stats()
    print(f"\n{sep}")
    print(f"  {variant_id} distributed trainer 结束")
    print(f"  累计训练批次: {tr_stats['total_batches']}, 轮次: {tr_stats['round_num']}, "
          f"平均 Loss: {tr_stats['avg_loss']:.4f}")
    print(f"  模型发布次数: {publisher.published}")
    print(sep)
