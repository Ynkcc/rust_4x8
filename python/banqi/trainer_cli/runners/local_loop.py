"""banqi/trainer_cli/runners/local_loop.py — 单机双进程闭环（统一训练架构 L5 单机形态）。

编排：banqi-collector 子进程（Rust ONNX 推理自对弈 → LocalEpisodeStore 落盘）
     + TrainWorker（经 LocalEpisodeStore 队列语义消费，全程无 Python 推理回调）
     + RegistryPublisher 线程（新导出 onnx → LocalModelRegistry 指针 → collector 换网）。

与 selfplay 路径互不影响；TRAIN_MODE=local 时启用。
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import time
from typing import Optional

from banqi.config import Config, make_config
from banqi.infra import LocalEpisodeStore, LocalModelRegistry
from banqi.memory_guard import start_memory_guard
from banqi.tb_logger import close_summary_writer, init_summary_writer
from banqi.training import TrainWorker
from banqi.variant import get_variant

from .context import CountingQueue, log_meta_tb, setup_variant_logging


def _default_onnx_of(config: Config) -> str:
    onnx_path = getattr(config, "ONNX_PATH", "") or ""
    if onnx_path:
        return onnx_path
    base, _ = os.path.splitext(config.MODEL_PATH)
    return base + ".onnx"


class RegistryPublisher(threading.Thread):
    """watch Trainer 导出的 onnx 文件，变更即 publish 到 Registry 指针。"""

    def __init__(self, registry: LocalModelRegistry, model_path: str, stop: threading.Event,
                 tag: str) -> None:
        super().__init__(name="RegistryPublisher", daemon=True)
        self.registry = registry
        self.model_path = model_path
        self.stop = stop
        self.tag = tag
        self.published = 0

    def run(self) -> None:
        last_mtime: Optional[float] = None
        while not self.stop.is_set():
            try:
                mtime = os.path.getmtime(self.model_path)
                if mtime != last_mtime and os.path.getsize(self.model_path) > 0:
                    if last_mtime is not None:  # 首次观测不重复发布
                        self.registry.publish(self.model_path)
                        self.published += 1
                    last_mtime = mtime
            except FileNotFoundError:
                pass
            except Exception as exc:
                print(f"{self.tag} ⚠️ RegistryPublisher 异常: {exc}")
            self.stop.wait(2.0)


def run_local_loop(variant_id: str) -> None:
    variant = get_variant(variant_id)
    config: Config = make_config(variant_id)
    config._variant = variant
    tag = f"[{variant.id}][local]"

    log_file = setup_variant_logging(variant)
    print(f"{tag} 📝 运行日志记录至: {log_file}")

    tb_ok = False
    if config.TENSORBOARD_ENABLED:
        tb_log_dir = os.path.join(config.TENSORBOARD_LOG_DIR, time.strftime("%Y%m%d-%H%M%S"))
        tb_ok = init_summary_writer(log_dir=tb_log_dir, enabled=True)
        if tb_ok:
            log_meta_tb(config, variant_id, tb_log_dir)

    start_memory_guard()

    episode_dir = getattr(config, "EPISODE_STORE_DIR", "") or "outputs/episodes"
    pointer = getattr(config, "MODEL_REGISTRY_POINTER", "") or \
        f"outputs/{variant.id}_latest.onnx"
    collector_bin = getattr(config, "COLLECTOR_BIN", "") or "banqi-collector"

    store = LocalEpisodeStore(episode_dir, variant.id)
    registry = LocalModelRegistry(pointer)
    counting_q = CountingQueue(store)

    onnx_path = _default_onnx_of(config)
    sep = "=" * 56
    print(sep)
    print(f"  🚀 单机双进程闭环启动（变体 {variant_id}，Trainer + Rust Collector）")
    print(f"  COLLECTOR_BIN = {collector_bin}")
    print(f"  EPISODE_STORE = {episode_dir}/{variant.id}/*.jsonl.gz")
    print(f"  MODEL_POINTER = {pointer}（watch: {onnx_path}）")
    print(f"  MCTS_SIMS     = {config.MCTS_SIMS}, GAMES_PER_ITER = {config.GAMES_PER_ITER}")
    print(sep)

    thread_stop = threading.Event()

    def _handler(signum, frame):
        if thread_stop.is_set():
            sys.exit(1)
        thread_stop.set()
        print(f"\n{tag} 收到 Ctrl-C，将在当前批结束后优雅退出...")

    signal.signal(signal.SIGINT, _handler)

    collector_cmd = [
        collector_bin,
        "--variant", variant.id,
        "--backend", "local",
        "--model-path", pointer,
        "--out-dir", episode_dir,
        "--device", str(getattr(config, "INFER_DEVICE", "auto") or "auto"),
        "--mcts-sims", str(config.MCTS_SIMS),
        "--max-considered-actions", str(config.MAX_CONSIDERED_ACTIONS),
        "--games-per-iter", str(config.GAMES_PER_ITER),
    ]
    print(f"{tag} 启动 Collector: {' '.join(collector_cmd)}")
    collector = subprocess.Popen(collector_cmd)

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
            if collector.poll() is not None:
                print(f"{tag} ⚠️ Collector 进程退出（code={collector.returncode}），停止闭环")
                thread_stop.set()
                break
            if not train_worker.is_alive():
                print(f"{tag} ⚠️ TrainWorker 已退出，停止闭环")
                thread_stop.set()
                break
            thread_stop.wait(2.0)
    finally:
        thread_stop.set()
        if collector.poll() is None:
            collector.terminate()
            try:
                collector.wait(timeout=15)
            except subprocess.TimeoutExpired:
                collector.kill()
        train_worker.join(timeout=60)
        if train_worker.is_alive():
            train_worker.join(timeout=10)
        train_worker.finalize()
        publisher.join(timeout=3)
        close_summary_writer()

    tr_stats = train_worker.stats()
    print(f"\n{sep}")
    print(f"  {variant_id} local 闭环结束")
    print(f"  累计训练批次: {tr_stats['total_batches']}, 轮次: {tr_stats['round_num']}, "
          f"平均 Loss: {tr_stats['avg_loss']:.4f}")
    print(f"  模型发布次数: {publisher.published}")
    print(sep)
