"""banqi/trainer_cli/runners.py — 训练模式分派与运行编排。

main 按 config.TRAIN_MODE 分派：
  - "selfplay"     : 标准模型 MCTS 自对弈闭环（见 _run_selfplay）
  - "archive"      : 仅从冷存储归档数据训练（见 _run_offline）
  - "rule_selfplay": 纯规则（minimax/heuristic）自对弈生成数据训练（见 _run_offline）
"""

from __future__ import annotations

import os
import sys
import time
import signal
import multiprocessing
from typing import List

import torch

import banqi_4x8
from banqi.config import Config
from banqi.variant import get_variant, Variant
from banqi.tb_logger import (
    init_summary_writer, close_summary_writer, _log_meta_tb,
)
from banqi.utils import (
    start_memory_guard, SystemMonitor, HAS_MONITOR,
)
from banqi.training_service import TrainWorker
from banqi.self_play import sp_worker_main
from banqi.archive import archiver_worker_factory, ArchiverWorker

# 可选依赖
try:
    from banqi.pbt import PBT_ENABLED, pbt_enabled
except Exception:
    PBT_ENABLED = False
    pbt_enabled = lambda: False

try:
    from banqi.rule_self_play import rule_selfplay_spawn_main
except Exception:
    rule_selfplay_spawn_main = None

try:
    import psutil
    HAS_PSUTIL = True
except Exception:
    HAS_PSUTIL = False

HAS_TORCH = torch is not None


class _CountingQueue:
    """队列消费计数包装（观测自对弈吞吐）。"""

    def __init__(self, q):
        self.q = q
        self.consumed_games = 0
        self.consumed_samples = 0

    def get(self, *a, **k):
        item = self.q.get(*a, **k)
        if isinstance(item, dict):
            n = len(item.get("boards", []))
            if n:
                self.consumed_samples += n
                self.consumed_games += 1
        return item

    def put(self, *a, **k):
        return self.q.put(*a, **k)


def main(variant_id: str) -> None:
    """统一训练入口：按 config.TRAIN_MODE 分派训练模式。"""
    from banqi.config import make_config
    config: Config = make_config(variant_id)
    train_mode = (config.TRAIN_MODE or "selfplay").strip().lower()
    if train_mode == "selfplay":
        _run_selfplay(variant_id)
    elif train_mode in ("archive", "rule_selfplay"):
        _run_offline(variant_id, train_mode)
    else:
        raise ValueError(
            f"未知 TRAIN_MODE={train_mode!r}，可选: selfplay / archive / rule_selfplay"
        )


def _run_offline(variant_id: str, mode: str) -> None:
    """离线训练：从冷存储归档数据（MongoDB）消费训练，无自对弈闭环。

    mode = "archive"      -> 仅消费归档数据（Mongo GameDocument.samples）
    mode = "rule_selfplay"-> 启动纯规则自对弈进程生成数据（不调 MCTS 模型），
                             写入训练队列 + 可选归档。
    """
    variant = get_variant(variant_id)
    config: Config = Config.from_variant(variant_id)
    config._variant = variant
    tag = f"[{variant.id}]"

    if HAS_TORCH:
        threads_env = os.getenv(variant.env_prefix + "TORCH_THREADS")
        if threads_env:
            torch.set_num_threads(int(threads_env))
            print(f"{tag} torch.set_num_threads = {threads_env}")

    tb_log_dir = ""
    tb_ok = False
    if config.TENSORBOARD_ENABLED:
        tb_log_dir = os.path.join(config.TENSORBOARD_LOG_DIR, time.strftime("%Y%m%d-%H%M%S"))
        tb_ok = init_summary_writer(log_dir=tb_log_dir, enabled=True)
        if tb_ok:
            _log_meta_tb(config, variant_id, tb_log_dir)

    print("=" * 56)
    mode_label = "冷存储离线训练" if mode == "archive" else "纯规则自对弈训练"
    print(f"  🚀 {mode_label} 启动（变体 {variant_id}，单进程多线程）")
    print("=" * 56)
    print(f"  PREDICT_BATCH   = {config.PREDICT_BATCH}")
    print(f"  TRAIN_BATCH     = {config.TRAIN_BATCH}")
    print(f"  MONGO_URI       = {config.MONGO_URI if config.ARCHIVE_ENABLED else '（归档关闭）'}")
    print(f"  COLLECTION      = {config.DB_NAME}.{config.COLLECTION}")
    print(f"  INFER_DEVICE    = {config.INFER_DEVICE}")
    print(f"  TRAIN_DEVICE    = {config.TRAIN_DEVICE}（训练，auto 自动选择）")
    print(f"  VALUE_TARGET    = {config.VALUE_TARGET_MODE}（value 目标模式）")
    print(f"  INIT_FROM_CKPT  = {config.INIT_FROM_CHECKPOINT or '（无）'}")
    if mode == "rule_selfplay":
        print(f"  RULE_PROC       = {config.RULE_SELF_PLAY_PROCESSES}（规则自对弈进程）")
        print(f"  RULE_DEPTH      = {config.RULE_SELF_PLAY_DEPTH}（minimax 搜索深度）")
    print("=" * 56)

    start_memory_guard()
    stop_flag: List[bool] = [False]

    def _handler(signum, frame):
        if stop_flag[0]:
            print(f"\n{tag} 再次收到 Ctrl-C，强制退出")
            sys.exit(1)
        stop_flag[0] = True
        print(f"\n{tag} 收到 Ctrl-C，将在当前批结束后优雅退出...")

    signal.signal(signal.SIGINT, _handler)

    ctx = multiprocessing.get_context("spawn")
    stop_event = ctx.Event()
    data_q = ctx.Queue(maxsize=config.DATA_QUEUE_MAXSIZE)
    counting_q = _CountingQueue(data_q)

    use_archive = bool(config.ARCHIVE_ENABLED and variant.archive_dir is not None)
    archive_q = ctx.Queue(maxsize=config.ARCHIVE_QUEUE_MAXSIZE) if use_archive else None

    producers: List[object] = []
    if mode == "rule_selfplay":
        if rule_selfplay_spawn_main is None:
            print(f"{tag} ❌ rule_self_play 模块不可用，无法启动规则自对弈")
            return
        procs = [
            ctx.Process(
                target=rule_selfplay_spawn_main,
                args=(variant_id, wid, data_q, archive_q, stop_event),
                name=f"RuleSP-{wid}", daemon=True,
            )
            for wid in range(config.RULE_SELF_PLAY_PROCESSES)
        ]
        for p in procs:
            p.start()
        producers.extend(procs)
        print(f"{tag} 🚀 规则自对弈子进程 × {len(procs)} 已启动")
    else:
        # archive 模式：启动 Mongo 数据供给线程，而非子进程
        from banqi.archive import MongoDataProducer
        prod = MongoDataProducer(data_q, stop_event, variant, config)
        prod.start()
        producers.append(prod)
        print(f"{tag} 🚀 Mongo 归档数据供给线程已启动（集合 {config.DB_NAME}.{config.COLLECTION}）")

    workers = [TrainWorker(counting_q, stop_flag, variant)]
    if use_archive:
        archiver = archiver_worker_factory(archive_q, stop_flag, variant)
        if archiver is not None:
            workers.append(archiver)

    for w in workers:
        w.start()

    monitor = None
    if config.MONITOR_ENABLED and HAS_MONITOR:
        monitor = SystemMonitor(
            interval=config.MONITOR_INTERVAL, show_per_core=config.MONITOR_PER_CORE,
            csv_path=config.MONITOR_CSV_PATH, log_to_tb=bool(config.TENSORBOARD_LOG_SYS and tb_ok),
            stop_flag=stop_flag,
        )
        monitor.start()
        print(f"{tag} 📊 系统资源监控已启动（每 {config.MONITOR_INTERVAL:.0f}s 采样）")
    elif not HAS_MONITOR:
        print(f"{tag} ⚠️ 未安装 psutil/nvidia-ml-py，系统资源监控已跳过")

    print(f"{tag} 线程已启动，正在运行（Ctrl-C 优雅退出）...\n")

    start_t = time.time()

    def _request_stop(reason: str = "") -> None:
        if stop_flag[0]:
            return
        stop_flag[0] = True
        if stop_event is not None:
            stop_event.set()

    try:
        while not stop_flag[0]:
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
                print(f"{tag} 有数据供给进程/线程退出")
                _request_stop()
                break
            time.sleep(2)
    except KeyboardInterrupt:
        _request_stop()

    print(f"\n{tag} 正在优雅关闭各供给进程/线程...")
    for p in producers:
        if p.is_alive():
            p.join(timeout=10)
    for p in producers:
        if p.is_alive():
            print(f"{tag} ⚠️ 供给进程/线程 {p.name} 未在超时内退出，强制终止")
            if hasattr(p, "terminate"):
                p.terminate()
                p.join(timeout=5)
            else:
                stop_flag[0] = True
                p.join(timeout=5)
    train_worker: TrainWorker = workers[0]
    if train_worker.is_alive():
        train_worker.join(timeout=30)
    if train_worker.is_alive():
        stop_flag[0] = True
        train_worker.join(timeout=10)
    train_worker.finalize()
    if use_archive and len(workers) > 1:
        archiver_worker = workers[1]
        if archiver_worker.is_alive():
            archiver_worker.join(timeout=15)
    if monitor is not None and monitor.is_alive():
        monitor.join(timeout=3)
    close_summary_writer()

    tr_stats = train_worker.stats()
    history = train_worker.round_history_snapshot()
    sep = "=" * 56
    print(f"\n{sep}")
    print(f"  {variant_id} 变体 离线训练结束（mode={mode}）")
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


def _run_selfplay(variant_id: str) -> None:
    variant = get_variant(variant_id)
    config: Config = Config.from_variant(variant_id)
    config._variant = variant
    tag = f"[{variant.id}]"

    if HAS_TORCH:
        threads_env = os.getenv(variant.env_prefix + "TORCH_THREADS")
        if threads_env:
            torch.set_num_threads(int(threads_env))
            print(f"{tag} torch.set_num_threads = {threads_env}")

    tb_log_dir = ""
    tb_ok = False
    if config.TENSORBOARD_ENABLED:
        tb_log_dir = os.path.join(config.TENSORBOARD_LOG_DIR, time.strftime("%Y%m%d-%H%M%S"))
        tb_ok = init_summary_writer(log_dir=tb_log_dir, enabled=True)
        if tb_ok:
            _log_meta_tb(config, variant_id, tb_log_dir)

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
    stop_flag: List[bool] = [False]

    def _handler(signum, frame):
        if stop_flag[0]:
            print(f"\n{tag} 再次收到 Ctrl-C，强制退出")
            sys.exit(1)
        stop_flag[0] = True
        print(f"\n{tag} 收到 Ctrl-C，将在当前批结束后优雅退出...")

    signal.signal(signal.SIGINT, _handler)

    ctx = multiprocessing.get_context("spawn")
    stop_event = ctx.Event()
    data_q = ctx.Queue(maxsize=config.DATA_QUEUE_MAXSIZE)
    use_archive = bool(config.ARCHIVE_ENABLED and variant.archive_dir is not None)
    archive_q = ctx.Queue(maxsize=config.ARCHIVE_QUEUE_MAXSIZE) if use_archive else None

    counting_q = _CountingQueue(data_q)

    print(
        f"{tag} banqi_4x8 {variant.env_const_prefix or '标准'}常量: "
        f"BOARD=({build_const(variant, 'BOARD_CHANNELS')},"
        f"{build_const(variant, 'BOARD_ROWS')},{build_const(variant, 'BOARD_COLS')}), "
        f"SCALAR={build_const(variant, 'SCALAR_FEATURE_COUNT')}, "
        f"ACTION={build_const(variant, 'ACTION_SPACE_SIZE')}"
    )

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
          f"(推理设备={config.INFER_DEVICE}, 独立 GIL/CUDA)")

    workers = [TrainWorker(counting_q, stop_flag, variant)]
    if use_archive:
        workers.append(ArchiverWorker(archive_q, stop_flag, variant))

    for w in workers:
        w.start()

    monitor = None
    if config.MONITOR_ENABLED and HAS_MONITOR:
        monitor = SystemMonitor(
            interval=config.MONITOR_INTERVAL, show_per_core=config.MONITOR_PER_CORE,
            csv_path=config.MONITOR_CSV_PATH, log_to_tb=bool(config.TENSORBOARD_LOG_SYS and tb_ok),
            stop_flag=stop_flag,
        )
        monitor.start()
        print(f"{tag} 📊 系统资源监控已启动（每 {config.MONITOR_INTERVAL:.0f}s 采样）")
    elif not HAS_MONITOR:
        print(f"{tag} ⚠️ 未安装 psutil/nvidia-ml-py，系统资源监控已跳过"
              "（pip install psutil nvidia-ml-py）")

    print(f"{tag} 线程已启动，正在运行（Ctrl-C 优雅退出）...\n")

    start_t = time.time()
    try:
        while not stop_flag[0]:
            elapsed = time.time() - start_t
            if config.MAX_RUNTIME_SECONDS > 0 and elapsed >= config.MAX_RUNTIME_SECONDS:
                print(f"\n{tag} 达到运行时限 {config.MAX_RUNTIME_SECONDS}s，优雅停止...")
                stop_flag[0] = True
                stop_event.set()
                break
            threads_alive = [w.name for w in workers if w.is_alive()]
            if len(threads_alive) < len(workers):
                print(f"{tag} 有主进程线程退出: {threads_alive}")
                stop_flag[0] = True
                stop_event.set()
                break
            dead_procs = [p.name for p in procs if not p.is_alive()]
            if dead_procs:
                print(f"{tag} 有自对弈子进程退出: {dead_procs}，停止整个闭环")
                stop_flag[0] = True
                stop_event.set()
                break
            time.sleep(2)
    except KeyboardInterrupt:
        stop_flag[0] = True
        stop_event.set()

    print(f"\n{tag} 正在优雅关闭各线程/子进程...")
    for p in procs:
        p.join(timeout=15)
    for p in procs:
        if p.is_alive():
            print(f"{tag} ⚠️ 子进程 {p.name} 未在超时内退出，强制终止")
            p.terminate()
            p.join(timeout=5)
    train_worker: TrainWorker = workers[0]
    if train_worker.is_alive():
        train_worker.join(timeout=30)
    if train_worker.is_alive():
        stop_flag[0] = True
        train_worker.join(timeout=10)
    train_worker.finalize()
    if use_archive:
        archiver_worker: ArchiverWorker = workers[1]
        if archiver_worker.is_alive():
            archiver_worker.join(timeout=15)
    if monitor is not None and monitor.is_alive():
        monitor.join(timeout=3)
    close_summary_writer()

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


def build_const(variant, name: str) -> int:
    """读取 banqi_4x8 上带变体前缀的维度常量。"""
    return int(getattr(banqi_4x8, variant.env_const_prefix + name))
