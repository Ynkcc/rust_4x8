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
import queue
import threading
import multiprocessing
from typing import Dict, List, Optional

try:
    import torch
    HAS_TORCH = True
except ImportError:  # pragma: no cover
    HAS_TORCH = False

import banqi_4x8
from banqi.archiver import ArchiverWorker
from banqi.config import Config, make_config
from banqi.memory_guard import start_memory_guard
from banqi.rule_teacher import RuleTeacherWorker, rule_teacher_worker_main
from banqi.self_play import sp_worker_main
from banqi.tb_logger import (
    add_hparams,
    add_text,
    close_summary_writer,
    init_summary_writer,
)
from banqi.training_service import TrainWorker
from banqi.variant import Variant, get_variant

# 系统资源监控（psutil + pynvml），缺失依赖时静默禁用
try:
    from banqi.system_monitor import SystemMonitor
    HAS_MONITOR = True
except ImportError:  # pragma: no cover
    HAS_MONITOR = False
    SystemMonitor = None  # type: ignore[assignment,misc]

# 可选依赖
try:
    from banqi.pbt import PBT_ENABLED, pbt_enabled
except Exception:
    PBT_ENABLED = False
    pbt_enabled = lambda: False

try:
    import psutil
    HAS_PSUTIL = True
except Exception:
    HAS_PSUTIL = False


class _CountingQueue:
    """队列消费计数包装（观测自对弈吞吐）。"""

    def __init__(self, q: "multiprocessing.Queue") -> None:
        self._q = q
        self.consumed_games = 0
        self.consumed_samples = 0

    def get(self, *args, **kwargs):
        item = self._q.get(*args, **kwargs)
        if item is not None:
            self.consumed_games += 1
            self.consumed_samples += int(item.get("num_samples", 0))
        return item

    def get_nowait(self, *args, **kwargs):
        item = self._q.get_nowait(*args, **kwargs)
        if item is not None:
            self.consumed_games += 1
            self.consumed_samples += int(item.get("num_samples", 0))
        return item

    def put(self, *args, **kwargs):
        self._q.put(*args, **kwargs)


def main(variant_id: str) -> None:
    """统一训练入口：按 config.TRAIN_MODE 分派训练模式。"""
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


class _ArchiveFeederWorker(threading.Thread):
    """归档数据供给线程（`TRAIN_MODE="archive"` 的数据源）。

    从冷存储（本地 JSONL 优先，Mongo 兜底）加载历史 episode，
    周期性压入 data_q 供 `TrainWorker` 消费训练。不启动自对弈。
    """

    def __init__(
        self,
        variant: Variant,
        data_q: "queue.Queue",
        stop_flag: "List[bool]",
    ) -> None:
        super().__init__(name=f"ArchiveFeederWorker-{variant.id}", daemon=True)
        self.variant = variant
        self.cfg = make_config(variant.id)
        self.tag = f"[ArchiveFeeder-{variant.id}]"
        self.data_q = data_q
        self.stop_flag = stop_flag
        self.total_games = 0

    def _resolve_archive_dir(self) -> Optional[str]:
        """解析归档目录：优先 ARCHIVE_TRAIN_DIR，其次 variant.archive_dir。"""
        from banqi.storage import list_jsonl_files
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cands = [
            self.cfg.ARCHIVE_TRAIN_DIR or "",
            self.variant.archive_dir or "",
            os.path.join(here, "training_data", f"archive_{self.variant.id}"),
            os.path.join(here, "training_data", f"archive_{self.variant.id}_imitate"),
        ]
        for d in cands:
            if d and os.path.isdir(d) and list_jsonl_files(d):
                return d
        return None

    def _load_from_mongo(self, limit_games: Optional[int]) -> List[Dict]:
        """从 MongoDB 读取该变体归档局，转成与 `GameEpisode.to_dict()` 兼容的 episode dict。"""
        try:
            import pymongo
            client = pymongo.MongoClient(self.cfg.MONGO_URI, serverSelectionTimeoutMS=5000)
            client.admin.command("ping")
            col = client[self.cfg.DB_NAME][self.cfg.COLLECTION]
        except Exception as exc:  # pragma: no cover
            print(f"{self.tag} ⚠️ MongoDB 不可用（归档兜底跳过）: {exc}")
            return []

        episodes: List[Dict] = []
        try:
            query: Dict = {}
            cursor = col.find(query).limit(limit_games) if limit_games else col.find(query)
            for doc in cursor:
                samples = doc.get("samples") or []
                if not samples:
                    continue
                ep = {
                    "boards": [s["board_state"] for s in samples],
                    "scalars": [s["scalar_state"] for s in samples],
                    "policies": [s["policy_probs"] for s in samples],
                    "mcts_values": [s.get("mcts_value", 0.0) for s in samples],
                    "completed_qs": [s.get("completed_q", 0.0) for s in samples],
                    "root_visits": [s.get("root_visit_count", 0) for s in samples],
                    "game_results": [s.get("game_result_value", 0.0) for s in samples],
                    "action_masks": [s["action_mask"] for s in samples],
                    "health_diffs": [s.get("health_diff", 0.0) for s in samples],
                    "game_length": int(doc.get("game_length", len(samples))),
                    "winner": doc.get("winner"),
                    "num_samples": len(samples),
                }
                episodes.append(ep)
        finally:
            client.close()
        return episodes

    def _put(self, item: Dict) -> None:
        while not self.stop_flag[0]:
            try:
                self.data_q.put(item, timeout=0.5)
                return
            except Exception:  # queue.Full
                continue

    def run(self) -> None:
        from banqi.storage import load_jsonl_episodes
        archive_dir = self._resolve_archive_dir()

        if archive_dir is None:
            print(f"{self.tag} ⚠️ 未找到本地归档目录（{self.cfg.ARCHIVE_TRAIN_DIR or self.variant.archive_dir}），"
                  f"将尝试从 MongoDB 读取（{self.cfg.DB_NAME}.{self.cfg.COLLECTION}）...")
        else:
            print(f"{self.tag} 🗃️ 使用本地归档目录: {archive_dir}")

        limit_games = self.cfg.ARCHIVE_TRAIN_GAMES or None
        total_rounds = max(1, self.cfg.ARCHIVE_TRAIN_ROUNDS)
        poll = max(1.0, self.cfg.ARCHIVE_POLL_INTERVAL)

        for r in range(total_rounds):
            if self.stop_flag[0]:
                break
            try:
                t0 = time.time()
                if archive_dir is not None:
                    episodes = load_jsonl_episodes(archive_dir, limit_games=limit_games)
                else:
                    episodes = self._load_from_mongo(limit_games)
                if not episodes:
                    print(f"{self.tag} ⚠️ 归档为空，等待新数据...")
                    time.sleep(poll)
                    continue
                # 每次灌入全部（或限制量），并标记 round 号便于观测
                for ep in episodes:
                    if self.stop_flag[0]:
                        break
                    ep = dict(ep)
                    ep.setdefault("num_samples", len(ep.get("boards") or []))
                    ep.setdefault("iteration", r)
                    self._put(ep)
                    self.total_games += 1
                print(f"{self.tag} 📦 第 {r + 1}/{total_rounds} 轮灌入 {len(episodes)} 局"
                      f"（累计 {self.total_games}，耗时 {time.time() - t0:.1f}s）")
            except Exception as exc:  # pragma: no cover
                print(f"{self.tag} ⚠️ 归档加载失败: {exc}")
            time.sleep(poll)

        print(f"{self.tag} 归档供给完成，累计 {self.total_games} 局")

    def stats(self) -> Dict[str, int]:
        return {"total_games": self.total_games}


def _log_meta_tb(config: Config, variant_id: str, tb_log_dir: str) -> None:
    """TensorBoard 运行元信息：HParams + 文本标注（启动时调用一次）。

    add_hparams 在 log_dir/hparams 子目录写入超参数表，跨多次运行可在
    TensorBoard 的 HParams 面板中对比。metric_dict 传空（训练指标另行记录）。
    """
    add_text("meta/variant", variant_id, 0)
    add_text("meta/run_id", os.path.basename(tb_log_dir) if tb_log_dir else "", 0)
    add_text("meta/train_mode", config.TRAIN_MODE or "selfplay", 0)
    add_text("meta/value_target", config.VALUE_TARGET_MODE, 0)
    add_text(
        "meta/device",
        f"train={config.TRAIN_DEVICE} infer={config.INFER_DEVICE} "
        f"cpu_aux={config.INFER_CPU_AUX_WORKERS}",
        0,
    )
    add_text(
        "meta/augment",
        f"enabled={config.DATA_AUGMENT_ENABLED} transforms={config.DATA_AUGMENT_TRANSFORMS}",
        0,
    )
    hparams = {
        "variant": variant_id,
        "mcts_sims": config.MCTS_SIMS,
        "max_considered_actions": config.MAX_CONSIDERED_ACTIONS,
        "games_per_iter": config.GAMES_PER_ITER,
        "train_batch": config.TRAIN_BATCH,
        "learning_rate": config.LEARNING_RATE,
        "min_lr": config.MIN_LR,
        "lr_decay_steps": config.LR_DECAY_STEPS,
        "train_epochs_per_round": config.TRAIN_EPOCHS_PER_ROUND,
        "weight_decay": config.WEIGHT_DECAY,
        "max_buffer": config.MAX_SAMPLE_BUFFER_SIZE,
        "min_samples_to_start": config.MIN_SAMPLES_TO_START,
        "value_target": config.VALUE_TARGET_MODE,
        "data_augment": config.DATA_AUGMENT_ENABLED,
        "eval_match_rounds": config.EVAL_MATCH_ROUNDS,
        "eval_match_games": config.EVAL_MATCH_GAMES,
        "eval_match_opponents": config.EVAL_MATCH_OPPONENTS,
        "eval_match_vs_prev": config.EVAL_MATCH_VS_PREV,
    }
    add_hparams({k: str(v) for k, v in hparams.items()}, {})


class _TeeStream:
    """将 stdout/stderr 同时输出到控制台与日志文件。"""

    def __init__(self, original_stream, file_obj) -> None:
        self.original_stream = original_stream
        self.file_obj = file_obj
        self._is_tee = True

    def write(self, data: str) -> None:
        self.original_stream.write(data)
        self.file_obj.write(data)
        self.file_obj.flush()

    def flush(self) -> None:
        self.original_stream.flush()
        self.file_obj.flush()

    def isatty(self) -> bool:
        return getattr(self.original_stream, "isatty", lambda: False)()


def setup_variant_logging(variant: Variant) -> str:
    """初始化变体运行日志：在 variant.logs_dir 中创建日志文件并将 print/logging 同步落盘。"""
    import logging
    os.makedirs(variant.logs_dir, exist_ok=True)
    log_file = os.path.join(variant.logs_dir, f"train_{time.strftime('%Y%m%d_%H%M%S')}.log")

    if not getattr(sys.stdout, "_is_tee", False):
        f = open(log_file, "a", encoding="utf-8")
        sys.stdout = _TeeStream(sys.stdout, f)
        sys.stderr = _TeeStream(sys.stderr, f)

    root_logger = logging.getLogger()
    handler = logging.FileHandler(log_file, encoding="utf-8")
    formatter = logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s")
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.INFO)
    return log_file


def _run_offline(variant_id: str, train_mode: str) -> None:
    variant = get_variant(variant_id)
    config: Config = make_config(variant_id)
    config._variant = variant
    tag = f"[{variant.id}]"

    log_file = setup_variant_logging(variant)
    print(f"{tag} 📝 运行日志记录至: {log_file}")

    if HAS_TORCH:
        threads_env = os.getenv("TORCH_THREADS")
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
                RuleTeacherWorker(variant, data_q, lambda: stop_flag[0], worker_id=0)
            )
            print(f"{tag} 🚀 Rust 教师自对弈调度线程已启动"
                  f"（RULE_SELFPLAY_BACKEND=thread，Rust 内部并发="
                  f"{config.RULE_SELFPLAY_CONCURRENCY}）")
    else:
        producers.append(_ArchiveFeederWorker(variant, data_q, stop_flag))

    # ---- 训练线程 ----
    workers: List[threading.Thread] = [TrainWorker(data_q, stop_flag, variant)]

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
            stop_flag=stop_flag,
        )
        monitor.start()
        print(f"{tag} 📊 系统资源监控已启动（每 {config.MONITOR_INTERVAL:.0f}s 采样）")
    elif not HAS_MONITOR:
        print(f"{tag} ⚠️ 未安装 psutil/nvidia-ml-py，系统资源监控已跳过")

    print(f"{tag} 线程已启动，正在运行（Ctrl-C 优雅退出）...\n")

    # ---- 主线程：等待 / 达到时限 / 优雅退出 ----
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
                stop_flag[0] = True
                p.join(timeout=5)
    train_worker: TrainWorker = workers[0]  # type: ignore[assignment]
    if train_worker.is_alive():
        train_worker.join(timeout=30)
    if train_worker.is_alive():
        stop_flag[0] = True
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


def _run_selfplay(variant_id: str) -> None:
    variant = get_variant(variant_id)
    config: Config = make_config(variant_id)
    config._variant = variant
    tag = f"[{variant.id}]"

    log_file = setup_variant_logging(variant)
    print(f"{tag} 📝 运行日志记录至: {log_file}")

    if HAS_TORCH:
        threads_env = os.getenv("TORCH_THREADS")
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
        model_path = config.ONNX_PATH if model_backend == "onnx" else config.MODEL_PATH
        procs = [
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

    workers: List[threading.Thread] = [TrainWorker(counting_q, stop_flag, variant)]
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
        stop_flag[0] = True
        train_worker.join(timeout=10)
    train_worker.finalize()
    if use_archive:
        archiver_worker: ArchiverWorker = workers[1]  # type: ignore[assignment]
        if archiver_worker.is_alive():
            archiver_worker.join(timeout=15)
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


_const_dims_cache: Dict = {}


def build_const(variant, name: str) -> int:
    """从 Rust 统一 `variant_dims(variant_id)` API 取变体维度。

    替代按 env_const_prefix 拼接 `GAME4X4_*` 等模块级常量名（后者已不作为 Python 侧
    维度来源）。结果按变体缓存。
    """
    vid = variant.id
    if vid not in _const_dims_cache:
        _const_dims_cache[vid] = dict(banqi_4x8.variant_dims(vid))
    dims = _const_dims_cache[vid]
    key = {
        "BOARD_ROWS": "board_rows",
        "BOARD_COLS": "board_cols",
        "BOARD_CHANNELS": "board_channels",
        "SCALAR_FEATURE_COUNT": "scalar_feature_count",
        "ACTION_SPACE_SIZE": "action_space_size",
    }[name]
    return int(dims[key])
