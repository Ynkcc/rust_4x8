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

import multiprocessing
import os
import queue
import signal
import sys
import threading
import time
from typing import Dict, List, Optional

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
from banqi.rule_self_play import RuleSelfPlayWorker, rule_sp_worker_main
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


class _CountingQueue:
    """包装 multiprocessing.Queue：主进程侧统计已消费的局数/样本数（供结束统计）。

    子进程把 episode put 到原始 Queue；TrainWorker 在主进程 get 时计数。
    注意：只包装 get 侧，put 透传（子进程不需要计数包装）。
    """

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

    def qsize(self):
        """透传底层队列积压数（供 TB queue/backlog 监控）。"""
        return self._q.qsize()


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
        """从 MongoDB 读取该变体归档局，转成与 `GameEpisode.to_dict()` 兼容的 episode dict。

        归档默认写 Mongo（当 Mongo 可用且 ARCHIVE_ENABLED）；本地 JSONL 不可用时
        兜底从 Mongo 读取。Mongo 不可用则返回空列表。
        """
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
    add_text("meta/train_mode", config.TRAIN_MODE, 0)
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
        "temperature_steps": config.TEMPERATURE_STEPS,
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


def _run_offline(variant_id: str, train_mode: str) -> None:
    """离线训练：`TRAIN_MODE="archive"`（从归档）或 `"rule_selfplay"`（纯规则自对弈）。

    不启动神经网络模型自对弈；由数据供给线程（归档供给 / 纯规则自对弈）生产
    episode 压入 data_q，供 `TrainWorker` 消费训练。
    """
    variant = get_variant(variant_id)
    config: Config = make_config(variant_id)
    tag = f"[{variant.id}]"

    # ---- 进程级线程数（可选）----
    if HAS_TORCH:
        threads_env = os.getenv(variant.env_prefix + "TORCH_THREADS")
        if threads_env:
            torch.set_num_threads(int(threads_env))
            print(f"{tag} torch.set_num_threads = {threads_env}")

    # ---- TensorBoard 初始化 ----
    tb_log_dir = ""
    tb_ok = False
    if config.TENSORBOARD_ENABLED:
        tb_log_dir = os.path.join(
            config.TENSORBOARD_LOG_DIR, time.strftime("%Y%m%d-%H%M%S")
        )
        tb_ok = init_summary_writer(log_dir=tb_log_dir, enabled=True)
        if tb_ok:
            _log_meta_tb(config, variant_id, tb_log_dir)

    print("=" * 56)
    print(f"  🚀 离线训练启动（变体 {variant_id}，mode={train_mode}，单进程多线程）")
    print("=" * 56)
    print(f"  TRAIN_MODE   = {train_mode}")
    print(f"  TRAIN_DEVICE = {config.TRAIN_DEVICE}（训练）")
    print(f"  VALUE_TARGET = {config.VALUE_TARGET_MODE}")
    print(f"  AUGMENT      = {'✅ 开' if config.DATA_AUGMENT_ENABLED else '❌ 关'}")
    if train_mode == "archive":
        print(f"  ARCHIVE_DIR  = {config.ARCHIVE_TRAIN_DIR or '自动探测'}")
        print(f"  ARCHIVE_GAMES= {config.ARCHIVE_TRAIN_GAMES or '全部'}")
        print(f"  ROUNDS       = {config.ARCHIVE_TRAIN_ROUNDS}")
    else:
        print(f"  RULE_TYPE    = {config.RULE_SELFPLAY_TYPE}")
        print(f"  RULE_DEPTH/SIMS = {config.RULE_SELFPLAY_DEPTH}/{config.RULE_SELFPLAY_SIMS}")
        print(f"  GAMES/BATCH  = {config.RULE_SELFPLAY_GAMES}")
        print(f"  ROUNDS       = {config.RULE_SELFPLAY_ROUNDS}")
        print(f"  TEMPERATURE  = {config.RULE_SELFPLAY_TEMPERATURE}")
        backend = (config.RULE_SELFPLAY_BACKEND or "thread").strip().lower()
        print(f"  CONCURRENCY  = {config.RULE_SELFPLAY_CONCURRENCY} × {backend}"
              f"（多线程/多进程纯规则自对弈）")
    print("=" * 56)

    # ---- 内存看门守护 ----
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

    # ---- 数据队列与数据供给（按并发后端分派）----
    # rule_selfplay 后端：
    #   "thread"  多线程：线程安全 queue.Queue + N 个 RuleSelfPlayWorker 线程
    #   "process" 多进程：multiprocessing spawn Queue/Event + N 个 rule_sp_worker_main 子进程
    # archive 模式始终使用线程供给（_ArchiveFeederWorker）。
    rule_backend = "thread"
    if train_mode == "rule_selfplay":
        rule_backend = (config.RULE_SELFPLAY_BACKEND or "thread").strip().lower()
        if rule_backend not in ("thread", "process"):
            print(f"{tag} ⚠️ 未知 RULE_SELFPLAY_BACKEND={rule_backend!r}，回退到 'thread'")
            rule_backend = "thread"

    stop_event = None
    ctx = None
    if train_mode == "rule_selfplay" and rule_backend == "process":
        # 跨进程队列（spawn 子进程纯规则自对弈）
        ctx = multiprocessing.get_context("spawn")
        stop_event = ctx.Event()
        data_q = ctx.Queue(maxsize=config.DATA_QUEUE_MAXSIZE)  # type: ignore[assignment]
    else:
        # 线程内数据队列（多线程供给 / 归档供给）
        data_q = queue.Queue(maxsize=config.DATA_QUEUE_MAXSIZE)  # type: ignore[assignment]

    producers = []  # 线程或进程对象的统一列表
    concurrency = max(1, config.RULE_SELFPLAY_CONCURRENCY)
    if train_mode == "rule_selfplay":
        if rule_backend == "process":
            for wid in range(concurrency):
                producers.append(
                    ctx.Process(  # type: ignore[union-attr]
                        target=rule_sp_worker_main,
                        args=(variant_id, wid, data_q, stop_event),
                        name=f"RuleSPProc-{wid}",
                        daemon=True,
                    )
                )
            print(f"{tag} 🚀 纯规则自对弈子进程 × {concurrency} 已创建"
                  f"（RULE_SELFPLAY_BACKEND=process）")
        else:
            for wid in range(concurrency):
                producers.append(
                    RuleSelfPlayWorker(variant, data_q, lambda: stop_flag[0],
                                       worker_id=wid)
                )
            print(f"{tag} 🚀 纯规则自对弈线程 × {concurrency} 已创建"
                  f"（RULE_SELFPLAY_BACKEND=thread）")
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


def main(variant_id: str) -> None:
    """统一训练入口：按 config.TRAIN_MODE 分派训练模式。

    - "selfplay"     : 标准模型 MCTS 自对弈闭环（默认，见 _run_selfplay）
    - "archive"      : 仅从冷存储归档数据训练（见 _run_offline）
    - "rule_selfplay": 纯规则（minimax/heuristic）自对弈生成数据训练（见 _run_offline）
    """
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


def _run_selfplay(variant_id: str) -> None:
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

    # ---- 跨进程队列（spawn 多进程自对弈）----
    ctx = multiprocessing.get_context("spawn")
    stop_event = ctx.Event()
    data_q = ctx.Queue(maxsize=config.DATA_QUEUE_MAXSIZE)
    use_archive = bool(config.ARCHIVE_ENABLED and variant.archive_dir is not None)
    archive_q = (
        ctx.Queue(maxsize=config.ARCHIVE_QUEUE_MAXSIZE) if use_archive else None
    )

    # 主进程侧消费计数包装（供结束统计 / 吞吐观察）
    counting_q = _CountingQueue(data_q)

    print(
        f"{tag} banqi_4x8 {variant.env_const_prefix or '标准'}常量: "
        f"BOARD=({build_const(variant, 'BOARD_CHANNELS')},"
        f"{build_const(variant, 'BOARD_ROWS')},{build_const(variant, 'BOARD_COLS')}), "
        f"SCALAR={build_const(variant, 'SCALAR_FEATURE_COUNT')}, "
        f"ACTION={build_const(variant, 'ACTION_SPACE_SIZE')}"
    )

    # ---- 自对弈子进程组（每个独立 GIL + CUDA context，spawn）----
    procs = [
        ctx.Process(
            target=sp_worker_main,
            args=(variant_id, wid, data_q, archive_q, stop_event),
            name=f"SelfPlayProc-{wid}",
            daemon=True,
        )
        for wid in range(config.SELF_PLAY_PROCESSES)
    ]
    for p in procs:
        p.start()
    print(f"{tag} 🚀 自对弈子进程 × {len(procs)} 已启动 "
          f"(推理设备={config.INFER_DEVICE}, 独立 GIL/CUDA)")

    # ---- 主进程线程组（训练 / 归档；TrainWorker 不调 Rust，线程足够）----
    workers = [
        TrainWorker(counting_q, stop_flag, variant),
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

    # ---- 优雅关闭 ----
    print(f"\n{tag} 正在优雅关闭各线程/子进程...")
    for p in procs:
        p.join(timeout=15)
    for p in procs:  # 仍有存活子进程：强制终止（防 Rust 线程挂住退出）
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

    # ---- 结束统计（自对弈侧来自主进程消费计数）----
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


if __name__ == "__main__":
    vid = sys.argv[1] if len(sys.argv) > 1 else "4x8"
    main(vid)
