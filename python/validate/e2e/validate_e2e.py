"""
validate_e2e.py — 端到端 CPU 冒烟测试（纯 CPU）。

用极小配置跑通真实链路：banqi_4x8 自对弈 → episode_to_samples → DataBuffer →
run_training_epochs → save_checkpoint → FileSaver 归档。

验证三线程编排（SelfPlayWorker / TrainWorker / ArchiverWorker）可启动、消费、
训练、归档、优雅退出，全程无死锁。

通过运行时覆盖 config 单例实例属性来收紧参数（不改 config.py）。
按用户要求，checkpoint 输出到当前目录（覆盖 banqi_model_latest.*）。

运行：python python/validate/validate_e2e.py
"""

from __future__ import annotations

import os
import queue
import time

import numpy as np
import torch

import os
import sys

_VALIDATE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PYTHON_DIR = os.path.dirname(_VALIDATE_DIR)
_BANQI_DIR = os.path.join(_PYTHON_DIR, "banqi")
for _d in (_PYTHON_DIR, _BANQI_DIR, _VALIDATE_DIR):
    if _d not in sys.path:
        sys.path.insert(0, _d)

import validate_common  # noqa: F401
from validate_common import Reporter, run_part

from banqi.config import config


def _override_tiny_config() -> None:
    """运行时把 config 单例实例属性收紧到极小值（仅影响本进程，不改 config.py）。"""
    config.MCTS_SIMS = 2
    config.MAX_CONSIDERED_ACTIONS = 4
    config.TEMPERATURE_STEPS = 1
    config.GAMES_PER_ITER = 2
    config.NUM_WORKERS = 1
    config.GAMES_PER_WORKER = 1
    config.TRAIN_BATCH = 4
    config.TRAIN_EPOCHS_PER_ROUND = 1
    config.MIN_SAMPLES_TO_START = 4
    config.MAX_SAMPLE_BUFFER_SIZE = 1000
    config.QUEUE_FETCH_BATCH = 2
    config.DATA_QUEUE_MAXSIZE = 8
    config.ARCHIVE_QUEUE_MAXSIZE = 16
    config.ARCHIVE_BATCH = 2
    config.CHECKPOINT_EVERY_N_ROUNDS = 1
    config.LEARNING_RATE = 2e-4


def test_full_pipeline() -> None:
    rep = Reporter("end-to-end CPU smoke")
    _override_tiny_config()

    from banqi.self_play import SelfPlayWorker, build_predictor, build_self_play_config
    from banqi.training_service import TrainWorker
    from banqi.archiver import ArchiverWorker

    print("      极小配置: MCTS_SIMS=2, GAMES_PER_ITER=2, TRAIN_BATCH=4, "
          "MIN_SAMPLES=4")

    predictor, _device = build_predictor(config.MODEL_PATH, device_str="cpu")
    sp_cfg = build_self_play_config()

    data_q: "queue.Queue" = queue.Queue(maxsize=config.DATA_QUEUE_MAXSIZE)
    archive_q: "queue.Queue" = queue.Queue(maxsize=config.ARCHIVE_QUEUE_MAXSIZE)
    stop_flag = [False]

    workers = [
        SelfPlayWorker(predictor, sp_cfg, data_q, archive_q, stop_flag, worker_id=0),
        TrainWorker(data_q, stop_flag),
        ArchiverWorker(archive_q, stop_flag, mongo_uri=""),  # 强制降级 FileSaver
    ]
    for w in workers:
        w.start()

    # 运行若干秒后优雅退出
    deadline = time.time() + 40
    while time.time() < deadline:
        if train_worker_stats_ok(workers[1]):
            break
        time.sleep(0.5)
    stop_flag[0] = True

    # join 各线程（验证无死锁）
    for w in workers:
        w.join(timeout=15)
        rep.check(not w.is_alive(), f"{w.name} exited cleanly")

    # ---- 收集统计 ----
    sp_stats = workers[0].stats()
    tr_stats = workers[1].stats()
    ar_stats = workers[2].stats()
    rep.check(sp_stats["total_games"] >= 1, f"self-play games >= 1 ({sp_stats['total_games']})")
    rep.check(sp_stats["total_samples"] >= 1, f"samples produced ({sp_stats['total_samples']})")
    rep.check(tr_stats["total_batches"] >= 1,
              f"training batches >= 1 ({tr_stats['total_batches']})")
    rep.check(ar_stats["archived_games"] >= 1,
              f"archived games >= 1 ({ar_stats['archived_games']})")

    # checkpoint 文件生成
    rep.check(os.path.exists(config.MODEL_PATH), f".pt created: {config.MODEL_PATH}")
    rep.check(os.path.exists(config.STATE_DICT_PATH), f".pth created: {config.STATE_DICT_PATH}")

    print(f"      stats: games={sp_stats['total_games']}, samples={sp_stats['total_samples']}, "
          f"batches={tr_stats['total_batches']}, archived={ar_stats['archived_games']}")
    rep.summary()


def train_worker_stats_ok(worker) -> bool:
    """训练线程是否已至少完成一个训练轮次。"""
    try:
        return worker.stats()["total_batches"] >= 1
    except Exception:
        return False


def main() -> None:
    run_part("e2e: full CPU pipeline", test_full_pipeline)


if __name__ == "__main__":
    main()
