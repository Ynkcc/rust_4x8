"""
run_training_mini.py — 4x2 迷你暗棋 自对弈 + 训练闭环入口

单进程双线程模型：
  - SelfPlayWorkerMini（生产者）：调用 run_mini_* 生成 episode，压入数据队列
  - TrainWorker（消费者）：从数据队列消费，填充 replay buffer 并训练 MiniBanqiNet

运行到 config_mini.MAX_RUNTIME_SECONDS 后自动优雅停止并落盘 checkpoint，
预期约 20 分钟内收敛（loss 下降 + 对随机基线胜率提升）。

运行方式（需先 maturin develop --features pyo3）：
    python python/run_training_mini.py
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

import banqi_4x8

from config_mini import config
from self_play_mini import SelfPlayWorkerMini, build_predictor_mini, build_self_play_config
from training_service_mini import TrainWorker


def main() -> None:
    print("=" * 56)
    print("  🚀 4x2 迷你暗棋 自对弈 + 训练闭环启动（CPU）")
    print("=" * 56)
    print(f"  MCTS_SIMS       = {config.MCTS_SIMS}")
    print(f"  GAMES_PER_ITER  = {config.GAMES_PER_ITER} (workers={config.NUM_WORKERS})")
    print(f"  TRAIN_BATCH     = {config.TRAIN_BATCH}, LR = {config.LEARNING_RATE}")
    print(f"  MODEL_PATH      = {config.MODEL_PATH}")
    print(f"  STATE_DICT_PATH = {config.STATE_DICT_PATH}")
    print(f"  运行时限        = {config.MAX_RUNTIME_SECONDS}s")
    print("=" * 56)

    stop_flag: List[bool] = [False]

    def _handler(signum, frame):
        stop_flag[0] = True
        print("\n[Main] 收到信号，将在当前批结束后优雅退出...")

    signal.signal(signal.SIGINT, _handler)

    data_q: "queue.Queue" = queue.Queue(maxsize=config.DATA_QUEUE_MAXSIZE)

    predictor, infer_device = build_predictor_mini(config.MODEL_PATH, device_str=config.INFER_DEVICE)
    sp_cfg = build_self_play_config()
    print(
        f"[Main] banqi_4x8 MINI: BOARD=({banqi_4x8.MINI_BOARD_CHANNELS},"
        f"{banqi_4x8.MINI_BOARD_ROWS},{banqi_4x8.MINI_BOARD_COLS}), "
        f"SCALAR={banqi_4x8.MINI_SCALAR_FEATURE_COUNT}, "
        f"ACTION={banqi_4x8.MINI_ACTION_SPACE_SIZE}"
    )

    workers = [
        SelfPlayWorkerMini(predictor, sp_cfg, data_q, stop_flag, worker_id=0),
        TrainWorker(data_q, stop_flag),
    ]
    for w in workers:
        w.start()

    print("[Main] 线程已启动，训练进行中...\n")

    # 主线程：运行到时限或收到信号
    start_t = time.time()
    try:
        while not stop_flag[0]:
            elapsed = time.time() - start_t
            if elapsed >= config.MAX_RUNTIME_SECONDS:
                print(f"\n[Main] 达到运行时限 {config.MAX_RUNTIME_SECONDS}s，优雅停止...")
                stop_flag[0] = True
                break
            if not all(w.is_alive() for w in workers):
                print("[Main] 有线程退出")
                break
            time.sleep(2)
    except KeyboardInterrupt:
        stop_flag[0] = True

    # 优雅关闭
    print("\n[Main] 正在优雅关闭各线程...")
    sp_worker: SelfPlayWorkerMini = workers[0]
    train_worker: TrainWorker = workers[1]
    if sp_worker.is_alive():
        sp_worker.join(timeout=10)
    if train_worker.is_alive():
        train_worker.join(timeout=30)
    stop_flag[0] = True
    if train_worker.is_alive():
        train_worker.join(timeout=10)
    train_worker.finalize()

    # 结束统计
    sp_stats = sp_worker.stats()
    tr_stats = train_worker.stats()
    history = train_worker.round_history_snapshot()
    sep = "=" * 56
    print(f"\n{sep}")
    print("  4x2 迷你暗棋训练结束")
    print(f"{sep}")
    print(f"  累计自对弈局数:    {sp_stats['total_games']}")
    print(f"  累计样本数:        {sp_stats['total_samples']}")
    print(f"  训练轮次:          {tr_stats['round_num']}")
    print(f"  累计训练批次:      {tr_stats['total_batches']}")
    print(f"  平均 Loss:         {tr_stats['avg_loss']:.4f}")
    if history:
        first = history[0]
        last = history[-1]
        print(f"  Loss 变化:         {first['train_loss']:.4f} → {last['train_loss']:.4f} "
              f"(round {first['round']} → {last['round']})")
    print(f"{sep}")


if __name__ == "__main__":
    main()
