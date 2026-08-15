"""
run_training.py — 4x4 暗棋训练闭环入口

启动 SelfPlayWorker4x4（生产者）+ TrainWorker（消费者），
训练到 config.MAX_RUNTIME_SECONDS 后优雅停止并落盘 checkpoint。

用法：
    python python/game_4x4/run_training.py
    # 环境变量：
    #   G4X4_MAX_RUNTIME  训练时长（秒，默认 3600）
    #   G4X4_MODEL_PATH / G4X4_STATE_DICT_PATH  模型路径覆盖
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

from config import config
from self_play import SelfPlayWorker4x4, build_predictor4x4, build_self_play_config
from training_service import TrainWorker

START = time.time()


def main() -> int:
    print("=" * 64)
    print("  4x4 暗棋训练启动（CPU）")
    print("=" * 64)
    print(f"  MCTS sims         = {config.MCTS_SIMS}")
    print(f"  GAMES_PER_ITER    = {config.GAMES_PER_ITER}（{config.NUM_WORKERS} workers）")
    print(f"  训练时限          = {config.MAX_RUNTIME_SECONDS}s")
    print(f"  模型输出          = {config.MODEL_PATH}")
    print("=" * 64)

    stop_flag: List[bool] = [False]

    def _handler(signum, frame):
        stop_flag[0] = True
        print("\n[Main] 收到信号，将在当前批结束后优雅退出...")

    signal.signal(signal.SIGINT, _handler)

    data_q: "queue.Queue" = queue.Queue(maxsize=config.DATA_QUEUE_MAXSIZE)
    predictor, _ = build_predictor4x4(config.MODEL_PATH, device_str=config.INFER_DEVICE)
    sp_cfg = build_self_play_config()

    workers = [
        SelfPlayWorker4x4(predictor, sp_cfg, data_q, stop_flag, worker_id=0),
        TrainWorker(data_q, stop_flag),
    ]
    for w in workers:
        w.start()

    train_worker: TrainWorker = workers[1]
    sp_worker: SelfPlayWorker4x4 = workers[0]

    try:
        while not stop_flag[0]:
            elapsed = time.time() - START
            if elapsed >= config.MAX_RUNTIME_SECONDS:
                print(f"\n[Main] 达到训练时限 {config.MAX_RUNTIME_SECONDS}s，停止")
                stop_flag[0] = True
                break
            if not all(w.is_alive() for w in workers):
                print("[Main] 有线程退出")
                break
            time.sleep(5)
    except KeyboardInterrupt:
        stop_flag[0] = True

    print("\n[Main] 正在优雅关闭各线程...")
    if sp_worker.is_alive():
        sp_worker.join(timeout=10)
    if train_worker.is_alive():
        train_worker.join(timeout=30)
    stop_flag[0] = True
    if train_worker.is_alive():
        train_worker.join(timeout=10)
    train_worker.finalize()

    tr = train_worker.stats()
    history = train_worker.round_history_snapshot()
    sp_stats = sp_worker.stats()
    print("\n" + "=" * 64)
    print("  4x4 训练结束")
    print("=" * 64)
    print(f"  自对弈局数:   {sp_stats['total_games']}（样本 {sp_stats['total_samples']}）")
    print(f"  训练轮次:     {tr['round_num']}")
    print(f"  累计批次:     {tr['total_batches']:.0f}")
    print(f"  平均 Loss:    {tr['avg_loss']:.4f}")
    if history:
        print(f"  Loss 变化:    {history[0]['train_loss']:.4f} → {history[-1]['train_loss']:.4f}")
    print(f"  总耗时:       {time.time() - START:.0f}s")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    sys.exit(main())
