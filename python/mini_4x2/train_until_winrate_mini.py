"""
train_until_winrate_mini.py — 继续训练 mini 模型，直到「vs minimax(alpha-beta, depth=4) 胜率 > 80%」。

流程：
  1. 复用 SelfPlayWorkerMini（生产者）+ TrainWorker（消费者）续训（自动从现有 checkpoint 恢复）。
  2. 主线程周期性从磁盘加载最新 checkpoint，跑「模型(MCTS) vs minimax(depth=4)」对局验证。
  3. 胜率超过 TARGET_WINRATE（默认 0.80，按 EVAL_GAMES 局判定）→ 优雅停止并落盘。

用法：
    python python/train_until_winrate_mini.py
    # 环境变量：
    #   MINI_MM_DEPTH    minimax 深度（默认 4）
    #   MINI_EVAL_GAMES  每轮评估局数（默认 20）
    #   MINI_TARGET_WR   目标胜率（默认 0.80）
    #   MINI_EVAL_SIMS   评估时模型 MCTS 模拟数（默认 128，与训练搜索强度一致）
    #   MINI_EVAL_EVERY  每隔多少训练轮评估一次（默认 5）
    #   MINI_MAX_RUNTIME 保护性总时限（秒，默认 180 分钟）
"""
from __future__ import annotations

import os
import queue
import random
import signal
import sys
import time
from typing import List, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import torch

import banqi_4x8

from config_mini import config
from self_play_mini import SelfPlayWorkerMini, build_predictor_mini, build_self_play_config
from training_service_mini import TrainWorker
from nn_model_mini import MiniBanqiNet, load_model_weights

_HERE = os.path.dirname(os.path.abspath(__file__))
MINIMAX_DEPTH = int(os.getenv("MINI_MM_DEPTH", "4"))
EVAL_GAMES = int(os.getenv("MINI_EVAL_GAMES", "20"))
TARGET_WINRATE = float(os.getenv("MINI_TARGET_WR", "0.80"))
EVAL_SIMS = int(os.getenv("MINI_EVAL_SIMS", "128"))
EVAL_EVERY_ROUNDS = int(os.getenv("MINI_EVAL_EVERY", "5"))
MAX_RUNTIME = int(os.getenv("MINI_MAX_RUNTIME", str(180 * 60)))
MODEL_PATH = os.getenv("MINI_MODEL_PATH", os.path.join(_HERE, "banqi_mini_model_latest.pt"))
STATE_DICT_PATH = os.getenv("MINI_STATE_DICT_PATH", os.path.join(_HERE, "banqi_mini_model_latest.pth"))


class EvalPredictor:
    """从磁盘 checkpoint 加载的评估用 Predictor（与 verify_mini_vs_minimax 一致）。"""

    def __init__(self, model: MiniBanqiNet, device: "torch.device"):
        self.model = model.to(device).eval()
        self.device = device

    def __call__(self, boards: np.ndarray, scalars: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        with torch.inference_mode():
            b = torch.from_numpy(np.ascontiguousarray(boards)).to(self.device)
            s = torch.from_numpy(np.ascontiguousarray(scalars)).to(self.device)
            logits, value = self.model(b, s)
            return logits.cpu().numpy().astype(np.float32), value.cpu().numpy().reshape(-1).astype(np.float32)


def load_disk_model() -> EvalPredictor:
    device = torch.device("cpu")
    model = MiniBanqiNet()
    if os.path.exists(MODEL_PATH):
        load_model_weights(model, MODEL_PATH, device)
    elif os.path.exists(STATE_DICT_PATH):
        state = torch.load(STATE_DICT_PATH, map_location=device)
        model.load_state_dict(state["model_state_dict"])
    else:
        raise FileNotFoundError(f"未找到模型 {MODEL_PATH} / {STATE_DICT_PATH}")
    return EvalPredictor(model, device)


def play_one_game(predictor: EvalPredictor, model_is_red: bool, mm_depth: int) -> int:
    """模型 vs minimax 一局。返回 +1 模型胜 / 0 平 / -1 minimax 胜。"""
    env = banqi_4x8.MiniDarkChess()
    moves = 0
    while not env.terminated():
        if env.winner() is not None:
            break
        is_red_turn = env.current_player() == 1
        model_turn = (is_red_turn == model_is_red)
        if model_turn:
            action = env.mcts_search_action(
                predictor, EVAL_SIMS, 12, c_visit=1.0, c_scale=0.25
            )
        else:
            action = env.minimax_action(mm_depth)
        if action is None:
            break
        env.step(action)
        moves += 1
        if moves > 400:
            break
    winner = env.winner()  # 1=红胜, -1=黑胜, 0=平
    if winner == 0 or winner is None:
        return 0
    if model_is_red:
        return 1 if winner == 1 else -1
    return 1 if winner == -1 else -1


def evaluate_vs_minimax() -> dict:
    """从磁盘加载最新模型，跑 EVAL_GAMES 局 vs minimax(depth)，返回统计。"""
    predictor = load_disk_model()
    wins = draws = 0
    for i in range(EVAL_GAMES):
        model_is_red = (i % 2 == 0)
        w = play_one_game(predictor, model_is_red, MINIMAX_DEPTH)
        if w == 1:
            wins += 1
        elif w == 0:
            draws += 1
    winrate = wins / EVAL_GAMES
    return {
        "wins": wins,
        "draws": draws,
        "losses": EVAL_GAMES - wins - draws,
        "winrate": winrate,
        "games": EVAL_GAMES,
    }


def main() -> None:
    print("=" * 64)
    print("  🚀 继续训练 mini 模型，直到 vs minimax(depth=4) 胜率 > 80%")
    print("=" * 64)
    print(f"  minimax depth      = {MINIMAX_DEPTH}")
    print(f"  评估局数/轮        = {EVAL_GAMES}（目标胜率 {TARGET_WINRATE:.0%}）")
    print(f"  评估 MCTS sims     = {EVAL_SIMS}")
    print(f"  每 {EVAL_EVERY_ROUNDS} 轮评估一次")
    print(f"  保护性总时限       = {MAX_RUNTIME}s")
    print("=" * 64)

    stop_flag: List[bool] = [False]

    def _handler(signum, frame):
        stop_flag[0] = True
        print("\n[Main] 收到信号，将在当前批结束后优雅退出...")

    signal.signal(signal.SIGINT, _handler)

    data_q: "queue.Queue" = queue.Queue(maxsize=config.DATA_QUEUE_MAXSIZE)
    predictor, _ = build_predictor_mini(config.MODEL_PATH, device_str=config.INFER_DEVICE)
    sp_cfg = build_self_play_config()

    workers = [
        SelfPlayWorkerMini(predictor, sp_cfg, data_q, stop_flag, worker_id=0),
        TrainWorker(data_q, stop_flag),
    ]
    for w in workers:
        w.start()

    train_worker: TrainWorker = workers[1]
    start_t = time.time()
    last_eval_round = -1
    reached = False

    try:
        while not stop_flag[0]:
            elapsed = time.time() - start_t
            if elapsed >= MAX_RUNTIME:
                print(f"\n[Main] 达到保护性时限 {MAX_RUNTIME}s，停止（未达标）")
                stop_flag[0] = True
                break
            if not all(w.is_alive() for w in workers):
                print("[Main] 有线程退出")
                break

            history = train_worker.round_history_snapshot()
            round_num = history[-1]["round"] if history else 0
            if round_num >= last_eval_round + EVAL_EVERY_ROUNDS and round_num > 0:
                last_eval_round = round_num
                # 等待 checkpoint 落盘稳定
                time.sleep(1.0)
                try:
                    stats = evaluate_vs_minimax()
                except Exception as exc:  # noqa: BLE001
                    print(f"[Main] 评估失败（跳过本轮）: {exc}")
                    time.sleep(5)
                    continue
                print(f"\n[Main] Round#{round_num} 评估 vs minimax(depth={MINIMAX_DEPTH}): "
                      f"胜{stats['wins']} 平{stats['draws']} 负{stats['losses']} "
                      f"胜率={stats['winrate']:.2%}（目标 {TARGET_WINRATE:.0%}）\n")
                if stats["winrate"] > TARGET_WINRATE:
                    print(f"  ✅ 达标！胜率 {stats['winrate']:.2%} > {TARGET_WINRATE:.0%}")
                    reached = True
                    stop_flag[0] = True
                    break
                if elapsed > 60 and round_num % 10 == 0:
                    tr = train_worker.stats()
                    print(f"[Main] 进度: round={round_num} batches={tr['total_batches']:.0f} "
                          f"avg_loss={tr['avg_loss']:.4f} 已运行 {elapsed/60:.1f}min")
            time.sleep(3)
    except KeyboardInterrupt:
        stop_flag[0] = True

    # 优雅关闭
    print("\n[Main] 正在优雅关闭各线程...")
    sp_worker = workers[0]
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
    print("\n" + "=" * 64)
    print("  训练结束")
    print("=" * 64)
    print(f"  训练轮次:   {tr['round_num']}")
    print(f"  累计批次:   {tr['total_batches']:.0f}")
    print(f"  平均 Loss:  {tr['avg_loss']:.4f}")
    if history:
        print(f"  Loss 变化:  {history[0]['train_loss']:.4f} → {history[-1]['train_loss']:.4f}")
    if reached:
        print(f"  ✅ 已达标: vs minimax(depth={MINIMAX_DEPTH}) 胜率 > {TARGET_WINRATE:.0%}")
    else:
        print(f"  ⚠️ 未达标（受时限/信号中断）")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    sys.exit(main())
