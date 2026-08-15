"""
self_play.py — 4x4 暗棋的自对弈数据生产者

调用 PyO3 的 `run_game4x4_*_self_play_with_predictor` 驱动 Gumbel MCTS 自对弈，
生成训练局数据并压入数据队列。
"""
from __future__ import annotations

import os
import queue
import threading
import time
from typing import Dict, List, Tuple

import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:  # pragma: no cover
    HAS_TORCH = False

import banqi_4x8

from config import config
from constant import ACTION_SPACE_SIZE
from nn_model import Banqi4x4Net, load_model_weights
from tb_logger import add_scalar  # TensorBoard 训练日志（未启用时为 no-op）


class Predictor4x4:
    """4x4 推理端：eval/inference_mode + 分块推理，匹配 py_evaluator.rs 契约。"""

    def __init__(self, model: "Banqi4x4Net", device: "torch.device", model_path: str | None) -> None:
        self.model = model.to(device)
        self.device = device
        self.model_path: str | None = model_path
        self._mtime: float = 0.0
        self.model.eval()

    def _maybe_reload_weights(self, force: bool = False) -> None:
        if not HAS_TORCH or not self.model_path or not os.path.exists(self.model_path):
            return
        mtime = os.path.getmtime(self.model_path)
        if force or mtime > self._mtime:
            try:
                load_model_weights(self.model, self.model_path, self.device)
                self.model.eval()
                self._mtime = mtime
                print(f"[Predictor4x4] 已重载权重: {self.model_path}")
            except Exception as exc:  # pragma: no cover
                print(f"[Predictor4x4] 权重加载失败 (保持旧模型): {exc}")

    def __call__(self, board: np.ndarray, scalars: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """返回 (policy_logits (N,112) float32, values (N,) float32)。"""
        self._maybe_reload_weights()
        batch = board.shape[0]
        if not HAS_TORCH:
            return (np.zeros((batch, ACTION_SPACE_SIZE), dtype=np.float32),
                    np.zeros(batch, dtype=np.float32))
        if batch == 0:
            return (np.zeros((0, ACTION_SPACE_SIZE), dtype=np.float32),
                    np.zeros(0, dtype=np.float32))
        chunk = config.PREDICT_BATCH
        if batch <= chunk:
            return self._infer(board, scalars)
        policy_list: List[np.ndarray] = []
        value_list: List[np.ndarray] = []
        for i in range(0, batch, chunk):
            pl, vl = self._infer(board[i:i + chunk], scalars[i:i + chunk])
            policy_list.append(pl)
            value_list.append(vl)
        return (np.concatenate(policy_list, axis=0).astype(np.float32),
                np.concatenate(value_list, axis=0).astype(np.float32))

    def _infer(self, board: np.ndarray, scalars: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        with torch.inference_mode():
            b = torch.from_numpy(np.ascontiguousarray(board)).to(self.device, non_blocking=True)
            s = torch.from_numpy(np.ascontiguousarray(scalars)).to(self.device, non_blocking=True)
            logits, value = self.model(b, s)
            return (logits.cpu().numpy().astype(np.float32),
                    value.cpu().numpy().reshape(-1).astype(np.float32))


def _episode_to_dict(ep, iteration: int, worker_id: int) -> Dict:
    d = dict(ep.to_dict())
    d["iteration"] = iteration
    d["worker_id"] = worker_id
    return d


class SelfPlayWorker4x4(threading.Thread):
    """生产者线程：调用 run_game4x4_* 生成 episode，压入数据队列与归档队列。"""

    def __init__(
        self,
        predictor: Predictor4x4,
        sp_cfg,
        data_q: "queue.Queue",
        archive_q: "queue.Queue",
        stop_flag: "List[bool]",
        worker_id: int = 0,
    ) -> None:
        super().__init__(name="SelfPlayWorker4x4", daemon=True)
        self.predictor = predictor
        self.sp_cfg = sp_cfg
        self.data_q = data_q
        self.archive_q = archive_q
        self.stop_flag = stop_flag
        self.worker_id = worker_id
        self.total_games = 0
        self.total_samples = 0
        self.iteration = 0
        self._game_count = 0
        self.game_records: List[Dict] = []
        self._lock = threading.Lock()

    def _put(self, q: "queue.Queue", item: Dict) -> None:
        while not self.stop_flag[0]:
            try:
                q.put(item, timeout=0.5)
                return
            except queue.Full:
                continue

    def run(self) -> None:
        while not self.stop_flag[0]:
            t0 = time.time()
            if config.NUM_WORKERS > 1:
                episodes = banqi_4x8.run_game4x4_parallel_self_play_with_predictor(
                    predict_fn=self.predictor,
                    config=self.sp_cfg,
                    num_workers=config.NUM_WORKERS,
                    games_per_worker=config.GAMES_PER_WORKER,
                    worker_id=self.worker_id,
                )
            else:
                episodes = banqi_4x8.run_game4x4_self_play_with_predictor(
                    predict_fn=self.predictor,
                    config=self.sp_cfg,
                    num_games=config.GAMES_PER_ITER,
                    worker_id=self.worker_id,
                )
            batch_duration = time.time() - t0

            if not episodes:
                if self.stop_flag[0]:
                    break
                continue

            for ep in episodes:
                if self.stop_flag[0]:
                    break
                with self._lock:
                    ep_dict = _episode_to_dict(ep, self.iteration, self.worker_id)
                    self._log_game(ep_dict, batch_duration / max(len(episodes), 1))
                    self.total_games += 1
                    self.total_samples += len(ep_dict["samples"]) if "samples" in ep_dict else ep_dict["num_samples"]
                    self._advance_iteration()
                # 同时压入训练消费队列与冷存储归档队列
                self._put(self.data_q, ep_dict)
                self._put(self.archive_q, ep_dict)

    def _advance_iteration(self) -> None:
        self._game_count += 1
        if self._game_count >= config.GAMES_PER_ITER:
            self._game_count -= config.GAMES_PER_ITER
            self.iteration += 1

    def _log_game(self, ep: Dict, duration: float) -> None:
        """逐局统计记录（在 _lock 临界区内调用）+ TensorBoard 打点。

        每局文本日志由 Rust 侧（[PW-x]）打印，此处不重复输出。
        """
        self.game_records.append({
            "game_length": int(ep["game_length"]),
            "winner": int(ep["winner"]),
            "duration": float(duration),
        })
        # TensorBoard 记录（x 轴为累计对局数）
        if config.TENSORBOARD_ENABLED:
            game_idx = self.total_games + 1
            add_scalar("selfplay/game_length", int(ep["game_length"]), game_idx)
            add_scalar("selfplay/steps_per_sec", ep["game_length"] / max(duration, 1e-9), game_idx)

    def stats(self) -> Dict[str, int]:
        with self._lock:
            return {
                "iteration": self.iteration,
                "total_games": self.total_games,
                "total_samples": self.total_samples,
            }

    def game_records_snapshot(self) -> List[Dict]:
        with self._lock:
            return list(self.game_records)


def build_predictor4x4(model_path: str | None, device_str: str = "cpu") -> Tuple[Predictor4x4, "torch.device"]:
    if not HAS_TORCH:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device_str == "auto" else torch.device(device_str)
    print(f"[SelfPlay4x4] device = {device}")
    model = Banqi4x4Net()
    if HAS_TORCH:
        model = model.to(device)
    predictor = Predictor4x4(model, device, model_path)
    if model_path and os.path.exists(model_path):
        print(f"[SelfPlay4x4] 使用模型权重: {model_path}")
    else:
        print(f"[SelfPlay4x4] 使用全新初始化网络")
    return predictor, device


def build_self_play_config() -> "banqi_4x8.SelfPlayConfig":
    return banqi_4x8.SelfPlayConfig(
        mcts_sims=config.MCTS_SIMS,
        max_considered_actions=config.MAX_CONSIDERED_ACTIONS,
        temperature_steps=config.TEMPERATURE_STEPS,
    )
