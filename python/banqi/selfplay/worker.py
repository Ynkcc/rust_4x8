"""banqi/selfplay/worker.py — 自对弈生产者线程与多进程子进程入口。

SelfPlayWorker：线程，按变体分派 Rust 绑定生成 episode，压入训练队列与归档队列，
  并按配置（USE_BATCHED_SELF_PLAY / NUM_WORKERS）选择 batched / parallel / single 方案，
  记录逐局统计与 TensorBoard 标量。
sp_worker_main：多进程（spawn）子进程入口，独立 GIL + 独立 CUDA context，权重经
  Predictor mtime 热重载自动同步，从根本上消除多线程 GIL 串行推理瓶颈。
"""

from __future__ import annotations

import os
import queue
import threading
import time
from collections import deque
from typing import Dict, List, Optional

import numpy as np

try:
    import banqi_4x8
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "无法导入 banqi_4x8。请先执行: maturin develop --features pyo3"
    ) from exc

from banqi.config import make_config
from banqi.tb_logger import add_scalar  # TensorBoard 训练日志（未启用时为 no-op）
from banqi.variant import Variant, get_variant

from .predictor import Predictor

# Rust 绑定函数名分发表（按 variant.rust_prefix）：(单局, 并行, 批量)
_SPLAY_FNS: Dict[str, tuple] = {
    "": ("run_self_play_with_predictor",
         "run_parallel_self_play_with_predictor",
         "run_batched_self_play_with_predictor"),
    "mini": ("run_mini_self_play_with_predictor",
             "run_mini_parallel_self_play_with_predictor",
             "run_mini_batched_self_play_with_predictor"),
    "game4x4": ("run_game4x4_self_play_with_predictor",
                "run_game4x4_parallel_self_play_with_predictor",
                "run_game4x4_batched_self_play_with_predictor"),
}


def _splay_fns(variant: Variant) -> tuple:
    key = variant.rust_prefix
    if key not in _SPLAY_FNS:
        raise KeyError(f"未知 rust_prefix {key!r}，可选: {sorted(_SPLAY_FNS)}")
    return _SPLAY_FNS[key]


def _episode_to_dict(ep, iteration: int, worker_id: int) -> Dict:
    d = dict(ep.to_dict())
    d["iteration"] = iteration
    d["worker_id"] = worker_id
    return d


class SelfPlayWorker(threading.Thread):
    """生产者线程：生成 episode 并压入数据队列与归档队列（4x2 无归档时 archive_q=None）。"""

    def __init__(
        self,
        predictor: Predictor,
        sp_cfg,
        variant: Variant,
        data_q: "queue.Queue",
        archive_q: Optional["queue.Queue"],
        stop_flag: "List[bool]",
        worker_id: int = 0,
    ) -> None:
        super().__init__(name=f"SelfPlayWorker-{variant.id}", daemon=True)
        self.predictor = predictor
        self.sp_cfg = sp_cfg
        self.variant = variant
        self.cfg = make_config(variant.id)
        self.tag = f"[SP-{variant.id}]"
        self.data_q = data_q
        self.archive_q = archive_q
        self.stop_flag = stop_flag
        self.worker_id = worker_id
        self._fn_single, self._fn_parallel, self._fn_batched = _splay_fns(variant)

        # 统计
        self.total_games = 0
        self.total_samples = 0
        self.iteration = 0
        self._game_count = 0  # 当前迭代内局数
        self.game_records: List[Dict] = []
        self._iter_lock = threading.Lock()
        # 滚动窗口胜负结果（winner=1 红/-1 黑/0 平），供 TB 胜率统计
        self._recent_results: deque = deque(maxlen=100)

    def _put(self, q: "queue.Queue", item: Dict) -> None:
        """压入队列；若队列满则等待（优雅退出时不等待）。"""
        while not self.stop_flag[0]:
            try:
                q.put(item, timeout=0.5)
                return
            except queue.Full:
                continue

    def run(self) -> None:
        """主循环，与 data_collector.rs / py_data_collector.rs 迭代语义一致。"""
        cfg = self.cfg
        while not self.stop_flag[0]:
            t0 = time.time()
            if cfg.USE_BATCHED_SELF_PLAY and hasattr(banqi_4x8, self._fn_batched):
                episodes = getattr(banqi_4x8, self._fn_batched)(
                    predict_fn=self.predictor,
                    config=self.sp_cfg,
                    num_games=cfg.GAMES_PER_ITER,
                    concurrency=cfg.BATCH_CONCURRENCY,
                    worker_id=self.worker_id,
                )
            elif cfg.NUM_WORKERS > 1:
                episodes = getattr(banqi_4x8, self._fn_parallel)(
                    predict_fn=self.predictor,
                    config=self.sp_cfg,
                    num_workers=cfg.NUM_WORKERS,
                    games_per_worker=cfg.GAMES_PER_WORKER,
                    worker_id=self.worker_id,
                )
            else:
                episodes = getattr(banqi_4x8, self._fn_single)(
                    predict_fn=self.predictor,
                    config=self.sp_cfg,
                    num_games=cfg.GAMES_PER_ITER,
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
                with self._iter_lock:
                    ep_dict = _episode_to_dict(ep, self.iteration, self.worker_id)
                    self._log_game(ep_dict, batch_duration / max(len(episodes), 1))
                # 数据队列必压；归档队列可选（4x2 无归档）
                self._put(self.data_q, ep_dict)
                if self.archive_q is not None:
                    self._put(self.archive_q, ep_dict)
                with self._iter_lock:
                    self.total_games += 1
                    self.total_samples += len(ep_dict["samples"]) if "samples" in ep_dict else ep_dict["num_samples"]
                    self._advance_iteration()

            # 吞吐：每批结束后记录局/秒（x 轴为累计局数）
            if self.cfg.TENSORBOARD_ENABLED:
                add_scalar("selfplay/games_per_sec", len(episodes) / max(batch_duration, 1e-9),
                           self.total_games)

    def _advance_iteration(self) -> None:
        """与 data_collector.rs 迭代推进语义一致：每 GAMES_PER_ITER 局 iteration += 1。"""
        cfg = self.cfg
        self._game_count += 1
        if self._game_count >= cfg.GAMES_PER_ITER:
            self._game_count -= cfg.GAMES_PER_ITER
            self.iteration += 1
            print(f"{self.tag} 📍 完成迭代 {self.iteration - 1} → 进入迭代 {self.iteration}")

    def _log_game(self, ep: Dict, duration: float) -> None:
        winner_str = {1: "红胜", -1: "黑胜"}.get(ep["winner"], "平局")
        print(
            f"{self.tag} Game #{self.total_games + 1} (iter={self.iteration}): "
            f"步数={ep['game_length']}, 结果={winner_str}, "
            f"耗时={duration:.1f}s ({ep['game_length'] / max(duration, 1e-9):.1f} steps/s)"
        )
        self.game_records.append({
            "game_length": int(ep["game_length"]),
            "winner": int(ep["winner"]),
            "duration": float(duration),
        })
        self._recent_results.append(int(ep["winner"]))

        if self.cfg.TENSORBOARD_ENABLED:
            game_idx = self.total_games + 1
            add_scalar("selfplay/game_length", int(ep["game_length"]), game_idx)
            add_scalar("selfplay/steps_per_sec", ep["game_length"] / max(duration, 1e-9), game_idx)
            # 滚动窗口胜率/平局率（红方胜率，模型交替先后手时近似模型胜率）
            n_win = sum(1 for w in self._recent_results if w == 1)
            n_draw = sum(1 for w in self._recent_results if w == 0)
            n_r = len(self._recent_results)
            add_scalar("selfplay/win_rate", 100.0 * n_win / max(n_r, 1), game_idx)
            add_scalar("selfplay/draw_rate", 100.0 * n_draw / max(n_r, 1), game_idx)
            # 搜索健康度：根节点价值均值（模型自我评价漂移）+ root 访问分布熵
            mvs = ep.get("mcts_values") or []
            if mvs:
                add_scalar("search/root_value_mean", float(np.mean(mvs)), game_idx)
            rvs = ep.get("root_visits") or []
            if len(rvs) > 1:
                rv = np.asarray(rvs, dtype=np.float64)
                p = rv / max(rv.sum(), 1e-9)
                ent = float(-(p * np.log(p + 1e-12)).sum())
                add_scalar("search/root_visits_entropy", ent, game_idx)
            # 棋盘占用率（该局全部局面平均，empty 为最后一通道）
            occ = self._episode_occupancy(ep)
            if occ is not None:
                add_scalar("data/board_occupancy", occ, game_idx)

    @staticmethod
    def _episode_occupancy(ep: Dict) -> Optional[float]:
        """该局平均棋盘占用率 = 1 - empty 通道均值（features.rs 通道序）。"""
        boards = ep.get("boards")
        shape = ep.get("board_shape")
        if not boards or not shape or len(shape) < 3:
            return None
        try:
            bc, br, bcol = int(shape[0]), int(shape[1]), int(shape[2])
            arr = np.asarray(boards, dtype=np.float32).reshape(-1, bc, br, bcol)
            return float((1.0 - arr[:, -1, :, :]).mean())
        except Exception:  # noqa: BLE001 - 单局统计失败不影响主流程
            return None

    def stats(self) -> Dict[str, int]:
        with self._iter_lock:
            return {
                "iteration": self.iteration,
                "total_games": self.total_games,
                "total_samples": self.total_samples,
            }

    def game_records_snapshot(self) -> List[Dict]:
        """返回逐局统计记录的浅拷贝（供基线验证/监控线程安全读取）。"""
        with self._iter_lock:
            return list(self.game_records)


def _log_episode(tag: str, ep: Dict, duration: float, game_index: int) -> None:
    """打印单局日志（子进程用，不含 TensorBoard / 统计记录）。"""
    winner_str = {1: "红胜", -1: "黑胜"}.get(ep["winner"], "平局")
    print(
        f"{tag} Game #{game_index} (iter={ep.get('iteration', 0)}): "
        f"步数={ep['game_length']}, 结果={winner_str}, "
        f"耗时={duration:.1f}s ({ep['game_length'] / max(duration, 1e-9):.1f} steps/s)"
    )


def sp_worker_main(
    variant_id: str,
    worker_id: int,
    data_q,
    archive_q,
    stop_event,
    inner_scheme: str = "",
    games_per_iter: Optional[int] = None,
    inner_workers: int = 1,
) -> None:
    """多进程自对弈子进程入口（target，必须模块级，spawn 才能 pickle）。

    每个子进程拥有独立的 Python 解释器（独立 GIL）与 CUDA context，
    从根本上消除「多线程共享 GIL → 叶子评估串行」的吞吐瓶颈。

    权重同步：Predictor 自带 model_path mtime 热重载——训练侧保存 checkpoint 后，
    各子进程自动加载新权重，无需额外进程间通信。
    """
    import torch as _torch  # noqa: PLC0415

    from .config import build_predictor, build_self_play_config

    variant = get_variant(variant_id)
    cfg = make_config(variant_id)
    tag = f"[SP-{variant.id}#{worker_id}]"
    gpi = games_per_iter or cfg.GAMES_PER_ITER
    scheme = (inner_scheme or
              ("batched" if cfg.USE_BATCHED_SELF_PLAY else "parallel"))

    _torch.set_num_threads(1)  # 每进程 1 torch 线程，防多进程共享核超售
    predictor, device = build_predictor(variant, cfg.MODEL_PATH, cfg.INFER_DEVICE)
    sp_cfg = build_self_play_config(variant)
    fn_single, fn_parallel, fn_batched = _splay_fns(variant)
    print(f"{tag} 🚀 子进程启动: device={device}, pid={os.getpid()}, "
          f"scheme={scheme}, games/iter={gpi}, inner_workers={inner_workers}")

    total_games = 0
    iteration = 0
    game_count = 0
    while not stop_event.is_set():
        t0 = time.time()
        try:
            if scheme == "batched":
                episodes = getattr(banqi_4x8, fn_batched)(
                    predict_fn=predictor, config=sp_cfg,
                    num_games=gpi,
                    concurrency=cfg.BATCH_CONCURRENCY,
                    worker_id=worker_id,
                )
            elif scheme == "parallel" and inner_workers > 1:
                episodes = getattr(banqi_4x8, fn_parallel)(
                    predict_fn=predictor, config=sp_cfg,
                    num_workers=inner_workers,
                    games_per_worker=max(1, -(-gpi // inner_workers)),
                    worker_id=worker_id,
                )
            else:
                episodes = getattr(banqi_4x8, fn_single)(
                    predict_fn=predictor, config=sp_cfg,
                    num_games=gpi, worker_id=worker_id,
                )
        except Exception as exc:  # pragma: no cover
            print(f"{tag} ⚠️ 自对弈异常: {exc}，子进程退出")
            break

        batch_duration = time.time() - t0
        if not episodes:
            if stop_event.is_set():
                break
            continue

        for ep in episodes:
            if stop_event.is_set():
                break
            ep_dict = _episode_to_dict(ep, iteration, worker_id)
            _log_episode(tag, ep_dict, batch_duration / max(len(episodes), 1),
                         total_games + 1)
            # 压队列（带超时，退出时不因队列满而卡死；超时则丢弃该局）
            if not stop_event.is_set():
                try:
                    data_q.put(ep_dict, timeout=30.0)
                    if archive_q is not None:
                        archive_q.put(ep_dict, timeout=30.0)
                except queue.Full:  # pragma: no cover
                    print(f"{tag} ⚠️ 队列满，丢弃 1 局（stop 退出中）")
            total_games += 1
            game_count += 1
            if game_count >= gpi:
                game_count -= gpi
                iteration += 1
                print(f"{tag} 📍 完成迭代 {iteration - 1} → 进入迭代 {iteration}")

    print(f"{tag} 子进程退出，累计 {total_games} 局，{iteration} 个迭代")
