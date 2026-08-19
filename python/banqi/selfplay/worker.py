"""banqi/selfplay/worker.py — 自对弈生产者线程与多进程子进程入口。

SelfPlayWorker：线程，统一走唯一入口 `run_python_match`（Python predict_fn 单线程），
  并发度由 BATCH_CONCURRENCY 控制（现由 Rust 侧 MCTS 串行推进，吞吐受 GIL 约束），
  记录逐局统计与 TensorBoard 标量。
sp_worker_main：多进程（spawn）子进程入口，独立 GIL + 独立 CUDA context，权重经
  Predictor mtime 热重载自动同步，从根本上消除多线程 GIL 串行推理瓶颈。
MODEL_BACKEND="onnx" 时走 `run_native_match`（Rust 侧持 ONNX 模型，推理不经过 GIL），
  每批重新加载模型文件，天然实现权重热更新。

旧的按变体分发的 `run_*_self_play_with_predictor` / `RustOnnxCollector` /
`RustTorchCollector` 入口已彻底移除，统一经 `run_python_match` / `run_native_match`。
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

# 统一入口变体映射（variant.rust_prefix -> Rust 入口的 variant_id）
_VARIANT_MAP: Dict[str, str] = {
    "": "4x8",        # 4x8
    "mini": "4x2",    # 4x2
    "game4x4": "4x4", # 4x4
}


def _variant_id(variant: Variant) -> str:
    key = variant.rust_prefix
    if key not in _VARIANT_MAP:
        raise KeyError(f"未知 rust_prefix {key!r}，可选: {sorted(_VARIANT_MAP)}")
    return _VARIANT_MAP[key]


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
        self.variant_id = _variant_id(variant)

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
        """主循环，统一走 `run_python_match`（Python 推理单线程）。"""
        cfg = self.cfg
        while not self.stop_flag[0]:
            t0 = time.time()
            episodes = banqi_4x8.run_python_match(
                predict_fn=self.predictor,
                config=self.sp_cfg,
                num_games=cfg.GAMES_PER_ITER,
                variant_id=self.variant_id,
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


def _run_native_model_loop(
    variant: Variant,
    model_path: str,
    cfg,
    sp_cfg,
    data_q,
    archive_q,
    stop_event,
    gpi: int,
    worker_id: int,
    tag: str,
) -> None:
    """Rust 持有模型（.onnx / .pt）的 `run_native_match` 主循环（免 GIL）。

    每批经 `run_native_match` 重新加载模型文件，天然实现训练中权重热更新；
    产出 episode 语义与 `run_python_match` 一致（PyGameEpisode）。
    """
    variant_id = _variant_id(variant)
    total_games = 0
    iteration = 0
    game_count = 0
    while not stop_event.is_set():
        if not os.path.exists(model_path):
            print(f"{tag} ⚠️ 模型不存在: {model_path}，等待...")
            time.sleep(2.0)
            continue
        t0 = time.time()
        try:
            _, _, _, _, _, episodes = banqi_4x8.run_native_match(
                player_a=model_path,
                player_b=model_path,
                n=gpi,
                variant_id=variant_id,
                model_sims=cfg.MCTS_SIMS,
                heuristic_sims=None,
                seed=None,
                num_threads=max(1, cfg.BATCH_CONCURRENCY),
                config=sp_cfg,
                record_episodes=True,
            )
            episodes = list(episodes)
        except Exception as exc:  # pragma: no cover
            print(f"{tag} ⚠️ run_native_match 自对弈异常: {exc}，子进程退出")
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


def sp_worker_main(
    variant_id: str,
    worker_id: int,
    data_q,
    archive_q,
    stop_event,
    games_per_iter: Optional[int] = None,
) -> None:
    """多进程自对弈子进程入口（target，必须模块级，spawn 才能 pickle）。

    每个子进程拥有独立的 Python 解释器（独立 GIL）与 CUDA context，
    从根本上消除「多线程共享 GIL → 叶子评估串行」的吞吐瓶颈。

    权重同步：Predictor 自带 model_path mtime 热重载——训练侧保存 checkpoint 后，
    各子进程自动加载新权重，无需额外进程间通信。

    自对弈统一走唯一入口：
      - `run_python_match`（predict_fn 单线程，默认 / torchscript 后端）
      - `run_native_match`（MODEL_BACKEND="onnx" 时，Rust 持模型免 GIL）
    """
    import torch as _torch  # noqa: PLC0415

    from .config import build_predictor, build_self_play_config

    variant = get_variant(variant_id)
    cfg = make_config(variant_id)
    tag = f"[SP-{variant.id}#{worker_id}]"
    gpi = games_per_iter or cfg.GAMES_PER_ITER
    scheme = "batched"
    vid = _variant_id(variant)

    _torch.set_num_threads(1)  # 每进程 1 torch 线程，防多进程共享核超售
    sp_cfg = build_self_play_config(variant)

    # ---- MODEL_BACKEND="onnx"：优先走 Rust 持有 ONNX 模型的 run_native_match（免 GIL） ----
    # 模型在 Rust 侧用 ONNX Runtime 推理，不经过 GIL；每批重新加载 .onnx 文件实现热更新。
    use_onnx = (cfg.MODEL_BACKEND or "").strip().lower() == "onnx"
    if use_onnx and cfg.ONNX_PATH and os.path.exists(cfg.ONNX_PATH):
        print(f"{tag} 🚀 子进程启动: ONNX 后端（run_native_match 原生推理），pid={os.getpid()}, "
              f"scheme={scheme}, games/iter={gpi}")
        _run_native_model_loop(
            variant, cfg.ONNX_PATH, cfg, sp_cfg, data_q, archive_q, stop_event,
            gpi, worker_id, tag,
        )
        return
    if use_onnx:
        print(f"{tag} ⚠️ ONNX 模型缺失，回退 Python 推理路径")

    predictor, device = build_predictor(variant, cfg.MODEL_PATH, cfg.INFER_DEVICE)
    print(f"{tag} 🚀 子进程启动: device={device}, pid={os.getpid()}, "
          f"scheme={scheme}, games/iter={gpi}")

    total_games = 0
    iteration = 0
    game_count = 0
    while not stop_event.is_set():
        t0 = time.time()
        try:
            episodes = banqi_4x8.run_python_match(
                predict_fn=predictor, config=sp_cfg,
                num_games=gpi,
                variant_id=vid,
                worker_id=worker_id,
            )
            episodes = list(episodes)
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
