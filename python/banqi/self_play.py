"""banqi/self_play.py — 自对弈数据生产者（共享实现，4x2 / 4x4 / 4x8 通用）

通过 PyO3 绑定 (banqi_4x8) 驱动 Gumbel MCTS 自对弈，生成训练局数据。
本模块以线程形式运行（SelfPlayWorker），把生成的 episode dict 压入：
  - data_q    （训练消费队列）
  - archive_q （MongoDB 冷存储归档队列；变体无归档时传 None 跳过）

所有配置统一来自 `banqi.config.make_config(variant_id)`；
Rust 绑定函数按 `variant.rust_prefix` 分派（"" / "mini" / "game4x4"）。
"""

from __future__ import annotations

import os
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple

import numpy as np

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

from banqi.config import make_config
from banqi.constants import Constants, build_constants
from banqi.nn_model import BanqiNet, load_model_weights
from banqi.tb_logger import add_scalar  # TensorBoard 训练日志（未启用时为 no-op）
from banqi.variant import Variant

# Rust 绑定函数名分发表（按 variant.rust_prefix）：(单局, 并行, 批量)
_SPLAY_FNS: Dict[str, Tuple[str, str, str]] = {
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


def _splay_fns(variant: Variant) -> Tuple[str, str, str]:
    key = variant.rust_prefix
    if key not in _SPLAY_FNS:
        raise KeyError(f"未知 rust_prefix {key!r}，可选: {sorted(_SPLAY_FNS)}")
    return _SPLAY_FNS[key]


# ============================================================================
# 推理端 Predictor（热重载，匹配 py_evaluator.rs 契约；内部按 PREDICT_BATCH 分块）
# ============================================================================

class Predictor:
    """
    薄包装：
      - 确保模型在 eval / inference_mode
      - 输入/输出都是 numpy（Rust 侧转成 numpy 后传进来）
      - 简易模型热重载（检查 model_path 文件 mtime）
      - 对任意 batch 按 PREDICT_BATCH 分块推理，避免显存/内存峰值
      - TF32 + cudnn benchmark 优化吞吐（GPU 时）
    """

    def __init__(self, model: BanqiNet, device, model_path: Optional[str],
                 variant: Variant) -> None:
        if isinstance(device, str):
            device = torch.device(device)
        self.variant = variant
        self.action_space = build_constants(variant).ACTION_SPACE_SIZE
        self.cfg = make_config(variant.id)
        self.model = model.to(device)
        self.device = device
        self.model_path: Optional[str] = model_path
        self._mtime: float = 0.0
        self.model.eval()

        # 吞吐优化：TF32 + cudnn auto-tune
        if HAS_TORCH and device.type == "cuda":
            torch.backends.cudnn.benchmark = True
            torch.set_float32_matmul_precision("high")
            print(f"[SP-{variant.id}] 吞吐优化: TF32 + cudnn.benchmark 已启用")

        self._maybe_reload_weights(force=True)

    def _maybe_reload_weights(self, force: bool = False) -> None:
        if not HAS_TORCH or not self.model_path or not os.path.exists(self.model_path):
            return
        mtime = os.path.getmtime(self.model_path)
        if force or mtime > self._mtime:
            try:
                load_model_weights(self.model, self.model_path, self.device)
                self.model.eval()
                self._mtime = mtime
                print(f"[SP-{self.variant.id}] 已重载权重: {self.model_path}")
            except Exception as exc:  # pragma: no cover
                print(f"[SP-{self.variant.id}] 权重加载失败 (保持旧模型): {exc}")

    def __call__(self, board: np.ndarray, scalars: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """返回 (policy_logits (N, A) float32, values (N,) float32)，匹配绑定契约。"""
        self._maybe_reload_weights()
        batch = board.shape[0]
        if not HAS_TORCH:
            return (
                np.zeros((batch, self.action_space), dtype=np.float32),
                np.zeros(batch, dtype=np.float32),
            )
        if batch == 0:
            return (
                np.zeros((0, self.action_space), dtype=np.float32),
                np.zeros(0, dtype=np.float32),
            )

        chunk = self.cfg.PREDICT_BATCH
        if batch <= chunk:
            return self._infer(board, scalars)

        policy_list: List[np.ndarray] = []
        value_list: List[np.ndarray] = []
        for i in range(0, batch, chunk):
            pl, vl = self._infer(board[i: i + chunk], scalars[i: i + chunk])
            policy_list.append(pl)
            value_list.append(vl)
        return (
            np.concatenate(policy_list, axis=0).astype(np.float32),
            np.concatenate(value_list, axis=0).astype(np.float32),
        )

    def _infer(self, board: np.ndarray, scalars: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        with torch.inference_mode():
            b = torch.from_numpy(np.ascontiguousarray(board)).to(self.device, non_blocking=True)
            s = torch.from_numpy(np.ascontiguousarray(scalars)).to(self.device, non_blocking=True)
            logits, value = self.model(b, s)
            return (
                logits.cpu().numpy().astype(np.float32),
                value.cpu().numpy().reshape(-1).astype(np.float32),
            )


# ============================================================================
# GPU + CPU 混合推理 Predictor
# ============================================================================

class MultiDevicePredictor:
    """
    GPU + CPU 混合推理 Predictor（与 4x8 版一致）。

    把一批输入按比例拆成两份：GPU 推理线程处理大头，若干个 CPU 推理线程处理
    小头，并行推理后合并返回。用法与 `Predictor` 完全一致。
    """

    def __init__(
        self,
        gpu_predictor: Predictor,
        cpu_predictor: Predictor,
        cpu_fraction: float = 0.3,
        cpu_workers: int = 1,
        min_split_batch: int = 16,
    ) -> None:
        if not 0.0 < cpu_fraction < 1.0:
            raise ValueError(f"cpu_fraction 应在 (0, 1) 之间，实际为 {cpu_fraction}")
        self._gpu = gpu_predictor
        self._cpu = cpu_predictor
        self.cpu_fraction = cpu_fraction
        self.cpu_workers = max(1, cpu_workers)
        self.min_split_batch = max(2, min_split_batch)
        self._gpu_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gpu-infer")
        self._cpu_pool = ThreadPoolExecutor(max_workers=self.cpu_workers, thread_name_prefix="cpu-infer")

    def __call__(self, board: np.ndarray, scalars: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        batch = board.shape[0]
        if batch <= self.min_split_batch:
            return self._gpu(board, scalars)

        cpu_n = max(1, min(batch - 1, int(round(batch * self.cpu_fraction))))
        gpu_n = batch - cpu_n

        gpu_future = self._gpu_pool.submit(self._gpu, board[:gpu_n], scalars[:gpu_n])

        cpu_board, cpu_scalar = board[gpu_n:], scalars[gpu_n:]
        workers = min(self.cpu_workers, cpu_n)
        if workers <= 1:
            pl_c, vl_c = self._cpu(cpu_board, cpu_scalar)
        else:
            edges = [cpu_n * k // workers for k in range(workers + 1)]
            futures = [
                self._cpu_pool.submit(
                    self._cpu,
                    cpu_board[edges[i]: edges[i + 1]],
                    cpu_scalar[edges[i]: edges[i + 1]],
                )
                for i in range(workers)
            ]
            parts = [f.result() for f in futures]
            pl_c = np.concatenate([p[0] for p in parts], axis=0).astype(np.float32)
            vl_c = np.concatenate([p[1] for p in parts], axis=0).astype(np.float32)

        pl_g, vl_g = gpu_future.result()
        return (
            np.concatenate([pl_g, pl_c], axis=0).astype(np.float32),
            np.concatenate([vl_g, vl_c], axis=0).astype(np.float32),
        )


# ============================================================================
# episode 处理：to_dict() + 补充 iteration/worker_id
# ============================================================================

def _episode_to_dict(ep, iteration: int, worker_id: int) -> Dict:
    d = dict(ep.to_dict())
    d["iteration"] = iteration
    d["worker_id"] = worker_id
    return d


# ============================================================================
# 自对弈生产者线程
# ============================================================================

class SelfPlayWorker(threading.Thread):
    """
    生产者线程：按变体分派 Rust 绑定生成 episode，压入数据队列与归档队列。
      - data_q    : 训练消费者从中取局
      - archive_q : 归档线程从中取局写 Mongo/本地（None = 该变体不归档）
    退出：stop_flag 置真后，在当前批结束后优雅退出。
    """

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

        if self.cfg.TENSORBOARD_ENABLED:
            game_idx = self.total_games + 1
            add_scalar("selfplay/game_length", int(ep["game_length"]), game_idx)
            add_scalar("selfplay/steps_per_sec", ep["game_length"] / max(duration, 1e-9), game_idx)

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


# ============================================================================
# 便捷工厂：构建模型 + Predictor（无 CLI）
# ============================================================================

def build_predictor(variant: Variant, model_path: Optional[str],
                    device_str: str = "auto") -> Tuple[Predictor, "torch.device"]:
    """构建 Predictor。model_path 为 None 时使用全新初始化网络。"""
    if not HAS_TORCH:
        print(f"[SP-{variant.id}] 警告：未安装 PyTorch，将使用退化预测（均匀 logits）")
        device = torch.device("cpu")
    else:
        device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device_str == "auto"
            else torch.device(device_str)
        )
    print(f"[SP-{variant.id}] device = {device}")

    model = BanqiNet(variant)
    if HAS_TORCH:
        model = model.to(device)
    predictor = Predictor(model, device, model_path, variant)

    if model_path and os.path.exists(model_path):
        print(f"[SP-{variant.id}] 使用模型权重: {model_path}")
    else:
        print(f"[SP-{variant.id}] 未指定有效模型路径，使用全新初始化网络")
    return predictor, device


def build_mixed_predictor(
    variant: Variant,
    model_path: Optional[str],
    device_str: str = "auto",
    cpu_workers: int = 1,
    cpu_fraction: float = 0.3,
    min_split_batch: int = 16,
) -> Tuple[Predictor, "torch.device"]:
    """
    构建 GPU + CPU 混合推理 Predictor（MultiDevicePredictor）。

    主设备（INFER_DEVICE）必须解析为 CUDA 才有混合意义。若主设备不是 CUDA
    （例如 INFER_DEVICE=cpu），回退到普通单设备 Predictor。
    """
    if not HAS_TORCH:
        print(f"[SP-{variant.id}] 警告：未安装 PyTorch，无法启用 GPU+CPU 混合推理，回退单设备")
        return build_predictor(variant, model_path, device_str)

    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device_str == "auto"
        else torch.device(device_str)
    )
    if device.type != "cuda":
        print(f"[SP-{variant.id}] 主推理设备 = {device}（非 CUDA），跳过 GPU+CPU 混合推理")
        return build_predictor(variant, model_path, device_str)

    print(
        f"[SP-{variant.id}] 启用 GPU+CPU 混合推理: GPU={device}, "
        f"CPU线程={cpu_workers}, CPU比例={cpu_fraction:.2f}"
    )
    gpu_model = BanqiNet(variant).to(device)
    gpu_predictor = Predictor(gpu_model, device, model_path, variant)

    cpu_device = torch.device("cpu")
    cpu_model = BanqiNet(variant).to(cpu_device)
    cpu_predictor = Predictor(cpu_model, cpu_device, model_path, variant)

    return (
        MultiDevicePredictor(
            gpu_predictor,
            cpu_predictor,
            cpu_fraction=cpu_fraction,
            cpu_workers=cpu_workers,
            min_split_batch=min_split_batch,
        ),
        device,
    )


def build_self_play_config(variant: Variant) -> "banqi_4x8.SelfPlayConfig":
    """构建 SelfPlayConfig（与 py/mod.rs 契约一致），c_scale/gumbel_scale 支持 env 覆盖。"""
    cfg = make_config(variant.id)
    p = variant.env_prefix
    c_scale = float(os.getenv(p + "C_SCALE", os.getenv("C_SCALE", "1.0")))
    gumbel_scale = float(os.getenv(p + "GUMBEL_SCALE", os.getenv("GUMBEL_SCALE", "1.0")))
    return banqi_4x8.SelfPlayConfig(
        mcts_sims=cfg.MCTS_SIMS,
        max_considered_actions=cfg.MAX_CONSIDERED_ACTIONS,
        temperature_steps=cfg.TEMPERATURE_STEPS,
        c_scale=c_scale,
        gumbel_scale=gumbel_scale,
    )
