"""
self_play.py — 自对弈数据生产者（无 CLI 参数）

通过 PyO3 Python 绑定 (banqi_4x8) 驱动 Gumbel MCTS 自对弈，生成训练局数据。
本模块以线程形式运行（SelfPlayWorker），把生成的 episode dict 同时压入：
  - data_q    （训练消费队列）
  - archive_q （MongoDB 冷存储归档队列）

所有配置集中在 config.py，不再接受命令行参数。
"""

from __future__ import annotations

import os
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Tuple

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

from config import config
from constant import ACTION_SPACE_SIZE
from nn_model import BanqiNet, load_model_weights
from tb_logger import add_scalar  # TensorBoard 训练日志（未启用时为 no-op）


# ============================================================================
# 推理端 Predictor（热重载，匹配 py_evaluator.rs 契约；内部按 batch=32 分块）
# ============================================================================

class Predictor:
    """
    薄包装：
      - 确保模型在 eval / inference_mode
      - 输入/输出都是 numpy（Rust 侧转成 numpy 后传进来）
      - 简易模型热重载（检查 --model 文件 mtime）
      - 对任意 batch 按 PREDICT_BATCH 分块推理，避免显存/内存峰值
      - TF32 + cudnn benchmark 优化吞吐
    """

    def __init__(self, model: "BanqiNet", device: "torch.device", model_path: str | None) -> None:
        self.model = model.to(device)
        self.device = device
        self.model_path: str | None = model_path
        self._mtime: float = 0.0
        self.model.eval()

        # 吞吐优化：TF32 + cudnn auto-tune
        # torch.compile 在 Windows 上需要 Triton（不可用），跳过；TF32 + cudnn 已提供大部分加速
        if HAS_TORCH and device.type == "cuda":
            torch.backends.cudnn.benchmark = True
            torch.set_float32_matmul_precision("high")
            print("[Predictor] 吞吐优化: TF32 + cudnn.benchmark 已启用")

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
                print(f"[Predictor] 已重载权重: {self.model_path}")
            except Exception as exc:  # pragma: no cover
                print(f"[Predictor] 权重加载失败 (保持旧模型): {exc}")

    def __call__(self, board: np.ndarray, scalars: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """返回 (policy_logits (N,352) float32, values (N,) float32)，匹配绑定契约。"""
        self._maybe_reload_weights()
        batch = board.shape[0]
        if not HAS_TORCH:
            return (
                np.zeros((batch, ACTION_SPACE_SIZE), dtype=np.float32),
                np.zeros(batch, dtype=np.float32),
            )
        if batch == 0:
            return (
                np.zeros((0, ACTION_SPACE_SIZE), dtype=np.float32),
                np.zeros(0, dtype=np.float32),
            )

        chunk = config.PREDICT_BATCH
        if batch <= chunk:
            return self._infer(board, scalars)

        policy_list: List[np.ndarray] = []
        value_list: List[np.ndarray] = []
        for i in range(0, batch, chunk):
            pl, vl = self._infer(board[i : i + chunk], scalars[i : i + chunk])
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
    GPU + CPU 混合推理 Predictor。

    把一批输入按比例拆成两份：GPU 推理线程处理大头，若干个 CPU 推理线程处理
    小头，并行推理后合并返回。这样在单个 GPU 推理线程吞吐不足以喂饱 CPU MCTS
    自对弈时，用空闲的 CPU 算力分担推理，总吞吐 ≈ GPU 吞吐 + CPU 吞吐。

    用法与 `Predictor` 完全一致：直接 `predictor(board, scalars)`，
    返回 (policy_logits (N,352) float32, values (N,) float32)。

    线程模型：
      - GPU 侧：1 个专用线程（gpu_pool），串行化对同一个 GPU 模型的访问，
        避免多个 MCTS worker 并发触碰 CUDA 上下文；
      - CPU 侧：`cpu_workers` 个线程（cpu_pool），CPU 子批再均分并行推理。

    依赖 PyTorch 推理时（C++ 算子执行）会释放 GIL 的特性，GPU/CPU 两个推理
    线程的计算才能真正并行；numpy 搬运等 Python 层开销由 GIL 串行化，占比小。

    权重热重载：两个子 Predictor 各自检查 model_path 的 mtime 独立热重载，
    训练侧导出 checkpoint（原子 replace）后两者会先后加载同一份新权重。
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
        # batch 太小：线程/拼接开销超过拆分收益，直接 GPU 一次推理
        if batch <= self.min_split_batch:
            return self._gpu(board, scalars)

        cpu_n = max(1, min(batch - 1, int(round(batch * self.cpu_fraction))))
        gpu_n = batch - cpu_n

        # GPU 大头交给 gpu_pool（单线程，串行化 GPU 访问）
        gpu_future = self._gpu_pool.submit(self._gpu, board[:gpu_n], scalars[:gpu_n])

        # CPU 小头按 cpu_workers 均分，并行推理后按序拼接
        cpu_board, cpu_scalar = board[gpu_n:], scalars[gpu_n:]
        workers = min(self.cpu_workers, cpu_n)
        if workers <= 1:
            pl_c, vl_c = self._cpu(cpu_board, cpu_scalar)
        else:
            edges = [cpu_n * k // workers for k in range(workers + 1)]
            futures = [
                self._cpu_pool.submit(
                    self._cpu,
                    cpu_board[edges[i] : edges[i + 1]],
                    cpu_scalar[edges[i] : edges[i + 1]],
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
    生产者线程：持续调用 Rust 绑定生成 episode，压入数据队列与归档队列。
      - data_q    : 训练消费者从中取局
      - archive_q : 归档线程从中取局写 Mongo/本地
    退出：stop_flag 置真后，在当前批结束后优雅退出。
    """

    def __init__(
        self,
        predictor: Predictor,
        sp_cfg,
        data_q: "queue.Queue",
        archive_q: "queue.Queue",
        stop_flag: "List[bool]",
        worker_id: int = 0,
    ) -> None:
        super().__init__(name="SelfPlayWorker", daemon=True)
        self.predictor = predictor
        self.sp_cfg = sp_cfg
        self.data_q = data_q
        self.archive_q = archive_q
        self.stop_flag = stop_flag
        self.worker_id = worker_id

        # 统计
        self.total_games = 0
        self.total_samples = 0
        self.iteration = 0
        self._game_count = 0  # 当前迭代内局数
        # 逐局统计记录（供基线验证/监控读取；纯追加，不改默认行为）
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
        while not self.stop_flag[0]:
            t0 = time.time()
            if config.USE_BATCHED_SELF_PLAY and hasattr(
                banqi_4x8, "run_batched_self_play_with_predictor"
            ):
                # 批量自对弈：同时推进 BATCH_CONCURRENCY 局，合并成大 batch 推理，
                # 摊薄 GPU 推理固定开销，提升吞吐。每批目标局数 = 一次迭代局数。
                episodes = banqi_4x8.run_batched_self_play_with_predictor(
                    predict_fn=self.predictor,
                    config=self.sp_cfg,
                    num_games=config.GAMES_PER_ITER,
                    concurrency=config.BATCH_CONCURRENCY,
                    worker_id=self.worker_id,
                )
            elif config.NUM_WORKERS > 1:
                episodes = banqi_4x8.run_parallel_self_play_with_predictor(
                    predict_fn=self.predictor,
                    config=self.sp_cfg,
                    num_workers=config.NUM_WORKERS,
                    games_per_worker=config.GAMES_PER_WORKER,
                    worker_id=self.worker_id,
                )
            else:
                episodes = banqi_4x8.run_self_play_with_predictor(
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
                with self._iter_lock:
                    ep_dict = _episode_to_dict(ep, self.iteration, self.worker_id)
                    self._log_game(ep_dict, batch_duration / max(len(episodes), 1))
                # 同时压入数据队列与归档队列
                self._put(self.data_q, ep_dict)
                self._put(self.archive_q, ep_dict)
                with self._iter_lock:
                    self.total_games += 1
                    self.total_samples += len(ep_dict["samples"]) if "samples" in ep_dict else ep_dict["num_samples"]
                    self._advance_iteration()

    def _advance_iteration(self) -> None:
        """与 data_collector.rs 迭代推进语义一致：每 GAMES_PER_ITER 局 iteration += 1。"""
        self._game_count += 1
        if self._game_count >= config.GAMES_PER_ITER:
            self._game_count -= config.GAMES_PER_ITER
            self.iteration += 1
            print(f"[Worker-{self.worker_id}] 📍 完成迭代 {self.iteration - 1} → 进入迭代 {self.iteration}")

    def _log_game(self, ep: Dict, duration: float) -> None:
        winner_str = {1: "红胜", -1: "黑胜"}.get(ep["winner"], "平局")
        print(
            f"[Worker-{self.worker_id}] Game #{self.total_games + 1} (iter={self.iteration}): "
            f"步数={ep['game_length']}, 结果={winner_str}, "
            f"耗时={duration:.1f}s ({ep['game_length'] / max(duration, 1e-9):.1f} steps/s)"
        )
        # 逐局统计记录（在 _iter_lock 临界区内调用）
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

def build_predictor(model_path: str | None, device_str: str = "auto") -> Tuple[Predictor, "torch.device"]:
    """构建 Predictor。model_path 为 None 时使用全新初始化网络。"""
    if not HAS_TORCH:
        print("[SelfPlay] 警告：未安装 PyTorch，将使用退化预测（均匀 logits）")
        device = torch.device("cpu")
    else:
        device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device_str == "auto"
            else torch.device(device_str)
        )
    print(f"[SelfPlay] device = {device}")

    model = BanqiNet()
    if HAS_TORCH:
        model = model.to(device)
    predictor = Predictor(model, device, model_path)

    if model_path and os.path.exists(model_path):
        print(f"[SelfPlay] 使用模型权重: {model_path}")
    else:
        print(f"[SelfPlay] 未指定有效模型路径，使用全新初始化网络")
    return predictor, device


def build_mixed_predictor(
    model_path: str | None,
    device_str: str = "auto",
    cpu_workers: int = 1,
    cpu_fraction: float = 0.3,
    min_split_batch: int = 16,
) -> Tuple["Predictor", "torch.device"]:
    """
    构建 GPU + CPU 混合推理 Predictor（MultiDevicePredictor）。

    主设备（INFER_DEVICE）必须解析为 CUDA 才有混合意义：GPU 处理大头、
    CPU 处理小头。若主设备不是 CUDA（例如 INFER_DEVICE=cpu），回退到
    普通单设备 Predictor。
    """
    if not HAS_TORCH:
        print("[SelfPlay] 警告：未安装 PyTorch，无法启用 GPU+CPU 混合推理，回退单设备")
        return build_predictor(model_path, device_str)

    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device_str == "auto"
        else torch.device(device_str)
    )
    if device.type != "cuda":
        print(f"[SelfPlay] 主推理设备 = {device}（非 CUDA），跳过 GPU+CPU 混合推理")
        return build_predictor(model_path, device_str)

    print(
        f"[SelfPlay] 启用 GPU+CPU 混合推理: GPU={device}, "
        f"CPU线程={cpu_workers}, CPU比例={cpu_fraction:.2f}"
    )
    gpu_model = BanqiNet().to(device)
    gpu_predictor = Predictor(gpu_model, device, model_path)

    cpu_device = torch.device("cpu")
    cpu_model = BanqiNet().to(cpu_device)
    cpu_predictor = Predictor(cpu_model, cpu_device, model_path)

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


def build_self_play_config() -> "banqi_4x8.SelfPlayConfig":
    """构建 SelfPlayConfig（与 py/mod.rs 契约一致）。"""
    return banqi_4x8.SelfPlayConfig(
        mcts_sims=config.MCTS_SIMS,
        max_considered_actions=config.MAX_CONSIDERED_ACTIONS,
        temperature_steps=config.TEMPERATURE_STEPS,
    )
