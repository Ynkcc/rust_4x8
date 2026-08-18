"""banqi/selfplay/predictor.py — 推理端 Predictor（热重载 + 分块 + 混合设备）。

Predictor：薄包装，确保模型 eval / 输入输出 numpy（Rust 侧转成 numpy 传入）、
  简易模型热重载（检查 model_path mtime）、按 PREDICT_BATCH 分块推理以抑制显存峰值、
  TF32 + cudnn benchmark 优化吞吐。匹配 py_evaluator.rs 契约。
MultiDevicePredictor：GPU + CPU 混合推理，把一批按比例拆成两份并行推理后合并。
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Tuple

import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:  # pragma: no cover
    HAS_TORCH = False

from banqi.config import make_config
from banqi.constants import build_constants
from banqi.nn_model import BanqiNet, load_model_weights
from banqi.variant import Variant

# 模型热重载检查节流间隔：Rust 每步 MCTS 评估都会回调 __call__，若每次都做
# os.path.exists + os.path.getmtime 两个系统调用，会白白占用推理热路径。
RELOAD_CHECK_INTERVAL = 2.0


class Predictor:
    """薄包装：eval 模式、numpy I/O、权重热重载、PREDICT_BATCH 分块、吞吐优化。"""

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
        self._last_reload_check: float = 0.0
        self.model.eval()

        # 吞吐优化：TF32 + cudnn auto-tune
        if HAS_TORCH and device.type == "cuda":
            torch.backends.cudnn.benchmark = True
            torch.set_float32_matmul_precision("high")
            print(f"[SP-{variant.id}] 吞吐优化: TF32 + cudnn.benchmark 已启用")

        self._maybe_reload_weights(force=True)

    def _maybe_reload_weights(self, force: bool = False) -> None:
        if not HAS_TORCH or not self.model_path:
            return
        # 节流：force（初始化）与超过间隔的调用才做文件 stat，热路径零系统调用。
        now = time.monotonic()
        if not force and now - self._last_reload_check < RELOAD_CHECK_INTERVAL:
            return
        self._last_reload_check = now
        if not os.path.exists(self.model_path):
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
        # 模型输出恒为 float32：concatenate 保持 dtype，无需再 astype 拷贝一次
        return (
            np.concatenate(policy_list, axis=0),
            np.concatenate(value_list, axis=0),
        )

    def _infer(self, board: np.ndarray, scalars: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        with torch.inference_mode():
            b = torch.from_numpy(np.ascontiguousarray(board)).to(self.device, non_blocking=True)
            s = torch.from_numpy(np.ascontiguousarray(scalars)).to(self.device, non_blocking=True)
            logits, value = self.model(b, s)
            # 模型输出恒为 float32：直接 .numpy()，去掉冗余 astype 的全量拷贝
            return (
                logits.detach().cpu().numpy(),
                value.detach().cpu().numpy().reshape(-1),
            )


class MultiDevicePredictor:
    """GPU + CPU 混合推理 Predictor。

    把一批输入按比例拆成两份：GPU 推理线程处理大头，若干个 CPU 推理线程处理小头，
    并行推理后合并返回。用法与 `Predictor` 完全一致。
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
            pl_c = np.concatenate([p[0] for p in parts], axis=0)
            vl_c = np.concatenate([p[1] for p in parts], axis=0)

        pl_g, vl_g = gpu_future.result()
        # 各部分均为 float32（Predictor 输出恒为 float32），concatenate 保持 dtype
        return (
            np.concatenate([pl_g, pl_c], axis=0),
            np.concatenate([vl_g, vl_c], axis=0),
        )


class OnnxPredictor:
    """Python onnxruntime 推理 Predictor（MODEL_BACKEND="onnx" 时的回退方案）。

    契约与 `Predictor` 一致（匹配 py_evaluator.rs / py/mod.rs 绑定）：
      `__call__(board: (N, C, H, W) float32, scalars: (N, S) float32)`
      -> `(policy_logits (N, A) float32, values (N,) float32)`

    仅在 Rust 绑定 `RustOnnxCollector` 不可用（wheel 未启用 onnx+pyo3）时使用；
    推理走 onnxruntime，可通过 `ONNX_PROVIDERS` 指定执行提供者。
    """

    def __init__(
        self,
        model_path: str,
        action_space: int,
        providers: Optional[List[str]] = None,
        variant: Optional[Variant] = None,
    ) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "MODEL_BACKEND=onnx 需要 onnxruntime：pip install onnxruntime"
            ) from exc
        self.variant = variant
        self.action_space = int(action_space)
        self.model_path = model_path
        self._providers = providers or ["CPUExecutionProvider"]
        self._mtime: float = 0.0
        self._last_reload_check: float = 0.0
        self._ort = ort
        self._sess = ort.InferenceSession(model_path, providers=self._providers)
        self._reload_weights(force=True)

    def _tag(self) -> str:
        return f"[SP-{self.variant.id}]" if self.variant is not None else "[SP-?]"

    def _reload_weights(self, force: bool = False) -> None:
        """mtime 热重载（与 Predictor 语义一致）：训练侧保存 checkpoint 后自动刷新。"""
        now = time.monotonic()
        if not force and now - self._last_reload_check < RELOAD_CHECK_INTERVAL:
            return
        self._last_reload_check = now
        if not os.path.exists(self.model_path):
            return
        mtime = os.path.getmtime(self.model_path)
        if force or mtime > self._mtime:
            try:
                self._sess = self._ort.InferenceSession(
                    self.model_path, providers=self._providers
                )
                self._mtime = mtime
                print(f"{self._tag()} 已重载 ONNX 模型: {self.model_path}")
            except Exception as exc:  # pragma: no cover
                print(f"{self._tag()} ONNX 重载失败（保持旧模型）: {exc}")

    def __call__(
        self, board: np.ndarray, scalars: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """返回 (policy_logits (N, A) float32, values (N,) float32)，匹配绑定契约。"""
        self._reload_weights()
        if board.shape[0] == 0:
            return (
                np.zeros((0, self.action_space), dtype=np.float32),
                np.zeros(0, dtype=np.float32),
            )
        outputs = self._sess.run(
            None,
            {
                "board": np.ascontiguousarray(board, dtype=np.float32),
                "scalars": np.ascontiguousarray(scalars, dtype=np.float32),
            },
        )
        policy_logits = np.asarray(outputs[0], dtype=np.float32)
        value = np.asarray(outputs[1], dtype=np.float32).reshape(-1)
        return policy_logits, value
