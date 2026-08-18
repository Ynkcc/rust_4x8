"""benchmark/predictors.py — 基准用的推理包装 / 模拟 predictor。"""

from __future__ import annotations

import threading
import time
from typing import Any, Tuple

import numpy as np


class CountingPredictor:
    """包装任意 callable predictor，统计调用次数与送入的总样本数（线程安全）。

    Rust 侧串行/并行/批量自对弈都可能从多线程调用 predict_fn，因此统计需加锁。
    """

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.calls = 0
        self.samples = 0
        self._lock = threading.Lock()

    def __call__(self, board: np.ndarray, scalars: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        n = int(board.shape[0])
        with self._lock:
            self.calls += 1
            self.samples += n
        return self.inner(board, scalars)


class SimulatedPredictor:
    """模拟推理延迟：delay(batch) = 固定启动开销 + 每样本线性成本。

    不依赖 PyTorch / 真实模型，用于快速验证基准脚本流程（--simulated），
    或对比「小 batch 串行 vs 大 batch 并行」时模拟 GPU 推理的固定开销摊薄效果。
    """

    FIXED_OVERHEAD_S = 0.002   # 每次推理调用的固定开销（kernel 启动 / 拷贝）
    PER_SAMPLE_S = 0.0001      # 每个样本的线性推理成本

    def __init__(self, action_space_size: int) -> None:
        self.action_space_size = action_space_size

    def __call__(self, board: np.ndarray, scalars: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        batch = int(board.shape[0])
        time.sleep(self.FIXED_OVERHEAD_S + batch * self.PER_SAMPLE_S)
        rng = np.random.default_rng()
        policy = rng.standard_normal((batch, self.action_space_size)).astype(np.float32)
        values = rng.uniform(-1.0, 1.0, size=batch).astype(np.float32)
        return policy, values
