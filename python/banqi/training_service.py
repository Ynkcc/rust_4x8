"""
training_service.py — 向后兼容入口。

原单文件实现已拆分为 banqi/training/ 子包（buffer.py / losses.py / worker.py）。
本文件仅做 re-export，避免改动现有调用方（train.py 等）。
新增代码请直接 `from banqi.training import TrainWorker`。
"""

from banqi.training import (
    DataBuffer,
    episode_to_samples,
    TrainWorker,
    train_step,
    run_training_epochs,
)

__all__ = [
    "DataBuffer",
    "episode_to_samples",
    "TrainWorker",
    "train_step",
    "run_training_epochs",
]
