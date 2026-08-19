"""
self_play.py — 向后兼容入口。

原单文件实现已拆分为 banqi/selfplay/ 子包（predictor.py / worker.py / config.py）。
本文件仅做 re-export，避免改动现有调用方（trainer_cli/runners.py 等）。
新增代码请直接 `from banqi.selfplay import SelfPlayWorker, sp_worker_main`。
"""

from banqi.selfplay import (
    OnnxPredictor,
    Predictor,
    MultiDevicePredictor,
    SelfPlayWorker,
    sp_worker_main,
    build_predictor,
    build_onnx_predictor,
    build_mixed_predictor,
    build_self_play_config,
    _episode_to_dict,
    _log_episode,
    RELOAD_CHECK_INTERVAL,
)

__all__ = [
    "OnnxPredictor",
    "Predictor",
    "MultiDevicePredictor",
    "SelfPlayWorker",
    "sp_worker_main",
    "build_predictor",
    "build_onnx_predictor",
    "build_mixed_predictor",
    "build_self_play_config",
    "_episode_to_dict",
    "_log_episode",
    "RELOAD_CHECK_INTERVAL",
]
