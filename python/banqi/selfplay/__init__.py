"""banqi/selfplay — 自对弈子包。

拆分自原单文件 self_play.py，按职责分层：
  predictor.py : Predictor / MultiDevicePredictor（推理包装 + 热重载 + 混合设备）
  worker.py    : SelfPlayWorker 线程 + sp_worker_main 多进程子进程入口
  config.py    : build_predictor / build_mixed_predictor / build_self_play_config

自对弈统一经 Rust 唯一入口 `run_python_match` / `run_native_match`（见 worker.py）。
旧的 Rust 持有模型收集器（RustTorchCollector / RustOnnxCollector）已移除。

self_play.py 保留为向后兼容的 re-export 入口。
"""

from .predictor import (
    OnnxPredictor,
    Predictor,
    MultiDevicePredictor,
    RELOAD_CHECK_INTERVAL,
)
from .worker import (
    SelfPlayWorker,
    sp_worker_main,
    _episode_to_dict,
    _log_episode,
)
from .config import (
    build_predictor,
    build_onnx_predictor,
    build_mixed_predictor,
    build_self_play_config,
)

__all__ = [
    "OnnxPredictor",
    "Predictor",
    "MultiDevicePredictor",
    "RELOAD_CHECK_INTERVAL",
    "SelfPlayWorker",
    "sp_worker_main",
    "_episode_to_dict",
    "_log_episode",
    "build_predictor",
    "build_onnx_predictor",
    "build_mixed_predictor",
    "build_self_play_config",
]
