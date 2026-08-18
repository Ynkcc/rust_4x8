"""banqi/selfplay — 自对弈子包。

拆分自原单文件 self_play.py，按职责分层：
  predictor.py : Predictor / MultiDevicePredictor（推理包装 + 热重载 + 混合设备）
  worker.py    : SelfPlayWorker 线程 + sp_worker_main 多进程子进程入口
  config.py    : build_predictor / build_mixed_predictor / build_self_play_config /
                 build_rust_collector / rust_collector_run_batch

self_play.py 保留为向后兼容的 re-export 入口。
"""

from .predictor import Predictor, MultiDevicePredictor, RELOAD_CHECK_INTERVAL
from .worker import (
    SelfPlayWorker,
    sp_worker_main,
    _splay_fns,
    _episode_to_dict,
    _log_episode,
)
from .config import (
    build_predictor,
    build_mixed_predictor,
    build_self_play_config,
    build_rust_collector,
    rust_collector_run_batch,
)

__all__ = [
    "Predictor",
    "MultiDevicePredictor",
    "RELOAD_CHECK_INTERVAL",
    "SelfPlayWorker",
    "sp_worker_main",
    "_splay_fns",
    "_episode_to_dict",
    "_log_episode",
    "build_predictor",
    "build_mixed_predictor",
    "build_self_play_config",
    "build_rust_collector",
    "rust_collector_run_batch",
]
