"""
tb_logger.py — TensorBoard 训练日志封装

单进程内共享一个全局 SummaryWriter，供训练 / 自对弈 / 系统监控各线程写入，
避免各自创建 writer 产生重复 event 文件。

用法（训练入口调用一次初始化，各线程随时写入）：
    from tb_logger import init_summary_writer, add_scalar, close_summary_writer

    init_summary_writer(log_dir="runs/20260814-100000")  # 启动时
    add_scalar("train/loss", 0.12, step=100)             # 任意线程（线程安全）
    close_summary_writer()                               # 结束时

依赖：tensorboard（requirements.txt 已有）。未安装或初始化失败时，
所有调用自动降级为 no-op，不影响训练。
"""

from __future__ import annotations

import threading
from typing import Optional

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:  # pragma: no cover
    HAS_TENSORBOARD = False

_writer: Optional[SummaryWriter] = None
_lock = threading.Lock()


def init_summary_writer(
    log_dir: str = "runs",
    enabled: bool = True,
    flush_secs: int = 30,
) -> bool:
    """
    初始化全局 SummaryWriter（幂等，保留首次创建的 writer）。

    参数:
        log_dir:   事件文件目录
        enabled:   总开关（对应 config.TENSORBOARD_ENABLED）
        flush_secs:事件刷新到磁盘的间隔秒数

    返回是否成功启用（True 表示后续 add_scalar 会真正写入）。
    """
    global _writer
    if not enabled or not HAS_TENSORBOARD:
        return False
    if _writer is not None:
        return True
    try:
        _writer = SummaryWriter(log_dir=log_dir, flush_secs=flush_secs)
        return True
    except Exception as exc:  # noqa: BLE001 - 初始化失败原因多样
        print(f"[TB] ⚠️ TensorBoard 初始化失败，日志记录已禁用: {exc}")
        _writer = None
        return False


def add_scalar(
    tag: str,
    scalar_value: float,
    global_step: Optional[int] = None,
    walltime: Optional[float] = None,
    **kwargs,
) -> None:
    """
    线程安全的标量写入；writer 未初始化时 no-op，写失败静默忽略。
    兼容 torch SummaryWriter 的 step= 关键字（作为 global_step 的别名）。
    """
    if "step" in kwargs:
        global_step = kwargs["step"]
    if _writer is None:
        return
    with _lock:
        try:
            _writer.add_scalar(tag, scalar_value, global_step, walltime)
        except Exception:  # noqa: BLE001 - 避免监控线程因日志异常而崩溃
            pass


def close_summary_writer() -> None:
    """关闭并 flush 全局 writer（幂等）。"""
    global _writer
    if _writer is None:
        return
    with _lock:
        try:
            _writer.close()
        except Exception:  # noqa: BLE001
            pass
        _writer = None
