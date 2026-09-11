"""banqi/infra/model_registry.py — ModelRegistry Protocol + Local 实现。

单机形态：Registry = 本地文件指针 `outputs/<variant>_latest.onnx`。
Trainer 导出新模型后 publish（原子替换指针文件），Collector 侧
Rust LocalRegistry watch 指针路径感知换网——两种模式「换网」语义一致。
"""

from __future__ import annotations

import os
import shutil
import time
from pathlib import Path
from typing import Optional, Protocol


class ModelRegistry(Protocol):
    """模型版本与准入接口（L4）。单机无准入（导出即生效）。"""

    def latest_model_path(self) -> Optional[str]:
        """当前 best 模型路径；尚无模型时返回 None。"""
        ...

    def publish(self, model_path: str) -> None:
        """把新导出的模型登记为 best（单机=复制到指针路径，原子替换）。"""
        ...


class LocalModelRegistry:
    def __init__(self, pointer_path: str) -> None:
        self.pointer_path = Path(pointer_path)

    def latest_model_path(self) -> Optional[str]:
        return str(self.pointer_path) if self.pointer_path.is_file() else None

    def publish(self, model_path: str) -> None:
        src = Path(model_path)
        if not src.is_file():
            raise FileNotFoundError(f"publish 目标不存在: {model_path}")
        self.pointer_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.pointer_path.with_suffix(self.pointer_path.suffix + ".tmp")
        # 同文件系统内先复制到临时名再原子 replace，避免 collector 读到半成品
        shutil.copyfile(src, tmp)
        os.replace(tmp, self.pointer_path)
        print(f"[registry] ✅ 已发布模型指针: {self.pointer_path} <- {model_path}")

    def wait_for_model(self, timeout: Optional[float] = None, poll: float = 2.0) -> str:
        """阻塞等待首个可用模型（collector 启动前置条件）。"""
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            path = self.latest_model_path()
            if path is not None:
                return path
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError(f"等待模型超时（{timeout}s）: {self.pointer_path}")
            time.sleep(poll)
