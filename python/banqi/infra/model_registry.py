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


def scheduler_variant(endpoint: Optional[str] = None) -> str:
    """从调度器 GetInfo 获取变体 id（trainer 启动时无需命令行传入变体）。"""
    import grpc

    from banqi.proto import scheduler_pb2, scheduler_pb2_grpc

    endpoint = endpoint or os.environ.get("SCHEDULER_ENDPOINT", "http://127.0.0.1:50051")
    target = endpoint.split("://", 1)[-1]
    stub = scheduler_pb2_grpc.SchedulerServiceStub(grpc.insecure_channel(target))
    variant = stub.GetInfo(scheduler_pb2.GetInfoRequest()).variant
    if not variant:
        raise ValueError(f"调度器未下发变体（GetInfo.variant 为空）: {endpoint}，请升级调度器并配置 SCHEDULER_VARIANT")
    return variant


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


class SchedulerModelRegistry:
    """分布式实现：经调度器签发预签名 URL 直传模型 + RegisterNetwork 登记。

    R2 凭据只在调度器持有，trainer 零存储配置：
    - publish：SignNetworkUpload（请求预签名 PUT）→ HTTP 直传 → RegisterNetwork
      （首个网络直接晋级，其后自动创建 gatekeeper 对打）；
    - parent_sha 记录上次成功登记的 sha，作为谱系信息上报。
    调度器地址：SCHEDULER_ENDPOINT（默认 http://127.0.0.1:50051）。
    """

    def __init__(self, endpoint: Optional[str] = None) -> None:
        import grpc

        from banqi.proto import scheduler_pb2, scheduler_pb2_grpc

        self.endpoint = endpoint or os.environ.get("SCHEDULER_ENDPOINT", "http://127.0.0.1:50051")
        self._pb2 = scheduler_pb2
        # grpc.insecure_channel 只接受 host:port，不接受 URL scheme 前缀
        target = self.endpoint.split("://", 1)[-1]
        self._channel = grpc.insecure_channel(target)
        self._stub = scheduler_pb2_grpc.SchedulerServiceStub(self._channel)
        self.last_sha: Optional[str] = None

    @staticmethod
    def sha256_of(path: str) -> str:
        import hashlib

        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()

    def latest_model_path(self) -> Optional[str]:
        """分布式形态下 trainer 不需要从 registry 取模型，返回 None。"""
        return None

    def publish(self, model_path: str) -> None:
        import urllib.request

        sha = self.sha256_of(model_path)
        with open(model_path, "rb") as f:
            body = f.read()

        # 1. 请求调度器签发预签名 PUT（对象键 networks/<sha>.bin）
        sign = self._stub.SignNetworkUpload(
            self._pb2.SignNetworkUploadRequest(
                trainer_id=f"trainer-{os.getpid()}",
                sha=sha,
                content_length=len(body),
                content_sha256=sha,
            )
        )
        if not sign.accepted:
            print(f"[registry] ⚠️ SignNetworkUpload 被拒绝: {sign.message}")
            return

        # 2. HTTP 直传 R2
        req = urllib.request.Request(sign.upload_url, data=body, method="PUT")
        with urllib.request.urlopen(req, timeout=600) as resp:
            if resp.status != 200:
                raise RuntimeError(f"模型直传失败: HTTP {resp.status} {sign.object_key}")
        print(f"[registry] ✅ 模型已直传: {sign.object_key} <- {model_path}")

        # 3. 登记网络（触发 gatekeeper 对打 / 首个网络晋级）
        ack = self._stub.RegisterNetwork(
            self._pb2.RegisterNetworkRequest(
                sha=sha,
                parent_sha=self.last_sha or "",
                notes=f"trainer publish {os.path.basename(model_path)}",
            )
        )
        if not ack.accepted:
            print(f"[registry] ⚠️ RegisterNetwork 被拒绝: {ack.message}")
            return
        self.last_sha = sha
        print(f"[registry] ✅ 已登记网络 sha={sha}: {ack.message} {ack.match_task_hint}".rstrip())
