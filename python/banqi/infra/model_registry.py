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


class SchedulerModelRegistry:
    """分布式实现：模型直传 R2 + gRPC RegisterNetwork 登记到 Go scheduler。

    - publish：sha256 命名上传 `networks/<sha>.bin`（与 Go r2.NetworkKey 一致），
      再调 RegisterNetwork（首个网络直接晋级，其后自动创建 gatekeeper 对打）；
    - parent_sha 记录上次成功登记的 sha，作为谱系信息上报。
    凭据走环境变量：R2_ACCOUNT_ID / R2_ACCESS_KEY_ID / R2_SECRET_ACCESS_KEY / R2_BUCKET；
    调度器地址：SCHEDULER_ENDPOINT（默认 http://127.0.0.1:50051）。
    """

    def __init__(self) -> None:
        import boto3

        account = os.environ["R2_ACCOUNT_ID"]
        self.bucket = os.environ["R2_BUCKET"]
        self.endpoint = os.environ.get("SCHEDULER_ENDPOINT", "http://127.0.0.1:50051")
        self._s3 = boto3.client(
            "s3",
            endpoint_url=f"https://{account}.r2.cloudflarestorage.com",
            aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
            aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
            region_name="auto",
        )
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
        import grpc

        from banqi.proto import scheduler_pb2, scheduler_pb2_grpc

        sha = self.sha256_of(model_path)
        key = f"networks/{sha}.bin"
        self._s3.upload_file(model_path, self.bucket, key)
        print(f"[registry] ✅ 模型已上传 R2: {key} <- {model_path}")

        with grpc.insecure_channel(self.endpoint) as channel:
            stub = scheduler_pb2_grpc.SchedulerServiceStub(channel)
            ack = stub.RegisterNetwork(
                scheduler_pb2.RegisterNetworkRequest(
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
