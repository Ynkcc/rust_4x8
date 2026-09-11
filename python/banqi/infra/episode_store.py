"""banqi/infra/episode_store.py — EpisodeStore Protocol + Local 实现。

单机形态：Collector（Rust banqi-collector）把 episode 以 jsonl.gz 落盘目录，
Trainer 通过本模块扫描目录消费，取代内存队列直喂路径。
字段契约由 Rust `serialize::episode_to_dict_json` 保证（与 PyO3/gRPC 路径一致），
Trainer 无差别消费，满足「两种模式 episode 序列化格式一致」的回归要求。
"""

from __future__ import annotations

import gzip
import io
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Protocol


class EpisodeStore(Protocol):
    """episode 数据出口/入口统一接口（L4）。"""

    def put(self, episodes: Iterable[Dict[str, Any]]) -> int:
        """写入一批 episode，返回写入局数。"""
        ...

    def iter_new_episodes(self) -> Iterator[Dict[str, Any]]:
        """迭代尚未消费过的 episode（按批次文件序）。"""
        ...


class LocalEpisodeStore:
    """本地目录实现：`<root>/<variant>/*.jsonl.gz`，每批一个文件。

    消费端通过 get/get_nowait/qsize 获得队列语义，可直接被
    CountingQueue / TrainWorker 包裹使用；内部按文件粒度去重（已消费
    文件不再读取），因此 Trainer 进程重启后会重读全部文件（冷启动续训
    由调用方用 --train-mode archive 或清理目录控制）。
    """

    def __init__(self, root: str, variant: str, poll_interval: float = 1.0) -> None:
        self.root = Path(root)
        self.variant = variant
        self.poll_interval = poll_interval
        self._consumed: set[str] = set()

    # ---- 写入端（Python 侧归档/回放等场景使用） ----

    def put(self, episodes: Iterable[Dict[str, Any]]) -> int:
        eps = list(episodes)
        if not eps:
            return 0
        out_dir = self.root / self.variant
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"py_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}.jsonl.gz"
        with gzip.open(path, "wt", encoding="utf-8") as f:
            for ep in eps:
                f.write(json.dumps(ep, ensure_ascii=False) + "\n")
        return len(eps)

    # ---- 读取端 ----

    def _list_batches(self) -> List[Path]:
        d = self.root / self.variant
        if not d.is_dir():
            return []
        return sorted(d.glob("*.jsonl.gz"))

    def iter_new_episodes(self) -> Iterator[Dict[str, Any]]:
        for batch in self._list_batches():
            name = batch.name
            if name in self._consumed:
                continue
            # 批次元数据回填：与内存队列路径的 episode dict（round_idx/worker_id）对齐
            iter_m = re.search(r"iter(\d+)", name)
            worker_m = re.search(r"_w(\d+)", name)
            round_idx = int(iter_m.group(1)) if iter_m else 0
            worker_id = int(worker_m.group(1)) if worker_m else 0
            try:
                with gzip.open(batch, "rt", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            ep = json.loads(line)
                            ep.setdefault("round_idx", round_idx)
                            ep.setdefault("worker_id", worker_id)
                            yield ep
            except Exception as exc:  # 半成品文件（collector 中断）跳过
                print(f"[EpisodeStore] ⚠️ 跳过损坏批次 {batch.name}: {exc}")
            self._consumed.add(name)

    def drain(self) -> List[Dict[str, Any]]:
        return list(self.iter_new_episodes())

    # ---- 队列语义（供 TrainWorker 直连） ----

    def get(self, timeout: Optional[float] = None) -> Dict[str, Any]:
        """阻塞直到有新 episode；timeout 到期抛 queue.Empty 语义的超时异常。"""
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            for ep in self.iter_new_episodes():
                return ep
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError(f"LocalEpisodeStore.get 超时（{timeout}s）：无新 episode")
            time.sleep(self.poll_interval)

    def get_nowait(self) -> Dict[str, Any]:
        it = self.iter_new_episodes()
        try:
            return next(it)
        except StopIteration:
            raise TimeoutError("LocalEpisodeStore.get_nowait：无新 episode") from None

    def qsize(self) -> int:
        return sum(1 for b in self._list_batches() if b.name not in self._consumed)


class R2EpisodeStore:
    """分布式实现：从 R2（S3 API）拉取 worker 直传的 episode 批次。

    对象键布局与 Go scheduler 一致：`episodes/<network_sha>/<data_id>.jsonl.gz`。
    按对象 key 粒度去重；Trainer 重启后会重读全部对象（去重态仅存内存）。
    凭据走环境变量：R2_ACCOUNT_ID / R2_ACCESS_KEY_ID / R2_SECRET_ACCESS_KEY / R2_BUCKET。
    """

    PREFIX = "episodes/"

    def __init__(self, poll_interval: float = 5.0) -> None:
        import boto3  # 延迟导入：仅分布式形态需要

        account = os.environ["R2_ACCOUNT_ID"]
        bucket = os.environ["R2_BUCKET"]
        self.bucket = bucket
        self.poll_interval = poll_interval
        self._s3 = boto3.client(
            "s3",
            endpoint_url=f"https://{account}.r2.cloudflarestorage.com",
            aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
            aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
            region_name="auto",
        )
        self._consumed: set[str] = set()

    # ---- 写入端（trainer 不产 episode，占位实现满足 Protocol） ----

    def put(self, episodes: Iterable[Dict[str, Any]]) -> int:
        raise NotImplementedError("R2EpisodeStore 仅作消费端；写入经 collector 直传")

    # ---- 读取端 ----

    def _list_keys(self) -> List[str]:
        keys: List[str] = []
        paginator = self._s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=self.PREFIX):
            for obj in page.get("Contents", []):
                if obj["Key"].endswith(".jsonl.gz"):
                    keys.append(obj["Key"])
        return sorted(keys)

    def iter_new_episodes(self) -> Iterator[Dict[str, Any]]:
        import gzip as _gzip

        for key in self._list_keys():
            if key in self._consumed:
                continue
            name = key.rsplit("/", 1)[-1]
            iter_m = re.search(r"iter(\d+)", name)
            round_idx = int(iter_m.group(1)) if iter_m else 0
            worker_m = re.search(r"_w([0-9a-f\-]+)", name)
            worker_id = worker_m.group(1) if worker_m else 0
            try:
                body = self._s3.get_object(Bucket=self.bucket, Key=key)["Body"].read()
                with _gzip.open(io.BytesIO(body), "rt", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            ep = json.loads(line)
                            ep.setdefault("round_idx", round_idx)
                            ep.setdefault("worker_id", worker_id)
                            yield ep
            except Exception as exc:
                print(f"[EpisodeStore] ⚠️ 跳过损坏对象 {key}: {exc}")
            self._consumed.add(key)

    def drain(self) -> List[Dict[str, Any]]:
        return list(self.iter_new_episodes())

    # ---- 队列语义（供 TrainWorker 直连） ----

    def get(self, timeout: Optional[float] = None) -> Dict[str, Any]:
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            for ep in self.iter_new_episodes():
                return ep
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError(f"R2EpisodeStore.get 超时（{timeout}s）：无新 episode")
            time.sleep(self.poll_interval)

    def get_nowait(self) -> Dict[str, Any]:
        it = self.iter_new_episodes()
        try:
            return next(it)
        except StopIteration:
            raise TimeoutError("R2EpisodeStore.get_nowait：无新 episode") from None

    def qsize(self) -> int:
        return sum(1 for k in self._list_keys() if k not in self._consumed)
