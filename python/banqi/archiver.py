"""banqi/archiver.py — 后台异步冷存储归档线程（共享实现）

按变体参数化：本地归档目录（variant.archive_dir）、MongoDB 库名（config.DB_NAME）、
日志前缀（[Archiver-{variant.id}]）均来自变体/统一配置。4x2 无归档线程（不创建即可）。

从归档队列批量消费 episode，异步写入 MongoDB（作为冷存储），
缺 pymongo 或连接失败时降级为本地 JSONL 归档。不阻塞自对弈与训练主流程。
归档数据始终为原始数据（不应用训练侧数据增强）。
"""

from __future__ import annotations

import os
import queue
import threading
from typing import List, Optional

from banqi.config import make_config
from banqi.storage import FileSaver, MongoSaver
from banqi.variant import Variant


class ArchiverWorker(threading.Thread):
    """
    后台归档线程：批量消费 episode 写入 Mongo（或本地 JSONL）。
      - 批量 insert_many，减少往返
      - Mongo 连接失败自动降级本地，不抛异常中断线程
    """

    def __init__(
        self,
        archive_q: "queue.Queue",
        stop_event: threading.Event,
        variant: Variant,
        mongo_uri: Optional[str] = None,
        local_archive_dir: Optional[str] = None,
    ) -> None:
        super().__init__(name=f"ArchiverWorker-{variant.id}", daemon=True)
        self.archive_q = archive_q
        self.stop_event = stop_event
        self.variant = variant
        self.cfg = make_config(variant.id)
        self.mongo_uri = mongo_uri if mongo_uri is not None else self.cfg.MONGO_URI
        self.tag = f"[Archiver-{variant.id}]"
        _py_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        raw_dir = local_archive_dir or variant.archive_dir or os.path.join("outputs", variant.id, "archive")
        self.local_archive_dir = raw_dir if os.path.isabs(raw_dir) else os.path.join(_py_dir, raw_dir)

        # 初始化保存器（Mongo 优先，失败降级本地）
        self.saver = self._init_saver()
        self.pending: List[dict] = []
        self.archived_games = 0
        self._lock = threading.Lock()

    def _init_saver(self):
        """优先 Mongo，失败则本地 JSONL。"""
        if self.mongo_uri:
            try:
                return MongoSaver(
                    self.mongo_uri,
                    db_name=self.cfg.DB_NAME,
                    collection=self.cfg.COLLECTION,
                )
            except Exception as exc:  # pragma: no cover
                print(f"{self.tag} ⚠️ MongoDB 连接失败，降级为本地归档: {exc}")
        os.makedirs(self.local_archive_dir, exist_ok=True)
        return FileSaver(self.local_archive_dir, save_format="jsonl")

    def _flush(self) -> None:
        """把缓存的一批 episode 写入保存器。"""
        if not self.pending:
            return
        try:
            # 以第一个 episode 的 iteration 近似作为批次迭代号
            iteration = self.pending[0].get("iteration", 0)
            worker_id = self.pending[0].get("worker_id", 0)
            self.saver.save_episodes(
                self.pending,
                iteration=iteration,
                worker_id=worker_id,
                game_start=self.archived_games,
            )
            with self._lock:
                self.archived_games += len(self.pending)
        except Exception as exc:  # pragma: no cover
            print(f"{self.tag} ⚠️ 归档失败（将丢弃本批）: {exc}")
        finally:
            self.pending = []

    def run(self) -> None:
        print(f"{self.tag} 🗄️  归档线程启动（Mongo={bool(self.mongo_uri)}, "
              f"batch={self.cfg.ARCHIVE_BATCH}）...")
        while not self.stop_event.is_set():
            try:
                item = self.archive_q.get(timeout=self.cfg.ARCHIVE_POLL_INTERVAL)
            except queue.Empty:
                continue
            self.pending.append(item)
            if len(self.pending) >= self.cfg.ARCHIVE_BATCH:
                self._flush()
        # 退出前排空残余
        self._flush()
        if hasattr(self.saver, "close"):
            self.saver.close()
        print(f"{self.tag} 归档线程退出，累计归档 {self.archived_games} 局")

    def stats(self) -> dict:
        with self._lock:
            return {
                "archived_games": self.archived_games,
                "pending": len(self.pending),
            }
