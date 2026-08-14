"""
archiver.py — 后台异步冷存储归档线程（无 CLI 参数）

从归档队列批量消费 episode，异步写入 MongoDB（作为冷存储），
缺 pymongo 或连接失败时降级为本地 JSONL 归档。不阻塞自对弈与训练主流程。
"""

from __future__ import annotations

import os
import queue
import threading
import time
from typing import List

from config import config
from storage import FileSaver, MongoSaver

# 本地归档目录（当 Mongo 不可用或未配置时的降级路径）
LOCAL_ARCHIVE_DIR = "./training_data/archive"


class ArchiverWorker(threading.Thread):
    """
    后台归档线程：批量消费 episode 写入 Mongo（或本地 JSONL）。
      - 批量 insert_many，减少往返
      - Mongo 连接失败自动降级本地，不抛异常中断线程
    """

    def __init__(
        self,
        archive_q: "queue.Queue",
        stop_flag: "List[bool]",
        mongo_uri: str = config.MONGO_URI,
    ) -> None:
        super().__init__(name="ArchiverWorker", daemon=True)
        self.archive_q = archive_q
        self.stop_flag = stop_flag
        self.mongo_uri = mongo_uri

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
                    db_name=config.DB_NAME,
                    collection=config.COLLECTION,
                )
            except Exception as exc:  # pragma: no cover
                print(f"[Archiver] ⚠️ MongoDB 连接失败，降级为本地归档: {exc}")
        os.makedirs(LOCAL_ARCHIVE_DIR, exist_ok=True)
        return FileSaver(LOCAL_ARCHIVE_DIR, save_format="jsonl")

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
            print(f"[Archiver] ⚠️ 归档失败（将丢弃本批）: {exc}")
        finally:
            self.pending = []

    def run(self) -> None:
        print(f"[Archiver] 🗄️  归档线程启动（Mongo={bool(self.mongo_uri)}, "
              f"batch={config.ARCHIVE_BATCH}）...")
        while not self.stop_flag[0]:
            try:
                item = self.archive_q.get(timeout=config.ARCHIVE_POLL_INTERVAL)
            except queue.Empty:
                continue
            self.pending.append(item)
            if len(self.pending) >= config.ARCHIVE_BATCH:
                self._flush()
        # 退出前排空残余
        self._flush()
        if hasattr(self.saver, "close"):
            self.saver.close()
        print(f"[Archiver] 归档线程退出，累计归档 {self.archived_games} 局")

    def stats(self) -> dict:
        with self._lock:
            return {
                "archived_games": self.archived_games,
                "pending": len(self.pending),
            }
