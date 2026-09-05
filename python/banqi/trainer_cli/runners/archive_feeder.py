"""banqi/trainer_cli/runners/archive_feeder.py — 冷存储归档数据供给线程。"""

from __future__ import annotations

import os
import queue
import threading
import time
from typing import Dict, List, Optional

from banqi.config import make_config
from banqi.variant import Variant


class ArchiveFeederWorker(threading.Thread):
    """归档数据供给线程（`TRAIN_MODE="archive"` 的数据源）。

    从冷存储（本地 JSONL 优先，Mongo 兜底）加载历史 episode，
    周期性压入 data_q 供 `TrainWorker` 消费训练。不启动自对弈。
    """

    def __init__(
        self,
        variant: Variant,
        data_q: "queue.Queue",
        stop_event: threading.Event,
    ) -> None:
        super().__init__(name=f"ArchiveFeederWorker-{variant.id}", daemon=True)
        self.variant = variant
        self.cfg = make_config(variant.id)
        self.tag = f"[ArchiveFeeder-{variant.id}]"
        self.data_q = data_q
        self.stop_event = stop_event
        self.total_games = 0

    def _resolve_archive_dir(self) -> Optional[str]:
        """解析归档目录：优先 ARCHIVE_TRAIN_DIR，其次 variant.archive_dir。"""
        from banqi.storage import list_jsonl_files
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cands = [
            self.cfg.ARCHIVE_TRAIN_DIR or "",
            self.variant.archive_dir or "",
            os.path.join(here, "training_data", f"archive_{self.variant.id}"),
            os.path.join(here, "training_data", f"archive_{self.variant.id}_imitate"),
        ]
        for d in cands:
            if d and os.path.isdir(d) and list_jsonl_files(d):
                return d
        return None

    def _load_from_mongo(self, limit_games: Optional[int]) -> List[Dict]:
        """从 MongoDB 读取该变体归档局，转成与 `GameEpisode.to_dict()` 兼容的 episode dict。"""
        try:
            import pymongo
            client = pymongo.MongoClient(self.cfg.MONGO_URI, serverSelectionTimeoutMS=5000)
            client.admin.command("ping")
            col = client[self.cfg.DB_NAME][self.cfg.COLLECTION]
        except Exception as exc:  # pragma: no cover
            print(f"{self.tag} ⚠️ MongoDB 不可用（归档兜底跳过）: {exc}")
            return []

        episodes: List[Dict] = []
        try:
            query: Dict = {}
            cursor = col.find(query).limit(limit_games) if limit_games else col.find(query)
            for doc in cursor:
                samples = doc.get("samples") or []
                if not samples:
                    continue
                ep = {
                    "boards": [s["board_state"] for s in samples],
                    "scalars": [s["scalar_state"] for s in samples],
                    "policies": [s["policy_probs"] for s in samples],
                    "mcts_values": [s.get("mcts_value", 0.0) for s in samples],
                    "completed_qs": [s.get("completed_q", 0.0) for s in samples],
                    "root_visits": [s.get("root_visit_count", 0) for s in samples],
                    "game_results": [s.get("game_result_value", 0.0) for s in samples],
                    "action_masks": [s["action_mask"] for s in samples],
                    "health_diffs": [s.get("health_diff", 0.0) for s in samples],
                    "game_length": int(doc.get("game_length", len(samples))),
                    "winner": doc.get("winner"),
                    "num_samples": len(samples),
                }
                episodes.append(ep)
        finally:
            client.close()
        return episodes

    def _put(self, item: Dict) -> None:
        while not self.stop_event.is_set():
            try:
                self.data_q.put(item, timeout=0.5)
                return
            except queue.Full:  # 队列满则重试；非 Full 异常直接抛出暴露
                continue

    def run(self) -> None:
        from banqi.storage import load_jsonl_episodes
        archive_dir = self._resolve_archive_dir()

        if archive_dir is None:
            print(f"{self.tag} ⚠️ 未找到本地归档目录（{self.cfg.ARCHIVE_TRAIN_DIR or self.variant.archive_dir}），"
                  f"将尝试从 MongoDB 读取（{self.cfg.DB_NAME}.{self.cfg.COLLECTION}）...")
        else:
            print(f"{self.tag} 🗃️ 使用本地归档目录: {archive_dir}")

        limit_games = self.cfg.ARCHIVE_TRAIN_GAMES or None
        total_rounds = max(1, self.cfg.ARCHIVE_TRAIN_ROUNDS)
        poll = max(1.0, self.cfg.ARCHIVE_POLL_INTERVAL)

        for r in range(total_rounds):
            if self.stop_event.is_set():
                break
            try:
                t0 = time.time()
                if archive_dir is not None:
                    episodes = load_jsonl_episodes(archive_dir, limit_games=limit_games)
                else:
                    episodes = self._load_from_mongo(limit_games)
                if not episodes:
                    print(f"{self.tag} ⚠️ 归档为空，等待新数据...")
                    time.sleep(poll)
                    continue
                # 每次灌入全部（或限制量），并标记 round 号便于观测
                for ep in episodes:
                    if self.stop_event.is_set():
                        break
                    ep = dict(ep)
                    ep.setdefault("num_samples", len(ep.get("boards") or []))
                    ep.setdefault("iteration", r)
                    self._put(ep)
                    self.total_games += 1
                print(f"{self.tag} 📦 第 {r + 1}/{total_rounds} 轮灌入 {len(episodes)} 局"
                      f"（累计 {self.total_games}，耗时 {time.time() - t0:.1f}s）")
            except Exception as exc:  # pragma: no cover
                print(f"{self.tag} ⚠️ 归档加载失败: {exc}")
            time.sleep(poll)

        print(f"{self.tag} 归档供给完成，累计 {self.total_games} 局")

    def stats(self) -> Dict[str, int]:
        return {"total_games": self.total_games}
