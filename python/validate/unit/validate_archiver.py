"""
validate_archiver.py — 验证归档与存储层（纯 CPU，不需要 MongoDB）。

检查项：
  1. to_json_safe：numpy 标量/数组/嵌套容器转 JSON 安全，可被 json.dumps
  2. FileSaver jsonl：写入 iter_XXXXXX_worker_XXX.jsonl，追加读回后字段等价
  3. FileSaver 降级路径：Mongo 不可用时 ArchiverWorker 走 FileSaver
  4. MongoSaver 文档结构与 Rust GameDocument/SampleDocument 一致（仅校验结构，不实际连库）
  5. ArchiverWorker 线程可启动、消费、flush、退出（无死锁）

运行：python python/validate/validate_archiver.py
"""

from __future__ import annotations

import json
import os
import queue
import tempfile

import numpy as np

import os
import sys

_VALIDATE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PYTHON_DIR = os.path.dirname(_VALIDATE_DIR)
_BANQI_DIR = os.path.join(_PYTHON_DIR, "banqi")
for _d in (_PYTHON_DIR, _BANQI_DIR, _VALIDATE_DIR):
    if _d not in sys.path:
        sys.path.insert(0, _d)

import validate_common  # noqa: F401
from validate_common import VARIANT, Reporter, make_episode, run_part

from banqi.storage import FileSaver, MongoSaver, to_json_safe


def test_to_json_safe() -> None:
    rep = Reporter("to_json_safe")
    obj = {
        "arr": np.array([1.5, 2.5], dtype=np.float32),
        "scalar_f": np.float32(3.5),
        "scalar_i": np.int64(7),
        "nested": {"a": [np.float64(1.0), np.array([1, 2])]},
        "tuple": (np.float32(1), np.int32(2)),
    }
    safe = to_json_safe(obj)
    dumped = json.dumps(safe, ensure_ascii=False)  # 不抛异常即可
    rep.check(isinstance(dumped, str) and len(dumped) > 0, "json.dumps succeeds")
    rep.check(isinstance(safe["arr"], list) and safe["arr"] == [1.5, 2.5],
              "ndarray -> list")
    rep.check(isinstance(safe["scalar_f"], float), "float32 -> float")
    rep.check(isinstance(safe["scalar_i"], int), "int64 -> int")
    rep.check(isinstance(safe["nested"]["a"], list), "nested list preserved")
    rep.summary()


def test_filesaver_jsonl() -> None:
    rep = Reporter("FileSaver jsonl")
    with tempfile.TemporaryDirectory() as d:
        saver = FileSaver(d, save_format="jsonl")
        ep = make_episode(num_steps=3, winner=1)
        saver.save_episodes([ep], iteration=2, worker_id=0, game_start=0)
        saver.save_episodes([ep], iteration=2, worker_id=0, game_start=1)
        saver.close()

        path = os.path.join(d, "iter_000002_worker_000.jsonl")
        rep.check(os.path.exists(path), f"jsonl file created: {os.path.basename(path)}")
        with open(path, encoding="utf-8") as f:
            lines = [json.loads(l) for l in f if l.strip()]
        rep.check(len(lines) == 2, f"two episodes appended ({len(lines)})")
        # 回读字段等价
        first = lines[0]
        rep.check(first["game_length"] == 3, "game_length roundtrip")
        rep.check(first["winner"] == 1, "winner roundtrip")
        rep.check(len(first["boards"]) == 3, "boards roundtrip")
        # 与原始 episode 数值一致
        rep.check(first["boards"][0] == np.asarray(ep["boards"][0]).tolist(),
                  "board values roundtrip")
    rep.summary()


def test_filesaver_json_format() -> None:
    rep = Reporter("FileSaver json")
    with tempfile.TemporaryDirectory() as d:
        saver = FileSaver(d, save_format="json")
        ep = make_episode(num_steps=2, winner=-1)
        saver.save_episodes([ep, ep], iteration=0, worker_id=0, game_start=5)
        saver.close()
        g0 = os.path.join(d, "games", "game_000005.json")
        g1 = os.path.join(d, "games", "game_000006.json")
        rep.check(os.path.exists(g0) and os.path.exists(g1),
                  "per-game json files created")
        with open(g0, encoding="utf-8") as f:
            data = json.load(f)
        rep.check(data["winner"] == -1, "json winner roundtrip")
    rep.summary()


def test_mongo_doc_structure() -> None:
    """不连库，仅用与 MongoSaver 相同的转换逻辑检查文档结构。
    通过 monkeypatch 拦截 collection.insert_many 来捕获文档。"""
    rep = Reporter("MongoSaver doc structure")
    import unittest.mock as mock

    ep = make_episode(num_steps=2, winner=1, rng=np.random.default_rng(0))
    captured: dict = {}

    def fake_init(self, uri, db_name="banqi_training", collection="games"):
        self._datetime = __import__("datetime").datetime
        self.client = mock.MagicMock()
        self.collection = mock.MagicMock()

    with mock.patch.object(MongoSaver, "__init__", fake_init):
        saver = MongoSaver("mongodb://fake:27017")
        saver.collection.insert_many.side_effect = lambda docs: captured.update(docs=docs)
        saver.save_episodes([ep], iteration=3)

    docs = captured["docs"]
    rep.check(len(docs) == 1, f"one doc saved (got {len(docs)})")
    doc = docs[0]
    rep.check(doc["iteration"] == 3, "doc iteration")
    rep.check(doc["game_length"] == 2, "doc game_length")
    rep.check(doc["winner"] == 1, "doc winner")
    rep.check("timestamp" in doc, "doc timestamp")
    samples = doc["samples"]
    rep.check(len(samples) == 2, "two samples in doc")
    s0 = samples[0]
    for key in ["board_state", "scalar_state", "policy_probs", "mcts_value",
                "completed_q", "root_visit_count", "game_result_value",
                "action_mask", "health_diff"]:
        rep.check(key in s0, f"sample key present: {key}")
    rep.check("step_in_game" not in s0, "sample step_in_game removed")
    rep.summary()


def test_archiver_worker_no_mongo() -> None:
    """ArchiverWorker 在无 Mongo（连接失败）时降级 FileSaver，且线程可正常退出。"""
    rep = Reporter("ArchiverWorker fallback + exit")
    import banqi.archiver as archiver_mod
    import threading

    archive_q = queue.Queue()
    stop_flag = [False]

    worker = archiver_mod.ArchiverWorker(archive_q, stop_flag, VARIANT, mongo_uri="")
    rep.check(isinstance(worker.saver, FileSaver), "fallback to FileSaver")

    ep = make_episode(num_steps=3, winner=1)
    archive_q.put(ep)
    # 手动 flush 触发保存
    worker.pending.append(ep)
    worker._flush()
    rep.check(worker.archived_games == 1, "archived_games incremented")

    # 线程运行并退出（用 stop_flag + 队列清空）
    t = threading.Thread(target=worker.run, daemon=True)
    t.start()
    # 等待 worker 处理完当前队列后，置 stop
    stop_flag[0] = True
    # 给 worker 一个 flush 机会
    archive_q.put(ep)
    t.join(timeout=5)
    rep.check(not t.is_alive(), "archiver thread exited cleanly")
    # 本地文件已生成
    archive_dir = worker.local_archive_dir
    rep.check(os.path.isdir(archive_dir), f"local archive dir exists: {archive_dir}")
    rep.summary()


def main() -> None:
    run_part("archiver: to_json_safe", test_to_json_safe)
    run_part("archiver: FileSaver jsonl", test_filesaver_jsonl)
    run_part("archiver: FileSaver json", test_filesaver_json_format)
    run_part("archiver: MongoSaver doc structure", test_mongo_doc_structure)
    run_part("archiver: worker fallback + exit", test_archiver_worker_no_mongo)


if __name__ == "__main__":
    main()
