# storage.py — 可复用存储层（4x4 训练用）
#
# 集中存放 JSON 安全序列化、本地文件归档（FileSaver）与 MongoDB 归档（MongoSaver），
# 供 self_play / archiver 复用。
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

DEFAULT_DB_NAME = "banqi_4x4"
DEFAULT_COLLECTION = "games"


# ============================================================================
# JSON 安全序列化
# ============================================================================

def to_json_safe(obj):
    """把 numpy 标量 / ndarray / 嵌套容器转换为可 JSON 序列化的纯 Python 对象。"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, dict):
        return {str(k): to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_json_safe(v) for v in obj]
    return obj


# ============================================================================
# 本地文件保存器
# ============================================================================

class FileSaver:
    """
    本地数据保存器：
      - save_format="jsonl": 每局一行 JSON，追加写
          output/iter_{iteration}_worker_{worker_id}.jsonl
      - save_format="json":  每局一个 JSON 文件，归档到
          output/games/game_{game_count}.json
    """

    def __init__(self, output_dir: str, save_format: str = "jsonl") -> None:
        self.output_dir = output_dir
        self.save_format = save_format
        os.makedirs(output_dir, exist_ok=True)
        if save_format == "json":
            os.makedirs(os.path.join(output_dir, "games"), exist_ok=True)
        self._open_fp: Optional[object] = None
        self._open_key: Optional[Tuple[int, int]] = None

    def _get_fp(self, iteration: int, worker_id: int):
        key = (iteration, worker_id)
        if self.save_format == "jsonl":
            if self._open_key != key:
                self.close()
                path = os.path.join(
                    self.output_dir, f"iter_{iteration:06d}_worker_{worker_id:03d}.jsonl"
                )
                self._open_fp = open(path, "a", encoding="utf-8")
                self._open_key = key
            return self._open_fp
        return None

    def save_episodes(
        self,
        episode_dicts: List[Dict],
        iteration: int,
        worker_id: int,
        game_start: int,
    ) -> None:
        if self.save_format == "jsonl":
            fp = self._get_fp(iteration, worker_id)
            for ep in episode_dicts:
                fp.write(json.dumps(to_json_safe(ep), ensure_ascii=False))
                fp.write("\n")
            fp.flush()
        else:  # json 归档
            games_dir = os.path.join(self.output_dir, "games")
            for idx, ep in enumerate(episode_dicts):
                path = os.path.join(games_dir, f"game_{game_start + idx:06d}.json")
                with open(path, "w", encoding="utf-8") as fp:
                    json.dump(to_json_safe(ep), fp, ensure_ascii=False)

    def close(self) -> None:
        if self._open_fp is not None:
            self._open_fp.close()
            self._open_fp = None
            self._open_key = None


# ============================================================================
# MongoDB 保存器
# ============================================================================

class MongoSaver:
    """
    可选 MongoDB 保存器：把 episode_dict 转换为与 mongodb_storage.rs 的
    GameDocument / SampleDocument 一致的结构后 insert_many。
    连接失败时抛异常，由上层降级为本地保存。
    """

    def __init__(self, uri: str, db_name: str = DEFAULT_DB_NAME,
                 collection: str = DEFAULT_COLLECTION) -> None:
        import pymongo  # 延迟导入，缺失时上层捕获降级
        from datetime import datetime

        self._datetime = datetime
        self.client = pymongo.MongoClient(uri, serverSelectionTimeoutMS=5000)
        self.collection = self.client[db_name][collection]
        self.client.admin.command("ping")
        print(f"[MongoSaver] 连接成功: {uri} -> {db_name}.{collection}")

    def save_episodes(self, episode_dicts: List[Dict], iteration: int, **_) -> None:
        documents: List[Dict] = []
        for ep in episode_dicts:
            samples = []
            for step_idx, (board, scalar, policy, mcts_val, completed_q,
                            root_visit, game_result, mask) in enumerate(zip(
                ep["boards"], ep["scalars"], ep["policies"], ep["mcts_values"],
                ep["completed_qs"], ep["root_visits"], ep["game_results"],
                ep["action_masks"],
            )):
                samples.append({
                    "board_state": list(board),
                    "scalar_state": list(scalar),
                    "policy_probs": list(policy),
                    "mcts_value": float(mcts_val),
                    "completed_q": float(completed_q),
                    "root_visit_count": int(root_visit),
                    "game_result_value": float(game_result),
                    "action_mask": list(mask),
                    "step_in_game": step_idx,
                })
            documents.append({
                "iteration": int(iteration),
                "game_length": int(ep["game_length"]),
                "winner": ep["winner"],
                "samples": samples,
                "timestamp": self._datetime.utcnow(),
            })
        if documents:
            self.collection.insert_many(documents)
            print(f"  [MongoSaver] 已保存 {len(documents)} 局 (iteration={iteration})")

    def close(self) -> None:
        try:
            self.client.close()
        except Exception:  # pragma: no cover
            pass
