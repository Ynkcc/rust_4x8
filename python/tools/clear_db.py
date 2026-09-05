"""python/tools/clear_db.py — 清空 MongoDB 中的对局归档数据

目标集合一律来自统一配置（banqi.config）：按变体读取 MONGO_URI / DB_NAME / COLLECTION，
不再硬编码库名，也不再使用已废弃的 MONGODB_URI 别名。

默认覆盖全部变体，并额外扫描服务端 `banqi*` 前缀库（清理配置外的遗留库，如 banqi_smoke），
用 --no-orphan 或 --variant 可关闭该扫描。清空方式为 drop 集合（含索引）。

用法：
    python tools/clear_db.py --dry-run           # 仅列出待清空集合与文档数
    python tools/clear_db.py --yes               # 非交互清空
    python tools/clear_db.py --variant 4x4       # 仅清 4x4 变体对应集合
"""

import argparse
import os
import sys

_PYTHON_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PYTHON_DIR not in sys.path:
    sys.path.insert(0, _PYTHON_DIR)

from pymongo import MongoClient

from banqi.config import make_config
from banqi.variant import VARIANTS

DB_PREFIX = "banqi"


def _collect_targets(client, variant_ids, include_orphans):
    """返回 {(db, collection): [所属变体 id]}，空列表表示配置外的遗留库。"""
    targets: dict[tuple[str, str], list[str]] = {}
    for vid in variant_ids:
        cfg = make_config(vid)
        targets.setdefault((cfg.DB_NAME, cfg.COLLECTION), []).append(vid)
    if include_orphans:
        for db_name in client.list_database_names():
            if db_name.startswith(DB_PREFIX):
                for col in client[db_name].list_collection_names():
                    targets.setdefault((db_name, col), [])
    return targets


def main() -> None:
    ap = argparse.ArgumentParser(description="清空 MongoDB 中的对局归档数据（drop 集合）")
    ap.add_argument(
        "--variant", action="append", choices=sorted(VARIANTS),
        help="仅清空指定变体的归档集合，可重复传入；默认全部变体",
    )
    ap.add_argument("--no-orphan", action="store_true", help="不扫描配置之外的 banqi* 遗留库")
    ap.add_argument("--dry-run", action="store_true", help="仅列出待清空集合与文档数，不删除")
    ap.add_argument("--yes", action="store_true", help="跳过交互确认（非交互执行）")
    args = ap.parse_args()

    variant_ids = args.variant or sorted(VARIANTS)
    uri = make_config(variant_ids[0]).MONGO_URI
    print(f"[clear_db] 连接 MongoDB: {uri}")
    client = MongoClient(uri, serverSelectionTimeoutMS=5000)
    try:
        client.admin.command("ping")
    except Exception as exc:
        print(f"[clear_db] ❌ 无法连接 MongoDB（uri={uri}）: {exc!r}")
        raise SystemExit(1)

    targets = _collect_targets(
        client, variant_ids, include_orphans=not (args.no_orphan or args.variant)
    )

    rows = []
    total = 0
    for (db_name, col), owners in sorted(targets.items()):
        n = client[db_name][col].count_documents({})
        total += n
        origin = f"变体 {','.join(owners)}" if owners else "配置外遗留"
        rows.append((db_name, col, n))
        print(f"  {db_name}.{col}: {n} 条  [{origin}]")
    print(f"[clear_db] 合计 {len(rows)} 个集合 / {total} 条文档")

    if args.dry_run:
        print("[clear_db] dry-run 结束，未做任何删除")
        return

    if not rows:
        print("[clear_db] 无匹配集合，退出")
        return

    if not args.yes:
        answer = input(f"[clear_db] 确认 drop 上述 {len(rows)} 个集合？输入 yes 继续: ")
        if answer.strip().lower() != "yes":
            print("[clear_db] 已取消")
            return

    for db_name, col, n in rows:
        client[db_name][col].drop()
        print(f"[clear_db] 已 drop {db_name}.{col}（原 {n} 条）")
    print(f"[clear_db] 完成，共 drop {len(rows)} 个集合 / {total} 条文档")


if __name__ == "__main__":
    main()
