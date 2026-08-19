#!/usr/bin/env python3
"""
play_and_record.py — 跑一局 AI 自对弈，记录原始数据，并输出人类可读的文字棋谱。

流程:
  1. 加载模型（默认 python/banqi_model_latest.pt；缺失时用随机初始化网络并警告）
  2. 调用 Rust 统一入口 banqi_4x8.run_python_match 生成一局 episode
     （concurrency=1 等价旧串行 run_self_play_with_predictor）
  3. 用 storage.FileSaver 记录原始 episode（JSONL 或 JSON）
  4. 调用 Rust 绑定 banqi_4x8.describe_record 解析该局记录，
     生成中文文字棋谱（含每手棋盘、双方血量、已阵亡棋子、合法/实际行动），输出 .txt 文件

说明:
  - 解析与棋谱渲染全部由 Rust 侧 describe_record 完成（逐手还原棋盘、
    重建 DarkChessEnv、校验 action_masks 与 actions），Python 侧不再自行解码/推断。
  - describe_record 内部使用 assert 校验，字段缺失或不一致会抛 PanicException；
    因 ep_dict 直接来自 Rust to_dict()，字段必然齐全一致，正常不会触发。

用法:
    python play_and_record.py
    python play_and_record.py --model ../xxx.pt --mcts-sims 128
    python play_and_record.py --out-dir output --save-format json --verbose
"""

from __future__ import annotations
import os
import sys

_TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
_PYTHON_DIR = os.path.dirname(_TOOLS_DIR)
_BANQI_DIR = os.path.join(_PYTHON_DIR, "banqi")
for _d in (_PYTHON_DIR, _BANQI_DIR):
    if _d not in sys.path:
        sys.path.insert(0, _d)


import argparse
import os
import sys

try:
    import banqi_4x8
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "无法导入 banqi_4x8。请先执行: maturin develop --features pyo3"
    ) from exc

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from config import config  # noqa: E402
from self_play import build_predictor  # noqa: E402
from storage import FileSaver  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="跑一局 AI 自对弈并输出文字棋谱")
    parser.add_argument("--model", default=None, help="模型权重路径（TorchScript .pt），默认 python/banqi_model_latest.pt")
    parser.add_argument("--mcts-sims", type=int, default=64, help="MCTS 模拟次数（默认 64）")
    parser.add_argument("--max-considered", type=int, default=16, help="候选动作数（默认 16）")
    parser.add_argument("--temperature-steps", type=int, default=12, help="温度采样步数（默认 12）")
    parser.add_argument("--out-dir", default=None, help="输出目录（默认 python/output）")
    parser.add_argument("--save-format", choices=["jsonl", "json"], default="jsonl",
                        help="原始数据保存格式（默认 jsonl）")
    parser.add_argument("--verbose", action="store_true", help="打印棋谱预览（describe_record 已含逐手细节）")
    args = parser.parse_args()

    model_path = args.model
    if model_path is None:
        default_model = os.path.join(_HERE, "banqi_model_latest.pt")
        if os.path.exists(default_model):
            model_path = default_model
        else:
            print(f"[WARN] 未找到默认模型 {default_model}，使用随机初始化网络")
            model_path = None

    out_dir = args.out_dir or os.path.join(_HERE, "output")
    os.makedirs(out_dir, exist_ok=True)

    # 1. 构建 Predictor（加载权重；推理用 CPU，不占 GPU）
    predictor, device = build_predictor(model_path, device_str=config.INFER_DEVICE)
    print(f"[Record] device = {device}, model = {model_path}")

    # 2. 跑一局
    sp_cfg = banqi_4x8.SelfPlayConfig(
        mcts_sims=args.mcts_sims,
        max_considered_actions=args.max_considered,
        temperature_steps=args.temperature_steps,
    )
    print(f"[Record] 开始自对弈一局 (mcts_sims={args.mcts_sims}) ...")
    episodes = banqi_4x8.run_python_match(
        predict_fn=predictor,
        config=sp_cfg,
        num_games=1,
        concurrency=1,
        worker_id=0,
        variant_id="4x8",
    )
    if not episodes:
        print("[Record] 未生成有效对局（空局）")
        return 1
    ep_dict = dict(episodes[0].to_dict())
    ep_dict["model_path"] = model_path or "(随机初始化)"
    ep_dict["mcts_sims"] = args.mcts_sims
    winner = ep_dict.get("winner")
    print(f"[Record] 对局完成: 步数={ep_dict['game_length']}, 样本数={ep_dict['num_samples']}, "
          f"结果={({1: '红胜', -1: '黑胜'}.get(winner, '平局'))}")

    # 3. 记录原始数据
    saver = FileSaver(out_dir, save_format=args.save_format)
    saver.save_episodes([ep_dict], iteration=0, worker_id=0, game_start=0)
    saver.close()
    if args.save_format == "jsonl":
        record_path = os.path.join(out_dir, "iter_000000_worker_000.jsonl")
    else:
        record_path = os.path.join(out_dir, "games", "game_000000.json")
    print(f"[Record] 原始数据已保存: {record_path}")

    # 4. 用 Rust 端 describe_record 解析并渲染文字棋谱
    text = banqi_4x8.describe_record(ep_dict)
    txt_path = os.path.join(out_dir, "game_000000.txt")
    with open(txt_path, "w", encoding="utf-8") as fp:
        fp.write(text)
    print(f"[Record] 文字棋谱已保存: {txt_path}")
    print()
    # 打印前 12 行预览
    print("\n".join(text.splitlines()[:12]))
    print("... (完整棋谱见 txt 文件)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
