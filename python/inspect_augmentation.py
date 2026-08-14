"""
inspect_augmentation.py — 人工核查「数据对称增强」逻辑正确性

流程：
  1. 用均匀 logits 的退化 predictor 跑一局随机自对弈（MCTS 完全随机探索）；
  2. 对原始局以及 hflip / vflip / rot180 变换后的局分别调用 Rust 绑定
     describe_record —— 该函数会用 boards/scalars 逐手重建环境、重新生成
     action_mask 并与记录逐元素断言一致（增强若错误会直接 panic）；
  3. 抽取「倒数第 10 步」的局面描述写入文本文件，供人工对比镜像关系。

人工观察要点（4x8 棋盘，行 0..3，列 a..h）：
  - hflip  棋盘 = identity 的左右镜像（列 c -> 7-c）
  - vflip  棋盘 = identity 的上下镜像（行 r -> 3-r）
  - rot180 棋盘 = identity 的中心对称（(r,c) -> (3-r, 7-c)）
  - 双方血量 / 已阵亡棋子：4 种变换下应完全一致
  - 合法行动 / 实际行动：坐标应随镜像同步变换

运行：
    python python/inspect_augmentation.py [输出文件]
默认输出: python/augmentation_inspect.txt
"""

from __future__ import annotations

import re
import sys
from typing import Dict, List

import numpy as np

try:
    import banqi_4x8
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "无法导入 banqi_4x8，请先执行: maturin develop --features pyo3"
    ) from exc

from data_augmentation import SYMMETRY_TRANSFORMS, transform_episode


def make_random_game(num_games: int = 1) -> dict:
    """跑一局随机自对弈：uniform logits -> MCTS 完全随机探索，等价随机游戏。"""
    def predict_fn(board, scalars):
        n = board.shape[0]
        return (np.zeros((n, 352), dtype=np.float32), np.zeros(n, dtype=np.float32))

    cfg = banqi_4x8.SelfPlayConfig(
        mcts_sims=16, max_considered_actions=24, temperature_steps=16,
    )
    episodes = banqi_4x8.run_self_play_with_predictor(
        predict_fn=predict_fn, config=cfg, num_games=num_games, worker_id=0,
    )
    return episodes[0].to_dict()


def to_record(ep: dict) -> dict:
    """把 episode dict 转成 describe_record 所需的 Python list 格式。"""
    return {
        "boards": [b.tolist() if hasattr(b, "tolist") else b for b in ep["boards"]],
        "scalars": [s.tolist() if hasattr(s, "tolist") else s for s in ep["scalars"]],
        "action_masks": [
            m.tolist() if hasattr(m, "tolist") else m for m in ep["action_masks"]
        ],
        "actions": ep["actions"],
    }


def split_steps(text: str) -> List[tuple]:
    """把 describe_record 输出按「第N手」切分为 [(手数, 块文本), ...]。"""
    blocks = re.split(r"(?m)^第(\d+)手\n", text)
    steps = []
    for i in range(1, len(blocks) - 1, 2):
        steps.append((int(blocks[i]), blocks[i + 1]))
    return steps


def extract_board_block(step_text: str) -> List[str]:
    """从一步的文本中截取棋盘块（列标签行 + 4 数据行）。"""
    marker = "棋盘（数字=行，字母=列）:"
    if marker not in step_text:
        return []
    after = step_text.split(marker, 1)[1].lstrip()
    board_lines = []
    for ln in after.splitlines():
        if not ln.strip():
            break
        board_lines.append(ln)
    return board_lines


def render_side_by_side(boards: Dict[str, List[str]]) -> List[str]:
    """把 4 种变换的棋盘并排显示，便于一眼观察镜像关系。"""
    if not boards:
        return []
    labels = list(boards)
    rows = max(len(v) for v in boards.values())
    out = []
    width = 26  # 每块棋盘（含行号）约 26 显示列，用等宽分隔
    header = "    " + " | ".join(label.center(width) for label in labels)
    out.append(header)
    for i in range(rows):
        cells = []
        for t in labels:
            v = boards[t]
            cells.append(v[i].ljust(width) if i < len(v) else " " * width)
        out.append("    " + " | ".join(cells))
    return out


def main() -> None:
    out_path = sys.argv[1] if len(sys.argv) > 1 else "augmentation_inspect.txt"

    ep = make_random_game()
    n = ep["game_length"]
    target = max(1, n - 9)  # 倒数第 10 步（1-based 手数）
    print(f"随机对弈一局完成: 步数={n}, winner={ep['winner']}，检查第 {target} 手")

    lines: List[str] = []
    lines.append("=" * 78)
    lines.append("  4x8 暗棋 · 数据对称增强人工核查")
    lines.append("=" * 78)
    lines.append(f"随机自对弈一局: 总步数 {n}, winner={ep['winner']}")
    lines.append(f"检查目标: 倒数第 10 步 = 第 {target} 手")
    lines.append("")
    lines.append("增强正确的判断标准（棋盘 4x8，行 0..3，列 a..h）:")
    lines.append("  - hflip   = identity 左右镜像 (c -> 7-c)")
    lines.append("  - vflip   = identity 上下镜像 (r -> 3-r)")
    lines.append("  - rot180  = identity 中心对称 ((r,c) -> (3-r, 7-c))")
    lines.append("  - 双方血量 / 已阵亡棋子: 4 种变换下应完全一致")
    lines.append("  - 合法行动 / 实际行动坐标: 应随镜像同步变换")
    lines.append("")

    # 先收集各变换下目标步的棋盘块（用于并排对照）
    boards_by_t: Dict[str, List[str]] = {}
    blocks_by_t: Dict[str, str] = {}

    for t in SYMMETRY_TRANSFORMS:
        ep_t = transform_episode(ep, t)
        # describe_record 内部逐手断言重建 mask 与记录一致，增强错误会 panic
        text = banqi_4x8.describe_record(to_record(ep_t))
        steps = split_steps(text)
        found = next((b for num, b in steps if num == target), None)
        if found is None:
            print(f"⚠️ {t}: 未找到第 {target} 手（步数不足?）", file=sys.stderr)
            continue
        blocks_by_t[t] = found
        boards_by_t[t] = extract_board_block(found)

    # 并排棋盘对照
    lines.append("-" * 78)
    lines.append("【一】并排棋盘对照（各行 = 同一棋盘在 4 种变换下的呈现）")
    lines.append("-" * 78)
    lines.extend(render_side_by_side(boards_by_t))
    lines.append("")

    # 每个变换的完整局面描述
    lines.append("-" * 78)
    lines.append("【二】各变换下第 %d 手的完整局面描述（describe_record 原文）" % target)
    lines.append("-" * 78)
    for t in SYMMETRY_TRANSFORMS:
        if t not in blocks_by_t:
            continue
        lines.append("=" * 78)
        lines.append(f"[{t}] 第 {target} 手")
        lines.append("=" * 78)
        lines.append(blocks_by_t[t].rstrip())
        lines.append("")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"✅ 已写入: {out_path}")
    print(f"   describe_record 对原始局及 3 个变换局均无断言错误 → 增强逻辑自洽")
    print(f"   人工观察: cat {out_path}")


if __name__ == "__main__":
    main()
