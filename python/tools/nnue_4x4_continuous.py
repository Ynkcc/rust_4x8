"""nnue_4x4_continuous.py — 4x4 NNUE 长期持续训练驱动。

每个 cycle：
  1. 用当前 best 模型做 Expectimax 自对弈，样本追加到滚动归档 JSONL；
  2. 全量归档重训（合并训练比纯增量更稳）；
  3. 新模型 vs 当前 best 打 N_EVAL 局（深度展开，可控预算）；
  4. 胜率 >= 晋升阈值则更新 best，否则丢弃本轮产物。

用法:
    python tools/nnue_4x4_continuous.py                # 前台运行
    nohup python tools/nnue_4x4_continuous.py &        # 后台长期运行
可选参数见 --help。状态与产物都在 outputs/4x4/nnue_loop/（best.nnue / archive.jsonl / history.log）。
"""

from __future__ import annotations

import argparse
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_PYTHON_DIR = os.path.dirname(_HERE)
for _d in (_PYTHON_DIR, _PYTHON_DIR + "/validate/e2e"):
    if _d not in sys.path:
        sys.path.insert(0, _d)

import banqi_4x8  # noqa: E402
from banqi.nnue.exporter import export_random  # noqa: E402
from banqi.nnue.train import NnueSampleDataset, train_nnue  # noqa: E402
from validate_nnue_loop_4x4 import arena_budgeted  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="4x4 NNUE 持续训练循环")
    p.add_argument("--cycles", type=int, default=0, help="循环次数，0=无限")
    p.add_argument("--games", type=int, default=128, help="每 cycle 自对弈局数")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--node-budget", type=int, default=20_000)
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--archive-max-games", type=int, default=3000,
                   help="归档保留最近 N 局（滚动窗口，防陈旧数据拖累）")
    p.add_argument("--eval-games", type=int, default=30, help="评审局数（new vs best）")
    p.add_argument("--eval-depth", type=int, default=4)
    p.add_argument("--eval-budget", type=int, default=8_000)
    p.add_argument("--promote-threshold", type=float, default=0.5,
                   help="胜率>=该值才晋升 best")
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--out-dir", type=str, default=None)
    args = p.parse_args()

    out = args.out_dir or os.path.join(_PYTHON_DIR, "outputs", "4x4", "nnue_loop")
    os.makedirs(out, exist_ok=True)
    best = os.path.join(out, "best.nnue")
    archive = os.path.join(out, "archive.jsonl")
    history = os.path.join(out, "history.log")

    def log(msg: str) -> None:
        line = f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(history, "a") as f:
            f.write(line + "\n")

    # 初始化：无 best 则用随机基座（从零冷启动）
    if not os.path.exists(best):
        export_random(best, feature_dim=278, output_scale=0.1)
        log("冷启动：导出随机基座 best.nnue")
    else:
        log(f"恢复训练：沿用现有 {best}")

    cycle = 0
    while args.cycles == 0 or cycle < args.cycles:
        cycle += 1
        t0 = time.time()
        log(f"=== cycle {cycle} 开始 (games={args.games}) ===")

        # 1. 自对弈（best 模型）
        chunk = os.path.join(out, f"_chunk_c{cycle}.jsonl")
        stats = banqi_4x8.run_expectimax_self_play(
            best, n_games=args.games, num_workers=args.workers,
            node_budget=args.node_budget, max_depth=args.depth,
            threads_per_search=1, seed=args.seed + cycle,
            out_jsonl=chunk, variant_id="4x4",
        )
        # 2. 追加归档并裁剪滚动窗口（按局裁剪，每行一局）
        with open(archive, "a") as f:
            f.write(open(chunk).read())
        os.remove(chunk)
        with open(archive) as f:
            lines = f.readlines()
        if len(lines) > args.archive_max_games:
            with open(archive, "w") as f:
                f.writelines(lines[-args.archive_max_games:])
        log(f"自对弈 {stats['games']} 局 (A{stats['a_wins']}/和{stats['draws']}/B{stats['b_wins']})，"
            f"归档共 {min(len(lines), args.archive_max_games)} 局")

        # 3. 全量重训
        ds = NnueSampleDataset([archive], value_source="completed_q")
        cand = os.path.join(out, "candidate.nnue")
        train_nnue(ds, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                   output_nnue=cand, checkpoint=os.path.join(out, "candidate.pth"))

        # 4. 评审：candidate vs best
        w, d, l = arena_budgeted(cand, best, args.eval_games, args.eval_depth, args.eval_budget)
        promote = w >= args.promote_threshold
        log(f"评审 candidate vs best: 胜 {w:.2f} / 和 {d:.2f} / 负 {l:.2f} → "
            f"{'晋升 ✓' if promote else '保留 best ✗'}（耗时 {time.time()-t0:.0f}s）")
        if promote:
            os.replace(cand, best)
            # 同步保留带版本号的快照
            os.replace(os.path.join(out, "candidate.pth"),
                       os.path.join(out, f"best_c{cycle}.pth"))
        else:
            for f_ in ("candidate.nnue", "candidate.pth"):
                fp = os.path.join(out, f_)
                if os.path.exists(fp):
                    os.remove(fp)


if __name__ == "__main__":
    main()
