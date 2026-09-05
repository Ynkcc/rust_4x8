"""banqi/trainer_cli/runners/expectimax_sidecar.py — Expectimax 强自对弈旁路。

checkpoint 事件驱动的周期性任务（默认关闭）：
- 监听 TrainWorker 的 ckpt_event，每 N 次 checkpoint 触发一次
  Rust 原生 Expectimax+NNUE 强自对弈（rust_bridge.run_expectimax_self_play）；
- 产出高质量 NNUE episode JSONL（搜索根值远准于 MCTS 快照，供 NNUE 精调）
  与对局统计（可作强对手评估参考）；
- 使用 NnueDistillWorker 最新导出的 .nnue（outputs/nnue/<variant>_latest.nnue），
  文件不存在时等待，形成「蒸馏 → 强自对弈 → 精调数据」的松耦合闭环。

成本说明：expectimax 为强搜索引擎，吞吐远低于 MCTS 闭环，故设计为
低频 sidecar（默认 EVERY_N_CHECKPOINTS=20），不参与常驻数据生产。
"""

from __future__ import annotations

import glob
import os
import threading
import time
from typing import Optional

from banqi.config import Config
from banqi.variant import Variant


class ExpectimaxSidecar(threading.Thread):
    """Expectimax+NNUE 强自对弈旁路 worker（checkpoint 事件触发）。"""

    def __init__(
        self,
        variant: Variant,
        cfg: Config,
        stop_event: threading.Event,
        ckpt_event: threading.Event,
        tag: str = "[EXPMAX]",
    ) -> None:
        super().__init__(name=f"ExpectimaxSidecar-{variant.id}", daemon=True)
        self.variant = variant
        self.cfg = cfg
        self.stop_event = stop_event
        self.ckpt_event = ckpt_event
        self.tag = tag

        # 与 NnueDistillWorker 共享输出目录（latest .nnue 所在处）
        self.nnue_dir = (getattr(cfg, "NNUE_DISTILL_OUTPUT_DIR", "")
                         or os.path.join("models", "nnue"))
        data_dir = (getattr(cfg, "NNUE_DISTILL_DATA_DIR", "")
                    or os.path.join("data", "nnue"))
        os.makedirs(data_dir, exist_ok=True)
        self._data_dir = data_dir

        self.ckpt_seen = 0
        self.runs = 0
        self.disabled = False
        self.last_stats: Optional[dict] = None

    # ------------------------------------------------------------------ #
    def run(self) -> None:
        every = max(int(getattr(self.cfg, "EXPECTIMAX_SIDECAR_EVERY_N_CHECKPOINTS", 20)), 1)
        print(f"{self.tag} ⚡ Expectimax 旁路已启动（每 {every} 次 checkpoint 触发，"
              f"games={getattr(self.cfg, 'EXPECTIMAX_SIDECAR_GAMES', 200)}）")
        while not self.stop_event.is_set():
            if self.disabled:
                break
            self.ckpt_event.wait(timeout=2.0)
            if self.stop_event.is_set():
                break
            self.ckpt_event.clear()
            self.ckpt_seen += 1
            if self.ckpt_seen % every != 0:
                continue
            self._run_expectimax()
        self._log_stats("退出")

    def _latest_nnue(self) -> Optional[str]:
        pattern = os.path.join(self.nnue_dir, f"{self.variant.id}_latest.nnue")
        if os.path.isfile(pattern):
            return pattern
        # 回退：任一版本化 .nnue 中取最新
        candidates = sorted(
            glob.glob(os.path.join(self.nnue_dir, f"{self.variant.id}_v*.nnue"))
        )
        return candidates[-1] if candidates else None

    def _run_expectimax(self) -> None:
        nnue_path = self._latest_nnue()
        if nnue_path is None:
            print(f"{self.tag} ⏳ 尚无 .nnue（等待 NnueDistillWorker 首次蒸馏），跳过")
            return
        try:
            from banqi.rust_bridge import run_expectimax_self_play

            if run_expectimax_self_play is None:
                raise RuntimeError("扩展模块未导出 run_expectimax_self_play")
        except Exception as exc:  # noqa: BLE001 — Rust 扩展未编译 expectimax 入口时降级
            print(f"{self.tag} ⚠️ run_expectimax_self_play 不可用，旁路自禁用（不影响主闭环）: {exc}")
            self.disabled = True
            return

        out = os.path.join(
            self._data_dir,
            f"expectimax_{self.variant.id}_{time.strftime('%Y%m%d_%H%M%S')}.jsonl",
        )
        print(f"{self.tag} 🚀 启动 Expectimax 强自对弈 (nnue={nnue_path}, out={out})")
        t0 = time.time()
        try:
            stats = run_expectimax_self_play(
                nnue_path,
                n_games=int(getattr(self.cfg, "EXPECTIMAX_SIDECAR_GAMES", 200)),
                num_workers=max(int(getattr(self.cfg, "EXPECTIMAX_SIDECAR_WORKERS", 4)), 1),
                node_budget=int(getattr(self.cfg, "EXPECTIMAX_SIDECAR_NODE_BUDGET", 500_000)),
                max_depth=int(getattr(self.cfg, "EXPECTIMAX_SIDECAR_MAX_DEPTH", 8)),
                threads_per_search=1,
                seed=None,
                out_jsonl=out,
                variant_id=self.variant.id,
            )
        except Exception as exc:  # noqa: BLE001 — 失败不阻塞主闭环
            print(f"{self.tag} ⚠️ Expectimax 强自对弈失败: {exc}")
            return
        self.runs += 1
        self.last_stats = dict(stats)
        total = stats.get("a_wins", 0) + stats.get("b_wins", 0) + stats.get("draws", 0)
        print(f"{self.tag} ✅ 强自对弈完成: {stats.get('games', total)} 局 "
              f"(A={stats.get('a_wins', 0)}, B={stats.get('b_wins', 0)}, "
              f"和={stats.get('draws', 0)}), steps={stats.get('steps', '-')}, "
              f"耗时={time.time() - t0:.0f}s, 数据: {out}")

    # ------------------------------------------------------------------ #
    def _log_stats(self, when: str) -> None:
        print(f"{self.tag} {when}: 触发 {self.ckpt_seen} 次 checkpoint, "
              f"强自对弈 {self.runs} 轮")
