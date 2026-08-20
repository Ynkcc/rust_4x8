"""banqi/rule_teacher.py — Rust 教师自对弈数据生产者（4x2 / 4x4 / 4x8 通用）。

纯规则教师自对弈在 Rust 侧实现（src/bridge/python/self_play/teacher.rs），Python
侧仅做编排：按变体分派到 Rust 导出的 `run_*_heuristic_self_play` /
`run_*_minimax_self_play` 绑定，把返回的 `PyGameEpisode`（Rust 导出的 Episode
格式）序列化为 dict 后压入训练队列，由 `TrainWorker` 消费训练。不依赖神经网络。

用途：`TRAIN_MODE="rule_selfplay"` 时的数据源，用于：
  - 模仿学习预热：让网络先学习强规则教师（minimax / 启发式）的走子与评估，
    避免从随机自对弈冷启动时目标噪声过大。
  - 数据增强：为模型自对弈补充高质量规则对局数据。

并发后端（RULE_SELFPLAY_BACKEND）：
  - "thread"  （默认）：多线程。启动 RULE_SELFPLAY_CONCURRENCY 个
    `RuleTeacherWorker` 线程；Rust 绑定调用释放 GIL，可并行。
  - "process"：多进程 spawn。每个 worker 独立进程 / GIL，彻底并行吃满多核。
    通过 `rule_teacher_worker_main` 进程入口 + multiprocessing.Queue 通信。

Episode 格式：Rust 导出的 `PyGameEpisode.to_dict()` 字段与 `TrainWorker` /
`DataBuffer` 兼容（含 boards / scalars / policies / mcts_values / completed_qs /
root_visits / game_results / health_diffs / action_masks / actions / game_length /
winner / board_shape / scalar_shape / action_space 等）。策略头验证的 ground
truth 采用 `actions`（教师实际走出的最优动作，见 training/buffer.py 的 fallback）。
"""

from __future__ import annotations

import multiprocessing
import queue
import threading
import time
from typing import Dict, List, Optional

try:
    import banqi_4x8
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "无法导入 banqi_4x8。请先执行: maturin develop --features pyo3"
    ) from exc

from banqi.config import make_config
from banqi.variant import Variant

# Rust 统一入口变体映射（variant.rust_prefix -> run_native_match 的 variant_id）
_VARIANT_MAP: Dict[str, str] = {
    "": "4x8",        # 4x8
    "mini": "4x2",    # 4x2
    "game4x4": "4x4", # 4x4
}

# 子批大小（局）：避免一次生成整批 RULE_SELFPLAY_GAMES（如 4x4 500 局）期间
# 队列长期为空、训练永不启动。按此拆分后边生成边入队，形成流水线。
# 4x4 启发式教师并发 4 约 2.5s/局，子批 25 局 ≈ 1 分钟即可产出第一批数据。
_SUB_BATCH = 25


def _variant_id(variant: Variant) -> str:
    key = variant.rust_prefix
    if key not in _VARIANT_MAP:
        raise KeyError(f"未知 rust_prefix {key!r}，可选: {sorted(_VARIANT_MAP)}")
    return _VARIANT_MAP[key]


def build_teacher_config(variant: Variant, sims: int) -> "banqi_4x8.SelfPlayConfig":
    """构建教师自对弈 SelfPlayConfig（启发式路径以 RULE_SELFPLAY_SIMS 作为 MCTS 模拟数）。"""
    return banqi_4x8.SelfPlayConfig(
        mcts_sims=sims,
        max_considered_actions=16,
        c_scale=1.0,
        gumbel_scale=1.0,
        playout_cap_random_enabled=False,
    )


def _episode_to_dict(ep, iteration: int, worker_id: int) -> Dict:
    """把 Rust 导出的 `PyGameEpisode` 序列化为与 `TrainWorker` 兼容的 episode dict。"""
    d = dict(ep.to_dict())
    d["iteration"] = iteration
    # worker.py 用 "round_idx" 作为轮次标记（eval_match / checkpoint 周期判定），
    # 必须显式写入，否则缺失时 worker 侧回退到 0，导致轮次语义与周期评估失效。
    d["round_idx"] = iteration
    d["worker_id"] = worker_id
    return d


def generate_teacher_batch(
    variant: Variant,
    rule_type: str,
    depth: int,
    sims: int,
    temperature: float,
    num_games: int,
    concurrency: int,
    worker_id: int,
) -> List[Dict]:
    """调用 Rust 统一 `run_native_match` 生成一批规则教师 episode。

    `rule_type` : "minimax" | "heuristic"（由 config.RULE_SELFPLAY_TYPE 决定）。
    双方选手均为同一规则教师（自对弈），`record_episodes=True` 收集训练数据。
    """
    del worker_id
    variant_id = _variant_id(variant)
    cfg = build_teacher_config(variant, sims)
    if rule_type == "minimax":
        player_a = player_b = f"minimax{depth}"
    else:
        player_a = player_b = f"heuristic{sims}"
    _, _, _, _, _, episodes = banqi_4x8.run_native_match(
        player_a=player_a,
        player_b=player_b,
        n=num_games,
        variant_id=variant_id,
        model_sims=64,
        heuristic_sims=None,
        seed=None,
        num_threads=concurrency,
        config=cfg,
        record_episodes=True,
    )
    return list(episodes)


class RuleTeacherWorker(threading.Thread):
    """Rust 教师自对弈生产者线程（`TRAIN_MODE="rule_selfplay"` + `RULE_SELFPLAY_BACKEND="thread"`）。

    用法与 `SelfPlayWorker` 一致：调用 Rust 教师绑定生成 episode dict 压入 `data_q`，
    由 `TrainWorker` 消费训练。不依赖神经网络，只复用 `config` 的规则参数。
    """

    def __init__(
        self,
        variant: Variant,
        data_q: "queue.Queue",
        stop_event: threading.Event,
        worker_id: int = 0,
    ) -> None:
        super().__init__(
            name=f"RuleTeacherWorker-{variant.id}-{worker_id}", daemon=True
        )
        self.variant = variant
        self.cfg = make_config(variant.id)
        self.tag = f"[RuleT-{variant.id}-{worker_id}]"
        self.data_q = data_q
        self.stop_event = stop_event
        self.worker_id = worker_id
        self.total_games = 0
        self.round_counter = 0

    def _put(self, item: Dict) -> None:
        while not self.stop_event.is_set():
            try:
                self.data_q.put(item, timeout=0.5)
                return
            except queue.Full:  # 队列满则重试；非 Full 异常直接抛出暴露
                continue

    def run(self) -> None:
        cfg = self.cfg
        rule_type = cfg.RULE_SELFPLAY_TYPE
        depth = cfg.RULE_SELFPLAY_DEPTH
        sims = cfg.RULE_SELFPLAY_SIMS
        temperature = cfg.RULE_SELFPLAY_TEMPERATURE
        concurrency = max(1, cfg.RULE_SELFPLAY_CONCURRENCY)  # Rust 侧 rayon 线程数（并发完全交给 Rust 内层）
        games_per_batch = max(1, cfg.RULE_SELFPLAY_GAMES)
        sub_batch = max(1, min(_SUB_BATCH, games_per_batch))
        total_rounds = max(0, getattr(cfg, "RULE_SELFPLAY_ROUNDS", 0))
        print(f"{self.tag} 🚀 Rust 教师自对弈线程启动: rule={rule_type} "
              f"depth={depth} sims={sims} temp={temperature} "
              f"inner_concurrency={concurrency} games/batch={games_per_batch} "
              f"sub_batch={sub_batch} total_rounds={total_rounds}（不依赖神经网络）")
        while not self.stop_event.is_set():
            if total_rounds > 0 and self.round_counter >= total_rounds:
                print(f"{self.tag} 纯规则自对弈完成，共生成 {self.round_counter} 轮（累计 {self.total_games} 局）")
                break
            t0 = time.time()
            produced = 0
            # 子批化：把本轮 games_per_batch 局拆成多个小子批，每个子批生成完
            # 立即入队，让 TrainWorker 边收数据边训练，避免整批生成期间队列长期为空。
            while produced < games_per_batch and not self.stop_event.is_set():
                n = min(sub_batch, games_per_batch - produced)
                sub_gen = 0
                try:
                    eps = generate_teacher_batch(
                        self.variant, rule_type, depth, sims, temperature,
                        num_games=n, concurrency=concurrency,
                        worker_id=self.worker_id,
                    )
                except Exception as exc:  # pragma: no cover
                    print(f"{self.tag} ⚠️ Rust 教师自对弈异常: {exc}")
                    time.sleep(0.5)
                    continue
                for ep in eps:
                    if self.stop_event.is_set():
                        break
                    d = _episode_to_dict(ep, self.round_counter, self.worker_id)
                    if not d.get("num_samples", 0):
                        continue
                    self._put(d)
                    self.total_games += 1
                    sub_gen += 1
                produced += sub_gen
                dur = time.time() - t0
                print(f"{self.tag} 📊 第 {self.round_counter + 1}/{total_rounds if total_rounds > 0 else '∞'} 轮子批完成 {sub_gen} 局"
                      f"（本轮 {produced}/{games_per_batch}，累计 {self.total_games}，"
                      f"耗时 {dur:.1f}s）")
                # 短暂让出，让 TrainWorker 有机会消费与训练
                time.sleep(0.2)
            self.round_counter += 1
            time.sleep(2.0)

    def stats(self) -> Dict[str, int]:
        return {
            "total_games": self.total_games,
        }


def rule_teacher_worker_main(
    variant_id: str,
    worker_id: int,
    data_q: "multiprocessing.Queue",
    stop_event: "multiprocessing.Event",
) -> None:
    """多进程模式下的 Rust 教师自对弈生产者入口（spawn 子进程 target）。

    每个 worker 独立进程，拥有独立 GIL，可彻底并行吃满多核。
    与线程版共享 `generate_teacher_batch` 生成逻辑，通过 `data_q`（multiprocessing
    队列）把 episode dict 传回主进程，由 `TrainWorker` 消费训练。
    """
    from banqi.variant import get_variant

    variant = get_variant(variant_id)
    cfg = make_config(variant_id)
    tag = f"[RuleT-{variant_id}-{worker_id}]"
    rule_type = cfg.RULE_SELFPLAY_TYPE
    depth = cfg.RULE_SELFPLAY_DEPTH
    sims = cfg.RULE_SELFPLAY_SIMS
    temperature = cfg.RULE_SELFPLAY_TEMPERATURE
    num_workers = max(1, cfg.RULE_SELFPLAY_CONCURRENCY)
    games_per_batch = max(1, cfg.RULE_SELFPLAY_GAMES)
    games_per_worker = max(1, games_per_batch // num_workers)
    sub_batch = max(1, min(_SUB_BATCH, games_per_worker))
    total_rounds = max(0, getattr(cfg, "RULE_SELFPLAY_ROUNDS", 0))
    total_games = 0
    round_counter = 0

    print(f"{tag} 🚀 Rust 教师自对弈子进程启动: rule={rule_type} "
          f"depth={depth} sims={sims} temp={temperature} "
          f"games/worker={games_per_worker} "
          f"sub_batch={sub_batch} total_rounds={total_rounds}（不依赖神经网络）")
    while not stop_event.is_set():
        if total_rounds > 0 and round_counter >= total_rounds:
            print(f"{tag} 纯规则自对弈完成，共生成 {round_counter} 轮（累计 {total_games} 局）")
            break
        t0 = time.time()
        produced = 0
        # 子批化：与线程版一致，避免整批生成期间队列长期为空
        while produced < games_per_worker and not stop_event.is_set():
            n = min(sub_batch, games_per_worker - produced)
            sub_gen = 0
            try:
                eps = generate_teacher_batch(
                    variant, rule_type, depth, sims, temperature,
                    num_games=n, concurrency=1,
                    worker_id=worker_id,
                )
            except Exception as exc:  # pragma: no cover
                print(f"{tag} ⚠️ Rust 教师自对弈异常: {exc}")
                time.sleep(0.5)
                continue
            for ep in eps:
                if stop_event.is_set():
                    break
                d = _episode_to_dict(ep, round_counter, worker_id)
                if not d.get("num_samples", 0):
                    continue
                # multiprocessing.Queue.put 会阻塞（无 timeout），循环里已检查 stop_event
                data_q.put(d)
                total_games += 1
                sub_gen += 1
            produced += sub_gen
            dur = time.time() - t0
            print(f"{tag} 📊 第 {round_counter + 1}/{total_rounds if total_rounds > 0 else '∞'} 轮子批完成 {sub_gen} 局"
                  f"（本轮 {produced}/{games_per_worker}，累计 {total_games}，"
                  f"耗时 {dur:.1f}s）")
            time.sleep(0.2)
        round_counter += 1
        time.sleep(2.0)
