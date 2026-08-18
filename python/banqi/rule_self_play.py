"""banqi/rule_self_play.py — 纯规则自对弈数据生成（共享实现，4x2 / 4x4 / 4x8 通用）

不依赖神经网络：用纯规则策略（minimax 或 启发式 MCTS）驱动对局，生成与
`GameEpisode` 兼容的 episode dict（字段对齐 `banqi.training_service.episode_to_samples`，
可直接喂给 `TrainWorker` / `DataBuffer`）。

用途：`TRAIN_MODE="rule_selfplay"` 时的数据源，用于：
  - 模仿学习预热：让网络先学习强规则教师（minimax / 启发式 MCTS）的走子与评估，
    避免从随机自对弈冷启动时目标噪声过大。
  - 数据增强：为模型自对弈补充高质量规则对局数据。

并发后端（`RULE_SELFPLAY_BACKEND`）：
  - "thread"  （默认）：多线程。同一进程内启动 RULE_SELFPLAY_CONCURRENCY 个
    `RuleSelfPlayWorker` 线程。Rust 绑定调用会释放 GIL，可并行；无需 spawn 子进程，
    轻量，适合 4x2 等小变体。
  - "process"：多进程。用 multiprocessing spawn 子进程，每 worker 独立进程/ GIL，
    彻底并行吃满多核。通过 `rule_sp_worker_main` 进程入口 + multiprocessing.Queue 通信。

策略目标设计：
  - minimax：`minimax_action(depth)` 返回单一最优动作（含 expectiminimax 搜索值），
    policy 采用"价值加权先验"——最优动作高概率、其余合法动作按温度分摊小概率。
  - heuristic：`heuristic_mcts_action(sims)` 返回启发式 Gumbel MCTS 选中动作，
    policy 同样按选中动作 + 温度平滑构造。
  - value 目标：minimax 用其搜索值；heuristic 用启发式评估值；终局用 ±1/0。

纯 Python 实现，无需 Rust 侧额外绑定（复用 chess_env.rs 已暴露的
`minimax_action` / `heuristic_mcts_action`）。
"""

from __future__ import annotations

import multiprocessing
import queue
import threading
import time
from typing import Callable, Dict, List, Optional

import numpy as np

try:
    import banqi_4x8
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "无法导入 banqi_4x8。请先执行: maturin develop --features pyo3"
    ) from exc

from banqi.config import make_config
from banqi.constants import build_constants
from banqi.variant import Variant

# 环境类名（与 banqi.eval 的 _ENV_CLASS_NAMES 对应）
_ENV_CLASS_NAMES = {
    "": "DarkChess",       # 4x8
    "game4x4": "Game4x4",  # 4x4
    "mini": "MiniDarkChess",  # 4x2
}


def get_env_class(variant: Variant):
    """返回变体的 Rust 环境类（延迟 import）。"""
    name = _ENV_CLASS_NAMES.get(variant.rust_prefix)
    if name is None:
        raise ValueError(f"未知 rust_prefix {variant.rust_prefix!r}（变体 {variant.id}）")
    return getattr(banqi_4x8, name)


# ---------------------------------------------------------------------------
# 策略目标构造
# ---------------------------------------------------------------------------

def _softmax_policy(env, action: Optional[int], action_space: int,
                    temperature: float) -> np.ndarray:
    """构造 policy 分布：选中动作高先验 + 温度平滑。

    - action 为 None（终局无动作）时返回均匀零向量（调用方应避免此情形）。
    - 选中动作 logit 设 1.0，其余合法动作 logit 设 0.0，再按温度 softmax；
      温度越小越接近 one-hot，越大越均匀（探索）。
    """
    probs = np.zeros(action_space, dtype=np.float32)
    if action is None:
        return probs
    legal = env.legal_moves()
    if not legal:
        return probs
    logits = np.full(action_space, -1e9, dtype=np.float32)
    for a in legal:
        logits[a] = 0.0
    logits[action] = 1.0  # 选中动作高先验
    t = max(temperature, 1e-3)
    exp = np.exp((logits - logits.max()) / t)
    exp[logits <= -1e8] = 0.0
    total = exp.sum()
    if total <= 0:
        probs[action] = 1.0
        return probs
    probs[:] = exp / total
    return probs


# ---------------------------------------------------------------------------
# 单局纯规则自对弈
# ---------------------------------------------------------------------------

def generate_rule_episode(
    variant: Variant,
    rule_type: str = "heuristic",
    depth: int = 3,
    sims: int = 64,
    temperature: float = 0.5,
    max_moves: Optional[int] = None,
) -> Dict:
    """用纯规则策略生成一局，返回与 `GameEpisode.to_dict()` 兼容的 episode dict。

    参数：
      rule_type : "minimax" | "heuristic"
      depth     : minimax 搜索深度（仅 rule_type="minimax"）
      sims      : 启发式 MCTS 模拟数（仅 rule_type="heuristic"）
      temperature: 走子温度（0=贪心选最优，越大越随机）
      max_moves : 步数上限（默认取环境 max_steps）
    """
    env_cls = get_env_class(variant)
    C = build_constants(variant)
    env = env_cls()
    aspace = C.ACTION_SPACE_SIZE

    boards: List[List[float]] = []
    scalars: List[List[float]] = []
    policies: List[List[float]] = []
    mcts_values: List[float] = []
    completed_qs: List[float] = []
    root_visits: List[int] = []
    game_results: List[float] = []
    action_masks: List[List[int]] = []
    actions: List[int] = []
    teacher_actions: List[int] = []  # 温度采样前的启发式/规则最优动作（策略头验证 ground truth）
    health_diffs: List[float] = []

    if max_moves is None:
        max_moves = env_cls.max_steps() if hasattr(env_cls, "max_steps") else 400

    steps = 0
    winner: Optional[int] = None
    while not env.terminated():
        if env.winner() is not None:
            winner = env.winner()
            break
        b, s = env.observation()
        mask = env.legal_moves()
        amask = [0] * aspace
        for a in mask:
            amask[a] = 1

        # 选动作（minimax / 启发式 MCTS，均不依赖神经网络）
        if rule_type == "minimax":
            action = env.minimax_action(depth)
        else:
            action = env.heuristic_mcts_action(sims)

        if action is None:
            winner = env.winner()
            break
        # 记录温度采样前的启发式/规则最优动作（作为策略头验证 ground truth）
        teacher_action = int(action)
        # 温度采样：以概率 temperature 走随机合法动作（增加探索），否则走规则最优
        if temperature > 1e-3 and mask and np.random.rand() < temperature:
            action = int(mask[np.random.randint(len(mask))])

        policy = _softmax_policy(env, action, aspace, temperature)

        boards.append(b)
        scalars.append(s)
        policies.append(policy.tolist())
        # 价值目标：规则自对弈下每步 value 用「终局结果 ±1/0」（AlphaZero 标准），
        # 待对局结束后统一回填，使任何 VALUE_TARGET_MODE 下目标一致且非零。
        mcts_values.append(0.0)
        completed_qs.append(0.0)
        root_visits.append(0)
        game_results.append(0.0)
        action_masks.append(amask)
        actions.append(int(action))
        teacher_actions.append(teacher_action)
        health_diffs.append(0.0)

        env.step(int(action))
        steps += 1
        if steps >= max_moves:
            break

    if winner is None:
        winner = env.winner() if env.winner() is not None else 0
    game_result = 1.0 if winner == 1 else (-1.0 if winner == -1 else 0.0)
    for i in range(len(game_results)):
        game_results[i] = game_result
        mcts_values[i] = game_result
        completed_qs[i] = game_result

    n = len(boards)
    return {
        "boards": boards,
        "scalars": scalars,
        "policies": policies,
        "mcts_values": mcts_values,
        "completed_qs": completed_qs,
        "root_visits": root_visits,
        "game_results": game_results,
        "action_masks": action_masks,
        "actions": actions,
        "teacher_actions": teacher_actions,
        "health_diffs": health_diffs,
        "game_length": n,
        "winner": winner,
        "health_diff_red": 0.0,
        "num_samples": n,
    }


# ---------------------------------------------------------------------------
# 多线程：多个 RuleSelfPlayWorker 线程持续用纯规则自对弈生产 episode 压入队列
# ---------------------------------------------------------------------------

# 停止信号统一抽象：线程模式传 `lambda: stop_flag[0]`，进程模式传 `stop_event.is_set`
_StoppedFn = Callable[[], bool]


class RuleSelfPlayWorker(threading.Thread):
    """纯规则自对弈生产者线程（`TRAIN_MODE="rule_selfplay"` + `RULE_SELFPLAY_BACKEND="thread"`）。

    用法与 `SelfPlayWorker` 一致：把生成的 episode dict 压入 `data_q`，
    由 `TrainWorker` 消费训练。不依赖神经网络，只复用 `config` 的规则参数。

    多线程并发：由 `train._run_offline` 按 `RULE_SELFPLAY_CONCURRENCY` 启动多个本线程
    实例（每个独立 `worker_id`），共享同一线程安全 `queue.Queue`。
    """

    def __init__(
        self,
        variant: Variant,
        data_q: "queue.Queue",
        stopped: _StoppedFn,
        worker_id: int = 0,
    ) -> None:
        super().__init__(
            name=f"RuleSelfPlayWorker-{variant.id}-{worker_id}", daemon=True
        )
        self.variant = variant
        self.cfg = make_config(variant.id)
        self.tag = f"[RuleSP-{variant.id}-{worker_id}]"
        self.data_q = data_q
        self.stopped = stopped
        self.worker_id = worker_id
        self.total_games = 0

    def _put(self, item: Dict) -> None:
        while not self.stopped():
            try:
                self.data_q.put(item, timeout=0.5)
                return
            except Exception:  # queue.Full
                continue

    def run(self) -> None:
        cfg = self.cfg
        rule_type = cfg.RULE_SELFPLAY_TYPE
        depth = cfg.RULE_SELFPLAY_DEPTH
        sims = cfg.RULE_SELFPLAY_SIMS
        temperature = cfg.RULE_SELFPLAY_TEMPERATURE
        games_per_batch = max(1, cfg.RULE_SELFPLAY_GAMES)
        print(f"{self.tag} 🚀 纯规则自对弈线程启动: rule={rule_type} "
              f"depth={depth} sims={sims} temp={temperature} "
              f"games/batch={games_per_batch}（不依赖神经网络）")
        while not self.stopped():
            t0 = time.time()
            generated = 0
            for _ in range(games_per_batch):
                if self.stopped():
                    break
                try:
                    ep = generate_rule_episode(
                        self.variant, rule_type=rule_type,
                        depth=depth, sims=sims, temperature=temperature,
                    )
                except Exception as exc:  # pragma: no cover
                    print(f"{self.tag} ⚠️ 规则自对弈异常: {exc}")
                    time.sleep(0.5)
                    continue
                if not ep.get("num_samples", 0):
                    continue
                ep["iteration"] = self.total_games // max(1, cfg.GAMES_PER_ITER)
                ep["worker_id"] = self.worker_id
                self._put(ep)
                self.total_games += 1
                generated += 1
            dur = time.time() - t0
            print(f"{self.tag} 📊 生成 {generated} 局（累计 {self.total_games}，"
                  f"耗时 {dur:.1f}s），等待训练消费...")
            # 让 TrainWorker 有时间消费与训练，避免压满队列
            time.sleep(2.0)

    def stats(self) -> Dict[str, int]:
        return {
            "total_games": self.total_games,
        }


# ---------------------------------------------------------------------------
# 多进程：rule_sp_worker_main 进程入口（RULE_SELFPLAY_BACKEND="process"）
# ---------------------------------------------------------------------------

def rule_sp_worker_main(
    variant_id: str,
    worker_id: int,
    data_q: "multiprocessing.Queue",
    stop_event: "multiprocessing.Event",
) -> None:
    """多进程模式下的纯规则自对弈生产者入口（spawn 子进程 target）。

    每个 worker 独立进程，拥有独立 GIL，可彻底并行吃满多核。
    与线程版共享 `generate_rule_episode` 生成逻辑，通过 `data_q`（multiprocessing
    队列）把 episode dict 传回主进程，由 `TrainWorker` 消费训练。
    """
    from banqi.variant import get_variant

    variant = get_variant(variant_id)
    cfg = make_config(variant_id)
    tag = f"[RuleSP-{variant_id}-{worker_id}]"
    rule_type = cfg.RULE_SELFPLAY_TYPE
    depth = cfg.RULE_SELFPLAY_DEPTH
    sims = cfg.RULE_SELFPLAY_SIMS
    temperature = cfg.RULE_SELFPLAY_TEMPERATURE
    games_per_batch = max(1, cfg.RULE_SELFPLAY_GAMES)
    total_games = 0

    print(f"{tag} 🚀 纯规则自对弈子进程启动: rule={rule_type} "
          f"depth={depth} sims={sims} temp={temperature} "
          f"games/batch={games_per_batch}（不依赖神经网络）")
    while not stop_event.is_set():
        t0 = time.time()
        generated = 0
        for _ in range(games_per_batch):
            if stop_event.is_set():
                break
            try:
                ep = generate_rule_episode(
                    variant, rule_type=rule_type,
                    depth=depth, sims=sims, temperature=temperature,
                )
            except Exception as exc:  # pragma: no cover
                print(f"{tag} ⚠️ 规则自对弈异常: {exc}")
                time.sleep(0.5)
                continue
            if not ep.get("num_samples", 0):
                continue
            ep["iteration"] = total_games // max(1, cfg.GAMES_PER_ITER)
            ep["worker_id"] = worker_id
            # multiprocessing.Queue.put 会阻塞（无 timeout），循环里已检查 stop_event
            data_q.put(ep)
            total_games += 1
            generated += 1
        dur = time.time() - t0
        print(f"{tag} 📊 生成 {generated} 局（累计 {total_games}，"
              f"耗时 {dur:.1f}s），等待训练消费...")
        time.sleep(2.0)
