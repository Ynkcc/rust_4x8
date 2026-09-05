"""banqi/trainer_cli/runners/ — 训练模式分派与运行编排（包）。

main 按 config.TRAIN_MODE 分派：
  - "selfplay"     : 标准模型 MCTS 自对弈闭环（runners/selfplay.py）
  - "archive"      : 仅从冷存储归档数据训练（runners/offline.py）
  - "rule_selfplay": 纯规则（minimax/heuristic）自对弈生成数据训练（runners/offline.py）

共享基础设施（可选依赖探测 / 日志落盘 / TB 元信息 / 队列计数 / 变体维度缓存）
统一在 runners/context.py；归档数据供给线程在 runners/archive_feeder.py。
"""

from __future__ import annotations

from banqi.config import Config, make_config

from .context import build_const, setup_variant_logging, log_meta_tb
from .offline import run_offline
from .selfplay import run_selfplay

__all__ = [
    "main",
    "build_const",
    "setup_variant_logging",
    "log_meta_tb",
]


def main(variant_id: str) -> None:
    """统一训练入口：按 config.TRAIN_MODE 分派训练模式。"""
    config: Config = make_config(variant_id)
    train_mode = (config.TRAIN_MODE or "selfplay").strip().lower()
    if train_mode == "selfplay":
        run_selfplay(variant_id)
    elif train_mode in ("archive", "rule_selfplay"):
        run_offline(variant_id, train_mode)
    else:
        raise ValueError(
            f"未知 TRAIN_MODE={train_mode!r}，可选: selfplay / archive / rule_selfplay"
        )
