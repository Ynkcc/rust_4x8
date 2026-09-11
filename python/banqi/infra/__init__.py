"""banqi/infra — L4 协作接口（EpisodeStore / ModelRegistry）。

Protocol + 各后端实现：
- local_*：单机形态（目录/jsonl.gz 握手）
- r2_* / scheduler_*：分布式形态（后续接入）

Trainer 与 Collector 只依赖 Protocol，不感知后端差异。
"""

from .episode_store import EpisodeStore, LocalEpisodeStore
from .model_registry import LocalModelRegistry, ModelRegistry

__all__ = [
    "EpisodeStore",
    "LocalEpisodeStore",
    "ModelRegistry",
    "LocalModelRegistry",
]
