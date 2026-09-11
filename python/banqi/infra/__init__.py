"""banqi/infra — L4 协作接口（EpisodeStore / ModelRegistry）。

Protocol + 各后端实现：
- local_*：单机形态（目录/jsonl.gz 握手）
- scheduler_*：分布式形态（R2 凭据只在调度器持有，trainer/worker 经
  gRPC 签发的预签名 URL 上下行，零存储配置）

Trainer 与 Collector 只依赖 Protocol，不感知后端差异。
"""

from .episode_store import EpisodeStore, LocalEpisodeStore, SchedulerEpisodeStore
from .model_registry import (
    LocalModelRegistry,
    ModelRegistry,
    SchedulerModelRegistry,
    scheduler_variant,
)

__all__ = [
    "EpisodeStore",
    "LocalEpisodeStore",
    "SchedulerEpisodeStore",
    "ModelRegistry",
    "LocalModelRegistry",
    "SchedulerModelRegistry",
    "scheduler_variant",
]
