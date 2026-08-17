"""banqi/config.py — 统一训练配置（4x2 / 4x4 / 4x8 共用）

把原先三份 config.py（4x8 顶层 / game_4x4 / mini_4x2）合并为一份 Config，
字段为三者并集；不同变体的默认值由 `make_config(variant_id)` 按变体注入。

环境变量覆盖规则（统一）：
  1. 优先读「变体前缀 + 字段名」：G4X4_XXX / MINI_XXX（4x8 无前缀）
  2. 其次读兼容旧名（如 G4X4_DATA_AUGMENT、G4X4_LR、MONGODB_URI 等历史变量）
  3. 最后读无前缀字段名（DATA_AUGMENT_ENABLED、MONITOR_ENABLED…）

用法：
    from banqi.config import make_config
    cfg = make_config("4x4")          # 含全部字段
    cfg.VALUE_TARGET_MODE             # "mcts" 等
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Optional

from banqi.variant import get_variant

# python/ 目录（banqi/config.py 位于 python/banqi/）
_PY_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 历史环境变量别名：字段名 -> 依次尝试的旧变量名（保留既有脚本的 env 兼容）
_LEGACY_ENV: Dict[str, List[str]] = {
    "DATA_AUGMENT_ENABLED": ["DATA_AUGMENT_ENABLED", "G4X4_DATA_AUGMENT"],
    "DATA_AUGMENT_KEEP_ORIGINAL": ["DATA_AUGMENT_KEEP_ORIGINAL", "G4X4_AUGMENT_KEEP_ORIGINAL"],
    "DATA_AUGMENT_TRANSFORMS": ["DATA_AUGMENT_TRANSFORMS", "G4X4_AUGMENT_TRANSFORMS"],
    "LEARNING_RATE": ["LEARNING_RATE", "G4X4_LR"],
    "TRAIN_EPOCHS_PER_ROUND": ["TRAIN_EPOCHS_PER_ROUND", "G4X4_EPOCHS_PER_ROUND"],
    "MAX_SAMPLE_BUFFER_SIZE": ["MAX_SAMPLE_BUFFER_SIZE", "G4X4_BUFFER_SIZE"],
    "GAMES_PER_ITER": ["GAMES_PER_ITER", "G4X4_GAMES_PER_ITER"],
    "MAX_RUNTIME_SECONDS": ["MAX_RUNTIME_SECONDS", "G4X4_MAX_RUNTIME", "MINI_MAX_RUNTIME"],
    "MONITOR_ENABLED": ["MONITOR_ENABLED", "G4X4_MONITOR"],
    "MONITOR_INTERVAL": ["MONITOR_INTERVAL", "G4X4_MONITOR_INTERVAL"],
    "MONITOR_PER_CORE": ["MONITOR_PER_CORE", "G4X4_MONITOR_PER_CORE"],
    "MONITOR_CSV_PATH": ["MONITOR_CSV_PATH", "G4X4_MONITOR_CSV"],
    "TENSORBOARD_ENABLED": ["TENSORBOARD_ENABLED", "G4X4_TB"],
    "TENSORBOARD_LOG_DIR": ["TENSORBOARD_LOG_DIR", "G4X4_TB_LOG_DIR"],
    "TENSORBOARD_LOG_SYS": ["TENSORBOARD_LOG_SYS", "G4X4_TB_LOG_SYS"],
    "ARCHIVE_ENABLED": ["ARCHIVE_ENABLED", "G4X4_ARCHIVE"],
    "MONGO_URI": ["MONGO_URI", "G4X4_MONGO_URI", "MONGODB_URI"],
    "DB_NAME": ["DB_NAME", "G4X4_DB_NAME"],
    "MODEL_PATH": ["MODEL_PATH", "G4X4_MODEL_PATH", "MINI_MODEL_PATH"],
    "STATE_DICT_PATH": ["STATE_DICT_PATH", "G4X4_STATE_DICT_PATH", "MINI_STATE_DICT_PATH"],
    "VALUE_TARGET_MODE": ["VALUE_TARGET_MODE", "G4X4_VALUE_TARGET"],
    "VALUE_ANNEAL_START": ["VALUE_ANNEAL_START", "G4X4_ANNEAL_START"],
    "VALUE_ANNEAL_INCREMENT": ["VALUE_ANNEAL_INCREMENT", "G4X4_ANNEAL_INCREMENT"],
    "VALUE_ANNEAL_STEP_ROUNDS": ["VALUE_ANNEAL_STEP_ROUNDS", "G4X4_ANNEAL_STEP"],
    "VALUE_DRIFT_EVAL_ROUNDS": ["VALUE_DRIFT_EVAL_ROUNDS", "G4X4_VALUE_DRIFT_EVAL"],
    "VALUE_DRIFT_NUM_POSITIONS": ["VALUE_DRIFT_NUM_POSITIONS", "G4X4_VALUE_DRIFT_N"],
    "ARCHIVE_PREFILL_GAMES": ["ARCHIVE_PREFILL_GAMES", "G4X4_ARCHIVE_PREFILL"],
    "ARCHIVE_PREFILL_DIR": ["ARCHIVE_PREFILL_DIR", "G4X4_ARCHIVE_PREFILL_DIR"],
}


# --------------------------------------------------------------------------- #
# 类型转换
# --------------------------------------------------------------------------- #

def _cast_int(v: str) -> int:
    return int(v)


def _cast_float(v: str) -> float:
    return float(v)


def _cast_bool(v: str) -> bool:
    """统一布尔语义：0 / false / no / off / 空 视为 False，其余 True。"""
    return v.strip().lower() not in ("", "0", "false", "no", "off")


def _cast_str(v: str) -> str:
    return v


def _cast_str_or_none(v: str) -> Optional[str]:
    v = v.strip()
    return v if v else None


# --------------------------------------------------------------------------- #
# 变体默认值
# --------------------------------------------------------------------------- #

_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "4x8": {
        "PREDICT_BATCH": 128,
        "MCTS_SIMS": 128,
        "MAX_CONSIDERED_ACTIONS": 24,
        "TEMPERATURE_STEPS": 16,
        "GAMES_PER_ITER": 100,
        "NUM_WORKERS": 2,
        "GAMES_PER_WORKER": 50,
        "USE_BATCHED_SELF_PLAY": False,
        "BATCH_CONCURRENCY": 4,
        "TRAIN_BATCH": 64,
        "LEARNING_RATE": 2e-4,
        "MIN_LR": 5e-6,
        "LR_DECAY_STEPS": 60000,
        "TRAIN_EPOCHS_PER_ROUND": 3,
        "WEIGHT_DECAY": 1e-4,
        "MAX_SAMPLE_BUFFER_SIZE": 100000,
        "MIN_SAMPLES_TO_START": 1000,
        "QUEUE_FETCH_BATCH": 8,
        "DATA_AUGMENT_ENABLED": True,
        "DATA_AUGMENT_KEEP_ORIGINAL": True,
        "DATA_AUGMENT_TRANSFORMS": "hflip,vflip,rot180",
        "MODEL_PATH": "banqi_model_latest.pt",
        "STATE_DICT_PATH": "banqi_model_latest.pth",
        "INFER_DEVICE": "cpu",
        "TRAIN_DEVICE": "auto",
        "INFER_CPU_AUX_WORKERS": 0,
        "INFER_CPU_FRACTION": 0.3,
        "INFER_MIN_SPLIT_BATCH": 16,
        "DATA_QUEUE_MAXSIZE": 128,
        "ARCHIVE_QUEUE_MAXSIZE": 256,
        "CHECKPOINT_EVERY_N_ROUNDS": 2,
        "MONITOR_ENABLED": True,
        "MONITOR_INTERVAL": 10.0,
        "MONITOR_PER_CORE": False,
        "MONITOR_CSV_PATH": None,
        "TENSORBOARD_ENABLED": True,
        "TENSORBOARD_LOG_DIR": "runs",
        "TENSORBOARD_LOG_SYS": True,
        "MONGO_URI": "mongodb://localhost:27017",
        "DB_NAME": "banqi_training",
        "COLLECTION": "games",
        "ARCHIVE_BATCH": 32,
        "ARCHIVE_POLL_INTERVAL": 1.0,
        "MAX_RUNTIME_SECONDS": 0,                # 0 = 不限时
        "ARCHIVE_ENABLED": True,
        "ARCHIVE_PREFILL_GAMES": 0,
        "ARCHIVE_PREFILL_DIR": "",
        "VALUE_TARGET_MODE": "mcts",
        "VALUE_ANNEAL_START": 0.2,
        "VALUE_ANNEAL_INCREMENT": 0.2,
        "VALUE_ANNEAL_STEP_ROUNDS": 10,
        "VALUE_DRIFT_EVAL_ROUNDS": 0,
        "VALUE_DRIFT_NUM_POSITIONS": 0,
    },
    "4x4": {
        "PREDICT_BATCH": 64,
        "MCTS_SIMS": 256,
        "MAX_CONSIDERED_ACTIONS": 16,
        "TEMPERATURE_STEPS": 12,
        "GAMES_PER_ITER": 40,
        "NUM_WORKERS": 4,
        "GAMES_PER_WORKER": 10,
        "USE_BATCHED_SELF_PLAY": False,
        "BATCH_CONCURRENCY": 4,
        "TRAIN_BATCH": 32,
        "LEARNING_RATE": 5e-4,
        "MIN_LR": 1e-5,
        "LR_DECAY_STEPS": 300000,
        "TRAIN_EPOCHS_PER_ROUND": 2,
        "WEIGHT_DECAY": 1e-4,
        "MAX_SAMPLE_BUFFER_SIZE": 16000,
        "MIN_SAMPLES_TO_START": 128,
        "QUEUE_FETCH_BATCH": 8,
        "DATA_AUGMENT_ENABLED": True,
        "DATA_AUGMENT_KEEP_ORIGINAL": True,
        "DATA_AUGMENT_TRANSFORMS": "hflip,vflip,rot180,rot90,rot270,diag,anti_diag",
        "MODEL_PATH": os.path.join(_PY_DIR, "game_4x4", "banqi4x4_model_latest.pt"),
        "STATE_DICT_PATH": os.path.join(_PY_DIR, "game_4x4", "banqi4x4_model_latest.pth"),
        "INFER_DEVICE": "cpu",
        "TRAIN_DEVICE": "cpu",
        "INFER_CPU_AUX_WORKERS": 0,
        "INFER_CPU_FRACTION": 0.3,
        "INFER_MIN_SPLIT_BATCH": 16,
        "DATA_QUEUE_MAXSIZE": 256,
        "ARCHIVE_QUEUE_MAXSIZE": 256,
        "CHECKPOINT_EVERY_N_ROUNDS": 2,
        "MONITOR_ENABLED": True,
        "MONITOR_INTERVAL": 10.0,
        "MONITOR_PER_CORE": False,
        "MONITOR_CSV_PATH": None,
        "TENSORBOARD_ENABLED": True,
        "TENSORBOARD_LOG_DIR": "runs_4x4",
        "TENSORBOARD_LOG_SYS": True,
        "MONGO_URI": "mongodb://localhost:27017",
        "DB_NAME": "banqi_4x4",
        "COLLECTION": "games",
        "ARCHIVE_BATCH": 32,
        "ARCHIVE_POLL_INTERVAL": 1.0,
        "MAX_RUNTIME_SECONDS": 60 * 60,
        "ARCHIVE_ENABLED": True,
        "ARCHIVE_PREFILL_GAMES": 400,
        "ARCHIVE_PREFILL_DIR": "",
        "VALUE_TARGET_MODE": "mcts",
        "VALUE_ANNEAL_START": 0.2,
        "VALUE_ANNEAL_INCREMENT": 0.2,
        "VALUE_ANNEAL_STEP_ROUNDS": 10,
        "VALUE_DRIFT_EVAL_ROUNDS": 5,
        "VALUE_DRIFT_NUM_POSITIONS": 500,
    },
    "4x2": {
        "PREDICT_BATCH": 64,
        "MCTS_SIMS": 128,
        "MAX_CONSIDERED_ACTIONS": 16,
        "TEMPERATURE_STEPS": 6,
        "GAMES_PER_ITER": 60,
        "NUM_WORKERS": 4,
        "GAMES_PER_WORKER": 15,
        "USE_BATCHED_SELF_PLAY": False,
        "BATCH_CONCURRENCY": 4,
        "TRAIN_BATCH": 32,
        "LEARNING_RATE": 2e-3,
        "MIN_LR": 1e-5,
        "LR_DECAY_STEPS": 12000,
        "TRAIN_EPOCHS_PER_ROUND": 8,
        "WEIGHT_DECAY": 1e-4,
        "MAX_SAMPLE_BUFFER_SIZE": 4000,
        "MIN_SAMPLES_TO_START": 256,
        "QUEUE_FETCH_BATCH": 8,
        "DATA_AUGMENT_ENABLED": False,
        "DATA_AUGMENT_KEEP_ORIGINAL": True,
        "DATA_AUGMENT_TRANSFORMS": "hflip",
        "MODEL_PATH": os.path.join(_PY_DIR, "mini_4x2", "banqi_mini_model_latest.pt"),
        "STATE_DICT_PATH": os.path.join(_PY_DIR, "mini_4x2", "banqi_mini_model_latest.pth"),
        "INFER_DEVICE": "cpu",
        "TRAIN_DEVICE": "cpu",
        "INFER_CPU_AUX_WORKERS": 0,
        "INFER_CPU_FRACTION": 0.3,
        "INFER_MIN_SPLIT_BATCH": 16,
        "DATA_QUEUE_MAXSIZE": 256,
        "ARCHIVE_QUEUE_MAXSIZE": 256,
        "CHECKPOINT_EVERY_N_ROUNDS": 2,
        "MONITOR_ENABLED": False,
        "MONITOR_INTERVAL": 10.0,
        "MONITOR_PER_CORE": False,
        "MONITOR_CSV_PATH": None,
        "TENSORBOARD_ENABLED": False,
        "TENSORBOARD_LOG_DIR": "runs_mini",
        "TENSORBOARD_LOG_SYS": True,
        "MONGO_URI": "mongodb://localhost:27017",
        "DB_NAME": "banqi_mini",
        "COLLECTION": "games",
        "ARCHIVE_BATCH": 32,
        "ARCHIVE_POLL_INTERVAL": 1.0,
        "MAX_RUNTIME_SECONDS": 18 * 60,
        "ARCHIVE_ENABLED": False,
        "ARCHIVE_PREFILL_GAMES": 0,
        "ARCHIVE_PREFILL_DIR": "",
        "VALUE_TARGET_MODE": "mcts",
        "VALUE_ANNEAL_START": 0.2,
        "VALUE_ANNEAL_INCREMENT": 0.2,
        "VALUE_ANNEAL_STEP_ROUNDS": 10,
        "VALUE_DRIFT_EVAL_ROUNDS": 0,
        "VALUE_DRIFT_NUM_POSITIONS": 0,
    },
}


# 字段 -> 类型转换（未列出的按 str 处理）
_CASTS: Dict[str, Callable[[str], Any]] = {
    "PREDICT_BATCH": _cast_int,
    "MCTS_SIMS": _cast_int,
    "MAX_CONSIDERED_ACTIONS": _cast_int,
    "TEMPERATURE_STEPS": _cast_int,
    "GAMES_PER_ITER": _cast_int,
    "NUM_WORKERS": _cast_int,
    "GAMES_PER_WORKER": _cast_int,
    "USE_BATCHED_SELF_PLAY": _cast_bool,
    "BATCH_CONCURRENCY": _cast_int,
    "TRAIN_BATCH": _cast_int,
    "LEARNING_RATE": _cast_float,
    "MIN_LR": _cast_float,
    "LR_DECAY_STEPS": _cast_int,
    "TRAIN_EPOCHS_PER_ROUND": _cast_int,
    "WEIGHT_DECAY": _cast_float,
    "MAX_SAMPLE_BUFFER_SIZE": _cast_int,
    "MIN_SAMPLES_TO_START": _cast_int,
    "QUEUE_FETCH_BATCH": _cast_int,
    "DATA_AUGMENT_ENABLED": _cast_bool,
    "DATA_AUGMENT_KEEP_ORIGINAL": _cast_bool,
    "DATA_AUGMENT_TRANSFORMS": _cast_str,
    "INFER_DEVICE": _cast_str,
    "TRAIN_DEVICE": _cast_str,
    "INFER_CPU_AUX_WORKERS": _cast_int,
    "INFER_CPU_FRACTION": _cast_float,
    "INFER_MIN_SPLIT_BATCH": _cast_int,
    "DATA_QUEUE_MAXSIZE": _cast_int,
    "ARCHIVE_QUEUE_MAXSIZE": _cast_int,
    "CHECKPOINT_EVERY_N_ROUNDS": _cast_int,
    "MONITOR_ENABLED": _cast_bool,
    "MONITOR_INTERVAL": _cast_float,
    "MONITOR_PER_CORE": _cast_bool,
    "MONITOR_CSV_PATH": _cast_str_or_none,
    "TENSORBOARD_ENABLED": _cast_bool,
    "TENSORBOARD_LOG_DIR": _cast_str,
    "TENSORBOARD_LOG_SYS": _cast_bool,
    "ARCHIVE_BATCH": _cast_int,
    "ARCHIVE_POLL_INTERVAL": _cast_float,
    "MAX_RUNTIME_SECONDS": _cast_int,
    "ARCHIVE_ENABLED": _cast_bool,
    "ARCHIVE_PREFILL_GAMES": _cast_int,
    "ARCHIVE_PREFILL_DIR": _cast_str,
    "VALUE_TARGET_MODE": _cast_str,
    "VALUE_ANNEAL_START": _cast_float,
    "VALUE_ANNEAL_INCREMENT": _cast_float,
    "VALUE_ANNEAL_STEP_ROUNDS": _cast_int,
    "VALUE_DRIFT_EVAL_ROUNDS": _cast_int,
    "VALUE_DRIFT_NUM_POSITIONS": _cast_int,
}

# 全部字段顺序（4x8 默认值为全集，其余变体从自身默认表取）
_FIELD_NAMES: List[str] = list(_DEFAULTS["4x8"].keys())


def _alias_applies(prefix: str, alias: str) -> bool:
    """旧别名是否适用于该变体：G4X4_* / MINI_* 只归对应变体，其余通用。"""
    if alias.startswith("G4X4_"):
        return prefix == "G4X4_"
    if alias.startswith("MINI_"):
        return prefix == "MINI_"
    return True


def _resolve_env(variant_id: str, name: str, default: Any) -> Any:
    """按「变体前缀 + 字段名」→ 适用的旧别名 → 通用字段名 的顺序读环境变量。"""
    p = get_variant(variant_id).env_prefix
    keys: List[str] = []
    if p:
        keys.append(p + name)
    keys.extend(
        a for a in _LEGACY_ENV.get(name, []) if _alias_applies(p, a)
    )
    if name not in keys:
        keys.append(name)
    for k in keys:
        if k in os.environ:
            return _CASTS.get(name, _cast_str)(os.environ[k])
    return default


@dataclass
class Config:
    """一个变体的完整训练配置。所有字段由 make_config 解析后填充。"""

    variant_id: str = ""
    PREDICT_BATCH: int = 128
    MCTS_SIMS: int = 128
    MAX_CONSIDERED_ACTIONS: int = 16
    TEMPERATURE_STEPS: int = 8
    GAMES_PER_ITER: int = 100
    NUM_WORKERS: int = 2
    GAMES_PER_WORKER: int = 50
    USE_BATCHED_SELF_PLAY: bool = False
    BATCH_CONCURRENCY: int = 4
    TRAIN_BATCH: int = 64
    LEARNING_RATE: float = 2e-4
    MIN_LR: float = 1e-5
    LR_DECAY_STEPS: int = 60000
    TRAIN_EPOCHS_PER_ROUND: int = 3
    WEIGHT_DECAY: float = 1e-4
    MAX_SAMPLE_BUFFER_SIZE: int = 100000
    MIN_SAMPLES_TO_START: int = 1000
    QUEUE_FETCH_BATCH: int = 8
    DATA_AUGMENT_ENABLED: bool = True
    DATA_AUGMENT_KEEP_ORIGINAL: bool = True
    DATA_AUGMENT_TRANSFORMS: str = "hflip,vflip,rot180"
    MODEL_PATH: str = "banqi_model_latest.pt"
    STATE_DICT_PATH: str = "banqi_model_latest.pth"
    INFER_DEVICE: str = "cpu"
    TRAIN_DEVICE: str = "auto"
    INFER_CPU_AUX_WORKERS: int = 0
    INFER_CPU_FRACTION: float = 0.3
    INFER_MIN_SPLIT_BATCH: int = 16
    DATA_QUEUE_MAXSIZE: int = 128
    ARCHIVE_QUEUE_MAXSIZE: int = 256
    CHECKPOINT_EVERY_N_ROUNDS: int = 2
    MONITOR_ENABLED: bool = True
    MONITOR_INTERVAL: float = 10.0
    MONITOR_PER_CORE: bool = False
    MONITOR_CSV_PATH: Optional[str] = None
    TENSORBOARD_ENABLED: bool = True
    TENSORBOARD_LOG_DIR: str = "runs"
    TENSORBOARD_LOG_SYS: bool = True
    MONGO_URI: str = "mongodb://localhost:27017"
    DB_NAME: str = "banqi_training"
    COLLECTION: str = "games"
    ARCHIVE_BATCH: int = 32
    ARCHIVE_POLL_INTERVAL: float = 1.0
    MAX_RUNTIME_SECONDS: int = 0
    ARCHIVE_ENABLED: bool = True
    ARCHIVE_PREFILL_GAMES: int = 0
    ARCHIVE_PREFILL_DIR: str = ""
    VALUE_TARGET_MODE: str = "mcts"
    VALUE_ANNEAL_START: float = 0.2
    VALUE_ANNEAL_INCREMENT: float = 0.2
    VALUE_ANNEAL_STEP_ROUNDS: int = 10
    VALUE_DRIFT_EVAL_ROUNDS: int = 0
    VALUE_DRIFT_NUM_POSITIONS: int = 0

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


_config_cache: Dict[str, Config] = {}


def make_config(variant_id: str) -> Config:
    """构造（并缓存）指定变体的 Config，应用变体默认值与环境变量覆盖。"""
    if variant_id not in _config_cache:
        defaults = _DEFAULTS.get(variant_id, _DEFAULTS["4x8"])
        c = Config(variant_id=variant_id)
        for name in _FIELD_NAMES:
            setattr(c, name, _resolve_env(variant_id, name, defaults[name]))
        _config_cache[variant_id] = c
    return _config_cache[variant_id]


if __name__ == "__main__":
    from banqi.variant import VARIANTS
    for vid in VARIANTS:
        c = make_config(vid)
        print(f"[banqi.config] {vid}: sims={c.MCTS_SIMS} batch={c.TRAIN_BATCH} "
              f"lr={c.LEARNING_RATE} games={c.GAMES_PER_ITER} "
              f"augment={c.DATA_AUGMENT_ENABLED} tb={c.TENSORBOARD_ENABLED} "
              f"archive={c.ARCHIVE_ENABLED} runtime={c.MAX_RUNTIME_SECONDS}s "
              f"model={os.path.basename(c.MODEL_PATH)}")
    print("[banqi.config] all OK")
