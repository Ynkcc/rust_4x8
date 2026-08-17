"""banqi/config.py — 统一训练配置（4x2 / 4x4 / 4x8 共用）

把原先三份 config.py（4x8 顶层 / game_4x4 / mini_4x2）合并为一份 Config，
字段为三者并集；所有配置值一律来自本地配置文件，源码内不内嵌任何默认值。

配置来源（优先级从低到高）：
  1. 本地配置文件 config.local.yaml（也可用环境变量 BANQI_CONFIG 指定
     其他 YAML 文件）。该文件为必需：缺失时 make_config 直接报错，
     请先运行 `python -m banqi.config --write-template` 生成。
  2. 环境变量（最高优先）

config.default.yaml 仅是生成 config.local.yaml 的模板（--write-template），
运行时不会读取，也不作为任何兜底。

环境变量覆盖规则（统一）：
  1. 优先读「变体前缀 + 字段名」：G4X4_XXX / MINI_XXX（4x8 无前缀）
  2. 其次读兼容旧名（如 G4X4_DATA_AUGMENT、G4X4_LR、MONGODB_URI 等历史变量）
  3. 最后读无前缀字段名（DATA_AUGMENT_ENABLED、MONITOR_ENABLED…）

路径字段（MODEL_PATH / STATE_DICT_PATH）：配置文件中写相对 python/ 目录的
相对路径即可，绝对路径也可直接用。

用法：
    from banqi.config import make_config
    cfg = make_config("4x4")          # 含全部字段
    cfg.VALUE_TARGET_MODE             # "mcts" 等

生成本地配置模板：
    python -m banqi.config --write-template   # 生成 config.local.yaml
"""

from __future__ import annotations

import os
import warnings
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Optional

from banqi.variant import get_variant

# python/ 目录（banqi/config.py 位于 python/banqi/）
_PY_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# banqi/ 包目录：本地配置文件所在位置
_CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_YAML = os.path.join(_CONFIG_DIR, "config.default.yaml")  # 仅作模板
_LOCAL_YAML = os.path.join(_CONFIG_DIR, "config.local.yaml")      # 唯一运行时来源

# 相对 python/ 目录解析的路径字段（兼容旧版 os.path.join(_PY_DIR, ...)）
_PATH_FIELDS = ("MODEL_PATH", "STATE_DICT_PATH")

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


# 字段 -> 类型转换（仅环境变量字符串需要；未列出的按 str 处理）
_CASTS: Dict[str, Callable[[str], Any]] = {
    "PREDICT_BATCH": _cast_int,
    "MCTS_SIMS": _cast_int,
    "MAX_CONSIDERED_ACTIONS": _cast_int,
    "TEMPERATURE_STEPS": _cast_int,
    "GAMES_PER_ITER": _cast_int,
    "NUM_WORKERS": _cast_int,
    "GAMES_PER_WORKER": _cast_int,
    "SELF_PLAY_PROCESSES": _cast_int,
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


# --------------------------------------------------------------------------- #
# YAML 配置文件加载（唯一来源：config.local.yaml 或 BANQI_CONFIG 指定文件）
# --------------------------------------------------------------------------- #

def _load_yaml(path: str) -> Optional[Dict[str, Any]]:
    """读取 YAML 配置文件。

    返回 None 表示文件不存在；文件存在但解析失败或非字典结构则直接报错
    （配置错误应当暴露，不做任何静默兜底）。
    """
    try:
        import yaml  # type: ignore
    except ImportError:
        raise RuntimeError(
            f"[banqi.config] 需要 PyYAML 才能读取配置文件 {path}\n"
            f"  请安装: pip install pyyaml"
        )
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise RuntimeError(f"[banqi.config] 配置文件解析失败: {path}\n  {e}")
    return data if isinstance(data, dict) else None


_config_data: Optional[Dict[str, Dict[str, Any]]] = None
_config_path: Optional[str] = None


def _get_config_data() -> Dict[str, Dict[str, Any]]:
    """惰性加载本地配置文件（首次 make_config 时）。缺失则报错，绝不兜底。"""
    global _config_data, _config_path
    if _config_data is not None:
        return _config_data
    path = os.environ.get("BANQI_CONFIG", "").strip() or _LOCAL_YAML
    data = _load_yaml(path)
    if data is None:
        raise RuntimeError(
            f"[banqi.config] 缺少配置文件: {path}\n"
            f"  本地配置文件是必需的。请先运行:\n"
            f"    python -m banqi.config --write-template\n"
            f"  生成 config.local.yaml，再修改其中的参数。"
        )
    known = set(Config.__dataclass_fields__) - {"variant_id"}
    cleaned: Dict[str, Dict[str, Any]] = {}
    for vid, fields in data.items():
        if not isinstance(fields, dict):
            _warn_unknown(f"变体 {vid!r} 的配置必须是字段字典")
            continue
        kept: Dict[str, Any] = {}
        for k, v in fields.items():
            if k in known:
                kept[k] = v
            else:
                _warn_unknown(k)
        cleaned[vid] = kept
    _config_data = cleaned
    _config_path = path
    return cleaned


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
    """一个变体的完整训练配置。

    字段为四个变体的并集，且不设任何内嵌默认值——所有值由 make_config
    从本地配置文件读取并填充（缺失字段直接报错，无源码兜底）。
    """

    variant_id: str
    PREDICT_BATCH: int
    MCTS_SIMS: int
    MAX_CONSIDERED_ACTIONS: int
    TEMPERATURE_STEPS: int
    GAMES_PER_ITER: int
    NUM_WORKERS: int
    GAMES_PER_WORKER: int
    SELF_PLAY_PROCESSES: int
    USE_BATCHED_SELF_PLAY: bool
    BATCH_CONCURRENCY: int
    TRAIN_BATCH: int
    LEARNING_RATE: float
    MIN_LR: float
    LR_DECAY_STEPS: int
    TRAIN_EPOCHS_PER_ROUND: int
    WEIGHT_DECAY: float
    MAX_SAMPLE_BUFFER_SIZE: int
    MIN_SAMPLES_TO_START: int
    QUEUE_FETCH_BATCH: int
    DATA_AUGMENT_ENABLED: bool
    DATA_AUGMENT_KEEP_ORIGINAL: bool
    DATA_AUGMENT_TRANSFORMS: str
    MODEL_PATH: str
    STATE_DICT_PATH: str
    INFER_DEVICE: str
    TRAIN_DEVICE: str
    INFER_CPU_AUX_WORKERS: int
    INFER_CPU_FRACTION: float
    INFER_MIN_SPLIT_BATCH: int
    DATA_QUEUE_MAXSIZE: int
    ARCHIVE_QUEUE_MAXSIZE: int
    CHECKPOINT_EVERY_N_ROUNDS: int
    MONITOR_ENABLED: bool
    MONITOR_INTERVAL: float
    MONITOR_PER_CORE: bool
    MONITOR_CSV_PATH: Optional[str]
    TENSORBOARD_ENABLED: bool
    TENSORBOARD_LOG_DIR: str
    TENSORBOARD_LOG_SYS: bool
    MONGO_URI: str
    DB_NAME: str
    COLLECTION: str
    ARCHIVE_BATCH: int
    ARCHIVE_POLL_INTERVAL: float
    MAX_RUNTIME_SECONDS: int
    ARCHIVE_ENABLED: bool
    ARCHIVE_PREFILL_GAMES: int
    ARCHIVE_PREFILL_DIR: str
    VALUE_TARGET_MODE: str
    VALUE_ANNEAL_START: float
    VALUE_ANNEAL_INCREMENT: float
    VALUE_ANNEAL_STEP_ROUNDS: int
    VALUE_DRIFT_EVAL_ROUNDS: int
    VALUE_DRIFT_NUM_POSITIONS: int

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


# 已知字段集合与字段顺序（均由 dataclass 声明决定）
_KNOWN_FIELDS = set(Config.__dataclass_fields__) - {"variant_id"}
_FIELD_NAMES: List[str] = [f for f in Config.__dataclass_fields__ if f != "variant_id"]
_unknown_warned: set = set()


def _warn_unknown(field: str) -> None:
    if field not in _unknown_warned:
        _unknown_warned.add(field)
        warnings.warn(f"[banqi.config] 忽略未知配置字段: {field!r}")


_config_cache: Dict[str, Config] = {}


def make_config(variant_id: str) -> Config:
    """构造（并缓存）指定变体的 Config。

    配置只来自本地配置文件（config.local.yaml 或 BANQI_CONFIG）与
    环境变量；源码不提供任何默认值，配置文件缺失/字段缺失直接报错。
    """
    if variant_id not in _config_cache:
        data = _get_config_data()
        fields = data.get(variant_id)
        if fields is None:
            raise RuntimeError(
                f"[banqi.config] 配置文件 {_config_path} 中缺少变体 {variant_id!r}\n"
                f"  可用变体: {sorted(data)}"
            )
        missing = sorted(_KNOWN_FIELDS - set(fields))
        if missing:
            raise RuntimeError(
                f"[banqi.config] 配置文件 {_config_path} 中变体 {variant_id!r} 缺少字段: {missing}\n"
                f"  请参考 config.default.yaml 补全（或重新生成 config.local.yaml）。"
            )
        c = object.__new__(Config)
        c.variant_id = variant_id
        for name in _FIELD_NAMES:
            value = fields[name]
            # 路径字段：相对 python/ 目录解析（绝对路径直接使用）
            if (
                name in _PATH_FIELDS
                and isinstance(value, str)
                and value
                and not os.path.isabs(value)
            ):
                value = os.path.join(_PY_DIR, value)
            setattr(c, name, _resolve_env(variant_id, name, value))
        _config_cache[variant_id] = c
    return _config_cache[variant_id]


if __name__ == "__main__":
    import sys
    import shutil

    if "--write-template" in sys.argv:
        if not os.path.isfile(_DEFAULT_YAML):
            print(f"[banqi.config] 模板文件缺失: {_DEFAULT_YAML}")
            sys.exit(1)
        shutil.copyfile(_DEFAULT_YAML, _LOCAL_YAML)
        print(f"[banqi.config] 已生成本地配置文件: {_LOCAL_YAML}")
        print("  修改其中的参数即可生效（环境变量优先级仍最高）。")
        sys.exit(0)

    from banqi.variant import VARIANTS
    for vid in VARIANTS:
        c = make_config(vid)
        print(f"[banqi.config] {vid}: sims={c.MCTS_SIMS} batch={c.TRAIN_BATCH} "
              f"lr={c.LEARNING_RATE} games={c.GAMES_PER_ITER} "
              f"augment={c.DATA_AUGMENT_ENABLED} tb={c.TENSORBOARD_ENABLED} "
              f"archive={c.ARCHIVE_ENABLED} runtime={c.MAX_RUNTIME_SECONDS}s "
              f"model={os.path.basename(c.MODEL_PATH)}")
    print("[banqi.config] all OK")
