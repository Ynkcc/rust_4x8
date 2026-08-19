"""banqi/config.py — 统一训练配置（4x2 / 4x4 / 4x8 共用）

把原先三份 config.py（4x8 顶层 / game_4x4 / mini_4x2）合并为一份 Config，
字段为三者并集；所有配置值一律来自本地配置文件，源码内不内嵌任何默认值。

配置文件采用「两段式」结构（config.default.yaml 模板）：
  common:  与模型变体无关的公共配置（设备 / 基础设施 / 通用超参），三变体共用。
  4x8/4x4/4x2: 仅保留真正因变体而异的字段。
运行时 make_config 会把 common 与所选变体的字段合并（变体字段覆盖 common），
合并后每个变体都必须覆盖全部字段，否则报错。

配置来源（优先级从低到高）：
  1. 本地配置文件 config.local.yaml（也可用环境变量 BANQI_CONFIG 指定
     其他 YAML 文件）。该文件为必需：缺失时 make_config 直接报错，
     请先运行 `python -m banqi.config --write-template` 生成。
  2. 环境变量（最高优先）

config.default.yaml 仅是生成 config.local.yaml 的模板（--write-template），
运行时不会读取，也不作为任何兜底。

环境变量覆盖规则（统一）：所有字段一律以「无前缀字段名」读取，名称直接
对应 Config 字段（如 LEARNING_RATE、DATA_AUGMENT_ENABLED、MONGO_URI…）。
不再支持变体前缀（G4X4_* / MINI_*）与历史旧名别名（MONGODB_URI 等）。

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

# python/ 目录（banqi/config.py 位于 python/banqi/）
_PY_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# banqi/ 包目录：本地配置文件所在位置
_CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_YAML = os.path.join(_CONFIG_DIR, "config.default.yaml")  # 仅作模板
_LOCAL_YAML = os.path.join(_CONFIG_DIR, "config.local.yaml")      # 唯一运行时来源

# 相对 python/ 目录解析的路径字段（统一转换为绝对路径，避免依赖 CWD）
_PATH_FIELDS = (
    "OUTPUT_DIR",
    "MODEL_PATH",
    "STATE_DICT_PATH",
    "ONNX_PATH",
    "TENSORBOARD_LOG_DIR",
    "ARCHIVE_TRAIN_DIR",
    "MONITOR_CSV_PATH",
    "ARCHIVE_PREFILL_DIR",
)




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
    "GAMES_PER_ITER": _cast_int,
    "NUM_WORKERS": _cast_int,
    "GAMES_PER_WORKER": _cast_int,
    "SELF_PLAY_PROCESSES": _cast_int,
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
    "CKPT_SAVE_EVERY": _cast_int,
    "CKPT_EXPORT_EVERY": _cast_int,

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
    "HEALTH_VALUE_HEAD_ENABLED": _cast_bool,
    "HEALTH_LOSS_WEIGHT": _cast_float,
    "HEALTH_UTILITY_WEIGHT": _cast_float,
    "HEALTH_UTILITY_CONFIDENCE_EXP": _cast_float,
    "EVAL_MATCH_ROUNDS": _cast_int,
    "EVAL_MATCH_GAMES": _cast_int,
    "EVAL_MATCH_OPPONENTS": _cast_str,
    "EVAL_MATCH_VS_PREV": _cast_bool,
    "TRAIN_MODE": _cast_str,
    "ARCHIVE_TRAIN_DIR": _cast_str,
    "ARCHIVE_TRAIN_GAMES": _cast_int,
    "ARCHIVE_TRAIN_ROUNDS": _cast_int,
    "RULE_SELFPLAY_TYPE": _cast_str,
    "RULE_SELFPLAY_DEPTH": _cast_int,
    "RULE_SELFPLAY_SIMS": _cast_int,
    "RULE_SELFPLAY_GAMES": _cast_int,
    "RULE_SELFPLAY_ROUNDS": _cast_int,
    "RULE_SELFPLAY_CONCURRENCY": _cast_int,
    "RULE_SELFPLAY_TEMPERATURE": _cast_float,
    "RULE_SELFPLAY_BACKEND": _cast_str,
    # ---- 模型后端（TorchScript / ONNX）----
    "MODEL_BACKEND": _cast_str,
    "ONNX_PROVIDERS": _cast_str,

    # ---- 权重指数滑动平均 (EMA) & 算力分配随机化 ----
    "EMA_ENABLED": _cast_bool,
    "EMA_DECAY": _cast_float,
    "PLAYOUT_CAP_RANDOM_ENABLED": _cast_bool,
    "FAST_MCTS_SIMS": _cast_int,
    "FULL_SEARCH_PROB": _cast_float,
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


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """深层递归合并字典，override 覆盖 base。"""
    res = dict(base)
    for k, v in override.items():
        if k in res and isinstance(res[k], dict) and isinstance(v, dict):
            res[k] = _deep_merge(res[k], v)
        else:
            res[k] = v
    return res


def _flatten_fields(d: Dict[str, Any], result: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """递归遍历嵌套字典，将所有标量键（大写名）展平为一维字段字典。"""
    if result is None:
        result = {}
    for k, v in d.items():
        if isinstance(v, dict):
            _flatten_fields(v, result)
        else:
            result[k] = v
    return result


_config_data: Optional[Dict[str, Dict[str, Any]]] = None
_config_path: Optional[str] = None

# 顶层公共配置区块：与模型变体无关，三变体共用，变体字段覆盖之
_COMMON_KEY = "common"


def _get_config_data() -> Dict[str, Dict[str, Any]]:
    """惰性加载本地配置文件（首次 make_config 时）。缺失则报错，绝不兜底。

    支持嵌套模块分组结构（Scheme B）：
      common:  公共模块分组配置（作为所有变体的基础）；
      4x8/4x4/4x2: 变体相关分组配置，深层覆盖 common 中的同名项。
    """
    global _config_data, _config_path
    if _config_data is not None:
        return _config_data
    path = os.environ.get("BANQI_CONFIG", "").strip() or _LOCAL_YAML
    raw_data = _load_yaml(path)
    if raw_data is None:
        raise RuntimeError(
            f"[banqi.config] 缺少配置文件: {path}\n"
            f"  本地配置文件是必需的。请先运行:\n"
            f"    python -m banqi.config --write-template\n"
            f"  生成 config.local.yaml，再修改其中的参数。"
        )
    known = set(Config.__dataclass_fields__) - {"variant_id"}

    if _COMMON_KEY not in raw_data:
        raise RuntimeError(
            f"[banqi.config] 配置文件 {path} 缺少 {_COMMON_KEY!r} 区块\n"
            f"  请重新生成 config.local.yaml（python -m banqi.config --write-template）。"
        )
    if not isinstance(raw_data[_COMMON_KEY], dict):
        raise RuntimeError(
            f"[banqi.config] 配置文件 {path} 中 {_COMMON_KEY!r} 必须是字典结构"
        )

    common_raw = raw_data[_COMMON_KEY]
    cleaned: Dict[str, Dict[str, Any]] = {}

    for vid, variant_raw in raw_data.items():
        if vid == _COMMON_KEY:
            continue
        if not isinstance(variant_raw, dict):
            _warn_unknown(f"变体 {vid!r} 的配置必须是字典结构")
            continue
        
        # 1. 递归深层合并 common 与变体差异配置
        merged_tree = _deep_merge(common_raw, variant_raw)
        
        # 2. 展平为扁平的大写字段字典
        flat_fields = _flatten_fields(merged_tree)
        
        # 3. 自动计算与推导字段
        if "NUM_WORKERS" in flat_fields and "GAMES_PER_WORKER" in flat_fields:
            flat_fields["GAMES_PER_ITER"] = int(flat_fields["NUM_WORKERS"]) * int(flat_fields["GAMES_PER_WORKER"])
        
        # 历史兼容字段自动映射
        if "CKPT_SAVE_EVERY" in flat_fields:
            flat_fields["CHECKPOINT_EVERY_N_ROUNDS"] = flat_fields["CKPT_SAVE_EVERY"]

        # 校验未知字段与已知字段过滤
        kept: Dict[str, Any] = {}
        for k, v in flat_fields.items():
            if k in known:
                kept[k] = v
            else:
                _warn_unknown(k)
        cleaned[vid] = kept

    _config_data = cleaned
    _config_path = path
    return cleaned


def _resolve_env(name: str, default: Any) -> Any:
    """统一：直接以字段名读取环境变量（无变体前缀、无旧别名）。"""
    if name in os.environ:
        return _CASTS.get(name, _cast_str)(os.environ[name])
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
    GAMES_PER_ITER: int
    NUM_WORKERS: int
    GAMES_PER_WORKER: int
    SELF_PLAY_PROCESSES: int
    SELF_PLAY_BACKEND: str       # 自对弈并发后端：process | thread（保留兼容，实际推理侧由 INFER_SIDE 决定）
    INFER_SIDE: str              # 推理执行侧：python（多进程，Python run_python_match 推理）| rust（多线程，Rust run_native_match 管理线程，透传 num_threads）
    BATCH_CONCURRENCY: int
    TRAIN_BATCH: int
    LEARNING_RATE: float
    MIN_LR: float
    LR_DECAY_STEPS: int
    TRAIN_EPOCHS_PER_ROUND: int
    WEIGHT_DECAY: float
    EMA_ENABLED: bool             # 是否启用 EMA
    EMA_DECAY: float              # EMA 滑动衰减率 (如 0.999)
    PLAYOUT_CAP_RANDOM_ENABLED: bool # 是否启用算力分配随机化
    FAST_MCTS_SIMS: int           # Fast Search MCTS 模拟数
    FULL_SEARCH_PROB: float       # Full Search 出现概率 (如 0.25)
    MAX_SAMPLE_BUFFER_SIZE: int
    MIN_SAMPLES_TO_START: int
    QUEUE_FETCH_BATCH: int
    DATA_AUGMENT_ENABLED: bool
    DATA_AUGMENT_KEEP_ORIGINAL: bool
    DATA_AUGMENT_TRANSFORMS: str
    OUTPUT_DIR: str
    MODEL_PATH: str
    STATE_DICT_PATH: str
    # ---- 模型后端切换 ----
    # MODEL_BACKEND: "torchscript"（默认，.pt TorchScript）| "onnx"（.onnx，ONNX Runtime）。
    #   - "onnx" 时，自对弈优先走 Rust 绑定 RustOnnxCollector（推理不经过 GIL）；
    #     若 wheel 未启用 onnx+pyo3 绑定，则回退到 Python onnxruntime 推理。
    MODEL_BACKEND: str
    ONNX_PATH: str            # ONNX 模型路径（相对 python/ 目录，MODEL_BACKEND="onnx" 时使用）
    ONNX_PROVIDERS: str       # onnxruntime 执行提供者，逗号分隔（如 "CUDAExecutionProvider,CPUExecutionProvider"）
    INFER_DEVICE: str
    TRAIN_DEVICE: str
    INFER_CPU_AUX_WORKERS: int
    INFER_CPU_FRACTION: float
    INFER_MIN_SPLIT_BATCH: int
    DATA_QUEUE_MAXSIZE: int
    ARCHIVE_QUEUE_MAXSIZE: int
    CHECKPOINT_EVERY_N_ROUNDS: int
    CKPT_SAVE_EVERY: int            # 完整 checkpoint (.pth) 落盘间隔轮次
    CKPT_EXPORT_EVERY: int          # TorchScript (.pt) / ONNX (.onnx) 导出间隔轮次
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
    # ============ 血量差异价值头（离散分类，可选） ============
    # 网络新增第三头预测终局整型血量差分布（K=2*INITIAL_HEALTH+1 分桶）。
    # HEALTH_VALUE_HEAD_ENABLED=false 时模型结构与旧版逐位等价。
    HEALTH_VALUE_HEAD_ENABLED: bool       # 是否启用血量差异分类头
    HEALTH_LOSS_WEIGHT: float             # 训练时血量头交叉熵 loss 权重 α
    HEALTH_UTILITY_WEIGHT: float          # MCTS 复合效用中血量期望权重 λ（0=禁用血量影响搜索，P3 使用）
    HEALTH_UTILITY_CONFIDENCE_EXP: float  # λ 随 |v_win| 的自适应幂指数；0=常量 λ（P3 使用）
    # ============ 对战评估（TensorBoard eval/*，通用） ============
    # 周期性把当前模型与规则对手（启发式 MCTS / minimax）及上一轮模型对弈，
    # 记录 eval/* 指标到 TensorBoard。EVAL_MATCH_ROUNDS=0 时关闭。
    EVAL_MATCH_ROUNDS: int            # 对战评估周期（训练轮，0=关闭）
    EVAL_MATCH_GAMES: int             # 每周期对弈局数（交替先后手）
    EVAL_MATCH_OPPONENTS: str         # 对手列表，逗号分隔：heuristic64 / minimax3
    EVAL_MATCH_VS_PREV: bool          # 是否与上一轮训练后模型对头（守门）
    # ============ 训练模式（TRAIN_MODE 分流） ============
    # TRAIN_MODE: 标准模型自对弈闭环 / 归档训练 / 纯规则自对弈训练
    #   - "selfplay"   : 默认。模型 MCTS 自对弈生成数据 + 训练（现有闭环）
    #   - "archive"    : 仅从冷存储归档数据训练，不启动模型自对弈
    #   - "rule_selfplay": 用纯规则（minimax/heuristic）自对弈生成数据训练，不依赖模型
    TRAIN_MODE: str
    # ---- 归档训练（TRAIN_MODE="archive"）----
    ARCHIVE_TRAIN_DIR: str            # 归档数据目录；空=自动探测 variant.archive_dir
    ARCHIVE_TRAIN_GAMES: int          # 从归档加载多少局用于训练（0=全部）
    ARCHIVE_TRAIN_ROUNDS: int         # 归档训练总轮数
    # ---- 纯规则自对弈训练（TRAIN_MODE="rule_selfplay"）----
    RULE_SELFPLAY_TYPE: str           # 规则类型：minimax | heuristic
    RULE_SELFPLAY_DEPTH: int          # minimax 搜索深度（仅 RULE_SELFPLAY_TYPE="minimax"）
    RULE_SELFPLAY_SIMS: int           # 启发式 MCTS 模拟数（仅 RULE_SELFPLAY_TYPE="heuristic"）
    RULE_SELFPLAY_GAMES: int          # 每轮生成局数
    RULE_SELFPLAY_ROUNDS: int         # 纯规则自对弈训练总轮数
    RULE_SELFPLAY_CONCURRENCY: int    # 纯规则自对弈生成并发 worker 数（线程或进程，见 BACKEND）
    RULE_SELFPLAY_TEMPERATURE: float  # 走子温度（温度越高越随机；0=贪心）
    RULE_SELFPLAY_BACKEND: str        # 并发后端：thread（多线程，默认）| process（多进程 spawn）

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
            setattr(c, name, _resolve_env(name, value))
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

