"""banqi — 4x2 / 4x4 / 4x8 暗棋共享训练组件包。

分层（自底向上）：
    声明源   : variant（变体唯一声明）/ actions / config
    派生     : constants（由 Variant 派生全部维度常量）
    模型     : nn_model（策略-价值网络）/ nnue（NNUE 评估网络 + 导出/训练）
    推理     : selfplay/predictor（Predictor 体系）/ predictor（Rust bin 嵌入入口）
    自对弈   : selfplay/{worker,config} / rule_teacher
    训练     : training/{buffer,losses,eval,worker}
    编排     : trainer_cli/{cli,config_resolver,runners}
    基础设施 : checkpoint / storage / archiver / tb_logger / system_monitor / memory_guard

使用：
    from banqi.variant import get_variant, VARIANTS
    from banqi.constants import build_constants, verify_against_bindings
    from banqi.nn_model import BanqiNet
"""

from __future__ import annotations

from typing import Any

__version__ = "0.1.0"

_LAZY: dict[str, Any] = {
    "Variant": ("banqi.variant", "Variant"),
    "VARIANTS": ("banqi.variant", "VARIANTS"),
    "get_variant": ("banqi.variant", "get_variant"),
}


def __getattr__(name: str) -> Any:
    """延迟暴露子模块符号，避免 eager import 副作用。"""
    if name in _LAZY:
        mod_name, attr = _LAZY[name]
        import importlib

        mod = importlib.import_module(mod_name)
        value = getattr(mod, attr)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["__version__", "Variant", "VARIANTS", "get_variant"]
