"""banqi — 4x2 / 4x4 / 4x8 暗棋共享训练组件包。

P1 范围：variant（变体描述符）/ constants（派生常量）/ nn_model（参数化网络）/
checkpoint（通用 checkpoint）/ data_augmentation（参数化对称增强）。

后续阶段（P2）将把 self_play / training_service / storage / archiver /
tb_logger / system_monitor / train 一并并入本包。

使用：
    from banqi.variant import get_variant, VARIANTS
    from banqi.constants import build_constants, verify_against_bindings
    from banqi.nn_model import BanqiNet
    from banqi.data_augmentation import make_augmentor
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
