"""
train.py — 向后兼容入口。

原单文件实现已拆分为 banqi/trainer_cli/ 子包（config_resolver.py / runners.py /
cli.py）。本文件仅做 re-export 与直接运行入口，避免改动现有调用方。
新增代码请直接 `from banqi.trainer_cli import main`。
"""

import sys

from banqi.trainer_cli import main, build_const, parse_args, programmatic_entry

__all__ = ["main", "build_const", "parse_args", "programmatic_entry"]


if __name__ == "__main__":
    vid = sys.argv[1] if len(sys.argv) > 1 else "4x8"
    main(vid)
