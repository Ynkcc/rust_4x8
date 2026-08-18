"""banqi/trainer_cli/__main__.py — 支持 `python -m banqi.trainer_cli [variant]`。"""

import sys

from .cli import parse_args, programmatic_entry


if __name__ == "__main__":
    # 支持 `python -m banqi.trainer_cli 4x8 [--train-mode ...]`，
    # 把位置参数 variant 透传给 argparse。
    argv = sys.argv[1:]
    if argv and not argv[0].startswith("-"):
        programmatic_entry(argv)
    else:
        programmatic_entry(argv)
