"""4x4 变体训练入口（薄壳）：统一闭环在 banqi.train。"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(os.path.dirname(_HERE)))  # python/（banqi 所在）

from banqi.train import main  # noqa: E402

if __name__ == "__main__":
    main("4x4")
