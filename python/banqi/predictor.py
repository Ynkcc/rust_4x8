"""
banqi/predictor.py — PyO3 独立 Rust bin (banqi-py-collector) 使用的神经网络推理入口（公共）。

由原 python/predictor_entry.py 的推理部分拆分而来，参数化为任意变体，
供 4x2 / 4x4 / 4x8 的 Rust 数据采集 bin 复用。推理核心复用
banqi.self_play.Predictor（热重载 + PREDICT_BATCH 分块 + GPU/CPU 混合推理）。

Rust bin 会：
1. 以 Python 嵌入模式启动
2. import 本模块
3. 调用 `predict(board, scalars)` 做神经网络推理（内部按 PREDICT_BATCH 分块）
4. 可选调用 `save_episodes(episode_dicts)` 保存整局记录

环境变量 (由 Rust bin 读取)：
    PY_PREDICTOR_MODULE = ./python/banqi/predictor.py   (默认)
    PY_PREDICT_FUNC    = predict                         (默认)
    PY_SAVE_FUNC       = save_episodes                   (默认)
    VARIANT_ID         = 4x8                             (默认，可选 4x4 / 4x2)
    MODEL_PATH         = 模型权重路径（缺省用全新初始化网络）
    OUTPUT_DIR         = ./training_data/py_collected    (默认，若没有 PY_SAVE_FUNC 则写入 JSON 到这里)
    MCTS_SIMS          = 64
    GAMES_PER_ITERATION = 100
    WORKER_ID          = CLI argv[1]
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List, Tuple

import numpy as np

# Rust 嵌入加载场景：bin 只把模块父目录与 cwd 加入 sys.path，而本模块要
# import banqi 包（python/ 下的包）。这里把 python/ 目录补进 sys.path，
# 使无论 cwd 在何处都能正常加载；作为库被正常 import 时该 append 无害。
_PY_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PY_DIR not in sys.path:
    sys.path.insert(0, _PY_DIR)

from banqi.self_play import build_predictor  # noqa: E402
from banqi.storage import to_json_safe  # noqa: E402
from banqi.variant import get_variant  # noqa: E402

VARIANT_ID = os.environ.get("VARIANT_ID", "4x8")
INFER_DEVICE = os.environ.get("INFER_DEVICE", "auto")


# ---------------------------------------------------------------------------
# 全局模型实例（带简易热重载，见 banqi.self_play.Predictor）
# ---------------------------------------------------------------------------

_PREDICTOR = None
_PREDICTOR_DEVICE = None


def _ensure_predictor():
    """惰性构建全局 Predictor（首次 predict 时）。"""
    global _PREDICTOR, _PREDICTOR_DEVICE
    if _PREDICTOR is None:
        _PREDICTOR, _PREDICTOR_DEVICE = build_predictor(
            get_variant(VARIANT_ID),
            os.environ.get("MODEL_PATH") or None,
            INFER_DEVICE,
        )
        print(f"[banqi.predictor] 变体={VARIANT_ID} 推理设备={_PREDICTOR_DEVICE}")
    return _PREDICTOR


# ---------------------------------------------------------------------------
# Rust 回调：预测接口 (输入 numpy，返回 numpy；内部按 PREDICT_BATCH 分块)
# ---------------------------------------------------------------------------

def predict(board: np.ndarray, scalars: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rust MCTS 每次评估都会调用这里。
    参数:
        board:  (N, BOARD_CHANNELS, BOARD_ROWS, BOARD_COLS) float32
        scalars:(N, SCALAR_FEATURE_COUNT) float32
    返回:
        policy_logits: (N, ACTION_SPACE_SIZE) float32
        values:        (N,) float32

    注意：Rust 侧 envs.len() 可能任意，Predictor 内部按 PREDICT_BATCH 分块
    送入模型再拼接，避免一次性推理过大 batch 导致显存/内存峰值。
    """
    return _ensure_predictor()(board, scalars)


# ---------------------------------------------------------------------------
# Rust 回调（可选）：把一局或多局完整数据交给 Python 处理/保存
# ---------------------------------------------------------------------------

def save_episodes(episodes: List[Dict[str, Any]]) -> None:
    """
    episodes: 每局一个 dict, 包含:
        game_length, winner, iteration, worker_id,
        boards, scalars, policies, mcts_values, completed_qs,
        root_visits, game_results, action_masks
    默认实现：追加写 jsonl；若要存到数据库 / 训练样本池，可在此修改。
    """
    out_dir = os.environ.get("OUTPUT_DIR", "./training_data/py_collected")
    os.makedirs(out_dir, exist_ok=True)

    worker_id = episodes[0].get("worker_id", 0) if episodes else 0
    iteration = episodes[0].get("iteration", 0) if episodes else 0
    jsonl_path = os.path.join(
        out_dir, f"iter_{iteration:06d}_worker_{worker_id:03d}.jsonl"
    )
    with open(jsonl_path, "a", encoding="utf-8") as fp:
        for ep in episodes:
            fp.write(json.dumps(to_json_safe(ep), ensure_ascii=False))
            fp.write("\n")

    print(
        f"[banqi.predictor] append {len(episodes)} episodes -> {jsonl_path}"
        f" (total now {_count_lines(jsonl_path)} lines)"
    )


def _count_lines(path: str) -> int:
    try:
        with open(path, "rb") as fp:
            return sum(1 for _ in fp)
    except OSError:
        return 0


if __name__ == "__main__":
    from banqi.constants import build_constants

    C = build_constants(get_variant(VARIANT_ID))
    bs = 4
    dummy_board = np.random.randn(
        bs, C.TOTAL_INPUT_CHANNELS, C.BOARD_ROWS, C.BOARD_COLS
    ).astype(np.float32)
    dummy_scalars = np.random.randn(bs, C.SCALAR_FEATURE_COUNT).astype(np.float32)
    pl, vl = predict(dummy_board, dummy_scalars)
    print("predict() output shapes:", pl.shape, vl.shape)
    print("expected:               ", (bs, C.ACTION_SPACE_SIZE), (bs,))
    assert pl.shape == (bs, C.ACTION_SPACE_SIZE)
    assert vl.shape == (bs,)
    print("OK")
