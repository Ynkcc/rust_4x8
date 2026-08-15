# config.py — 4x4 暗棋训练配置（CPU 训练用）
#
# 参考 4x2 迷你（mini_4x2/config_mini.py）的成熟参数，针对 4x4 状态空间
# （16 格 / 每方 8 子 / 动作空间 112）适度放大。
import os

# 本文件所在目录（模型文件与脚本同目录存放，便于从任意 cwd 运行）
_HERE = os.path.dirname(os.path.abspath(__file__))


class Config4x4:
    # =========================================================================
    # 推理端（Predictor）
    # =========================================================================
    PREDICT_BATCH = 64

    # =========================================================================
    # 自对弈
    # =========================================================================
    # 4x4 状态空间（16 格/112 动作）比 4x2 大得多，MCTS sims 需在数据质量与
    # 吞吐间平衡：默认 48（每局约 5-10s，CPU 可接受），可用 G4X4_MCTS_SIMS 覆盖。
    MCTS_SIMS = int(os.getenv("G4X4_MCTS_SIMS", "48"))
    MAX_CONSIDERED_ACTIONS = 16
    TEMPERATURE_STEPS = 8      # 4x4 局更长，探索步数略增
    GAMES_PER_ITER = int(os.getenv("G4X4_GAMES_PER_ITER", "40"))
    NUM_WORKERS = 4            # CPU 12 核并行自对弈
    GAMES_PER_WORKER = 10      # 总对局 = NUM_WORKERS × GAMES_PER_WORKER = 40

    USE_BATCHED_SELF_PLAY = False
    BATCH_CONCURRENCY = 4

    # =========================================================================
    # 训练
    # =========================================================================
    TRAIN_BATCH = 32
    LEARNING_RATE = 2e-3
    MIN_LR = 1e-5
    LR_DECAY_STEPS = 15000
    TRAIN_EPOCHS_PER_ROUND = 8
    MAX_SAMPLE_BUFFER_SIZE = 4000   # 更新鲜的数据，避免旧弱模型拖累
    MIN_SAMPLES_TO_START = 128
    QUEUE_FETCH_BATCH = 8

    # =========================================================================
    # 数据增强（4x4 对称性有限，关闭）
    # =========================================================================
    DATA_AUGMENT_ENABLED = False
    DATA_AUGMENT_KEEP_ORIGINAL = True
    DATA_AUGMENT_TRANSFORMS = "hflip"

    # 模型文件（与脚本同目录，支持环境变量覆盖）
    MODEL_PATH = os.getenv("G4X4_MODEL_PATH", os.path.join(_HERE, "banqi4x4_model_latest.pt"))
    STATE_DICT_PATH = os.getenv("G4X4_STATE_DICT_PATH", os.path.join(_HERE, "banqi4x4_model_latest.pth"))

    # =========================================================================
    # 设备（CPU）
    # =========================================================================
    INFER_DEVICE = "cpu"
    TRAIN_DEVICE = "cpu"

    # =========================================================================
    # 队列 / 线程
    # =========================================================================
    DATA_QUEUE_MAXSIZE = 256
    ARCHIVE_QUEUE_MAXSIZE = 256
    CHECKPOINT_EVERY_N_ROUNDS = 2

    # =========================================================================
    # 运行时限（秒）：训练到该时长后自动优雅停止并落盘 checkpoint。
    # =========================================================================
    MAX_RUNTIME_SECONDS = int(os.getenv("G4X4_MAX_RUNTIME", str(60 * 60)))

    # =========================================================================
    # 系统监控（精简输出）
    # =========================================================================
    MONITOR_ENABLED = False
    TENSORBOARD_ENABLED = False
    TENSORBOARD_LOG_DIR = "runs_4x4"

    # =========================================================================
    # 归档（JSONL 降级）
    # =========================================================================
    MONGO_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
    DB_NAME = "banqi_4x4"
    COLLECTION = "games"
    ARCHIVE_BATCH = 32
    ARCHIVE_POLL_INTERVAL = 1.0


# 单例
config = Config4x4()
