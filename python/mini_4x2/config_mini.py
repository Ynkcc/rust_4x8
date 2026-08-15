# config_mini.py — 4x2 迷你暗棋训练配置（CPU 快速收敛用）
#
# 目标：在约 20 分钟内让极小网络收敛（loss 下降 + 对随机基线胜率提升）。
# 针对 CPU（无 GPU）+ 极小状态空间（8 格 / 40 动作）调优：
#   - 小的 MCTS 模拟数（sims=32）+ 小的 batch（32），单局/单轮开销极低
#   - 较大初始 LR 加速收敛
#   - 关闭数据增强（4x2 对称性有限）
#   - 归档降级 JSONL（无需 mongod）
import os

# 本文件所在目录（模型文件与脚本同目录存放，便于从任意 cwd 运行）
_HERE = os.path.dirname(os.path.abspath(__file__))


class ConfigMini:
    # =========================================================================
    # 推理端（Predictor）
    # =========================================================================
    PREDICT_BATCH = 64

    # =========================================================================
    # 自对弈
    # =========================================================================
    # 提高搜索强度：MCTS sims 32→128，使自对弈数据的 policy/value 标签质量
    # 接近 minimax(depth=4) 的水平，突破「数据上限 < minimax」的结构性瓶颈。
    MCTS_SIMS = 128           # 强搜索：数据质量是网络策略上限的决定因素
    MAX_CONSIDERED_ACTIONS = 16
    TEMPERATURE_STEPS = 6
    GAMES_PER_ITER = 60       # 每轮生成的对局数
    NUM_WORKERS = 4           # CPU 12 核，用多 worker 并行自对弈摊薄推理开销
    GAMES_PER_WORKER = 15     # 总对局 = NUM_WORKERS × GAMES_PER_WORKER = 60

    # 批量自对弈（把多棵树的叶子评估合成一个大 batch）
    USE_BATCHED_SELF_PLAY = False
    BATCH_CONCURRENCY = 4

    # =========================================================================
    # 训练
    # =========================================================================
    TRAIN_BATCH = 32
    LEARNING_RATE = 2e-3
    MIN_LR = 1e-5
    # 提高训练强度：每轮更多 epochs + 更小更新鲜的 buffer。
    # 关键修复：旧 buffer(20000) 中早期弱模型数据长期占据，导致训练被拖累、
    # loss 回升；缩小到 4000 让训练更偏重新（强）数据。
    LR_DECAY_STEPS = 12000
    TRAIN_EPOCHS_PER_ROUND = 8
    MAX_SAMPLE_BUFFER_SIZE = 4000
    MIN_SAMPLES_TO_START = 256
    QUEUE_FETCH_BATCH = 8

    # =========================================================================
    # 数据增强（4x2 对称性有限，关闭）
    # =========================================================================
    DATA_AUGMENT_ENABLED = False
    DATA_AUGMENT_KEEP_ORIGINAL = True
    DATA_AUGMENT_TRANSFORMS = "hflip"

    # 模型文件（与脚本同目录，支持环境变量覆盖）
    MODEL_PATH = os.getenv("MINI_MODEL_PATH", os.path.join(_HERE, "banqi_mini_model_latest.pt"))
    STATE_DICT_PATH = os.getenv("MINI_STATE_DICT_PATH", os.path.join(_HERE, "banqi_mini_model_latest.pth"))

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
    # 目标 20 分钟内收敛，默认跑 18 分钟（含余量）。
    # =========================================================================
    MAX_RUNTIME_SECONDS = int(os.getenv("MINI_MAX_RUNTIME", str(18 * 60)))

    # =========================================================================
    # 系统监控 / TensorBoard（精简输出，避免刷屏）
    # =========================================================================
    MONITOR_ENABLED = False
    TENSORBOARD_ENABLED = False
    TENSORBOARD_LOG_DIR = "runs_mini"

    # =========================================================================
    # 归档（JSONL 降级）
    # =========================================================================
    MONGO_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
    DB_NAME = "banqi_mini"
    COLLECTION = "games"
    ARCHIVE_BATCH = 32
    ARCHIVE_POLL_INTERVAL = 1.0


# 单例
config = ConfigMini()
