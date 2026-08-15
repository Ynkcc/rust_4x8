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
    # 注意：CosineAnnealingLR 会在 T_max 达到后周期性回升 LR，导致训练后期
    # loss 反弹。每轮 ≈ (buffer/32)*8 个 batch ≈ 1000 batch，默认 1 小时
    # （MAX_RUNTIME_SECONDS=3600）会话约 40000 batch，因此 T_max 需 ≥ 会话
    # 总 batch 数，确保整个会话只走一个衰减周期、不中途回升。
    LR_DECAY_STEPS = 40000
    TRAIN_EPOCHS_PER_ROUND = 8
    MAX_SAMPLE_BUFFER_SIZE = 4000   # 更新鲜的数据，避免旧弱模型拖累
    MIN_SAMPLES_TO_START = 128
    QUEUE_FETCH_BATCH = 8

    # =========================================================================
    # 数据增强（data_augmentation.py）
    # 4x4 方盘 D4 对称群全部 8 个空间自同构可用（比 4x8 长盘的 4 个还多），
    # 默认启用全部 7 个非恒等变换，每条样本随机选一个生成增强样本（×2）。
    # 仅作用于训练侧 replay buffer；冷存储归档始终保存原始数据。
    # =========================================================================
    DATA_AUGMENT_ENABLED = os.getenv("G4X4_DATA_AUGMENT", "1") != "0"
    DATA_AUGMENT_KEEP_ORIGINAL = os.getenv("G4X4_AUGMENT_KEEP_ORIGINAL", "1") == "1"
    DATA_AUGMENT_TRANSFORMS = os.getenv(
        "G4X4_AUGMENT_TRANSFORMS",
        "hflip,vflip,rot180,rot90,rot270,diag,anti_diag",
    )

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
    # 系统资源监控（system_monitor.py，psutil + pynvml）
    # CPU 训练下仅 CPU/内存采样有意义，GPU 部分在无驱动时自动降级跳过。
    # =========================================================================
    MONITOR_ENABLED = os.getenv("G4X4_MONITOR", "1") != "0"
    MONITOR_INTERVAL = float(os.getenv("G4X4_MONITOR_INTERVAL", "10.0"))  # 采样间隔（秒）
    MONITOR_PER_CORE = os.getenv("G4X4_MONITOR_PER_CORE", "0") == "1"     # 显示每核 CPU
    MONITOR_CSV_PATH = os.getenv("G4X4_MONITOR_CSV") or None              # CSV 落盘路径

    # =========================================================================
    # TensorBoard 训练日志（tb_logger.py）
    # 记录 train loss、lr、自对弈吞吐（selfplay/*）、系统资源（sys/*）。
    # 查看方式: tensorboard --logdir <TENSORBOARD_LOG_DIR>
    # =========================================================================
    TENSORBOARD_ENABLED = os.getenv("G4X4_TB", "1") != "0"
    TENSORBOARD_LOG_DIR = os.getenv("G4X4_TB_LOG_DIR", "runs_4x4")
    TENSORBOARD_LOG_SYS = os.getenv("G4X4_TB_LOG_SYS", "1") == "1"  # 系统资源写入 TB

    # =========================================================================
    # 冷存储归档（archiver.py）
    # MongoDB 优先，连接失败自动降级为本地 JSONL（./training_data/archive_4x4）。
    # 归档数据始终为原始数据（不应用训练侧数据增强）。
    # =========================================================================
    ARCHIVE_ENABLED = os.getenv("G4X4_ARCHIVE", "1") != "0"
    MONGO_URI = os.getenv("G4X4_MONGO_URI", os.getenv("MONGODB_URI", "mongodb://localhost:27017"))
    DB_NAME = os.getenv("G4X4_DB_NAME", "banqi_4x4")
    COLLECTION = "games"
    ARCHIVE_BATCH = 32
    ARCHIVE_POLL_INTERVAL = 1.0


# 单例
config = Config4x4()
