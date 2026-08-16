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
    # 吞吐间平衡。实测（CPU 32 核）：sims 48→256 吞吐几乎不变（瓶颈在 Python
    # 推理开销而非树搜索），因此用高 sims 换取高质量 Gumbel 训练目标。
    MCTS_SIMS = int(os.getenv("G4X4_MCTS_SIMS", "256"))
    MAX_CONSIDERED_ACTIONS = 16
    TEMPERATURE_STEPS = 12     # 4x4 局更长（~25 步），前 ~一半步数保持探索
    GAMES_PER_ITER = int(os.getenv("G4X4_GAMES_PER_ITER", "40"))
    NUM_WORKERS = 4            # CPU 12 核并行自对弈
    GAMES_PER_WORKER = 10      # 总对局 = NUM_WORKERS × GAMES_PER_WORKER = 40

    USE_BATCHED_SELF_PLAY = False
    BATCH_CONCURRENCY = 4

    # =========================================================================
    # 训练
    # =========================================================================
    TRAIN_BATCH = 32
    # 精化用较低 LR（5e-4），避免在数据量不足时大幅改动权重导致过拟合/退化。
    # 冷启动（buffer 全空）时用 lr=2e-3 训练量已够；一旦从强先验精化必须小步长。
    LEARNING_RATE = float(os.getenv("G4X4_LR", "5e-4"))
    MIN_LR = 1e-5
    # 注意：CosineAnnealingLR 会在 T_max 达到后周期性回升 LR，导致训练后期
    # loss 反弹。T_max 设很大，确保整个会话只走一个衰减周期、不中途回升。
    LR_DECAY_STEPS = 300000
    # 关键修复：每轮只训练 1-2 epoch。此前 8 epoch 导致"训练量 >> 数据量"
    # （每轮仅 ~1120 新样本，却训练 ~1000 batch × 8 轮），每个样本被训练
    # ~100 次，灾难性过拟合近期自对弈分布 → 棋力退化（35%→10%）。
    TRAIN_EPOCHS_PER_ROUND = int(os.getenv("G4X4_EPOCHS_PER_ROUND", "2"))
    # 扩大 buffer 至 16000：配合冷存储预填充保留历史多样性，避免每轮被冲刷。
    MAX_SAMPLE_BUFFER_SIZE = int(os.getenv("G4X4_BUFFER_SIZE", "16000"))
    MIN_SAMPLES_TO_START = 128
    QUEUE_FETCH_BATCH = 8

    # 冷存储预填充：训练启动时从归档加载最近 N 局复用（0=关闭）。
    # 默认 400 局（约 1.4 万样本），保证训练一开始就有多样化历史数据。
    ARCHIVE_PREFILL_GAMES = int(os.getenv("G4X4_ARCHIVE_PREFILL", "400"))
    ARCHIVE_PREFILL_DIR = os.getenv("G4X4_ARCHIVE_PREFILL_DIR", "")  # 空=自动选择归档目录

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
