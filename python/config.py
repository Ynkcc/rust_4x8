# config.py — 统一配置（无 CLI 参数）
#
# 所有自对弈 / 训练 / 归档 / 推理的全局配置集中于此。
# 可通过环境变量覆盖部分路径，其余为纯常量。
import os


class Config:
    # =========================================================================
    # 推理端（Predictor / predictor_entry）预测分块 batch
    # =========================================================================
    # Rust 侧按 envs.len() 一次性传入，这里分块送模型。
    # RTX 4060 (8GB) 下 batch=128 可充分利用 GPU，模型仅 760K 参数显存宽裕。
    PREDICT_BATCH = 128

    # =========================================================================
    # 自对弈（self_play）
    # =========================================================================
    # 说明：
    #  - 4x8 暗棋一局最多 100 步（MAX_STEPS_PER_EPISODE），且机会节点（翻棋）多，
    #    因此 MCTS 模拟数与温度探索窗口都应按"长局 + 高随机"特性选取。
    #  - scheduler.step() 在每个训练 batch 后调用一次，LR_DECAY_STEPS 需按
    #    预计总 batch 数估算，不能取小，否则 LR 过早衰减到 eta_min 导致模型"石化"。
    # Gumbel AlphaZero 在小 sim 数下仍有效；128 sim 兼顾质量与吞吐（vs 基线 24 sim / 7.1 samples/s），
    # 可显著减少日志中 "MCTS 实际模拟数为 0、提前终止搜索" 的终局搜索不足现象。
    MCTS_SIMS = 128
    MAX_CONSIDERED_ACTIONS = 24
    TEMPERATURE_STEPS = 16
    GAMES_PER_ITER = 100
    # 多 worker 并行自对弈：交错 CPU MCTS 遍历与 GPU 推理，消除 CPU/GPU 交替空闲
    # 2 worker 通常可让 CPU 利用率从 <30% 提升到 60%+，吞吐近似翻倍
    NUM_WORKERS = 2
    GAMES_PER_WORKER = 50  # 总对局数 = NUM_WORKERS × GAMES_PER_WORKER = 100

    # 批量自对弈：若为 True，则改用 run_batched_self_play_with_predictor，
    # 同时驱动 BATCH_CONCURRENCY 局游戏，把多棵树的 MCTS 叶子评估合并成一个大 batch
    # 送给网络，摊薄 GPU 推理固定开销、提升吞吐。
    # 注意：当前 RTX 4060 / 760K 小模型下实测无明显提升（GPU 推理本身受限，
    # 固定开销占比小，瓶颈在 Python/numpy/GIL 数据搬运而非算力利用），
    # 故默认关闭；保留为实验特性，待换更大模型 / 推理更慢场景再评估启用。
    USE_BATCHED_SELF_PLAY = False
    BATCH_CONCURRENCY = 4        # 单线程协调器同时推进的对局数（越大单批越大）

    # =========================================================================
    # 训练（training_service）
    # =========================================================================
    TRAIN_BATCH = 64
    LEARNING_RATE = 2e-4
    MIN_LR = 5e-6
    # 每轮约 (buffer/64)*TRAIN_EPOCHS_PER_ROUND 个 batch
    LR_DECAY_STEPS = 60000
    TRAIN_EPOCHS_PER_ROUND = 3
    MAX_SAMPLE_BUFFER_SIZE = 100000
    MIN_SAMPLES_TO_START = 1000
    # 每次从数据队列批量取出的局数
    QUEUE_FETCH_BATCH = 8

    # =========================================================================
    # 训练数据对称增强（data_augmentation.py）
    # =========================================================================
    # 仅作用于训练侧 replay buffer；冷存储（MongoDB/JSONL 归档）始终保存原始数据。
    DATA_AUGMENT_ENABLED = os.getenv("DATA_AUGMENT_ENABLED", "1") != "0"  # 是否启用
    DATA_AUGMENT_KEEP_ORIGINAL = os.getenv("DATA_AUGMENT_KEEP_ORIGINAL", "1") == "1"  # 增强时同时保留原始样本
    DATA_AUGMENT_TRANSFORMS = os.getenv("DATA_AUGMENT_TRANSFORMS", "hflip,vflip,rot180")  # 候选变换（逗号分隔）

    # 模型文件
    MODEL_PATH = "banqi_model_latest.pt"      # TorchScript（供 Rust 推理）
    STATE_DICT_PATH = "banqi_model_latest.pth"  # state_dict（供训练恢复）

    # =========================================================================
    # 推理 / 训练设备分离
    # =========================================================================
    # 自对弈 MCTS 推理（Predictor）默认用 CPU，不占用 GPU 算力/显存，
    # 把 GPU 完全留给训练；训练默认 auto（CUDA 可用则用 CUDA，否则回退 CPU）。
    # 推理与训练是两个独立模型实例，通过 checkpoint 文件（.pt/.pth）同步权重。
    INFER_DEVICE = os.getenv("INFER_DEVICE", "cpu")   # 推理设备（cpu / cuda / auto）
    TRAIN_DEVICE = os.getenv("TRAIN_DEVICE", "auto")  # 训练设备（auto / cuda / cpu）

    # =========================================================================
    # GPU + CPU 混合推理（CPU 辅助推理线程）
    # =========================================================================
    # 部分设备上单个 GPU 推理线程的吞吐不足以喂饱 CPU MCTS 自对弈（推理请求
    # 在队列中堆积，CPU 利用率低）。启用后，每批推理按 INFER_CPU_FRACTION
    # 比例拆成两份，GPU 推理线程处理大头、INFER_CPU_AUX_WORKERS 个 CPU 推理
    # 线程处理小头，并行推理后合并，总吞吐 ≈ GPU 吞吐 + CPU 吞吐。
    # 注意：仅当 INFER_DEVICE 解析为 CUDA 时生效（INFER_DEVICE=cpu 时无需混合）；
    #       模型为 760K 小参数量，CPU 推理速度可观，混合后提升明显。
    INFER_CPU_AUX_WORKERS = int(os.getenv("INFER_CPU_AUX_WORKERS", "0"))   # CPU 辅助推理线程数（0=关闭）
    INFER_CPU_FRACTION = float(os.getenv("INFER_CPU_FRACTION", "0.3"))     # 每批拆分给 CPU 的比例 (0~1)
    INFER_MIN_SPLIT_BATCH = int(os.getenv("INFER_MIN_SPLIT_BATCH", "16"))  # batch 小于该值时不拆分，直接 GPU

    # =========================================================================
    # 队列 / 线程
    # =========================================================================
    DATA_QUEUE_MAXSIZE = 128       # 数据队列上限（episode 数）
    ARCHIVE_QUEUE_MAXSIZE = 256    # 归档队列上限
    CHECKPOINT_EVERY_N_ROUNDS = 2  # 每 N 轮训练导出一次 checkpoint

    # =========================================================================
    # 系统资源监控（system_monitor.py，psutil + pynvml）
    # =========================================================================
    # 训练期间周期性采样并打印 CPU / 内存 / GPU 用量；
    # 无 NVIDIA 驱动或未安装 pynvml 时 GPU 部分自动降级跳过。
    MONITOR_ENABLED = os.getenv("MONITOR_ENABLED", "1") != "0"
    MONITOR_INTERVAL = float(os.getenv("MONITOR_INTERVAL", "10.0"))  # 采样间隔（秒）
    MONITOR_PER_CORE = os.getenv("MONITOR_PER_CORE", "0") == "1"     # 显示每核 CPU
    MONITOR_CSV_PATH = os.getenv("MONITOR_CSV_PATH") or None         # CSV 落盘路径

    # =========================================================================
    # TensorBoard 训练日志（tb_logger.py）
    # =========================================================================
    # 记录 train loss、lr、自对弈吞吐（selfplay/*）、系统资源（sys/*）。
    # 查看方式: tensorboard --logdir <TENSORBOARD_LOG_DIR>
    TENSORBOARD_ENABLED = os.getenv("TENSORBOARD_ENABLED", "1") != "0"
    TENSORBOARD_LOG_DIR = os.getenv("TENSORBOARD_LOG_DIR", "runs")
    TENSORBOARD_LOG_SYS = os.getenv("TENSORBOARD_LOG_SYS", "1") == "1"  # 系统资源写入 TB

    # =========================================================================
    # MongoDB 冷存储归档（archiver）
    # =========================================================================
    MONGO_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
    DB_NAME = "banqi_training"
    COLLECTION = "games"
    ARCHIVE_BATCH = 32             # 归档批量写入大小
    ARCHIVE_POLL_INTERVAL = 1.0    # 归档线程空闲轮询间隔（秒）


# 单例，避免重复 import 各自实例
config = Config()
