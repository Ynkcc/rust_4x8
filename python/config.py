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
    # Gumbel AlphaZero 在小 sim 数下仍有效；64 sim 兼顾质量与吞吐（vs 基线 24 sim / 7.1 samples/s）
    MCTS_SIMS = 64
    MAX_CONSIDERED_ACTIONS = 16
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

    # 验证集配置（固定留出）
    # VAL_SIZE: 从最早到达的自对弈数据中固定留出的验证样本数，填满后不再追加、
    #           永不进入训练 buffer、也永不被滚动窗口覆盖（真正的 held-out）。
    VAL_SIZE = 2000
    VAL_EVAL_MIN_BATCHES = 10

    # 模型文件
    MODEL_PATH = "banqi_model_latest.pt"      # TorchScript（供 Rust 推理）
    STATE_DICT_PATH = "banqi_model_latest.pth"  # state_dict（供训练恢复）

    # =========================================================================
    # 队列 / 线程
    # =========================================================================
    DATA_QUEUE_MAXSIZE = 128       # 数据队列上限（episode 数）
    ARCHIVE_QUEUE_MAXSIZE = 256    # 归档队列上限
    CHECKPOINT_EVERY_N_ROUNDS = 2  # 每 N 轮训练导出一次 checkpoint

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
