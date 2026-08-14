# config.py — 统一配置（无 CLI 参数）
#
# 所有自对弈 / 训练 / 归档 / 推理的全局配置集中于此。
# 可通过环境变量覆盖部分路径，其余为纯常量。
import os


class Config:
    # =========================================================================
    # 推理端（Predictor / predictor_entry）预测分块 batch
    # =========================================================================
    # Rust 侧按 envs.len() 一次性传入，这里分块送模型，避免大 batch 显存/内存峰值。
    PREDICT_BATCH = 32

    # =========================================================================
    # 自对弈（self_play）
    # =========================================================================
    MCTS_SIMS = 64
    MAX_CONSIDERED_ACTIONS = 16
    TEMPERATURE_STEPS = 12
    GAMES_PER_ITER = 100
    NUM_WORKERS = 1
    GAMES_PER_WORKER = 1

    # =========================================================================
    # 训练（training_service）
    # =========================================================================
    TRAIN_BATCH = 32
    LEARNING_RATE = 2e-4
    MIN_LR = 1e-6
    LR_DECAY_STEPS = 5000
    TRAIN_EPOCHS_PER_ROUND = 3
    MAX_SAMPLE_BUFFER_SIZE = 50000
    MIN_SAMPLES_TO_START = 2000
    # 每次从数据队列批量取出的局数
    QUEUE_FETCH_BATCH = 8

    # 验证集配置
    VAL_SPLIT = 0.1
    VAL_BUFFER_CAPACITY = 5000
    VAL_EVAL_MIN_BATCHES = 10

    # 模型文件
    MODEL_PATH = "banqi_model_latest.pt"      # TorchScript（供 Rust 推理）
    STATE_DICT_PATH = "banqi_model_latest.pth"  # state_dict（供训练恢复）

    # =========================================================================
    # 队列 / 线程
    # =========================================================================
    DATA_QUEUE_MAXSIZE = 64        # 数据队列上限（episode 数）
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
