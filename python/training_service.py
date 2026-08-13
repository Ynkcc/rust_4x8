import time
import os
import math
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F
import numpy as np
from pymongo import MongoClient
import random

from nn_model import BanqiNet
from constant import (
    TOTAL_INPUT_CHANNELS,
    BOARD_ROWS,
    BOARD_COLS,
    SCALAR_FEATURE_COUNT,
    ACTION_SPACE_SIZE
)

# --- Configuration ---
MONGO_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
DB_NAME = "banqi_training"
COLLECTION_NAME = "games"
META_COLLECTION = "training_meta"
MODEL_PATH = "banqi_model_latest.pt"
STATE_DICT_PATH = "banqi_model_latest.pth"
BATCH_SIZE = 512
LEARNING_RATE = 2e-4
MIN_LR = 1e-6
LR_DECAY_STEPS = 5000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Buffer 配置
MAX_SAMPLE_BUFFER_SIZE = 50000
MIN_SAMPLES_TO_START = 2000
FETCH_LIMIT = 2000
TRAIN_EPOCHS_PER_ROUND = 3

# 闭环迭代配置
CLOSED_LOOP = True
POLL_INTERVAL_SEC = 10
SAVE_EVERY_N_ROUNDS = 2

# 验证集配置
VAL_SPLIT = 0.1
VAL_BUFFER_CAPACITY = 5000
VAL_EVAL_MIN_BATCHES = 10

# MongoDB 客户端单例
_mongo_client = None
_mongo_db = None


def get_mongo_db():
    global _mongo_client, _mongo_db
    if _mongo_client is None:
        _mongo_client = MongoClient(MONGO_URI)
        _mongo_db = _mongo_client[DB_NAME]
    return _mongo_db


def get_mongo_collection():
    return get_mongo_db()[COLLECTION_NAME]


class DataBuffer:
    """向量化缓冲区，优化内存并加速 Tensor 转换"""
    def __init__(self, capacity):
        self.capacity = capacity
        self.boards = []
        self.scalars = []
        self.probs = []
        self.values = []
        self.masks = []
        self.root_visits = []

    def add_samples(self, samples):
        for s in samples:
            board = np.array(s['board_state'], dtype=np.float32).reshape(
                TOTAL_INPUT_CHANNELS, BOARD_ROWS, BOARD_COLS
            )
            self.boards.append(board)
            scalar_arr = np.array(s['scalar_state'], dtype=np.float32)
            if scalar_arr.shape[0] > SCALAR_FEATURE_COUNT:
                scalar_arr = scalar_arr[:SCALAR_FEATURE_COUNT]
            self.scalars.append(scalar_arr)
            self.probs.append(np.array(s['policy_probs'], dtype=np.float32))
            val = s.get('game_result_value', s.get('mcts_value', 0.0))
            self.values.append(val)
            self.masks.append(np.array(s['action_mask'], dtype=np.float32))
            self.root_visits.append(int(s.get('root_visit_count', 0)))

        if len(self.boards) > self.capacity:
            excess = len(self.boards) - self.capacity
            self.boards = self.boards[excess:]
            self.scalars = self.scalars[excess:]
            self.probs = self.probs[excess:]
            self.values = self.values[excess:]
            self.masks = self.masks[excess:]
            self.root_visits = self.root_visits[excess:]

    def __len__(self):
        return len(self.boards)

    def get_batch(self, indices):
        b = torch.from_numpy(np.stack([self.boards[i] for i in indices]))
        s = torch.from_numpy(np.stack([self.scalars[i] for i in indices]))
        p = torch.from_numpy(np.stack([self.probs[i] for i in indices]))
        v = torch.tensor([self.values[i] for i in indices], dtype=torch.float32)
        m = torch.from_numpy(np.stack([self.masks[i] for i in indices]))
        return b, s, p, v, m


def get_last_processed_id(db):
    meta = db[META_COLLECTION].find_one({"type": "progress"})
    return meta['last_id'] if meta else None


def save_progress(db, last_id):
    db[META_COLLECTION].update_one(
        {"type": "progress"},
        {"$set": {"last_id": last_id, "updated_at": time.time()}},
        upsert=True
    )


def save_checkpoint(model, optimizer, scheduler):
    """
    保存完整训练状态:
    1. .pth: model + optimizer + scheduler 状态（用于断点恢复训练）
    2. .pt: TorchScript（供 Rust 推理加载）
    """
    pt_temp_path = MODEL_PATH + ".tmp"
    pth_temp_path = STATE_DICT_PATH + ".tmp"

    try:
        model.eval()

        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'model_config': {
                'input_channels': TOTAL_INPUT_CHANNELS,
                'board_rows': BOARD_ROWS,
                'board_cols': BOARD_COLS,
                'scalar_features': SCALAR_FEATURE_COUNT,
                'action_space': ACTION_SPACE_SIZE
            }
        }, pth_temp_path)
        os.replace(pth_temp_path, STATE_DICT_PATH)

        with torch.no_grad():
            example_board = torch.randn(1, TOTAL_INPUT_CHANNELS, BOARD_ROWS, BOARD_COLS, device=DEVICE)
            example_scalars = torch.randn(1, SCALAR_FEATURE_COUNT, device=DEVICE)
            traced_model = torch.jit.trace(model, (example_board, example_scalars))
            traced_model.save(pt_temp_path)

        os.replace(pt_temp_path, MODEL_PATH)
        print(f"[Training] ✅ Checkpoint 保存成功: {STATE_DICT_PATH} + {MODEL_PATH}")
    except Exception as e:
        print(f"[Training] ❌ Checkpoint 保存失败: {e}")
        for tmp in [pt_temp_path, pth_temp_path]:
            if os.path.exists(tmp):
                os.remove(tmp)


def load_checkpoint(model, optimizer, scheduler):
    """
    从 .pth 恢复完整训练状态（model + optimizer + scheduler）。
    如果 .pth 不完整或缺失，回退到仅加载权重 (.pt / 全新模型)。
    """
    state_loaded = False

    if os.path.exists(STATE_DICT_PATH):
        try:
            checkpoint = torch.load(STATE_DICT_PATH, map_location=DEVICE)
            model.load_state_dict(checkpoint['model_state_dict'])
            if 'optimizer_state_dict' in checkpoint:
                try:
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                except Exception as e_opt:
                    print(f"[Training] ⚠️ Optimizer 状态加载失败 ({e_opt})，保持新初始化")
            if 'scheduler_state_dict' in checkpoint:
                try:
                    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                except Exception as e_sch:
                    print(f"[Training] ⚠️ Scheduler 状态加载失败 ({e_sch})，保持新初始化")
            print(f"[Training] ✅ 从 {STATE_DICT_PATH} 恢复完整训练状态")
            state_loaded = True
        except Exception as e:
            print(f"[Training] ⚠️ 完整 .pth 加载失败 ({e})，尝试仅加载权重...")

    if not state_loaded and os.path.exists(MODEL_PATH):
        try:
            jit_model = torch.jit.load(MODEL_PATH, map_location=DEVICE)
            model.load_state_dict(jit_model.state_dict())
            print(f"[Training] ✅ 从 {MODEL_PATH} 加载模型权重 (TorchScript 回退)")
        except Exception as e2:
            print(f"[Training] ⚠️ 权重加载失败 ({e2})，使用全新模型")

    if not state_loaded and not os.path.exists(MODEL_PATH) and not os.path.exists(STATE_DICT_PATH):
        print("[Training] 📝 初始化全新模型（无 checkpoint）")


def train_step(model, optimizer, batch_data, device):
    model.train()
    boards_t, scalars_t, target_probs_t, target_values_t, masks_t = batch_data

    boards_t = boards_t.to(device)
    scalars_t = scalars_t.to(device)
    target_probs_t = target_probs_t.to(device)
    target_values_t = target_values_t.to(device).view(-1, 1)
    masks_t = masks_t.to(device)

    optimizer.zero_grad()
    logits, values = model(boards_t, scalars_t)

    masked_logits = logits + (masks_t - 1.0) * 1e9
    log_probs = F.log_softmax(masked_logits, dim=1)
    policy_loss = -torch.sum(target_probs_t * log_probs, dim=1).mean()

    value_loss = F.mse_loss(values, target_values_t)
    total_loss = policy_loss + value_loss

    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    return total_loss.item(), policy_loss.item(), value_loss.item()


@torch.no_grad()
def evaluate(model, buffer, batch_size, device):
    model.eval()
    indices = list(range(len(buffer)))
    random.shuffle(indices)
    num_batches = len(indices) // batch_size
    if num_batches == 0:
        return None

    total_loss_sum = 0.0
    policy_loss_sum = 0.0
    value_loss_sum = 0.0

    for step in range(num_batches):
        batch_indices = indices[step * batch_size : (step + 1) * batch_size]
        boards, scalars, target_probs, target_values, masks = buffer.get_batch(batch_indices)

        boards = boards.to(device)
        scalars = scalars.to(device)
        target_probs = target_probs.to(device)
        target_values = target_values.to(device).view(-1, 1)
        masks = masks.to(device)

        logits, values = model(boards, scalars)
        masked_logits = logits + (masks - 1.0) * 1e9
        log_probs = F.log_softmax(masked_logits, dim=1)
        policy_loss = -torch.sum(target_probs * log_probs, dim=1).mean()
        value_loss = F.mse_loss(values, target_values)
        total_loss = policy_loss + value_loss

        total_loss_sum += total_loss.item()
        policy_loss_sum += policy_loss.item()
        value_loss_sum += value_loss.item()

    return (
        total_loss_sum / num_batches,
        policy_loss_sum / num_batches,
        value_loss_sum / num_batches,
    )


def run_training_epochs(model, optimizer, scheduler, buffer, num_epochs):
    """
    在完整 replay buffer 上训练指定个 epoch。
    scheduler.step() 按 batch 步进以匹配 CosineAnnealingLR 的 T_max (batch 数)。
    返回 (epoch 平均 loss 列表, 累计训练 batch 数)。
    """
    total_batches = 0
    epoch_results = []

    for epoch in range(num_epochs):
        indices = list(range(len(buffer)))
        random.shuffle(indices)
        num_batches = len(indices) // BATCH_SIZE
        if num_batches == 0:
            break

        batch_total_l, batch_pol_l, batch_val_l = 0.0, 0.0, 0.0
        for step in range(num_batches):
            batch_indices = indices[step * BATCH_SIZE : (step + 1) * BATCH_SIZE]
            batch_data = buffer.get_batch(batch_indices)
            tl, pl, vl = train_step(model, optimizer, batch_data, DEVICE)
            scheduler.step()
            batch_total_l += tl
            batch_pol_l += pl
            batch_val_l += vl
            total_batches += 1

        avg_l = batch_total_l / num_batches
        avg_p = batch_pol_l / num_batches
        avg_v = batch_val_l / num_batches
        epoch_results.append((avg_l, avg_p, avg_v))

        if num_epochs > 1:
            print(f"[Training]   Epoch {epoch+1}/{num_epochs} | {num_batches} 批次 | "
                  f"Loss: {avg_l:.4f} (Pol: {avg_p:.4f}, Val: {avg_v:.4f})")

    return epoch_results, total_batches


def main():
    print(f"[Training] Starting service on {DEVICE}")

    # 1. 模型 + 优化器 + 调度器
    model = BanqiNet().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=LR_DECAY_STEPS,
        eta_min=MIN_LR,
    )
    print(f"[Training] CosineAnnealingLR: lr_init={LEARNING_RATE:.2e}, "
          f"lr_min={MIN_LR:.2e}, T_max={LR_DECAY_STEPS} steps")

    # 2. 恢复 checkpoint（权重 + optimizer + scheduler）
    load_checkpoint(model, optimizer, scheduler)

    # 立即导出一次，确保 Rust 侧有可用的 .pt（如果模型是全新的就导出初始）
    save_checkpoint(model, optimizer, scheduler)

    # 3. 数据库连接和缓冲区
    db = get_mongo_db()
    collection = db[COLLECTION_NAME]
    buffer = DataBuffer(MAX_SAMPLE_BUFFER_SIZE)
    val_buffer = DataBuffer(VAL_BUFFER_CAPACITY)

    print(f"[Training] 🚀 开始训练（MinSamples={MIN_SAMPLES_TO_START}, "
          f"Epochs/Round={TRAIN_EPOCHS_PER_ROUND}, ClosedLoop={CLOSED_LOOP}）...")

    try:
        last_id = get_last_processed_id(db)
        if last_id is not None:
            print(f"[Training] 📍 从断点继续，last_id={last_id}")

        # 初始填充 replay buffer：取最新的 FETCH_LIMIT 局
        initial_cursor = collection.find({}).sort('_id', -1).limit(FETCH_LIMIT)
        initial_docs = list(initial_cursor)
        # 初始数据按插入顺序（升序）加入，保持 last_id 指向最大的 _id
        initial_docs_sorted = sorted(initial_docs, key=lambda d: d['_id'])
        if initial_docs_sorted:
            count_init = 0
            for doc in initial_docs_sorted:
                if 'samples' in doc and doc['samples']:
                    buffer.add_samples(doc['samples'])
                    count_init += len(doc['samples'])
            last_id = initial_docs_sorted[-1]['_id']
            print(f"[Training] 📥 初始加载 {len(initial_docs_sorted)} 局，{count_init} 样本 → Buffer={len(buffer)}")

        round_num = 0
        total_batches_trained = 0
        total_loss_sum = 0.0
        total_policy_loss_sum = 0.0
        total_value_loss_sum = 0.0

        # 4. 主循环：拉新 → 训练 → 持久化
        while True:
            query = {"_id": {"$gt": last_id}} if last_id else {}
            new_cursor = collection.find(query).sort('_id', 1).limit(FETCH_LIMIT)
            new_docs = list(new_cursor)

            if not new_docs:
                if not CLOSED_LOOP:
                    print("[Training] ✅ 离线模式：已处理完所有数据，退出")
                    break
                # 闭环模式：等待新数据，期间每 SAVE_EVERY_N_ROUNDS 间隔尝试导出一次
                last_save_round = round_num
                saved_this_wait = False
                wait_started = time.time()
                while not new_docs:
                    time.sleep(POLL_INTERVAL_SEC)
                    elapsed_rounds_equiv = int((time.time() - wait_started) / 60) + round_num
                    if elapsed_rounds_equiv - last_save_round >= SAVE_EVERY_N_ROUNDS and not saved_this_wait:
                        save_checkpoint(model, optimizer, scheduler)
                        saved_this_wait = True
                    new_cursor = collection.find(query).sort('_id', 1).limit(FETCH_LIMIT)
                    new_docs = list(new_cursor)
                continue

            # 拆分 train / val，追加到各自 buffer
            split_point = int(len(new_docs) * (1.0 - VAL_SPLIT))
            train_docs = new_docs[:split_point]
            val_docs = new_docs[split_point:]

            count_train = 0
            for doc in train_docs:
                if 'samples' in doc and doc['samples']:
                    buffer.add_samples(doc['samples'])
                    count_train += len(doc['samples'])
            count_val = 0
            for doc in val_docs:
                if 'samples' in doc and doc['samples']:
                    val_buffer.add_samples(doc['samples'])
                    count_val += len(doc['samples'])

            last_id = new_docs[-1]['_id']
            print(f"[Training] 📥 Round#{round_num} 加载 {len(new_docs)} 局 → "
                  f"train: {count_train}, val: {count_val} → Buffer={len(buffer)}")

            # 最少样本检查（既保证 BATCH_SIZE 也保证 MIN_SAMPLES_TO_START）
            min_required = max(BATCH_SIZE, MIN_SAMPLES_TO_START)
            if len(buffer) < min_required:
                print(f"[Training] ⚠️ Buffer={len(buffer)} < {min_required}，暂不训练，等待更多")
                save_progress(db, last_id)
                round_num += 1
                continue

            # 对完整 Buffer 训练多个 epoch
            epoch_results, batches_in_round = run_training_epochs(
                model, optimizer, scheduler, buffer, TRAIN_EPOCHS_PER_ROUND
            )

            total_batches_trained += batches_in_round
            round_total = sum(r[0] for r in epoch_results)
            round_pol = sum(r[1] for r in epoch_results)
            round_val = sum(r[2] for r in epoch_results)
            total_loss_sum += round_total
            total_policy_loss_sum += round_pol
            total_value_loss_sum += round_val

            if epoch_results:
                last_avg_l, last_avg_p, last_avg_v = epoch_results[-1]
                cur_lr = optimizer.param_groups[0]['lr']
                print(f"[Training] ✅ Round#{round_num} 结束 | {batches_in_round} 批次 | "
                      f"Loss: {last_avg_l:.4f} (Pol: {last_avg_p:.4f}, Val: {last_avg_v:.4f}) "
                      f"| lr={cur_lr:.2e}")

            # 验证集评估
            min_val_samples = BATCH_SIZE * VAL_EVAL_MIN_BATCHES
            if len(val_buffer) >= min_val_samples:
                val_result = evaluate(model, val_buffer, BATCH_SIZE, DEVICE)
                if val_result is not None:
                    vl, vp, vv = val_result
                    train_ref = epoch_results[-1][0] if epoch_results else 0.0
                    flag = " ⚠️ 过拟合?" if vl > train_ref + 0.1 else ""
                    print(f"[Training] 📊 验证集: Loss={vl:.4f} (Pol: {vp:.4f}, Val: {vv:.4f}){flag}")

            # 持久化进度 + 定时 checkpoint
            save_progress(db, last_id)
            round_num += 1

            if round_num % SAVE_EVERY_N_ROUNDS == 0:
                save_checkpoint(model, optimizer, scheduler)

        # 循环结束 → 输出总体统计
        if total_batches_trained > 0:
            overall_avg_loss = total_loss_sum / total_batches_trained
            overall_avg_pol = total_policy_loss_sum / total_batches_trained
            overall_avg_val = total_value_loss_sum / total_batches_trained
            print(f"\n[Training] ✅ 训练完成！总计 {total_batches_trained} 批次")
            print(f"[Training] 平均 Loss: {overall_avg_loss:.4f} "
                  f"(Pol: {overall_avg_pol:.4f}, Val: {overall_avg_val:.4f})")
        else:
            print("[Training] ⚠️ 本轮运行未执行足够训练批次")

        save_checkpoint(model, optimizer, scheduler)
        print("[Training] 🎉 最终 Checkpoint 已保存")

    except KeyboardInterrupt:
        print("[Training] Stopping (KeyboardInterrupt)...")
        save_checkpoint(model, optimizer, scheduler)
    except Exception as e:
        import traceback
        print(f"[Training] ❌ Error: {e}")
        traceback.print_exc()
        save_checkpoint(model, optimizer, scheduler)


if __name__ == "__main__":
    main()
