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

# 引入你的模型定义
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
META_COLLECTION = "training_meta"              # 用于持久化训练进度
MODEL_PATH = "banqi_model_latest.pt"           # TorchScript 模型，供 Rust 加载
STATE_DICT_PATH = "banqi_model_latest.pth"     # State Dict，供 Python 训练
BATCH_SIZE = 512            # 适当增大 Batch Size 以稳定梯度
LEARNING_RATE = 2e-4        # 初始学习率
MIN_LR = 1e-6               # Cosine 退火下限
LR_DECAY_STEPS = 5000       # Cosine 退火总步数（步数，不是 epoch）
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Buffer 配置
MAX_SAMPLE_BUFFER_SIZE = 50000  # Replay Buffer 容量（超出后 FIFO 淘汰最旧样本）
MIN_SAMPLES_TO_START = 2000
FETCH_LIMIT = 2000               # 每次拉取的游戏数（增大以快速填充 Buffer）
TRAIN_EPOCHS_PER_ROUND = 3       # 每轮对完整 Buffer 训练的 epoch 数

# 闭环迭代配置
CLOSED_LOOP = True               # True: 无新数据时等待；False: 训练完退出
POLL_INTERVAL_SEC = 10           # 等待新数据的轮询间隔
SAVE_EVERY_N_ROUNDS = 2          # 每训练多少轮重新导出模型（让 Rust 端加载）

# 验证集配置
VAL_SPLIT = 0.1                  # 每次 fetch 最新数据中的 10% 做验证集
VAL_BUFFER_CAPACITY = 5000       # 验证 Buffer 容量（足够做有意义的估计）
VAL_EVAL_MIN_BATCHES = 10        # 验证 Buffer 至少要有这么多 batch 才评估

# MongoDB 客户端单例
_mongo_client = None
_mongo_db = None

def get_mongo_db():
    """获取 MongoDB 数据库连接"""
    global _mongo_client, _mongo_db
    if _mongo_client is None:
        _mongo_client = MongoClient(MONGO_URI)
        _mongo_db = _mongo_client[DB_NAME]
    return _mongo_db

def get_mongo_collection():
    """获取 MongoDB 集合,复用客户端连接"""
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
        """批量添加样本到缓冲区"""
        for s in samples:
            # 确保 board_state 是正确的 4D 形状: [Channels, Rows, Cols]
            board = np.array(s['board_state'], dtype=np.float32).reshape(
                TOTAL_INPUT_CHANNELS, BOARD_ROWS, BOARD_COLS
            )
            self.boards.append(board)
            scalar_arr = np.array(s['scalar_state'], dtype=np.float32)
            # 向后兼容: 旧数据库里 scalar 末尾拼了 352 维 action_mask，
            # 现在 SCALAR_FEATURE_COUNT 已经不包含 mask 了，截断即可。
            if scalar_arr.shape[0] > SCALAR_FEATURE_COUNT:
                scalar_arr = scalar_arr[:SCALAR_FEATURE_COUNT]
            self.scalars.append(scalar_arr)
            self.probs.append(np.array(s['policy_probs'], dtype=np.float32))
            # 优先使用真实游戏结果
            val = s.get('game_result_value', s.get('mcts_value', 0.0))
            self.values.append(val)
            self.masks.append(np.array(s['action_mask'], dtype=np.float32))
            self.root_visits.append(int(s.get('root_visit_count', 0)))
        
        # FIFO 淘汰旧数据
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
        """快速提取批次数据并构建 Tensor"""
        b = torch.from_numpy(np.stack([self.boards[i] for i in indices]))
        s = torch.from_numpy(np.stack([self.scalars[i] for i in indices]))
        p = torch.from_numpy(np.stack([self.probs[i] for i in indices]))
        v = torch.tensor([self.values[i] for i in indices], dtype=torch.float32)
        m = torch.from_numpy(np.stack([self.masks[i] for i in indices]))
        return b, s, p, v, m

def get_last_processed_id(db):
    """从数据库获取上次训练到的游戏 ID"""
    meta = db[META_COLLECTION].find_one({"type": "progress"})
    return meta['last_id'] if meta else None

def save_progress(db, last_id):
    """持久化训练进度到数据库"""
    db[META_COLLECTION].update_one(
        {"type": "progress"},
        {"$set": {"last_id": last_id, "updated_at": time.time()}},
        upsert=True
    )

def save_model(model):
    """
    保存模型为两种格式：
    1. .pth (state_dict) - 用于 Python 训练恢复
    2. .pt (TorchScript) - 供 Rust 推理加载
    """
    pt_temp_path = MODEL_PATH + ".tmp"
    pth_temp_path = STATE_DICT_PATH + ".tmp"
    
    try:
        model.eval()
        
        # 1. 保存 State Dict (.pth)
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_config': {
                'input_channels': TOTAL_INPUT_CHANNELS,
                'board_rows': BOARD_ROWS,
                'board_cols': BOARD_COLS,
                'scalar_features': SCALAR_FEATURE_COUNT,
                'action_space': ACTION_SPACE_SIZE
            }
        }, pth_temp_path)
        os.replace(pth_temp_path, STATE_DICT_PATH)
        
        # 2. 保存 TorchScript (.pt)
        with torch.no_grad():
            # 创建示例输入用于 Tracing
            # 必须与 Rust 端 tensor 维度完全一致: 
            # Board: [1, 16, 4, 8], Scalars: [1, 242]
            example_board = torch.randn(1, TOTAL_INPUT_CHANNELS, BOARD_ROWS, BOARD_COLS, device=DEVICE)
            example_scalars = torch.randn(1, SCALAR_FEATURE_COUNT, device=DEVICE)
            
            # 使用 Trace 导出 TorchScript
            traced_model = torch.jit.trace(model, (example_board, example_scalars))
            traced_model.save(pt_temp_path)
            
        # 原子性替换，防止读取到损坏文件
        os.replace(pt_temp_path, MODEL_PATH)
        
        print(f"[Training] ✅ 模型保存成功: {STATE_DICT_PATH} (训练) + {MODEL_PATH} (推理)")
    except Exception as e:
        print(f"[Training] ❌ 模型保存失败: {e}")
        # 清理临时文件
        for tmp in [pt_temp_path, pth_temp_path]:
            if os.path.exists(tmp):
                os.remove(tmp)

def train_step(model, optimizer, batch_data, device):
    """
    执行单步训练
    
    Args:
        batch_data: (boards, scalars, target_probs, target_values, masks) 元组
    
    Logic:
    1. Policy Loss: CrossEntropy(Network_Logits, Improved_Policy_Target)
       - Improved_Policy_Target 来自 Rust 端的 Gumbel 搜索结果
    2. Value Loss: MSE(Network_Value, Game_Result)
       - Game_Result 是真实胜负 (1, -1, 0)，而非 MCTS 估值
    """
    model.train()
    
    boards_t, scalars_t, target_probs_t, target_values_t, masks_t = batch_data
    
    # 搬运到设备
    boards_t = boards_t.to(device)
    scalars_t = scalars_t.to(device)
    target_probs_t = target_probs_t.to(device)
    target_values_t = target_values_t.to(device).view(-1, 1)
    masks_t = masks_t.to(device)

    # 前向传播
    optimizer.zero_grad()
    logits, values = model(boards_t, scalars_t)

    # Policy Loss (Cross Entropy with Mask)
    masked_logits = logits + (masks_t - 1.0) * 1e9
    log_probs = F.log_softmax(masked_logits, dim=1)
    policy_loss = -torch.sum(target_probs_t * log_probs, dim=1).mean()

    # Value Loss (MSE)
    value_loss = F.mse_loss(values, target_values_t)

    # 总损失
    total_loss = policy_loss + value_loss

    # 反向传播
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    return total_loss.item(), policy_loss.item(), value_loss.item()

@torch.no_grad()
def evaluate(model, buffer, batch_size, device):
    """在验证/测试 Buffer 上计算平均 Loss（不更新权重）"""
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

def main():
    print(f"[Training] Starting service on {DEVICE}")
    
    # 1. 初始化模型
    model = BanqiNet().to(DEVICE)
    
    # 优先加载 .pth (state_dict)，更适合继续训练
    if os.path.exists(STATE_DICT_PATH):
        try:
            checkpoint = torch.load(STATE_DICT_PATH, map_location=DEVICE)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"[Training] ✅ 从 {STATE_DICT_PATH} 加载模型权重")
        except Exception as e:
            print(f"[Training] ⚠️ 加载 .pth 失败 ({e})，尝试 .pt...")
            # 回退：尝试从 TorchScript 加载
            if os.path.exists(MODEL_PATH):
                try:
                    jit_model = torch.jit.load(MODEL_PATH, map_location=DEVICE)
                    model.load_state_dict(jit_model.state_dict())
                    print(f"[Training] ✅ 从 {MODEL_PATH} 加载模型权重 (TorchScript 回退)")
                except Exception as e2:
                    print(f"[Training] ⚠️ 加载失败 ({e2})，使用全新模型")
    elif os.path.exists(MODEL_PATH):
        # 只有 .pt 存在
        try:
            jit_model = torch.jit.load(MODEL_PATH, map_location=DEVICE)
            model.load_state_dict(jit_model.state_dict())
            print(f"[Training] ✅ 从 {MODEL_PATH} 加载模型权重")
        except Exception as e:
            print(f"[Training] ⚠️ 加载失败 ({e})，使用全新模型")
    else:
        print("[Training] 📝 创建全新模型")
    
    # 立即保存一次，确保 Rust 端有模型可用
    save_model(model)

    # 2. 优化器 + 学习率调度器
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=LR_DECAY_STEPS,
        eta_min=MIN_LR,
    )
    print(f"[Training] CosineAnnealingLR: lr_init={LEARNING_RATE:.2e}, "
          f"lr_min={MIN_LR:.2e}, T_max={LR_DECAY_STEPS} steps")
    
    # 3. 数据库连接和缓冲区
    db = get_mongo_db()
    collection = db[COLLECTION_NAME]
    buffer = DataBuffer(MAX_SAMPLE_BUFFER_SIZE)
    val_buffer = DataBuffer(VAL_BUFFER_CAPACITY)
    
    print(f"[Training] 🚀 开始训练...")
    
    try:
        last_id = None
        round_num = 0
        total_batches_trained = 0
        total_loss_sum = 0.0
        total_policy_loss_sum = 0.0
        total_value_loss_sum = 0.0
        
        # --- 持久化训练断点 ---
        last_id = get_last_processed_id(db)
        if last_id is not None:
            print(f"[Training] 📍 从断点继续，last_id={last_id}")

        # --- 加载已有数据填充 Replay Buffer ---
        initial_docs = list(collection.find({}).sort('_id', 1).limit(FETCH_LIMIT))
        if initial_docs:
            count_init = 0
            for doc in initial_docs:
                if 'samples' in doc and doc['samples']:
                    buffer.add_samples(doc['samples'])
                    count_init += len(doc['samples'])
            last_id = initial_docs[-1]['_id']
            print(f"[Training] 📥 初始加载 {len(initial_docs)} 局游戏，{count_init} 个样本 → Buffer={len(buffer)}")

        # --- 持续拉新并训练（Replay Buffer 不清空） ---
        while True:
            # 1. 拉取新数据（不清空 Buffer，而是追加）
            query = {"_id": {"$gt": last_id}} if last_id else {}
            cursor = collection.find(query).sort('_id', 1).limit(FETCH_LIMIT)
            new_docs = list(cursor)
            
            if not new_docs:
                if not CLOSED_LOOP:
                    break  # 离线模式：训练完退出
                # 闭环模式：等待 Rust 端生成更多自对弈数据
                last_save_round = round_num
                saved_this_wait = False
                while not new_docs:
                    time.sleep(POLL_INTERVAL_SEC)
                    # 在等待期间定期重新导出模型，让 Rust 端能加载更好的权重
                    if round_num - last_save_round >= SAVE_EVERY_N_ROUNDS and not saved_this_wait:
                        save_model(model)
                        saved_this_wait = True
                    cursor = collection.find(query).sort('_id', 1).limit(FETCH_LIMIT)
                    new_docs = list(cursor)
                continue  # 跳回顶部，重新训练新数据
            
            # 将游戏样本加载到缓冲区（分出一部分做验证集）
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
            print(f"[Training] 📥 加载 {len(new_docs)} 局游戏 → "
                  f"train: {count_train}, val: {count_val}")
            
            # 检查是否有足够数据训练
            if len(buffer) < BATCH_SIZE:
                print(f"[Training] ⚠️ 样本不足一个批次，跳过")
                continue
            
            # 训练这批数据
            indices = list(range(len(buffer)))
            random.shuffle(indices)
            
            num_batches = len(indices) // BATCH_SIZE
            batch_total_l, batch_pol_l, batch_val_l = 0.0, 0.0, 0.0
            
            for step in range(num_batches):
                batch_indices = indices[step * BATCH_SIZE : (step + 1) * BATCH_SIZE]
                batch_data = buffer.get_batch(batch_indices)
                
                tl, pl, vl = train_step(model, optimizer, batch_data, DEVICE)
                scheduler.step()
                
                batch_total_l += tl
                batch_pol_l += pl
                batch_val_l += vl
                total_batches_trained += 1
            
            # 累计损失
            total_loss_sum += batch_total_l
            total_policy_loss_sum += batch_pol_l
            total_value_loss_sum += batch_val_l
            
            # 输出这批数据的训练统计
            if num_batches > 0:
                avg_l = batch_total_l / num_batches
                avg_p = batch_pol_l / num_batches
                avg_v = batch_val_l / num_batches
                cur_lr = optimizer.param_groups[0]['lr']
                print(f"[Training] 训练 {num_batches} 批次 - "
                      f"Loss: {avg_l:.4f} (Pol: {avg_p:.4f}, Val: {avg_v:.4f}) "
                      f"| lr={cur_lr:.2e}")
                print(f"[Training] 训练 {num_batches} 批次 - Loss: {avg_l:.4f} (Pol: {avg_p:.4f}, Val: {avg_v:.4f})")

            # 4. 验证集评估
            min_val_samples = BATCH_SIZE * VAL_EVAL_MIN_BATCHES
            if len(val_buffer) >= min_val_samples:
                val_result = evaluate(model, val_buffer, BATCH_SIZE, DEVICE)
                if val_result is not None:
                    vl, vp, vv = val_result
                    flag = " ⚠️ 过拟合?" if vl > avg_l + 0.1 else ""
                    print(f"[Training] 📊 验证集: Loss={vl:.4f} "
                          f"(Pol: {vp:.4f}, Val: {vv:.4f}){flag}")
            round_num += 1

            # 闭环模式下定期导出模型，让 Rust 端能加载到最新权重
            if round_num % SAVE_EVERY_N_ROUNDS == 0:
                save_model(model)


            if new_docs:
                count_new = 0
                for doc in new_docs:
                    if 'samples' in doc and doc['samples']:
                        buffer.add_samples(doc['samples'])
                        count_new += len(doc['samples'])
                last_id = new_docs[-1]['_id']
                print(f"[Training] 📥 新增 {len(new_docs)} 局游戏 / {count_new} 样本 → Buffer={len(buffer)}")

            # 2. 检查是否有足够数据训练
            if len(buffer) < BATCH_SIZE:
                print(f"[Training] ⚠️ Buffer={len(buffer)} < BATCH_SIZE={BATCH_SIZE}，等待更多数据...")
                break

            # 3. 对完整 Buffer 训练多个 epoch
            for epoch in range(TRAIN_EPOCHS_PER_ROUND):
                indices = list(range(len(buffer)))
                random.shuffle(indices)

                num_batches = len(indices) // BATCH_SIZE
                batch_total_l, batch_pol_l, batch_val_l = 0.0, 0.0, 0.0

                for step in range(num_batches):
                    batch_indices = indices[step * BATCH_SIZE : (step + 1) * BATCH_SIZE]
                    batch_data = buffer.get_batch(batch_indices)

                    tl, pl, vl = train_step(model, optimizer, batch_data, DEVICE)

                    batch_total_l += tl
                    batch_pol_l += pl
                    batch_val_l += vl
                    total_batches_trained += 1

                total_loss_sum += batch_total_l
                total_policy_loss_sum += batch_pol_l
                total_value_loss_sum += batch_val_l

                if num_batches > 0:
                    avg_l = batch_total_l / num_batches
                    avg_p = batch_pol_l / num_batches
                    avg_v = batch_val_l / num_batches
                    print(f"[Training] Epoch {epoch+1}/{TRAIN_EPOCHS_PER_ROUND} "
                          f"| {num_batches} 批次 "
                          f"| Loss: {avg_l:.4f} (Pol: {avg_p:.4f}, Val: {avg_v:.4f})")

            # 4. 保存进度
            save_progress(db, last_id)
        
        # 输出总体训练统计
        if total_batches_trained > 0:
            overall_avg_loss = total_loss_sum / total_batches_trained
            overall_avg_pol = total_policy_loss_sum / total_batches_trained
            overall_avg_val = total_value_loss_sum / total_batches_trained
            print(f"\n[Training] ✅ 训练完成！总计 {total_batches_trained} 批次")
            print(f"[Training] 平均 Loss: {overall_avg_loss:.4f} (Pol: {overall_avg_pol:.4f}, Val: {overall_avg_val:.4f})")
        else:
            print("[Training] ⚠️ 没有足够数据进行训练")
        
        # 保存最终模型
        save_model(model)
        print("[Training] 🎉 模型已保存")

    except KeyboardInterrupt:
        print("[Training] Stopping...")
        save_model(model)
    except Exception as e:
        print(f"[Training] ❌ Error: {e}")
        save_model(model)

if __name__ == "__main__":
    main()