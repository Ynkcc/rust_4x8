import time
import subprocess
import os
import torch
import torch.optim as optim
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
MODEL_PATH = "banqi_model_latest.pt"  # Only one model file
BATCH_SIZE = 256
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Buffer 配置
MAX_SAMPLE_BUFFER_SIZE = 50000  # 最大样本缓冲区大小,防止内存无限增长


# MongoDB 客户端单例
_mongo_client = None

def get_mongo_collection():
    """获取 MongoDB 集合,复用客户端连接"""
    global _mongo_client
    if _mongo_client is None:
        _mongo_client = MongoClient(MONGO_URI)
    return _mongo_client[DB_NAME][COLLECTION_NAME]

def save_model(model):
    temp_path = MODEL_PATH + ".tmp"
    
    if isinstance(model, torch.jit.ScriptModule):
        model.save(temp_path)
    else:
        # 导出 TorchScript (供 Rust tch-rs 加载推理)
        model.eval()
        with torch.no_grad():
            # 创建示例输入
            example_board = torch.randn(1, TOTAL_INPUT_CHANNELS, BOARD_ROWS, BOARD_COLS, device=DEVICE)
            example_scalars = torch.randn(1, SCALAR_FEATURE_COUNT, device=DEVICE)
            
            # Trace 模型
            traced_model = torch.jit.trace(model, (example_board, example_scalars))
            traced_model.save(temp_path)
    
    # Atomic rename
    os.replace(temp_path, MODEL_PATH)
    
    print(f"[Training] Model saved: {MODEL_PATH}")

def train_step(model, optimizer, batch_samples):
    model.train()
    
    # 1. Prepare Tensors
    boards = []
    scalars_list = []
    target_probs = []
    target_values = []
    masks_list = []

    for sample in batch_samples:
        # sample structure matches what's saved in MongoDB
        # {'board_state': [...], 'scalar_state': [...], 'policy_probs': [...], 'mcts_value': f, 'action_mask': [...]}
        boards.append(sample['board_state'])
        scalars_list.append(sample['scalar_state'])
        target_probs.append(sample['policy_probs'])
        target_values.append(sample['mcts_value'])
        masks_list.append(sample['action_mask'])

    boards_t = torch.tensor(boards, dtype=torch.float32, device=DEVICE).view(-1, TOTAL_INPUT_CHANNELS, BOARD_ROWS, BOARD_COLS)
    scalars_t = torch.tensor(scalars_list, dtype=torch.float32, device=DEVICE).view(-1, SCALAR_FEATURE_COUNT)
    target_probs_t = torch.tensor(target_probs, dtype=torch.float32, device=DEVICE)
    target_values_t = torch.tensor(target_values, dtype=torch.float32, device=DEVICE).view(-1, 1)
    masks_t = torch.tensor(masks_list, dtype=torch.float32, device=DEVICE)

    # 2. Forward
    optimizer.zero_grad()
    logits, values = model(boards_t, scalars_t)

    # 3. Loss Calculation
    
    # Policy Loss: Cross Entropy with Masks
    # Apply masks to logits (set invalid to -inf)
    masked_logits = logits + (masks_t - 1.0) * 1e9
    log_probs = F.log_softmax(masked_logits, dim=1)
    # loss = - sum(target * log_prob)
    policy_loss = -torch.sum(target_probs_t * log_probs, dim=1).mean()

    # Value Loss: MSE
    value_loss = F.mse_loss(values, target_values_t)

    total_loss = policy_loss + value_loss

    # 4. Backward with Gradient Clipping
    total_loss.backward()
    
    # 梯度裁剪: 防止梯度爆炸,提高训练稳定性
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    optimizer.step()

    return total_loss.item(), policy_loss.item(), value_loss.item()

def main():
    # 1. Setup Model
    if os.path.exists(MODEL_PATH):
        try:
            # Try loading as TorchScript (JIT) model first
            model = torch.jit.load(MODEL_PATH, map_location=DEVICE)
            print("[Training] Loaded existing TorchScript model.")
        except Exception as e:
            print(f"[Training] Model corrupted or incompatible: {e}")
            print("[Training] Starting with fresh model.")
            model = BanqiNet().to(DEVICE)
    else:
        model = BanqiNet().to(DEVICE)
    
    save_model(model) # Ensure file exists for inference service


    # 3. Training Loop Setup
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    collection = get_mongo_collection()
    
    sample_buffer = []
    # Use ObjectID or timestamp to track new data. Here using _id is simpler for robustness.
    last_id = None 

    print("[Training] Starting loop...")
    try:
        while True:
            # --- Fetch Data ---
            query = {}
            if last_id:
                query['_id'] = {'$gt': last_id}
            
            # Fetch sorted by insertion order
            cursor = collection.find(query).sort('_id', 1).limit(2000) 
            
            new_docs = list(cursor)
            if new_docs:
                last_id = new_docs[-1]['_id']
                count_new = 0
                for doc in new_docs:
                    # Flatten samples from GameDocument
                    if 'samples' in doc and doc['samples']:
                        sample_buffer.extend(doc['samples'])
                        count_new += len(doc['samples'])
                print(f"[Training] Fetched {len(new_docs)} games, {count_new} samples. Buffer: {len(sample_buffer)}")
            
            # --- Check Batch Availability ---
            if len(sample_buffer) >= BATCH_SIZE:
                # Calculate number of full batches
                num_batches = len(sample_buffer) // BATCH_SIZE
                num_samples_to_use = num_batches * BATCH_SIZE
                
                print(f"[Training] Starting training on {num_samples_to_use} samples ({num_batches} batches)...")
                
                # Extract and Shuffle
                train_data = sample_buffer[:num_samples_to_use]
                random.shuffle(train_data)
                
                # Save remainder for next time
                sample_buffer = sample_buffer[num_samples_to_use:]
                
                # 限制 buffer 大小,防止内存无限增长
                if len(sample_buffer) > MAX_SAMPLE_BUFFER_SIZE:
                    excess = len(sample_buffer) - MAX_SAMPLE_BUFFER_SIZE
                    sample_buffer = sample_buffer[excess:]
                    print(f"[Training] Buffer size limited: removed {excess} oldest samples")
                
                # --- Train ---
                total_l, pol_l, val_l = 0, 0, 0
                for i in range(0, num_samples_to_use, BATCH_SIZE):
                    batch = train_data[i : i + BATCH_SIZE]
                    tl, pl, vl = train_step(model, optimizer, batch)
                    total_l += tl
                    pol_l += pl
                    val_l += vl
                
                # Log averages
                print(f"[Training] Finished. Avg Loss: {total_l/num_batches:.4f} (P: {pol_l/num_batches:.4f}, V: {val_l/num_batches:.4f})")



            else:
                # Wait for more data
                time.sleep(2)

    except KeyboardInterrupt:
        print("[Training] Stopping...")


if __name__ == "__main__":
    main()