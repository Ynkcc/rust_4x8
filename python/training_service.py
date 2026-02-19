import time
import os
import torch
import torch.optim as optim
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
LEARNING_RATE = 2e-4        # 略微调整学习率
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Buffer 配置
MAX_SAMPLE_BUFFER_SIZE = 50000  # 增加缓冲区容量
MIN_SAMPLES_TO_START = 2000
FETCH_LIMIT = 500                # 每次拉取的游戏数
MAX_STEPS_PER_ROUND = 100        # 每轮最大训练步数

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

    def add_samples(self, samples):
        """批量添加样本到缓冲区"""
        for s in samples:
            # 确保 board_state 是正确的 4D 形状: [Channels, Rows, Cols]
            board = np.array(s['board_state'], dtype=np.float32).reshape(
                TOTAL_INPUT_CHANNELS, BOARD_ROWS, BOARD_COLS
            )
            self.boards.append(board)
            self.scalars.append(np.array(s['scalar_state'], dtype=np.float32))
            self.probs.append(np.array(s['policy_probs'], dtype=np.float32))
            # 优先使用真实游戏结果
            val = s.get('game_result_value', s.get('mcts_value', 0.0))
            self.values.append(val)
            self.masks.append(np.array(s['action_mask'], dtype=np.float32))
        
        # FIFO 淘汰旧数据
        if len(self.boards) > self.capacity:
            excess = len(self.boards) - self.capacity
            self.boards = self.boards[excess:]
            self.scalars = self.scalars[excess:]
            self.probs = self.probs[excess:]
            self.values = self.values[excess:]
            self.masks = self.masks[excess:]

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

    # 2. 优化器
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 3. 数据库连接和缓冲区
    db = get_mongo_db()
    collection = db[COLLECTION_NAME]
    buffer = DataBuffer(MAX_SAMPLE_BUFFER_SIZE)
    
    print(f"[Training] 🚀 开始训练...")
    
    try:
        last_id = None
        total_batches_trained = 0
        total_loss_sum = 0.0
        total_policy_loss_sum = 0.0
        total_value_loss_sum = 0.0
        
        # --- 分批加载和训练 ---
        while True:
            # 清空缓冲区，准备加载新一批数据
            buffer.boards.clear()
            buffer.scalars.clear()
            buffer.probs.clear()
            buffer.values.clear()
            buffer.masks.clear()
            
            # 从数据库加载一批游戏
            query = {"_id": {"$gt": last_id}} if last_id else {}
            cursor = collection.find(query).sort('_id', 1).limit(FETCH_LIMIT)
            new_docs = list(cursor)
            
            if not new_docs:
                break  # 没有更多数据，训练结束
            
            # 将游戏样本加载到缓冲区
            count_new_samples = 0
            for doc in new_docs:
                if 'samples' in doc and doc['samples']:
                    buffer.add_samples(doc['samples'])
                    count_new_samples += len(doc['samples'])
            
            last_id = new_docs[-1]['_id']
            print(f"[Training] 📥 加载 {len(new_docs)} 局游戏，{count_new_samples} 个样本")
            
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
                print(f"[Training] 训练 {num_batches} 批次 - Loss: {avg_l:.4f} (Pol: {avg_p:.4f}, Val: {avg_v:.4f})")
        
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