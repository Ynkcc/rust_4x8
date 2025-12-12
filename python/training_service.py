import time
import subprocess
import os
import grpc
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from pymongo import MongoClient
import random

import banqi_pb2 ,banqi_pb2_grpc


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
MODEL_PATH = "banqi_model_15.ot"
BATCH_SIZE = 256
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INFERENCE_SERVER_CMD = ["python", "inference_service.py"]

def start_inference_service():
    """Starts the inference service as a subprocess."""
    print("[Training] Starting Inference Service...")
    # Using Popen to run it in background
    process = subprocess.Popen(INFERENCE_SERVER_CMD)
    time.sleep(3) # Wait for startup
    return process

def connect_inference_stub():
    channel = grpc.insecure_channel('localhost:50051')
    return banqi_pb2_grpc.InferenceServiceStub(channel)

def get_mongo_collection():
    client = MongoClient(MONGO_URI)
    return client[DB_NAME][COLLECTION_NAME]

def save_model(model):
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"[Training] Model saved to {MODEL_PATH}")

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

    # 4. Backward
    total_loss.backward()
    optimizer.step()

    return total_loss.item(), policy_loss.item(), value_loss.item()

def main():
    # 1. Setup Model
    model = BanqiNet().to(DEVICE)
    if os.path.exists(MODEL_PATH):
        try:
            model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
            print("[Training] Loaded existing model weights.")
        except:
            print("[Training] Model corrupted, starting fresh.")
            pass
    
    save_model(model) # Ensure file exists for inference service

    # 2. Start Inference Service
    inf_process = start_inference_service()
    inf_stub = connect_inference_stub()

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

                # --- Update Model & Inference Service ---
                save_model(model)
                try:
                    inf_stub.ReloadModel(banqi_pb2.Empty())
                    print("[Training] Inference service updated.")
                except grpc.RpcError as e:
                    print(f"[Training] Failed to update inference service: {e}")

            else:
                # Wait for more data
                time.sleep(2)

    except KeyboardInterrupt:
        print("[Training] Stopping...")
    finally:
        inf_process.terminate()

if __name__ == "__main__":
    main()