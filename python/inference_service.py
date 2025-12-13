import grpc
from concurrent import futures
import time
import torch
import torch.nn.functional as F
import os
import threading
import queue
import numpy as np

# Import generated proto code
import banqi_pb2, banqi_pb2_grpc

from nn_model import BanqiNet
from constant import (
    TOTAL_INPUT_CHANNELS,
    BOARD_ROWS,
    BOARD_COLS,
    SCALAR_FEATURE_COUNT,
    ACTION_SPACE_SIZE
)

MODEL_PATH = "banqi_model_15.ot"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Batch Configuration ---
MAX_BATCH_SIZE = 64        # 最大批次大小 (根据显存调整)
BATCH_TIMEOUT = 0.005      # 超时时间 (秒)，例如 5ms。如果在 5ms 内没凑够 Batch 也会强制推理

class InferenceService(banqi_pb2_grpc.InferenceServiceServicer):
    def __init__(self):
        print(f"[Inference] Initializing on {DEVICE}...")
        self.model = BanqiNet().to(DEVICE)
        self.load_weights()
        self.model.eval()

        # Batch Processing Queue
        self.request_queue = queue.Queue()
        self.shutdown_event = threading.Event()
        
        # Start the background worker thread
        self.worker_thread = threading.Thread(target=self._batch_worker, daemon=True)
        self.worker_thread.start()
        print(f"[Inference] Batch worker started (Batch Size: {MAX_BATCH_SIZE}, Timeout: {BATCH_TIMEOUT}s)")

    def load_weights(self):
        if os.path.exists(MODEL_PATH):
            try:
                state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
                self.model.load_state_dict(state_dict)
                print(f"[Inference] Model weights loaded from {MODEL_PATH}")
            except Exception as e:
                print(f"[Inference] Error loading weights: {e}")
        else:
            print(f"[Inference] No weights found at {MODEL_PATH}, using random initialization.")

    def _batch_worker(self):
        """
        Background loop that collects requests and runs inference in batches.
        """
        while not self.shutdown_event.is_set():
            batch_requests = []
            start_wait = time.time()

            # 1. Collect requests until MAX_BATCH_SIZE or TIMEOUT
            while len(batch_requests) < MAX_BATCH_SIZE:
                # Check how much time we have left to wait
                time_elapsed = time.time() - start_wait
                remaining_time = BATCH_TIMEOUT - time_elapsed

                if remaining_time <= 0 and len(batch_requests) > 0:
                    break # Timeout reached, process what we have

                try:
                    # If we have items, don't wait indefinitely, wait for the remaining time
                    # If batch is empty, we can wait longer (e.g., 0.1s) to avoid busy loop
                    timeout = remaining_time if len(batch_requests) > 0 else 0.05
                    req_item = self.request_queue.get(timeout=timeout)
                    batch_requests.append(req_item)
                except queue.Empty:
                    if len(batch_requests) > 0:
                        break # Queue empty but we have some items, process them
                    continue # Queue empty and no items, continue waiting

            if not batch_requests:
                continue

            # 2. Prepare Batch Tensors
            # batch_requests is a list of tuples: (request_proto, result_container_dict, completion_event)
            
            raw_boards = []
            raw_scalars = []
            raw_masks = []

            for req, _, _ in batch_requests:
                raw_boards.append(req.board)
                raw_scalars.append(req.scalars)
                raw_masks.append(req.masks)

            # Move to GPU once as a large batch (Much faster than individual moves)
            boards_t = torch.tensor(raw_boards, dtype=torch.float32, device=DEVICE).view(-1, TOTAL_INPUT_CHANNELS, BOARD_ROWS, BOARD_COLS)
            scalars_t = torch.tensor(raw_scalars, dtype=torch.float32, device=DEVICE).view(-1, SCALAR_FEATURE_COUNT)
            masks_t = torch.tensor(raw_masks, dtype=torch.float32, device=DEVICE).view(-1, ACTION_SPACE_SIZE)

            # 3. Inference
            try:
                with torch.no_grad():
                    logits, values = self.model(boards_t, scalars_t)
                    
                    # Apply Masking
                    masked_logits = logits + (masks_t - 1.0) * 1e9
                    policies = F.softmax(masked_logits, dim=1)
                    
                    # Move results back to CPU
                    policies_cpu = policies.cpu().numpy()
                    values_cpu = values.cpu().numpy()

                # 4. Distribute Results
                for i, (_, result_container, event) in enumerate(batch_requests):
                    result_container['policy'] = policies_cpu[i].tolist()
                    result_container['value'] = float(values_cpu[i])
                    event.set() # Notify the waiting Predict thread

            except Exception as e:
                print(f"[Inference] Error during batch processing: {e}")
                # Notify all waiting threads of failure (optional: handle gracefully)
                for _, result_container, event in batch_requests:
                    result_container['error'] = e
                    event.set()

    def Predict(self, request, context):
        # Create a synchronization mechanism for this specific request
        completion_event = threading.Event()
        result_container = {} # Using a dict to store the result by reference

        # Push to queue
        self.request_queue.put((request, result_container, completion_event))

        # Wait for the background worker to finish processing
        completion_event.wait()

        # Check for errors
        if 'error' in result_container:
            context.abort(grpc.StatusCode.INTERNAL, str(result_container['error']))

        return banqi_pb2.PredictResponse(
            policy=result_container['policy'], 
            value=result_container['value']
        )

    def ReloadModel(self, request, context):
        print("[Inference] Reloading model weights requested...")
        # Since model is shared with the thread, we should rely on PyTorch's internal thread safety 
        # or pause the worker. For simplicity, just loading state_dict is usually fine if atomic.
        # But to be safe, we can wait until queue is empty or just load it.
        self.load_weights()
        return banqi_pb2.Status(success=True, message="Weights reloaded")

def serve():
    # Increase max workers to allow many incoming connections waiting on the queue
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=100))
    banqi_pb2_grpc.add_InferenceServiceServicer_to_server(InferenceService(), server)
    server.add_insecure_port('[::]:50051')
    print("[Inference] Server started on port 50051")
    server.start()
    server.wait_for_termination()

if __name__ == '__main__':
    serve()