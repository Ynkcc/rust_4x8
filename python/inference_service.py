import grpc
from concurrent import futures
import time
import torch
import torch.nn.functional as F
import os

# Import generated proto code
# Note: You need to generate these using grpc_tools.protoc
import banqi_pb2 ,banqi_pb2_grpc

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

class InferenceService(banqi_pb2_grpc.InferenceServiceServicer):
    def __init__(self):
        print(f"[Inference] Initializing on {DEVICE}...")
        self.model = BanqiNet().to(DEVICE)
        self.load_weights()
        self.model.eval()

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

    def Predict(self, request, context):
        # 1. Prepare Inputs
        # Convert lists to tensors and reshape
        board_data = torch.tensor(request.board, dtype=torch.float32, device=DEVICE)
        board = board_data.view(1, TOTAL_INPUT_CHANNELS, BOARD_ROWS, BOARD_COLS)
        
        scalars_data = torch.tensor(request.scalars, dtype=torch.float32, device=DEVICE)
        scalars = scalars_data.view(1, SCALAR_FEATURE_COUNT)
        
        masks_data = torch.tensor(request.masks, dtype=torch.float32, device=DEVICE)
        masks = masks_data.view(1, ACTION_SPACE_SIZE)

        # 2. Inference
        with torch.no_grad():
            logits, value = self.model(board, scalars)
            
            # 3. Apply Masking and Softmax
            # Mask logic: set invalid action logits to -inf (using -1e9 for stability)
            # Rust equivalent: let masked_logits = &logits + (&mask_tensor - 1.0) * 1e9;
            masked_logits = logits + (masks - 1.0) * 1e9
            policy = F.softmax(masked_logits, dim=1)
            
            # Extract scalar value
            value_scalar = value.item()
            policy_list = policy.squeeze().tolist()

        return banqi_pb2.PredictResponse(policy=policy_list, value=value_scalar)

    def ReloadModel(self, request, context):
        print("[Inference] Reloading model weights requested...")
        self.load_weights()
        return banqi_pb2.Status(success=True, message="Weights reloaded")

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    banqi_pb2_grpc.add_InferenceServiceServicer_to_server(InferenceService(), server)
    server.add_insecure_port('[::]:50051')
    print("[Inference] Server started on port 50051")
    server.start()
    server.wait_for_termination()

if __name__ == '__main__':
    serve()