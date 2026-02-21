import torch
import torch.optim as optim
import torch.nn.functional as F

from nn_model import BanqiNet
from constant import (
    TOTAL_INPUT_CHANNELS,
    BOARD_ROWS,
    BOARD_COLS,
    SCALAR_FEATURE_COUNT,
    ACTION_SPACE_SIZE
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    # Policy Loss (Cross Entropy with Mask)
    masked_logits = logits + (masks_t - 1.0) * 1e9
    log_probs = F.log_softmax(masked_logits, dim=1)
    policy_loss = -torch.sum(target_probs_t * log_probs, dim=1).mean()

    # Value Loss (MSE)
    value_loss = F.mse_loss(values, target_values_t)

    total_loss = policy_loss + value_loss

    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    return total_loss.item(), policy_loss.item(), value_loss.item()


def run_overfit_test():
    print(f"Running Overfitting Test on {DEVICE}...")
    
    # 初始化模型，使用稍大的学习率加速收敛以验证过拟合能力
    model = BanqiNet().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    batch_size = 32

    # 1. 伪造随机的训练输入
    boards = torch.randn(batch_size, TOTAL_INPUT_CHANNELS, BOARD_ROWS, BOARD_COLS)
    scalars = torch.randn(batch_size, SCALAR_FEATURE_COUNT)
    
    # 2. 模拟 MCTS 目标概率与合法操作掩码 MASK
    target_probs = torch.zeros(batch_size, ACTION_SPACE_SIZE)
    masks = torch.zeros(batch_size, ACTION_SPACE_SIZE)
    
    for i in range(batch_size):
        # 假设每个样本只选择 1 个合法动作 (One-Hot)
        valid_actions = torch.randperm(ACTION_SPACE_SIZE)[:10]
        masks[i, valid_actions] = 1.0
        
        # 只给其中 1 个动作分配 1.0 的概率，其他 9 个合法动作分配 0 取消熵底限
        target_action = valid_actions[0]
        target_probs[i, target_action] = 1.0
        
    # 3. 真实游戏结果值：随机给 -1, 0, 或 1
    target_values = torch.randint(low=-1, high=2, size=(batch_size,), dtype=torch.float32)

    batch_data = (boards, scalars, target_probs, target_values, masks)

    epochs = 500
    print(f"\nStarting {epochs} epochs of overfitting on a single batch of {batch_size} samples...")
    print("Expected: Total Loss, Policy Loss and Value Loss should rapidly converge towards a very small number (close to 0).")

    for epoch in range(1, epochs + 1):
        total_l, pol_l, val_l = train_step(model, optimizer, batch_data, DEVICE)
        
        if epoch % 50 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d} - Total Loss: {total_l:.4f} "
                  f"| Policy Loss: {pol_l:.4f} | Value Loss: {val_l:.4f}")

if __name__ == "__main__":
    run_overfit_test()
