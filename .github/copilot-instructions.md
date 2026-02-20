# Copilot instructions for banqi_4x8 (4×8 Dark Chess)

Short, focused guidance for this repo. **Follow these rules strictly.**

## 🚨 Critical Rules (Mandatory)
1.  **Language**: All code comments and documentation MUST be in **Chinese**. NO exceptions.
2.  **MCTS Sensitivity**:
    *   **Do NOT remove** the header warning in `src/mcts/search.rs`.
    *   **Do NOT modify** the full expansion logic for chance nodes (reveal actions).
    *   **Do NOT remove** the player perspective flip logic (`value_from_perspective`).
3.  **Model Loading**: Rust uses `tch-rs` to load TorchScript models (`.pt`). Do NOT use `.ot` or other formats in Rust inference code.
4.  **Bitboards**: Use bitwise operations for board logic in `src/game_env/`. Avoid object allocations in hot paths.

## 🏗️ Architecture: Hybrid Rust + Python

This project uses a hybrid architecture for AlphaZero training:

### 1. Data Generation (Rust)
*   **High Performance**: Rust handles self-play and MCTS.
*   **Entry Point**: `src/data_collector.rs`.
*   **Components**:
    *   `src/mcts/`: **Gumbel AlphaZero MCTS** implementation (Top-K sampling in `search.rs`, Config in `config.rs`).
    *   `src/self_play.rs`: Manages game loops and data collection.
    *   `src/game_env/`: The core game logic (4x8 board, bitboards).
*   **Model**: Loads `banqi_model_traced.pt` via `tch::CModule`.
*   **Storage**: Saves game episodes to **MongoDB** (`banqi_training.games`).

### 2. Model Training (Python)
*   **Flexible Training**: Python handles the neural network training loop.
*   **Entry Point**: `python/training_service.py`.
*   **Components**:
    *   `python/nn_model.py`: PyTorch model definition (`BanqiNet`).
    *   `python/constant.py`: Shared constants (Features, Action Space).
*   **Workflow**:
    1.  Fetch games from MongoDB.
    2.  Train `BanqiNet` on batches.
    3.  Save weights (`.ot`) AND **trace** to TorchScript (`banqi_model_traced.pt`).
    4.  The Rust collectors automatically reload the new `.pt` model (on restart or signal).

## 📂 Key Files & Locations

| Component | Path | Description |
| :--- | :--- | :--- |
| **Env** | `src/game_env/` | 4x8 Board, Bitboards, Rules. `ACTION_SPACE_SIZE=352`. |
| **MCTS** | `src/mcts/search.rs` | Gumbel MCTS. **Touch with care.** |
| **Collector** | `src/data_collector.rs`| Main binary for self-play. Connects to MongoDB. |
| **Evaluator** | `src/local_evaluator.rs`| `LocalEvaluator` implements `Evaluator` trait using `tch-rs`. |
| **Training** | `python/training_service.py`| Training loop, Loss calculation, Model export. |
| **Model** | `python/nn_model.py` | PyTorch Network Architecture. |
| **UI** | `src/tauri_main.rs` | Tauri backend (if active). |

## 🔢 Data Shapes & Constants

*   **Board**: 4 rows × 8 cols × 32 pieces.
*   **Input Tensor**: `[Batch, 16, 4, 8]` (16 channels: Self/Opp/Hidden/Empty).
*   **Scalar Features**: `242` floats.
*   **Action Space**: `352` (Reveal=32, Move=104, Capture=216).
*   **Masks**: `[Batch, 352]` (1=Valid, 0=Invalid).

## ⚠️ Common Pitfalls

*   **Shape Mismatch**: When editing `nn_model.py`, ensure `forward` output matches `tch-rs` expectation in `LocalEvaluator::evaluate`.
*   **MongoDB Schema**: Rust serializes samples as flat lists; Python expects specific field names (`board_state`, `scalar_state`, `policy_probs`, `mcts_value`, `action_mask`).
*   **Device Handling**: `tch-rs` needs explicit device moving (`to_device`). Python does too.
*   **Blocking**: MCTS is synchronous. Do not introduce async unless refactoring the whole stack.

## 🛠️ Development Workflow

1.  **Modify Model**: Edit `python/nn_model.py`.
2.  **Train/Trace**: Run `python/training_service.py` -> generates `banqi_model_traced.pt`.
3.  **Run Collector**: `cargo run --release --bin banqi-data-collector` -> loads `.pt`, pushes to Mongo.
4.  **Visualize**: `cargo run --bin banqi-tauri` (if frontend checks needed).