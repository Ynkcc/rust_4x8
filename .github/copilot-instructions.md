# Copilot instructions for banqi_4x8 (4×8 Dark Chess)

Short, focused guidance to help an AI coding assistant be productive in this repo. Keep recommendations code-aware and safe.

## Quick Architecture
- game rules & bitboards: `src/game_env.rs` (32 positions, 4x8 grid, bitboard optimizations)
- policies & search: `src/mcts.rs` (two-phase expansions; chance nodes enumerate all reveal outcomes)
- AI wrappers & policies: `src/ai/` (`mcts_dl.rs`, `random.rs`, `reveal_first.rs`)
- neural net: `src/nn_model.rs` (torch/tch model; ACTION_SIZE=352, BOARD_H=4, BOARD_W=8)
- inference server: `src/inference.rs` (batching, `ChannelEvaluator`, `InferenceServer`)
- training loop & pipelines: `src/parallel_train.rs`, `src/train.rs`, `src/training.rs`
- Tauri/UI bridge: `src/tauri_main.rs` + `frontend/*`

## Critical Data Shapes & Conventions
- Board: `BOARD_ROWS=4`, `BOARD_COLS=8`. Total positions = 32.
- State stack: `STATE_STACK_SIZE` (set in `src/game_env.rs`) is usually 1. If you change it, update all `.view` calls (nn_model, inference, training, DB serialization).
- Board channels: `BOARD_CHANNELS = 2*NUM_PIECE_TYPES + 2` (16 channels: own pieces, opp pieces, hidden, empty).
- Scalar features: `SCALAR_FEATURE_COUNT = 3 + 2*SURVIVAL_VECTOR_SIZE + ACTION_SPACE_SIZE` (242 total: move counter, red/black HP, survival vectors, action mask).
- Action mapping: `action_lookup_tables()` (builds `action_to_coords` & `coords_to_action`) in `src/game_env.rs`. Any change to the action space must update these tables and frontend helpers (`get_move_action` in `src/tauri_main.rs` and `frontend/main_tauri.js`).
- Action counts: `REVEAL_ACTIONS_COUNT = 32`, `REGULAR_MOVE_ACTIONS_COUNT = 104`, `CANNON_ATTACK_ACTIONS_COUNT = 216`, `ACTION_SPACE_SIZE = 352`.

## Important Patterns & Safety Rules
- Do not remove the header comment at the top of `src/mcts.rs`. It exists as a sentinel and enforces chance-node semantics.
- MCTS always uses two-phase expansion for chance nodes — expand every reveal outcome. Do not collapse or short-circuit this; the tree assumes full enumeration.
- `ModelWrapper::gate` (in `ai/mcts_dl.rs`) is a Mutex used to serialize forward passes. Respect it when adding parallel or async calls.
- `hidden_pieces` in the environment is the canonical bag — cloning the env is required before mutating `hidden_pieces` to avoid desyncs with the MCTS tree.

## Training & Inference Notes
- There are multiple training binaries and pipelines; pick the right one for your task: `banqi-parallel-train` (parallel workers + batching), `banqi-train` (single process, DB-backed), `banqi-lr-finder`.
- Inference batching is in `src/inference.rs`. The server expects specific tensor shapes: [batch, 16, 4, 8] and masks of length 352.
- Replay storage: `src/database.rs` & `src/train.rs` serialize board/scalar/probs/masks to SQLite. Use little-endian float bytes; maintain schema compatibility.

## UI & Bridge (Tauri)
- Commands are registered in `invoke_handler![]` in `src/tauri_main.rs`. For any new command, add it to the Tauri handler and the frontend calls in `frontend/main_tauri.js`.
- If you change action encoding or coords, update the frontend `get_move_action` helper and UI grid (`frontend/styles.css`) to match.

## Developer Workflows & Checks
- Quick run targets:
  - GUI: `cargo run --bin banqi-tauri` (requires Tauri dev toolchain)
  - CLI demo: `cargo run --bin banqi_4x8`
  - Parallel training: `cargo run --bin banqi-parallel-train [optional_model.ot]`
  - Data-driven training: `cargo run --bin banqi-train`
  - Smoke verify a model: `cargo run --bin banqi-verify-trained -- banqi_model_latest.ot`
  - Tests & debug: `cargo run --bin test-observation` or `cargo run --bin banqi-verify`/`banqi-verify-samples`
- Quick sanity check after edits: run `cargo run --bin banqi_4x8` and confirm initial prints: `obs.board.shape()` and `obs.scalars.shape()` match expected dimensions and `ACTION_SPACE_SIZE`.

## Migration Checklist (3×4 → 4×8 / ACTION_SIZE changes)
1. Update `BOARD_ROWS`, `BOARD_COLS`, `ACTION_SPACE_SIZE` and `SCALAR_FEATURE_COUNT` in `src/game_env.rs`.
2. Recompute `action_lookup_tables()` to ensure `coords_to_action` & `action_to_coords` are consistent.
3. Update `nn_model.rs` input shapes (BOARD_H, BOARD_W, ACTION_SIZE), `BanqiNet` forward shapes, and `SCALAR_FEATURE_COUNT`. Keep hyperparameters under review.
4. Update `inference.rs`, `training.rs`, and `parallel_train.rs` to use the new mask/action sizes and tensor `view(...)` shapes.
5. Update DB serialization/deserialization helpers in `src/database.rs`/`save_samples_to_db()`/`load_samples_from_db()` to adjust board/scalar byte lengths.
6. Update frontend `get_move_action`, grid rendering, and the `project_reveal_probabilities` projection if necessary.

## Quick examples
- Add an action: modify `build_action_lookup_tables` in `src/game_env.rs` (push to `action_to_coords` and insert into `coords_to_action`) and ensure `ACTION_SPACE_SIZE` constant reflects new size.
- Batch inference: `InferenceServer` uses `view([batch, C, H, W])` — change `C`, `H`, `W` in `inference.rs` and `nn_model.rs` in lockstep.
- Writing model code: Use `ModelWrapper::load_from_file` + `MctsDlPolicy::new(model, &env, sims)` to re-attach AI agents in the UI.

## Gotchas & Warnings
- `STATE_STACK_SIZE` = 1 by default; changing it affects tensor dims across many files. Update all `.view()` calls.
- Keep `game_env.rs` bitboard optimizations; avoid adding heap-heavy operations there — it's performance sensitive.
- MCTS chance nodes rely on the `hidden_pieces` bag; mutating that vector elsewhere (e.g., debugging reveals) can desync the tree. Always clone the env before tampering with hidden data.
- UI/tauri commands for loading models are currently commented out — if you enable them, add command registration and JS requests accordingly.

## When stuck
- Trace flow: `DarkChessEnv::get_state` → `Evaluator` (`ai` or `tch`) → `MCTS` → `training/inference`.
- Use `cargo run --bin banqi-4x8` to spot shape/mask mismatches. Check `obs.board.shape()` & `obs.scalars.shape()` prints; mismatches indicate a shape drift.