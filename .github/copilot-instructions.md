# Copilot instructions for banqi_4x8 (4×8 Dark Chess)

Short, focused guidance to help an AI coding assistant be productive in this repo. Keep recommendations code-aware and safe.

## Quick Architecture
- game rules & bitboards: `src/game_env.rs` (32 positions, 4x8 grid, bitboard optimizations)
- policies & search: `src/mcts.rs` (two-phase expansions; chance nodes enumerate all reveal outcomes)
- AI wrappers & policies: `src/ai/` (`mcts_dl.rs`, `random.rs`, `reveal_first.rs`)
- neural net: `src/nn_model.rs` (torch/tch model; current code is for legacy 3×4 shape and ACTION_SIZE=46)
- inference server: `src/inference.rs` (batching, `ChannelEvaluator`, `InferenceServer`)
- training loop & pipelines: `src/parallel_train.rs`, `src/train.rs`, `src/training.rs`
- Tauri/UI bridge: `src/tauri_main.rs` + `frontend/*`

## Critical Data Shapes & Conventions
- Board: `BOARD_ROWS=4`, `BOARD_COLS=8`. Total positions = 32.
- State stack: `STATE_STACK_SIZE` (set in `src/game_env.rs`) is usually 1. If you change it, update all `.view` calls (nn_model, inference, training, DB serialization).
- Board channels: `BOARD_CHANNELS = 2*NUM_PIECE_TYPES + 2` (16 channels: own pieces, opp pieces, hidden, empty).
- Scalar features: `SCALAR_FEATURE_COUNT = 4 + 2*SURVIVAL_VECTOR_SIZE + ACTION_SPACE_SIZE` — action size affects scalar shape.
- Action mapping: `action_lookup_tables()` (builds `action_to_coords` & `coords_to_action`) in `src/game_env.rs`. Any change to the action space must update these tables and frontend helpers (`get_move_action` in `src/tauri_main.rs` and `frontend/main_tauri.js`).
- Action counts are computed as constants in `src/game_env.rs`: `REVEAL_ACTIONS_COUNT = 32`, `REGULAR_MOVE_ACTIONS_COUNT = 104`, `CANNON_ATTACK_ACTIONS_COUNT = 216`, `ACTION_SPACE_SIZE = 352`.

## Important Patterns & Safety Rules
- Do not remove the header comment at the top of `src/mcts.rs`. It exists as a sentinel and enforces chance-node semantics.
- MCTS always uses two-phase expansion for chance nodes — expand every reveal outcome. Do not collapse or short-circuit this; the tree assumes full enumeration.
- `ModelWrapper::gate` (in `ai/mcts_dl.rs`) is a Mutex used to serialize forward passes. Respect it when adding parallel or async calls.
- `hidden_pieces` in the environment is the canonical bag — cloning the env is required before mutating `hidden_pieces` to avoid desyncs with the MCTS tree.

## Training & Inference Notes
- There are multiple training binaries and pipelines; pick the right one for your task: `banqi-parallel-train` (parallel workers + batching), `banqi-train` (single process, DB-backed), `banqi-lr-finder`.
- Inference batching is in `src/inference.rs`. The server expects specific tensor shapes: currently [batch, 8, 3, 4] and masks of length 46 (legacy). If migrating to 4×8 and ACTION_SIZE=352, update shapes & mask widths here and in `nn_model.rs` and training code.
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
- `mcts.rs`'s chance node logic must be preserved; it is subtle and changes can bias search.
- UI/tauri commands for loading models are currently commented out — if you enable them, add command registration and JS requests accordingly.

## When stuck
- Trace flow: `DarkChessEnv::get_state` → `Evaluator` (`ai` or `tch`) → `MCTS` → `training/inference`.
- Use `cargo run --bin banqi-4x8` to spot shape/mask mismatches. Check `obs.board.shape()` & `obs.scalars.shape()` prints; mismatches indicate a shape drift.

If anything is unclear, point to a specific file/line and I'll expand with code examples or a small refactor change.
# Copilot instructions for banqi_4x8

## Big picture
- This crate hosts a full reinforcement-learning stack for 4×8 Dark Chess: `game_env.rs` models the rules/bitboards, `ai/` contains heuristics and MCTS+DL, `training.rs` & friends handle optimization, and `tauri_main.rs` exposes a desktop UI backed by `frontend/` assets.
- Multiple binaries exist (see `Cargo.toml`); choose the right entry point: `banqi-tauri` (UI), `banqi-4x8` (CLI demo), `banqi-parallel-train` (self-play pipeline), `banqi-train` (SQLite-driven training), plus verifier binaries for smoke tests.

## State encoding & action space
- `DarkChessEnv` builds a 32-slot board using bitboards; `Observation` packs `(STATE_STACK_SIZE=1, 16 channels, 4 rows, 8 cols)` plus a 242-length scalar (move counter, two survival vectors, step ratio, action mask snapshot).
- `action_to_coords` / `coords_to_action` map indexes ↔ board coordinates; `ACTION_SPACE_SIZE = 352` (32 reveals + 104 regular moves + 216 cannon attacks). Update both lookup maps and the frontend helper (`get_move_action`) together if you change this.
- Legacy training/inference code (`nn_model.rs`, `inference.rs`, `training.rs`) still assumes 46 actions and a 3×4 board; before editing, confirm which pipeline you are touching and keep tensor shapes consistent or stage the migration deliberately.

## Gameplay core (`src/game_env.rs`)
- Functions like `step`, `action_masks`, and `get_state` are performance-sensitive; they rely on precalculated bitboards and ray tables. Preserve inlining, avoid extra heap allocations, and update `update_reveal_probabilities` whenever hidden pool logic changes.
- Scenario helpers `setup_two_advisors` / `setup_hidden_threats` are currently no-ops (kept for compatibility). Any new scenario must reset the bitboards, histories, and probability tables, or validation will yield nonsense metrics.

## Search & policies
- `src/mcts.rs` implements chance nodes for reveal actions; keep the header warning comment and never collapse the two-phase expansion (chance node must enumerate every reveal outcome to stay unbiased).
- `ai/random.rs` & `ai/reveal_first.rs` provide baseline opponents; `ai/mcts_dl.rs` wraps `BanqiNet` with persistent MCTS tree reuse. Model access is serialized via `ModelWrapper::gate`—respect that mutex when adding async code.

## Training/inference stack
- `parallel_train.rs` starts one inference server thread plus many `SelfPlayWorker`s (usually 32) over mpsc channels; batching params are `inference_batch_size` and `batch_timeout_ms`. If you change these, also revisit GPU memory usage and worker counts.
- `inference.rs` expects `.ot` checkpoints saved by `tch`; it batches requests into `[batch, 8, 3, 4]` tensors. When migrating to the full 4×8 encoding, update `view(...)` shapes, mask widths, and the replay buffer serialization in `database.rs`.
- Replay data is persisted via SQLite (`database.rs`) or CSV logs (`training_log.rs`). Always convert floats to little-endian bytes before storage and keep schema migrations backward compatible, since older binaries read the same file.

## UI & Tauri bridge
- `src/tauri_main.rs` exposes commands (`reset_game`, `step_game`, `bot_move`, `get_move_action`, etc.) used by `frontend/main_tauri.js`; any new command must be registered in both the Rust `invoke_handler!` macro and the JS side.
- The frontend still renders a 3×4 grid and only labels generals/advisors/soldiers; when expanding piece coverage or the 4×8 layout, adjust `styles.css`, the grid template, and `getPieceText` in tandem.
- Reveals shown in the UI use `project_reveal_probabilities` to collapse 14-type odds into six buckets; keep this projection in sync with the server output to avoid mismatched tooltips.

## Developer workflows
- Build/run: `cargo run --bin banqi-tauri` for the GUI (requires Tauri tooling), `cargo run --bin banqi-4x8` for CLI simulation, `cargo run --bin banqi-parallel-train [optional_model.ot]` for self-play training, and `cargo run --bin banqi-verify-trained -- <model.ot>` to smoke-test checkpoints.
- Logs land in `runlog.txt` and `training_log.csv`; prefer appending structured info there instead of ad-hoc `println!` spam so downstream analysis scripts keep working.

## Gotchas
- `STATE_STACK_SIZE` is set to 1 everywhere; if you re-enable stacking, update tensor shapes, SQLite serialization, and any `.view([*, 8, 3, 4])` occurrences in lockstep.
- MCTS chance nodes rely on the `hidden_pieces` bag; mutating that vector elsewhere (e.g., debugging reveals) can desync the tree. Always clone the env before tampering with hidden data.
- Do not delete the sentinel comment block at the top of `mcts.rs`; it’s used as a regression guard for automated agents.
- When adding binaries or features, remember `tauri.conf.json` controls bundling—mirror environment variables or feature flags there if the UI needs them.

## When unsure
- Trace data flow: `DarkChessEnv::get_state` → evaluator (random/tch) → `MCTS` → training/inference. A break in any of these stages usually shows up as mismatched tensor shapes or zeroed action masks.
- If you need a quick sanity check, run `cargo run --bin banqi_4x8` (CLI) to watch random moves and ensure action encoding still lines up before touching training or UI layers.
