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
