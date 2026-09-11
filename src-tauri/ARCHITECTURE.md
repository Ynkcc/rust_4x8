# ARCHITECTURE — banqi-tauri（Tauri 桌面端）

> **仓库规划**：本目录计划拆分为独立仓库 `banqi-tauri`（已是独立 crate / workspace member）。
> 拆分前随主仓库维护；本文与主仓库 `docs/ARCHITECTURE.md` §5 对应，以本文为准。

## 1. 定位与结构

Tauri 2 桌面 GUI：`src-tauri/` 为独立 crate `banqi-tauri`（workspace member，path 依赖 `banqi_4x8`——**拆分时需改为 git 依赖**）。

| 条目 | 说明 |
|---|---|
| `Cargo.toml` / `build.rs`（GTK/WebKit 预检 + `tauri_build::build()`）/ `tauri.conf.json` / `icons/` | crate 配套 |
| `src/main.rs` | 入口，持 `AppState`（环境 + 各引擎） |
| `frontend/` | Vite + Vue 3 + TypeScript 工程（源码 `src/`，构建产物 `dist/`；Tauri 加载 `frontendDist=./frontend/dist`，dev 经 devUrl http://localhost:5173） |

## 2. 前端

- `src/api/types.ts` + `src/api/client.ts`：command 返回类型定义与 `invoke` 封装（`api.*`）；
- `src/composables/`：`useGame`（对局状态/走子/高亮/日志/机器人回合）、`useSettings`（变体/对手/模型列表与参数）、`useMctsTree`（搜索树懒加载缓存 + 布局）、`useToast`、`useLogs`；
- `src/components/`：`App`（三栏布局 + 抽屉）、`BoardView`、`ControlPanel`、`StatusPanel`、`LogPanel`、`PieceTray`、`BitboardPanel`、`MctsTreePanel`、`ToastHost`；`src/domain/pieces.ts` 棋子/位板常量。

## 3. `#[tauri::command]` 列表

- 对局：`reset_game`、`step_game`、`bot_move`、`get_game_state`、`get_move_action`、`get_opponent_type`
- 模型：`list_models`、`load_model`
- 引擎参数：`set_minimax_depth`、`set_mcts_iterations`、`set_engine_budget`、`set_heuristic_sims`、`set_nnue_depth`、`set_nnue_budget`
- MCTS 树可视化（懒加载）：`mcts_get_root`、`mcts_get_children`、`mcts_get_node_detail`、`mcts_search`。MctsDL/MctsOnnx 落子后整棵 `MctsArena<DarkChessEnv>` 常驻 `AppState.mcts_tree`，前端按需逐节点拉取子边渲染（SVG 树面板，机会节点 outcome 亦懒展开）

## 4. 构建/开发

- Rust：主仓库内 `cargo build -p banqi-tauri`（本目录内直接 `cargo build` 亦可；torch/onnx 对手需 `-p banqi-tauri --features torch,onnx`）；
- 前端（`frontend/` 内）：`npm run dev`（Vite，端口 5173）、`npm run build`（vue-tsc + vite build → `dist/`）。

## 5. 变更记录

- 2026-09-11：从主仓库 `docs/ARCHITECTURE.md` §5 拆出，作为未来独立仓库的架构文档。
