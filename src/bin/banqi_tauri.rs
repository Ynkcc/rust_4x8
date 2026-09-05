#[cfg(feature = "torch")]
use banqi_4x8::engine::{MctsDlPolicy, ModelWrapper};
use banqi_4x8::engine::{Policy, RandomPolicy, RevealFirstPolicy};
#[cfg(feature = "onnx")]
use banqi_4x8::inference::onnx::{OnnxMctsPolicy, OnnxModel};
use banqi_4x8::core::env::*;
use serde::{Deserialize, Serialize}; // Added Deserialize
use std::collections::HashMap;
#[cfg(any(feature = "torch", feature = "onnx"))]
use std::sync::Arc;
use std::sync::Mutex;
use tauri::{Manager, State};

// 游戏状态的可序列化版本
#[derive(Debug, Clone, Serialize)]
struct GameState {
    board: Vec<String>,
    current_player: String,
    move_counter: usize,
    total_step_counter: usize,
    dead_red: Vec<String>,
    dead_black: Vec<String>,
    hidden_red: Vec<String>,
    hidden_black: Vec<String>,
    action_masks: Vec<i32>,
    reveal_probabilities: Vec<f32>,
    bitboards: HashMap<String, Vec<bool>>,
    hp_red: i32,   // 红方血量
    hp_black: i32, // 黑方血量
    variant: String, // "dark" = 4x8 暗棋, "mini" = 4x2 迷你暗棋, "4x4" = 4x4 暗棋
}

#[derive(Debug, Clone, Serialize)]
struct StepResult {
    state: GameState,
    terminated: bool,
    truncated: bool,
    winner: Option<i32>,
}

// 对手类型枚举
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum OpponentType {
    PvP,           // 本地双人
    Random,        // 随机对手
    RevealFirst,   // 优先翻棋
    Minimax,       // 纯规则 Minimax（expectiminimax+alpha-beta，已升级：多特征评估/置换表/走子排序/静态搜索）
    Engine,        // 纯计算强引擎（αβ + Star1 + 置换表 + 迭代加深，节点预算可控）
    MctsHeuristic, // Gumbel MCTS + 纯计算启发式评估（无需 torch）
    MctsDL,        // MCTS + 深度学习（TorchScript，需 torch feature）
    MctsOnnx,      // MCTS + ONNX 深度学习（需 onnx feature，无需 libtorch）
}

// 应用状态：包含游戏环境和当前对手设置
struct AppState {
    game: Mutex<DarkChessEnv>,
    opponent_type: Mutex<OpponentType>,
    // Minimax 搜索深度（对应 verify_mini_vs_minimax.py 的 minimax(depth=4) 档位）
    minimax_depth: Mutex<usize>,
    // 强引擎节点预算
    engine_budget: Mutex<u64>,
    // 启发式 MCTS 模拟次数
    heuristic_sims: Mutex<usize>,
    // MCTS 配置
    mcts_num_simulations: Mutex<usize>,
    // 已加载的模型（可在选择 MctsDL 时构建策略）
    #[cfg(feature = "torch")]
    model: Mutex<Option<Arc<ModelWrapper>>>,
    // MCTS+DL 策略（基于 DarkChessEnv，4x8/4x4/4x2 共用）
    #[cfg(feature = "torch")]
    mcts_policy: Mutex<Option<MctsDlPolicy<DarkChessEnv>>>,
    // 已加载的 ONNX 模型（MctsOnnx 对手，无需 libtorch）
    #[cfg(feature = "onnx")]
    onnx_model: Mutex<Option<Arc<OnnxModel>>>,
    #[cfg(feature = "onnx")]
    onnx_policy: Mutex<Option<OnnxMctsPolicy<DarkChessEnv>>>,
}

// Tauri 命令：重置游戏
#[tauri::command]
fn reset_game(opponent: Option<String>, variant: Option<String>, state: State<AppState>) -> GameState {
    let mut game = state.game.lock().unwrap();
    let mut opp_type_lock = state.opponent_type.lock().unwrap();

    // 设置对手类型
    *opp_type_lock = match opponent.as_deref() {
        Some("Random") => OpponentType::Random,
        Some("RevealFirst") => OpponentType::RevealFirst,
        Some("Minimax") => OpponentType::Minimax,
        Some("Engine") => OpponentType::Engine,
        Some("MctsHeuristic") => OpponentType::MctsHeuristic,
        Some("MctsDL") => OpponentType::MctsDL,
        Some("MctsOnnx") => OpponentType::MctsOnnx,
        _ => OpponentType::PvP,
    };

    // 按变体重建环境：mini = 4x2 迷你暗棋（8 格 / 40 动作空间），
    // 4x4 = 4x4 暗棋（16 格 / 112 动作空间），
    // 其余 = 4x8 标准暗棋（32 格 / 192 动作空间）。构造器内部已 reset。
    *game = match variant.as_deref() {
        Some("mini") => DarkChessEnv::new_mini(),
        Some("4x4") => DarkChessEnv::new_4x4(),
        _ => DarkChessEnv::new(),
    };

    // 若选择 MctsDL / MctsOnnx 且已有对应模型，创建策略实例
    #[cfg(feature = "torch")]
    if *opp_type_lock == OpponentType::MctsDL {
        let model_opt = state.model.lock().unwrap().clone();
        let mut policy_lock = state.mcts_policy.lock().unwrap();
        if let Some(model) = model_opt {
            let sims = *state.mcts_num_simulations.lock().unwrap();
            *policy_lock = Some(MctsDlPolicy::new(model, &*game, sims));
        } else {
            *policy_lock = None; // 未加载模型，策略不可用
        }
    } else {
        #[cfg(feature = "torch")]
        {
            // 非 MctsDL 模式清空策略
            let mut policy_lock = state.mcts_policy.lock().unwrap();
            *policy_lock = None;
        }
    }
    #[cfg(feature = "onnx")]
    if *opp_type_lock == OpponentType::MctsOnnx {
        let model_opt = state.onnx_model.lock().unwrap().clone();
        let mut policy_lock = state.onnx_policy.lock().unwrap();
        if let Some(model) = model_opt {
            let sims = *state.mcts_num_simulations.lock().unwrap();
            *policy_lock = Some(OnnxMctsPolicy::new(model, &*game, sims));
        } else {
            *policy_lock = None; // 未加载 ONNX 模型，策略不可用
        }
    } else {
        #[cfg(feature = "onnx")]
        {
            // 非 MctsOnnx 模式清空 ONNX 策略
            let mut policy_lock = state.onnx_policy.lock().unwrap();
            *policy_lock = None;
        }
    }

    extract_game_state(&*game)
}

// Tauri 命令：执行动作
#[tauri::command]
fn step_game(action: usize, state: State<AppState>) -> Result<StepResult, String> {
    let mut game = state.game.lock().unwrap();

    match game.step(action, None) {
        Ok((_reward, terminated, truncated, winner)) => {
            let state_data = extract_game_state(&*game);
            Ok(StepResult {
                state: state_data,
                terminated,
                truncated,
                winner,
            })
        }
        Err(e) => Err(e),
    }
}

// Tauri 命令：执行 AI 动作
#[tauri::command]
async fn bot_move(state: State<'_, AppState>) -> Result<StepResult, String> {
    let opp_type = *state.opponent_type.lock().unwrap();

    // 如果处于 PvP，提示前端无需调用 AI
    if opp_type == OpponentType::PvP {
        return Err("当前为本地双人模式，无需 AI 行动".to_string());
    }

    // 调用策略模块选择动作。
    // Minimax / Engine / MctsHeuristic 为计算密集搜索，放后台线程执行避免阻塞 UI；
    // 其余策略廉价，直接同步执行。
    let snapshot = state.game.lock().unwrap().clone();
    let chosen_action = match opp_type {
        OpponentType::Minimax => {
            let depth = *state.minimax_depth.lock().unwrap();
            tauri::async_runtime::spawn_blocking(move || {
                banqi_4x8::engine::minimax::minimax_choose_action(&snapshot, depth)
            })
            .await
            .map_err(|e| format!("Minimax 搜索线程错误: {e}"))?
        }
        OpponentType::Engine => {
            let budget = *state.engine_budget.lock().unwrap();
            let cfg = banqi_4x8::core::expectimax::SearchConfig {
                node_budget: budget,
                ..Default::default()
            };
            tauri::async_runtime::spawn_blocking(move || {
                banqi_4x8::core::expectimax::search(&snapshot, &cfg).map(|r| r.action)
            })
            .await
            .map_err(|e| format!("引擎搜索线程错误: {e}"))?
        }
        OpponentType::MctsHeuristic => {
            let sims = *state.heuristic_sims.lock().unwrap();
            tauri::async_runtime::spawn_blocking(move || {
                let policy = banqi_4x8::engine::mcts_heuristic::HeuristicMctsPolicy::new(sims);
                policy.choose_action(&snapshot)
            })
            .await
            .map_err(|e| format!("启发式 MCTS 线程错误: {e}"))?
        }
        _ => {
            let game = state.game.lock().unwrap();
            match opp_type {
                OpponentType::RevealFirst => RevealFirstPolicy::choose_action(&*game),
                OpponentType::Random => RandomPolicy::choose_action(&*game),
                #[cfg(feature = "torch")]
                OpponentType::MctsDL => {
                    let mut policy_lock = state.mcts_policy.lock().unwrap();
                    if policy_lock.is_none() {
                        // 尝试基于已加载模型创建
                        let model_opt = state.model.lock().unwrap().clone();
                        if let Some(model) = model_opt {
                            let sims = *state.mcts_num_simulations.lock().unwrap();
                            *policy_lock = Some(MctsDlPolicy::new(model, &*game, sims));
                        } else {
                            return Err("未加载模型，无法执行 MCTS+DL 策略".into());
                        }
                    }
                    let policy = policy_lock.as_ref().unwrap();
                    policy.choose_action(&*game)
                }
                #[cfg(not(feature = "torch"))]
                OpponentType::MctsDL => return Err("MctsDL 需要启用 torch 特性".into()),
                #[cfg(feature = "onnx")]
                OpponentType::MctsOnnx => {
                    let mut policy_lock = state.onnx_policy.lock().unwrap();
                    if policy_lock.is_none() {
                        // 尝试基于已加载的 ONNX 模型创建
                        let model_opt = state.onnx_model.lock().unwrap().clone();
                        if let Some(model) = model_opt {
                            let sims = *state.mcts_num_simulations.lock().unwrap();
                            *policy_lock = Some(OnnxMctsPolicy::new(model, &*game, sims));
                        } else {
                            return Err("未加载 ONNX 模型，无法执行 MCTS+ONNX 策略".into());
                        }
                    }
                    let policy = policy_lock.as_ref().unwrap();
                    policy.choose_action(&*game)
                }
                #[cfg(not(feature = "onnx"))]
                OpponentType::MctsOnnx => return Err("MctsOnnx 需要启用 onnx 特性".into()),
                OpponentType::PvP => None,        // 已在上面返回 Err，这里兜底
                OpponentType::Minimax => unreachable!(),
                OpponentType::Engine => unreachable!(),
                OpponentType::MctsHeuristic => unreachable!(),
            }
        }
    }
    .ok_or_else(|| "AI 无棋可走".to_string())?;

    let mut game = state.game.lock().unwrap();
    match game.step(chosen_action, None) {
        Ok((_reward, terminated, truncated, winner)) => {
            let state_data = extract_game_state(&*game);
            Ok(StepResult {
                state: state_data,
                terminated,
                truncated,
                winner,
            })
        }
        Err(e) => Err(e),
    }
}

// Tauri 命令：获取当前状态
#[tauri::command]
fn get_game_state(state: State<AppState>) -> GameState {
    let game = state.game.lock().unwrap();
    extract_game_state(&*game)
}

// Tauri 命令：获取对手类型
#[tauri::command]
fn get_opponent_type(state: State<AppState>) -> OpponentType {
    *state.opponent_type.lock().unwrap()
}

// Tauri 命令：获取移动动作编号
#[tauri::command]
fn get_move_action(from_sq: usize, to_sq: usize, state: State<AppState>) -> Option<usize> {
    let game = state.game.lock().unwrap();
    game.get_action_for_coords(&vec![from_sq, to_sq])
}

// 辅助函数：获取棋子短名称
fn get_piece_short_name(piece: &Piece) -> String {
    let p_char = match piece.player {
        Player::Red => "R",
        Player::Black => "B",
    };
    let t_char = match piece.piece_type {
        PieceType::General => "Gen",
        PieceType::Advisor => "Adv",
        PieceType::Soldier => "Sol",
        PieceType::Cannon => "Can",
        PieceType::Horse => "Hor",
        PieceType::Chariot => "Car",
        PieceType::Elephant => "Ele",
    };
    format!("{}_{}", p_char, t_char)
}

// 辅助函数：从游戏环境中提取状态
fn extract_game_state(env: &DarkChessEnv) -> GameState {
    let board_slots = env.get_board_slots();
    let board: Vec<String> = board_slots
        .iter()
        .map(|slot| match slot {
            Slot::Empty => "Empty".to_string(),
            Slot::Hidden => "Hidden".to_string(),
            Slot::Revealed(piece) => get_piece_short_name(piece),
        })
        .collect();

    let current_player = match env.get_current_player() {
        Player::Red => "Red".to_string(),
        Player::Black => "Black".to_string(),
    };

    let dead_red: Vec<String> = env
        .get_dead_pieces(Player::Red)
        .iter()
        .map(|pt| format!("{:?}", pt))
        .collect();

    let dead_black: Vec<String> = env
        .get_dead_pieces(Player::Black)
        .iter()
        .map(|pt| format!("{:?}", pt))
        .collect();

    let hidden_red: Vec<String> = env
        .get_hidden_pieces(Player::Red)
        .iter()
        .map(|pt| format!("{:?}", pt))
        .collect();

    let hidden_black: Vec<String> = env
        .get_hidden_pieces(Player::Black)
        .iter()
        .map(|pt| format!("{:?}", pt))
        .collect();

    let action_masks = env.action_masks();
    let reveal_probabilities = project_reveal_probabilities(env.get_reveal_probabilities());
    let bitboards = env.get_bitboards();
    let hp_red = env.get_hp(Player::Red);
    let hp_black = env.get_hp(Player::Black);
    let variant = if env.config.cols == 2 {
        "mini".to_string()
    } else if env.config.cols == 4 && env.config.rows == 4 {
        "4x4".to_string()
    } else {
        "dark".to_string()
    };

    GameState {
        board,
        current_player,
        move_counter: env.get_move_counter(),
        total_step_counter: env.get_total_steps(),
        dead_red,
        dead_black,
        hidden_red,
        hidden_black,
        action_masks,
        reveal_probabilities,
        bitboards,
        hp_red,
        hp_black,
        variant,
    }
}

fn project_reveal_probabilities(raw: &[f32]) -> Vec<f32> {
    // 直接返回所有14个概率（红方7种棋子 + 黑方7种棋子）
    // 顺序: R_Sol, R_Can, R_Hor, R_Car, R_Ele, R_Adv, R_Gen, B_Sol, B_Can, B_Hor, B_Car, B_Ele, B_Adv, B_Gen
    raw.to_vec()
}

// ===================== 额外命令：模型与参数 =====================

#[derive(Debug, Clone, Serialize)]
struct ModelEntry {
    name: String,
    path: String,
}

/// 递归收集目录下的 .pt / .onnx 模型（忽略隐藏目录 / node_modules / target 等）。
fn collect_models(dir: &std::path::Path, depth: usize, out: &mut Vec<ModelEntry>) {
    if depth > 4 {
        return;
    }
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for e in entries.flatten() {
        let path = e.path();
        let Ok(ft) = e.file_type() else { continue };
        if ft.is_dir() {
            let name = e.file_name().to_string_lossy().to_string();
            if name.starts_with('.') || name == "node_modules" || name == "target" {
                continue;
            }
            collect_models(&path, depth + 1, out);
        } else if ft.is_file() {
            let is_model = path
                .extension()
                .map(|x| x == "pt" || x == "onnx")
                .unwrap_or(false);
            if is_model {
                let name = path
                    .file_name()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .to_string();
                out.push(ModelEntry {
                    name,
                    path: path.to_string_lossy().to_string(),
                });
            }
        }
    }
}

/// 列出 python/outputs 目录下的 .pt / .onnx 模型
#[tauri::command]
fn list_models() -> Vec<ModelEntry> {
    let mut out = Vec::new();
    let search_dir = std::path::Path::new("python/outputs");
    if search_dir.exists() {
        collect_models(search_dir, 0, &mut out);
    } else {
        let alt_dir = std::path::Path::new("outputs");
        if alt_dir.exists() {
            collect_models(alt_dir, 0, &mut out);
        }
    }
    out.sort_by(|a, b| a.path.cmp(&b.path));
    out
}

/// 载入模型：按扩展名分派（.pt → TorchScript（需 torch feature），.onnx → ONNX）。
#[cfg(any(feature = "torch", feature = "onnx"))]
#[tauri::command]
fn load_model(path: String, state: State<AppState>) -> Result<String, String> {
    let is_onnx = std::path::Path::new(&path)
        .extension()
        .map(|x| x == "onnx")
        .unwrap_or(false);
    if is_onnx {
        load_onnx_model_impl(&path, &state)
    } else {
        load_torch_model_impl(&path, &state)
    }
}

#[cfg(feature = "onnx")]
fn load_onnx_model_impl(path: &str, state: &State<AppState>) -> Result<String, String> {
    let model = OnnxModel::new(path, "auto").map_err(|e| format!("ONNX 模型加载失败: {e}"))?;
    let arc_model = Arc::new(model);
    {
        let mut model_lock = state.onnx_model.lock().unwrap();
        *model_lock = Some(arc_model.clone());
    }
    // 若当前为 MctsOnnx 且已有游戏，尝试重建策略
    if *state.opponent_type.lock().unwrap() == OpponentType::MctsOnnx {
        let sims = *state.mcts_num_simulations.lock().unwrap();
        let game = state.game.lock().unwrap();
        let mut pol_lock = state.onnx_policy.lock().unwrap();
        *pol_lock = Some(OnnxMctsPolicy::new(arc_model, &*game, sims));
    }
    Ok(format!("ONNX 模型已加载: {}", path))
}

#[cfg(not(feature = "onnx"))]
#[allow(dead_code)]
fn load_onnx_model_impl(path: &str, _state: &State<AppState>) -> Result<String, String> {
    Err(format!("需要启用 onnx 特性才能加载 ONNX 模型（{path}）"))
}

#[cfg(feature = "torch")]
fn load_torch_model_impl(path: &str, state: &State<AppState>) -> Result<String, String> {
    let wrapper = ModelWrapper::load_from_file(path)?;
    let arc_wrapper = Arc::new(wrapper);
    {
        let mut model_lock = state.model.lock().unwrap();
        *model_lock = Some(arc_wrapper.clone());
    }
    // 若当前为 MctsDL 且已有游戏，尝试重建策略
    if *state.opponent_type.lock().unwrap() == OpponentType::MctsDL {
        let sims = *state.mcts_num_simulations.lock().unwrap();
        let game = state.game.lock().unwrap();
        let mut pol_lock = state.mcts_policy.lock().unwrap();
        *pol_lock = Some(MctsDlPolicy::new(arc_wrapper, &*game, sims));
    }
    Ok(format!("模型已加载: {}", path))
}

#[cfg(not(feature = "torch"))]
#[allow(dead_code)]
fn load_torch_model_impl(path: &str, _state: &State<AppState>) -> Result<String, String> {
    Err(format!("需要启用 torch 特性才能加载 TorchScript 模型（{path}）"))
}

/// 无 torch / onnx 特性时的占位函数
#[cfg(not(any(feature = "torch", feature = "onnx")))]
#[tauri::command]
fn load_model(_path: String, _state: State<AppState>) -> Result<String, String> {
    Err("需要启用 torch 或 onnx 特性才能加载模型".into())
}

/// 设置 Minimax 搜索深度（默认 4，与 verify_mini_vs_minimax.py 的 minimax(depth=4) 对应）
#[tauri::command]
fn set_minimax_depth(depth: usize, state: State<AppState>) -> Result<usize, String> {
    if depth == 0 {
        return Err("搜索深度必须大于 0".into());
    }
    if depth > 10 {
        return Err("搜索深度过大（>10），可能导致搜索极慢或内存爆炸".into());
    }
    let mut d = state.minimax_depth.lock().unwrap();
    *d = depth;
    Ok(*d)
}

/// 设置 MCTS 每步搜索次数
#[tauri::command]
fn set_mcts_iterations(iters: usize, state: State<AppState>) -> Result<usize, String> {
    if iters == 0 {
        return Err("搜索次数必须大于 0".into());
    }
    let mut sims = state.mcts_num_simulations.lock().unwrap();
    *sims = iters;

    #[cfg(feature = "torch")]
    if let Some(policy) = state.mcts_policy.lock().unwrap().as_mut() {
        policy.set_iterations(iters);
    }
    #[cfg(feature = "onnx")]
    if let Some(policy) = state.onnx_policy.lock().unwrap().as_mut() {
        policy.set_iterations(iters);
    }

    Ok(*sims)
}

/// 设置纯计算强引擎（Engine）的节点预算
#[tauri::command]
fn set_engine_budget(budget: u64, state: State<AppState>) -> Result<u64, String> {
    if budget == 0 {
        return Err("节点预算必须大于 0".into());
    }
    let mut b = state.engine_budget.lock().unwrap();
    *b = budget;
    Ok(*b)
}

/// 设置启发式 MCTS（MctsHeuristic）的模拟次数
#[tauri::command]
fn set_heuristic_sims(sims: usize, state: State<AppState>) -> Result<usize, String> {
    if sims == 0 {
        return Err("模拟次数必须大于 0".into());
    }
    let mut s = state.heuristic_sims.lock().unwrap();
    *s = sims;
    Ok(*s)
}

pub fn run() {
    tauri::Builder::default()
        .setup(|app| {
            // 初始化游戏环境和状态
            let env = DarkChessEnv::new();
            app.manage(AppState {
                game: Mutex::new(env),
                opponent_type: Mutex::new(OpponentType::PvP),
                minimax_depth: Mutex::new(4),
                engine_budget: Mutex::new(300_000),
                heuristic_sims: Mutex::new(300),
                mcts_num_simulations: Mutex::new(200),
                #[cfg(feature = "torch")]
                model: Mutex::new(None),
                #[cfg(feature = "torch")]
                mcts_policy: Mutex::new(None),
                #[cfg(feature = "onnx")]
                onnx_model: Mutex::new(None),
                #[cfg(feature = "onnx")]
                onnx_policy: Mutex::new(None),
            });
            Ok(())
        })
        .plugin(tauri_plugin_shell::init())
        .invoke_handler(tauri::generate_handler![
            reset_game,
            step_game,
            bot_move,
            get_game_state,
            get_opponent_type,
            get_move_action,
            list_models,
            load_model,
            set_minimax_depth,
            set_mcts_iterations,
            set_engine_budget,
            set_heuristic_sims
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}

fn main() {
    run();
}
