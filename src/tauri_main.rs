// code_files/src/tauri_main.rs
use banqi_4x8::ai::{Policy, RandomPolicy, RevealFirstPolicy};
#[cfg(feature = "torch")]
use banqi_4x8::ai::{MctsDlPolicy, ModelWrapper};
use banqi_4x8::*;
use serde::{Deserialize, Serialize}; // Added Deserialize
use std::collections::HashMap;
use std::sync::Mutex;
#[cfg(feature = "torch")]
use std::sync::Arc;
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
    PvP,         // 本地双人
    Random,      // 随机对手
    RevealFirst, // 优先翻棋
    MctsDL,      // MCTS + 深度学习
}

// 应用状态：包含游戏环境和当前对手设置
struct AppState {
    game: Mutex<DarkChessEnv>,
    opponent_type: Mutex<OpponentType>,
    // MCTS 配置
    mcts_num_simulations: Mutex<usize>,
    // 已加载的模型（可在选择 MctsDL 时构建策略）
    #[cfg(feature = "torch")]
    model: Mutex<Option<Arc<ModelWrapper>>>,
    // MCTS+DL 策略
    #[cfg(feature = "torch")]
    mcts_policy: Mutex<Option<MctsDlPolicy>>,
}

// Tauri 命令：重置游戏
#[tauri::command]
fn reset_game(opponent: Option<String>, state: State<AppState>) -> GameState {
    let mut game = state.game.lock().unwrap();
    let mut opp_type_lock = state.opponent_type.lock().unwrap();

    // 设置对手类型
    *opp_type_lock = match opponent.as_deref() {
        Some("Random") => OpponentType::Random,
        Some("RevealFirst") => OpponentType::RevealFirst,
        Some("MctsDL") => OpponentType::MctsDL,
        _ => OpponentType::PvP,
    };

    game.reset();

    // 若选择 MctsDL 且已有模型，创建策略实例
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
    
    extract_game_state(&*game)
}

// Tauri 命令：执行动作
#[tauri::command]
fn step_game(action: usize, state: State<AppState>) -> Result<StepResult, String> {
    let mut game = state.game.lock().unwrap();

    match game.step(action, None) {
        Ok((_obs, _reward, terminated, truncated, winner)) => {
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
fn bot_move(state: State<AppState>) -> Result<StepResult, String> {
    let mut game = state.game.lock().unwrap();
    let opp_type = *state.opponent_type.lock().unwrap();

    // 如果处于 PvP，提示前端无需调用 AI
    if opp_type == OpponentType::PvP {
        return Err("当前为本地双人模式，无需 AI 行动".to_string());
    }

    #[allow(unused_variables)]
    let opp_type_ref = opp_type; // 用于后续可能的扩展

    // 调用策略模块选择动作
    let chosen_action = match opp_type {
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
        },
        #[cfg(not(feature = "torch"))]
        OpponentType::MctsDL => return Err("MctsDL 需要启用 torch 特性".into()),
        OpponentType::PvP => None,    // 已在上面返回 Err，这里兜底
    }
    .ok_or_else(|| "AI 无棋可走".to_string())?;

    match game.step(chosen_action, None) {
        Ok((_obs, _reward, terminated, truncated, winner)) => {
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
    path: String 
}

/// 列出当前目录下的 .pt 模型
#[tauri::command]
fn list_models() -> Vec<ModelEntry> {
    let mut out = Vec::new();
    if let Ok(entries) = std::fs::read_dir(".") {
        for e in entries.flatten() {
            if let Ok(ft) = e.file_type() {
                if ft.is_file() {
                    if let Some(ext) = e.path().extension() {
                        if ext == "pt" {
                            let path = e.path();
                            let name = path.file_name().unwrap_or_default().to_string_lossy().to_string();
                            out.push(ModelEntry { 
                                name, 
                                path: path.to_string_lossy().to_string() 
                            });
                        }
                    }
                }
            }
        }
    }
    out.sort_by(|a, b| a.name.cmp(&b.name));
    out
}

/// 载入模型（.pt TorchScript 模型）
#[cfg(feature = "torch")]
#[tauri::command]
fn load_model(path: String, state: State<AppState>) -> Result<String, String> {
    let wrapper = ModelWrapper::load_from_file(&path)?;
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

/// 无 torch 特性时的占位函数
#[cfg(not(feature = "torch"))]
#[tauri::command]
fn load_model(_path: String, _state: State<AppState>) -> Result<String, String> {
    Err("需要启用 torch 特性才能加载模型".into())
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
    
    Ok(*sims)
}

pub fn run() {
    tauri::Builder::default()
        .setup(|app| {
            // 初始化游戏环境和状态
            let env = DarkChessEnv::new();
            app.manage(AppState {
                game: Mutex::new(env),
                opponent_type: Mutex::new(OpponentType::PvP),
                mcts_num_simulations: Mutex::new(200),
                #[cfg(feature = "torch")]
                model: Mutex::new(None),
                #[cfg(feature = "torch")]
                mcts_policy: Mutex::new(None),
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
            set_mcts_iterations
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}

fn main() {
    run();
}
