use ndarray::{Array1, Array4};
use rand::seq::SliceRandom;
use rand::thread_rng;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::fmt;
use std::sync::OnceLock;

// ==============================================================================
// --- 常量定义 ---
// ==============================================================================

/// 棋盘行数 (4)
pub const BOARD_ROWS: usize = 4;
/// 棋盘列数 (8)
pub const BOARD_COLS: usize = 8;
/// 总位置数 (32)
pub const TOTAL_POSITIONS: usize = BOARD_ROWS * BOARD_COLS;
/// 棋子种类数 (7种: 帅/将, 仕/士, 相/象, 俥/车, 傌/马, 炮/包, 兵/卒)
pub const NUM_PIECE_TYPES: usize = 7;
/// 判和步数限制：连续多少步无吃子则判和
pub const MAX_CONSECUTIVE_MOVES_FOR_DRAW: usize = 24; // 参照 README.md

/// 状态堆叠帧数 (1 表示只使用当前帧，不包含历史帧)
pub const STATE_STACK_SIZE: usize = 1; 
/// 每局游戏最大总步数限制 (防止死循环)
const MAX_STEPS_PER_EPISODE: usize = 100;
/// 初始随机翻开的棋子数量 (用于加速开局)
const INITIAL_REVEALED_PIECES: usize = 4;

/// 初始血量 (减分制，每方从60分开始，吃子扣对方分)
const INITIAL_HEALTH_POINTS: i32 = 60;

// --- 棋子数量定义（每方） ---
const SOLDIERS_COUNT: usize = 5;
const CANNONS_COUNT: usize = 2;
const HORSES_COUNT: usize = 2;
const CHARIOTS_COUNT: usize = 2;
const ELEPHANTS_COUNT: usize = 2;
const ADVISORS_COUNT: usize = 2;
const GENERALS_COUNT: usize = 1;
const TOTAL_PIECES_PER_PLAYER: usize = SOLDIERS_COUNT
    + CANNONS_COUNT
    + HORSES_COUNT
    + CHARIOTS_COUNT
    + ELEPHANTS_COUNT
    + ADVISORS_COUNT
    + GENERALS_COUNT; // 16

/// 预定义的各棋子最大数量 (用于Scalar特征编码)
const PIECE_MAX_COUNTS: [usize; NUM_PIECE_TYPES] = [
    SOLDIERS_COUNT,
    CANNONS_COUNT,
    HORSES_COUNT,
    CHARIOTS_COUNT,
    ELEPHANTS_COUNT,
    ADVISORS_COUNT,
    GENERALS_COUNT,
];

/// 存活向量大小: 包含双方所有可能的棋子
const SURVIVAL_VECTOR_SIZE: usize = TOTAL_PIECES_PER_PLAYER;

/// Scalar 特征数量: 
/// 3个全局标量 (MoveCount, RedHP, BlackHP) + 2个存活向量(各16) + 动作掩码长度
pub const SCALAR_FEATURE_COUNT: usize = 3 + 2 * SURVIVAL_VECTOR_SIZE + ACTION_SPACE_SIZE;

/// 翻棋概率表大小: 2个玩家 * 7种棋子 = 14
const REVEAL_PROBABILITY_SIZE: usize = 2 * NUM_PIECE_TYPES;

// --- 方向常量 ---
const DIRECTION_UP: usize = 0;
const DIRECTION_DOWN: usize = 1;
const DIRECTION_LEFT: usize = 2;
const DIRECTION_RIGHT: usize = 3;
const NUM_DIRECTIONS: usize = 4;

/// MSB (Most Significant Bit) 位数 (64位整数的最高位索引基数)
const U64_BITS: usize = 64;

/// 棋盘状态张量的通道数: 
/// 己方7种 + 敌方7种 + 暗子1种 + 空位1种 = 16
pub const BOARD_CHANNELS: usize = 2 * NUM_PIECE_TYPES + 2; 

// --- 动作空间定义 ---
// 动作空间总大小 = 翻棋动作 + 移动动作 + 炮击动作

/// 翻棋动作数 (32个位置均可翻)
pub const REVEAL_ACTIONS_COUNT: usize = 32;
/// 常规移动动作数：相邻格移动
/// 4x8 网格共有 52 条无向相邻边，方向化为 104 个动作
pub const REGULAR_MOVE_ACTIONS_COUNT: usize = 104;
/// 炮的攻击动作数：
/// 水平 (每行21对组合×2方向=42, 4行=168) + 垂直 (每列3对组合×2方向=6, 8列=48) = 216
pub const CANNON_ATTACK_ACTIONS_COUNT: usize = 216;
/// 总动作空间大小: 32 + 104 + 216 = 352
pub const ACTION_SPACE_SIZE: usize =
    REVEAL_ACTIONS_COUNT + REGULAR_MOVE_ACTIONS_COUNT + CANNON_ATTACK_ACTIONS_COUNT;

// ==============================================================================
// --- Bitboard 辅助函数 ---
// 使用 u64 来表示 4x8=32 个格子的状态，位运算加速逻辑判断
// ==============================================================================

/// 棋盘全掩码 (低32位为1)
const BOARD_MASK: u64 = (1u64 << TOTAL_POSITIONS) - 1;

/// 创建一个 Bitboard，仅将第 x 位置为 1
#[inline]
const fn ull(x: usize) -> u64 {
    1u64 << x
}

/// 计算末尾零的数量，等效于找到最低位1的位置 (LSB)
#[inline]
fn trailing_zeros(bb: u64) -> usize {
    bb.trailing_zeros() as usize
}

/// 计算最高有效位的位置 (MSB)
#[inline]
fn msb_index(bb: u64) -> Option<usize> {
    if bb == 0 {
        None
    } else {
        Some(U64_BITS - 1 - bb.leading_zeros() as usize)
    }
}

/// 获取 LSB 索引并从 Bitboard 中清除该位 (用于遍历)
#[inline]
fn pop_lsb(bb: &mut u64) -> usize {
    let tz = bb.trailing_zeros() as usize;
    *bb &= *bb - 1;
    tz
}

/// 生成列掩码 (File Mask)，用于边界检查
const fn file_mask(file_col: usize) -> u64 {
    let mut m: u64 = 0;
    let mut r = 0;
    while r < BOARD_ROWS {
        let sq = r * BOARD_COLS + file_col;
        m |= ull(sq);
        r += 1;
    }
    m
}

#[allow(dead_code)]
const NOT_FILE_A: u64 = BOARD_MASK & !(file_mask(0)); // 非第一列 (用于左移检查)
#[allow(dead_code)]
const NOT_FILE_H: u64 = BOARD_MASK & !(file_mask(BOARD_COLS - 1)); // 非最后一列 (用于右移检查)

// ==============================================================================
// --- 基础数据结构 ---
// ==============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PieceType {
    Soldier = 0,  // 兵/卒 (等级最低，但可吃将)
    Cannon = 1,   // 炮/包 (特殊攻击)
    Horse = 2,    // 马/傌
    Chariot = 3,  // 车/俥
    Elephant = 4, // 象/相
    Advisor = 5,  // 士/仕
    General = 6,  // 将/帅 (等级最高)
}

impl PieceType {
    #[allow(dead_code)]
    fn from_usize(u: usize) -> Option<Self> {
        match u {
            0 => Some(PieceType::Soldier),
            1 => Some(PieceType::Cannon),
            2 => Some(PieceType::Horse),
            3 => Some(PieceType::Chariot),
            4 => Some(PieceType::Elephant),
            5 => Some(PieceType::Advisor),
            6 => Some(PieceType::General),
            _ => None,
        }
    }

    /// 获取棋子价值 (用于简单的启发式评估或血量扣除)
    pub fn value(&self) -> i32 {
        match self {
            PieceType::Soldier => 2,
            PieceType::Cannon => 5,
            PieceType::Horse => 5,
            PieceType::Chariot => 5,
            PieceType::Elephant => 5,
            PieceType::Advisor => 10,
            PieceType::General => 30,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Player {
    Red = 1,
    Black = -1,
}

impl Player {
    pub fn opposite(&self) -> Self {
        match self {
            Player::Red => Player::Black,
            Player::Black => Player::Red,
        }
    }

    pub fn val(&self) -> i32 {
        *self as i32
    }
}

impl fmt::Display for Player {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Player::Red => write!(f, "红方(Red)"),
            Player::Black => write!(f, "黑方(Black)"),
        }
    }
}

/// 棋子结构体
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Piece {
    pub piece_type: PieceType,
    pub player: Player,
}

impl Piece {
    pub fn new(piece_type: PieceType, player: Player) -> Self {
        Self { piece_type, player }
    }

    fn short_name(&self) -> String {
        let p_char = match self.player {
            Player::Red => "R",
            Player::Black => "B",
        };
        let t_char = match self.piece_type {
            PieceType::General => "Gen",
            PieceType::Cannon => "Can",
            PieceType::Horse => "Hor",
            PieceType::Chariot => "Cha",
            PieceType::Elephant => "Ele",
            PieceType::Advisor => "Adv",
            PieceType::Soldier => "Sol",
        };
        format!("{}_{}", p_char, t_char)
    }
}

/// 棋盘格状态枚举
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Slot {
    Empty,           // 空位
    Hidden,          // 暗子 (未翻开)
    Revealed(Piece), // 明子 (已翻开)
}

// ==============================================================================
// --- 预计算表 (Lookup Tables) ---
// 用于快速转换 动作索引 <-> 坐标，以及计算炮的攻击射线
// ==============================================================================

struct ActionLookupTables {
    action_to_coords: Vec<Vec<usize>>,
    coords_to_action: HashMap<Vec<usize>, usize>,
}

static ACTION_LOOKUP_TABLES: OnceLock<ActionLookupTables> = OnceLock::new();
static RAY_ATTACKS: OnceLock<Vec<Vec<u64>>> = OnceLock::new();

fn action_lookup_tables() -> &'static ActionLookupTables {
    ACTION_LOOKUP_TABLES.get_or_init(build_action_lookup_tables)
}

fn ray_attacks() -> &'static Vec<Vec<u64>> {
    RAY_ATTACKS.get_or_init(build_ray_attacks)
}

/// 构建动作映射表
fn build_action_lookup_tables() -> ActionLookupTables {
    let mut action_to_coords = Vec::with_capacity(ACTION_SPACE_SIZE);
    let mut coords_to_action = HashMap::with_capacity(ACTION_SPACE_SIZE);
    let mut idx = 0;

    // 1. 翻棋动作 (idx 0-31)
    for sq in 0..TOTAL_POSITIONS {
        let coords = vec![sq];
        action_to_coords.push(coords.clone());
        coords_to_action.insert(coords, idx);
        idx += 1;
    }

    // 2. 常规移动动作 (相邻格子)
    let moves = [(-1, 0), (1, 0), (0, -1), (0, 1)];
    for r1 in 0..BOARD_ROWS {
        for c1 in 0..BOARD_COLS {
            let from_sq = r1 * BOARD_COLS + c1;
            for (dr, dc) in moves.iter() {
                let r2 = r1 as i32 + dr;
                let c2 = c1 as i32 + dc;
                if r2 >= 0 && r2 < BOARD_ROWS as i32 && c2 >= 0 && c2 < BOARD_COLS as i32 {
                    let to_sq = (r2 as usize) * BOARD_COLS + (c2 as usize);
                    let coords = vec![from_sq, to_sq];
                    action_to_coords.push(coords.clone());
                    coords_to_action.insert(coords, idx);
                    idx += 1;
                }
            }
        }
    }

    // 3. 炮的跳跃攻击动作
    // 这里只预生成所有可能的"跳跃"坐标对 (中间隔1个以上格子)
    // 实际合法性由 action_masks 动态检查
    for r1 in 0..BOARD_ROWS {
        for c1 in 0..BOARD_COLS {
            let from_sq = r1 * BOARD_COLS + c1;

            // 水平方向
            for c2 in 0..BOARD_COLS {
                if (c1 as i32 - c2 as i32).abs() > 1 {
                    let to_sq = r1 * BOARD_COLS + c2;
                    let coords = vec![from_sq, to_sq];
                    if !coords_to_action.contains_key(&coords) {
                        action_to_coords.push(coords.clone());
                        coords_to_action.insert(coords, idx);
                        idx += 1;
                    }
                }
            }

            // 垂直方向
            for r2 in 0..BOARD_ROWS {
                if (r1 as i32 - r2 as i32).abs() > 1 {
                    let to_sq = r2 * BOARD_COLS + c1;
                    let coords = vec![from_sq, to_sq];
                    if !coords_to_action.contains_key(&coords) {
                        action_to_coords.push(coords.clone());
                        coords_to_action.insert(coords, idx);
                        idx += 1;
                    }
                }
            }
        }
    }

    if idx != ACTION_SPACE_SIZE {
        panic!("动作空间计算错误: 预期 {}, 实际 {}", ACTION_SPACE_SIZE, idx);
    }

    ActionLookupTables {
        action_to_coords,
        coords_to_action,
    }
}

/// 构建射线攻击表 (用于加速炮的攻击判定)
/// ray_attacks[direction][square] 返回从该位置向该方向发出的所有格子的 Bitboard
fn build_ray_attacks() -> Vec<Vec<u64>> {
    let mut ray_attacks = vec![vec![0u64; TOTAL_POSITIONS]; NUM_DIRECTIONS];
    for sq in 0..TOTAL_POSITIONS {
        let r = sq / BOARD_COLS;
        let c = sq % BOARD_COLS;

        // UP
        for i in (0..r).rev() {
            let target_sq = i * BOARD_COLS + c;
            ray_attacks[DIRECTION_UP][sq] |= ull(target_sq);
        }
        // DOWN
        for i in (r + 1)..BOARD_ROWS {
            let target_sq = i * BOARD_COLS + c;
            ray_attacks[DIRECTION_DOWN][sq] |= ull(target_sq);
        }
        // LEFT
        for i in (0..c).rev() {
            let target_sq = r * BOARD_COLS + i;
            ray_attacks[DIRECTION_LEFT][sq] |= ull(target_sq);
        }
        // RIGHT
        for i in (c + 1)..BOARD_COLS {
            let target_sq = r * BOARD_COLS + i;
            ray_attacks[DIRECTION_RIGHT][sq] |= ull(target_sq);
        }
    }
    ray_attacks
}

// ==============================================================================
// --- 环境结构体 (DarkChessEnv) ---
// ==============================================================================

/// 观察空间数据结构 (Neural Network Input)
#[derive(Debug, Clone)]
pub struct Observation {
    /// 棋盘特征张量: (Stack, Channels, H, W)
    pub board: Array4<f32>,
    /// 全局标量特征: (Stack, Features)
    pub scalars: Array1<f32>,
}

#[derive(Clone, Debug)]
pub struct DarkChessEnv {
    // --- 游戏核心状态 ---
    /// 棋盘格子状态 (Empty, Hidden, Revealed)
    board: Vec<Slot>,
    /// 当前玩家
    current_player: Player,
    /// 连续无吃子步数 (用于判和)
    move_counter: usize,
    /// 游戏总步数
    total_step_counter: usize,

    // --- 位棋盘 (Bitboards) 用于加速计算 ---
    /// 按 [Player][PieceType] 索引的位棋盘
    piece_bitboards: HashMap<Player, [u64; NUM_PIECE_TYPES]>,
    /// 按 [Player] 索引的明子位棋盘
    revealed_bitboards: HashMap<Player, u64>,
    /// 所有暗子位置
    hidden_bitboard: u64,
    /// 所有空位位置
    empty_bitboard: u64,

    // --- 游戏统计与记录 ---
    /// 阵亡棋子列表
    dead_pieces: HashMap<Player, Vec<PieceType>>,
    /// 玩家分数/血量 (减分制)
    scores: HashMap<Player, i32>,
    /// 上一步动作
    last_action: i32,

    // --- 历史堆叠 (State Stacking) ---
    board_history: VecDeque<Vec<f32>>,
    scalar_history: VecDeque<Vec<f32>>,

    // --- 概率相关 (Bag Model) ---
    /// 隐藏棋子池: 翻棋时从此池中抽取 (模拟未知状态)
    pub hidden_pieces: Vec<Piece>,
    /// 翻棋概率表 (用于 Observation 输入)
    reveal_probabilities: Vec<f32>,
}

impl DarkChessEnv {
    pub fn new() -> Self {
        let mut env = Self {
            board: vec![Slot::Empty; TOTAL_POSITIONS], // 初始填充，reset时会覆盖
            current_player: Player::Red,
            move_counter: 0,
            total_step_counter: 0,

            piece_bitboards: HashMap::new(),
            revealed_bitboards: HashMap::new(),
            hidden_bitboard: 0,
            empty_bitboard: 0,

            dead_pieces: HashMap::new(),
            scores: HashMap::new(),
            last_action: -1,

            board_history: VecDeque::with_capacity(STATE_STACK_SIZE),
            scalar_history: VecDeque::with_capacity(STATE_STACK_SIZE),

            hidden_pieces: Vec::new(),
            reveal_probabilities: vec![0.0; REVEAL_PROBABILITY_SIZE],
        };

        // 确保单例表已初始化
        action_lookup_tables();
        ray_attacks();
        
        env.reset();
        env
    }

    /// 根据动作编号获取对应的坐标列表（便于调试/可视化）
    pub fn get_coords_for_action(&self, action: usize) -> Option<&Vec<usize>> {
        action_lookup_tables().action_to_coords.get(action)
    }

    /// (已弃用) 自定义场景：仅保留红黑士
    pub fn setup_two_advisors(&mut self, _current_player: Player) {
        // no-op: 保持现有随机初始局面
    }

    /// (已弃用) 自定义场景：隐藏的威胁
    pub fn setup_hidden_threats(&mut self) {
        // no-op: 保持现有随机初始局面
    }

    /// 重置内部状态变量
    fn reset_internal_state(&mut self) {
        self.board = vec![Slot::Empty; TOTAL_POSITIONS];

        // 初始化 Bitboards
        self.piece_bitboards
            .insert(Player::Red, [0; NUM_PIECE_TYPES]);
        self.piece_bitboards
            .insert(Player::Black, [0; NUM_PIECE_TYPES]);

        self.revealed_bitboards.insert(Player::Red, 0);
        self.revealed_bitboards.insert(Player::Black, 0);

        self.hidden_bitboard = 0;
        self.empty_bitboard = 0;

        self.dead_pieces.insert(Player::Red, Vec::new());
        self.dead_pieces.insert(Player::Black, Vec::new());

        // 初始化血量（减分制）
        self.scores.insert(Player::Red, INITIAL_HEALTH_POINTS);
        self.scores.insert(Player::Black, INITIAL_HEALTH_POINTS);

        self.current_player = Player::Red;
        self.move_counter = 0;
        self.total_step_counter = 0;
        self.last_action = -1;

        self.board_history.clear();
        self.scalar_history.clear();
        self.hidden_pieces.clear();
        self.reveal_probabilities = vec![0.0; REVEAL_PROBABILITY_SIZE];
    }

    /// 初始化棋盘布局 (Shuffle Bag Model)
    fn initialize_board(&mut self) {
        let mut rng = thread_rng();

        // 1. 生成实际棋子池 (Bag)
        let mut pieces = Vec::new();
        for &player in &[Player::Red, Player::Black] {
            for _ in 0..GENERALS_COUNT { pieces.push(Piece::new(PieceType::General, player)); }
            for _ in 0..ADVISORS_COUNT { pieces.push(Piece::new(PieceType::Advisor, player)); }
            for _ in 0..ELEPHANTS_COUNT { pieces.push(Piece::new(PieceType::Elephant, player)); }
            for _ in 0..CHARIOTS_COUNT { pieces.push(Piece::new(PieceType::Chariot, player)); }
            for _ in 0..HORSES_COUNT { pieces.push(Piece::new(PieceType::Horse, player)); }
            for _ in 0..CANNONS_COUNT { pieces.push(Piece::new(PieceType::Cannon, player)); }
            for _ in 0..SOLDIERS_COUNT { pieces.push(Piece::new(PieceType::Soldier, player)); }
        }
        // 打乱棋子池
        pieces.shuffle(&mut rng);
        self.hidden_pieces = pieces; // 这些是等待被翻出来的棋子

        // 2. 填充棋盘
        // 暗棋初始状态全部为 Hidden，位置满，空位为0
        self.empty_bitboard = 0; 
        self.hidden_bitboard = BOARD_MASK; 

        for sq in 0..TOTAL_POSITIONS {
            self.board[sq] = Slot::Hidden;
        }

        // 计算初始概率表
        self.update_reveal_probabilities();

        // 3. 随机翻开 N 个 Hidden 位置 (加速开局)
        if TOTAL_POSITIONS > 0 {
            let mut hidden_indices: Vec<usize> = (0..TOTAL_POSITIONS).collect();
            hidden_indices.shuffle(&mut rng);
            let reveal_count = std::cmp::min(hidden_indices.len(), INITIAL_REVEALED_PIECES);

            for &idx in hidden_indices.iter().take(reveal_count) {
                // 这里调用 reveal_piece_at，它会更新 Bitboard
                self.reveal_piece_at(idx, None);
            }
        }
    }

    /// 重置环境
    pub fn reset(&mut self) -> Observation {
        self.reset_internal_state();
        self.initialize_board();

        let initial_board = self.get_board_state_tensor();
        let initial_scalar = self.get_scalar_state_vector();

        // 填充历史帧
        for _ in 0..STATE_STACK_SIZE {
            self.board_history.push_back(initial_board.clone());
            self.scalar_history.push_back(initial_scalar.clone());
        }

        self.get_state()
    }

    // --- 翻子逻辑 ---

    /// 翻开指定位置的棋子并更新 Bitboards
    /// specified_piece: 可选，用于 MCTS 确定化时指定翻出的棋子，否则从 hidden_pieces 中随机抽取
    fn reveal_piece_at(&mut self, sq: usize, specified_piece: Option<Piece>) {
        // 确保位置是 Hidden
        if !matches!(self.board[sq], Slot::Hidden) {
            panic!("尝试翻开非 Hidden 位置: {}", sq);
        }

        if self.hidden_pieces.is_empty() {
            panic!("逻辑错误：棋盘上有 Hidden 位置，但 hidden_pieces 池已空");
        }

        let idx = if let Some(target) = specified_piece {
            // 查找指定棋子
            self.hidden_pieces
                .iter()
                .position(|p| *p == target)
                .expect("指定的棋子不在隐藏棋子池中 (Cheat/Determinization Error)")
        } else {
            // 随机选择
            let mut rng = thread_rng();
            rng.gen_range(0..self.hidden_pieces.len())
        };

        // 从池中移除并使用
        let piece = self.hidden_pieces.swap_remove(idx);

        // 更新 Bitboards
        let mask = ull(sq);
        self.hidden_bitboard &= !mask;

        let p_bb = self.revealed_bitboards.get_mut(&piece.player).unwrap();
        *p_bb |= mask;

        let pt_bb =
            &mut self.piece_bitboards.get_mut(&piece.player).unwrap()[piece.piece_type as usize];
        *pt_bb |= mask;

        // 更新棋盘状态
        self.board[sq] = Slot::Revealed(piece);

        // 更新概率表
        self.update_reveal_probabilities();
    }

    /// 更新翻棋概率表 (用于 AI 输入)
    fn update_reveal_probabilities(&mut self) {
        let total_hidden = self.hidden_pieces.len();

        if total_hidden == 0 {
            self.reveal_probabilities = vec![0.0; REVEAL_PROBABILITY_SIZE];
            return;
        }

        let mut counts = vec![0; REVEAL_PROBABILITY_SIZE];
        for piece in &self.hidden_pieces {
            let idx = match (piece.player, piece.piece_type) {
                (Player::Red, PieceType::Soldier) => 0,
                (Player::Red, PieceType::Cannon) => 1,
                (Player::Red, PieceType::Horse) => 2,
                (Player::Red, PieceType::Chariot) => 3,
                (Player::Red, PieceType::Elephant) => 4,
                (Player::Red, PieceType::Advisor) => 5,
                (Player::Red, PieceType::General) => 6,
                (Player::Black, PieceType::Soldier) => 7,
                (Player::Black, PieceType::Cannon) => 8,
                (Player::Black, PieceType::Horse) => 9,
                (Player::Black, PieceType::Chariot) => 10,
                (Player::Black, PieceType::Elephant) => 11,
                (Player::Black, PieceType::Advisor) => 12,
                (Player::Black, PieceType::General) => 13,
            };
            counts[idx] += 1;
        }

        for i in 0..REVEAL_PROBABILITY_SIZE {
            self.reveal_probabilities[i] = counts[i] as f32 / total_hidden as f32;
        }
    }

    pub fn get_reveal_probabilities(&self) -> &Vec<f32> {
        &self.reveal_probabilities
    }

    // --- 核心 Step 逻辑 ---

    /// 执行动作，推进游戏状态
    pub fn step(
        &mut self,
        action: usize,
        reveal_piece: Option<Piece>, // 仅在MCTS模拟时用于指定翻出的棋子
    ) -> Result<(Observation, f32, bool, bool, Option<i32>), String> {
        let masks = self.action_masks();
        if masks[action] == 0 {
            return Err(format!("无效动作: {}", action));
        }

        self.last_action = action as i32;
        self.total_step_counter += 1;

        let lookup = action_lookup_tables();

        if action < REVEAL_ACTIONS_COUNT {
            // 翻棋动作
            let sq = lookup.action_to_coords[action][0];
            self.reveal_piece_at(sq, reveal_piece);
            self.move_counter = 0; // 翻棋重置判和计数
        } else {
            // 移动/攻击动作
            let coords = &lookup.action_to_coords[action];
            let from_sq = coords[0];
            let to_sq = coords[1];
            self.apply_move_action(from_sq, to_sq, reveal_piece);
        }

        // 切换玩家
        self.current_player = self.current_player.opposite();
        self.update_history();

        // 检查游戏结束条件
        let (terminated, truncated, winner) = self.check_game_over_conditions();
        Ok((self.get_state(), 0.0, terminated, truncated, winner))
    }

    /// 应用移动/攻击逻辑
    fn apply_move_action(&mut self, from_sq: usize, to_sq: usize, reveal_piece: Option<Piece>) {
        // 提取源棋子 (必须是 Revealed)
        let attacker = match std::mem::replace(&mut self.board[from_sq], Slot::Empty) {
            Slot::Revealed(p) => p,
            _ => panic!("Move action source is not a revealed piece!"),
        };

        // 如果目标是暗子 (炮击暗子的情况)，先翻开
        if matches!(self.board[to_sq], Slot::Hidden) {
            self.reveal_piece_at(to_sq, reveal_piece);
        }

        // 此时目标已经是 明子 或 空位
        let target_slot =
            std::mem::replace(&mut self.board[to_sq], Slot::Revealed(attacker.clone()));

        let attacker_mask = ull(from_sq);
        let defender_mask = ull(to_sq);
        let p = attacker.player;
        let pt = attacker.piece_type as usize;

        // 1. 更新源位置 (Attacker leaves from_sq)
        let my_revealed_bb = self.revealed_bitboards.get_mut(&p).unwrap();
        *my_revealed_bb &= !attacker_mask;

        let my_pt_bb = &mut self.piece_bitboards.get_mut(&p).unwrap()[pt];
        *my_pt_bb &= !attacker_mask;

        self.empty_bitboard |= attacker_mask;

        // 2. 更新目标位置 (Attacker moves to to_sq)
        *my_revealed_bb |= defender_mask;
        *my_pt_bb |= defender_mask;
        self.empty_bitboard &= !defender_mask; // 目标位置不再是空位

        match target_slot {
            Slot::Empty => {
                // 移动到空位，增加判和计数
                self.move_counter += 1;
            }
            Slot::Revealed(defender) => {
                // 3. 处理吃子 (Capture)
                let opp = defender.player;
                let opp_pt = defender.piece_type as usize;

                // 移除被吃子
                let opp_revealed_bb = self.revealed_bitboards.get_mut(&opp).unwrap();
                *opp_revealed_bb &= !defender_mask;

                let opp_pt_bb = &mut self.piece_bitboards.get_mut(&opp).unwrap()[opp_pt];
                *opp_pt_bb &= !defender_mask;

                self.dead_pieces
                    .get_mut(&defender.player)
                    .unwrap()
                    .push(defender.piece_type);

                // 更新被吃方血量（减分制）
                let score = self.scores.get_mut(&defender.player).unwrap();
                *score = score.saturating_sub(defender.piece_type.value());

                // 吃子重置判和计数
                self.move_counter = 0;
            }
            Slot::Hidden => {
                panic!("Unexpected Hidden slot after reveal");
            }
        }
    }

    fn update_history(&mut self) {
        let board_state = self.get_board_state_tensor();
        let scalar_state = self.get_scalar_state_vector();

        self.board_history.push_back(board_state);
        self.scalar_history.push_back(scalar_state);

        if self.board_history.len() > STATE_STACK_SIZE {
            self.board_history.pop_front();
            self.scalar_history.pop_front();
        }
    }

    // --- 状态特征提取 ---

    /// 生成棋盘状态张量 (扁平化)
    fn get_board_state_tensor(&self) -> Vec<f32> {
        let mut tensor = Vec::with_capacity(BOARD_CHANNELS * TOTAL_POSITIONS);
        let my = self.current_player;
        let opp = my.opposite();

        // 辅助闭包：将 bitboard 展开为 f32 向量
        let mut push_bitboard = |bb: u64| {
            for sq in 0..TOTAL_POSITIONS {
                tensor.push(if (bb & ull(sq)) != 0 { 1.0 } else { 0.0 });
            }
        };

        // 通道顺序:
        // 1. 己方7种棋子 (Revealed)
        for pt in 0..NUM_PIECE_TYPES {
            push_bitboard(self.piece_bitboards[&my][pt]);
        }
        // 2. 敌方7种棋子 (Revealed)
        for pt in 0..NUM_PIECE_TYPES {
            push_bitboard(self.piece_bitboards[&opp][pt]);
        }
        // 3. 暗子位置
        push_bitboard(self.hidden_bitboard);
        // 4. 空位位置
        push_bitboard(self.empty_bitboard);

        tensor
    }

    /// 生成标量特征向量
    fn get_scalar_state_vector(&self) -> Vec<f32> {
        let mut vec = Vec::with_capacity(SCALAR_FEATURE_COUNT);
        self.get_scalar_state_vector_into(&mut vec);
        vec
    }

    /// 填充 scalar 状态到提供的缓冲区，避免重复分配
    fn get_scalar_state_vector_into(&self, vec: &mut Vec<f32>) {
        vec.clear();
        vec.reserve(SCALAR_FEATURE_COUNT);

        let my = self.current_player;
        let opp = my.opposite();

        // 全局特征
        vec.push(self.move_counter as f32 / MAX_CONSECUTIVE_MOVES_FOR_DRAW as f32);
        vec.push(self.get_hp(my) as f32 / INITIAL_HEALTH_POINTS as f32);
        vec.push(self.get_hp(opp) as f32 / INITIAL_HEALTH_POINTS as f32);

        // 存活向量 (Bag Encoding)
        for &player in &[my, opp] {
            let bitboards = &self.piece_bitboards[&player];
            for pt in 0..NUM_PIECE_TYPES {
                let count = bitboards[pt].count_ones() as usize;
                let max_count = PIECE_MAX_COUNTS[pt];
                // One-hot-like encoding for counts
                vec.extend(std::iter::repeat(1.0).take(count));
                vec.extend(std::iter::repeat(0.0).take(max_count - count));
            }
        }

        // 动作掩码 (辅助网络学习合法动作)
        let action_masks = self.action_masks();
        vec.extend(action_masks.iter().map(|&x| x as f32));
    }

    pub fn get_state(&self) -> Observation {
        let mut board_data =
            Vec::with_capacity(STATE_STACK_SIZE * BOARD_CHANNELS * BOARD_ROWS * BOARD_COLS);
        for frame in &self.board_history {
            board_data.extend_from_slice(frame);
        }
        let board = Array4::from_shape_vec(
            (STATE_STACK_SIZE, BOARD_CHANNELS, BOARD_ROWS, BOARD_COLS),
            board_data,
        )
        .expect("Failed to reshape board array");

        let mut scalars_data = Vec::with_capacity(STATE_STACK_SIZE * SCALAR_FEATURE_COUNT);
        for frame in &self.scalar_history {
            scalars_data.extend_from_slice(frame);
        }
        let scalars = Array1::from_vec(scalars_data);

        Observation { board, scalars }
    }

    /// 使用提供的缓冲区填充 Observation，避免重复分配内存 (优化性能)
    pub fn get_state_into(&self, board_data: &mut Vec<f32>, scalars_data: &mut Vec<f32>) -> Observation {
        board_data.clear();
        board_data.reserve(STATE_STACK_SIZE * BOARD_CHANNELS * BOARD_ROWS * BOARD_COLS);
        for frame in &self.board_history {
            board_data.extend_from_slice(frame);
        }
        let board = Array4::from_shape_vec(
            (STATE_STACK_SIZE, BOARD_CHANNELS, BOARD_ROWS, BOARD_COLS),
            board_data.clone(),
        )
        .expect("Failed to reshape board array");

        scalars_data.clear();
        scalars_data.reserve(STATE_STACK_SIZE * SCALAR_FEATURE_COUNT);
        for frame in &self.scalar_history {
            scalars_data.extend_from_slice(frame);
        }
        let scalars = Array1::from_vec(scalars_data.clone());

        Observation { board, scalars }
    }

    // --- 游戏规则检查 ---

    /// 检查游戏是否结束
    /// 返回: (terminated, truncated, winner)
    fn check_game_over_conditions(&self) -> (bool, bool, Option<i32>) {
        // 1. 血量归零判定
        if self.scores[&Player::Red] <= 0 {
            return (true, false, Some(Player::Black.val()));
        }
        if self.scores[&Player::Black] <= 0 {
            return (true, false, Some(Player::Red.val()));
        }

        // 2. 全灭判定
        if self.dead_pieces[&Player::Red].len() == TOTAL_PIECES_PER_PLAYER {
            return (true, false, Some(Player::Black.val()));
        }
        if self.dead_pieces[&Player::Black].len() == TOTAL_PIECES_PER_PLAYER {
            return (true, false, Some(Player::Red.val()));
        }

        // 3. 无棋可走判定 (Stalemate -> Win for other side)
        let masks = self.get_action_masks_for_player(self.current_player);
        if masks.iter().all(|&x| x == 0) {
            return (true, false, Some(self.current_player.opposite().val()));
        }

        // 4. 平局条件：无吃子步数限制
        if self.move_counter >= MAX_CONSECUTIVE_MOVES_FOR_DRAW {
            return (true, false, Some(0));
        }

        // 5. 平局条件：总步数限制
        if self.total_step_counter >= MAX_STEPS_PER_EPISODE {
            return (false, true, Some(0));
        }

        (false, false, None)
    }

    // --- 动作掩码计算 (核心逻辑) ---

    pub fn action_masks(&self) -> Vec<i32> {
        let mut mask = vec![0; ACTION_SPACE_SIZE];
        self.action_masks_into(&mut mask);
        mask
    }

    pub fn action_masks_into(&self, mask: &mut [i32]) {
        self.get_action_masks_for_player_into(self.current_player, mask);
    }

    fn get_action_masks_for_player(&self, player: Player) -> Vec<i32> {
        let mut mask = vec![0; ACTION_SPACE_SIZE];
        self.get_action_masks_for_player_into(player, &mut mask);
        mask
    }

    /// 计算当前玩家的合法动作掩码
    fn get_action_masks_for_player_into(&self, player: Player, mask: &mut [i32]) {
        // 清空缓冲区
        for m in mask.iter_mut() {
            *m = 0;
        }
        let lookup = action_lookup_tables();

        // --------------------------------------------------------
        // 1. 翻棋动作 (Reveal)
        // --------------------------------------------------------
        let mut temp_hidden = self.hidden_bitboard;
        while temp_hidden != 0 {
            let sq = pop_lsb(&mut temp_hidden);
            if let Some(&idx) = lookup.coords_to_action.get(&vec![sq]) {
                mask[idx] = 1;
            }
        }

        // --------------------------------------------------------
        // 2. 准备位棋盘数据
        // --------------------------------------------------------
        let empty_bb = self.empty_bitboard;
        let my = player;
        let opp = player.opposite();

        let my_revealed_bb = *self.revealed_bitboards.get(&my).unwrap();
        let my_piece_bb = *self.piece_bitboards.get(&my).unwrap();
        let opp_piece_bb = *self.piece_bitboards.get(&opp).unwrap();

        // --------------------------------------------------------
        // 3. 常规移动 (Regular Moves)
        // 逻辑：计算所有可能的吃子目标，通过位移(Shift)计算可行步
        // --------------------------------------------------------

        // 计算目标集合 (Target Bitboards): 可以是空位 或 满足等级压制的敌方棋子
        let mut target_bbs: [u64; NUM_PIECE_TYPES] = [0; NUM_PIECE_TYPES];
        let mut cumulative_targets: u64 = empty_bb; // 初始包含空位

        // 按等级累积敌方棋子: Soldier(0) -> General(6)
        // 规则：等级高吃等级低 (General > Advisor > ... > Soldier)
        for pt in 0..NUM_PIECE_TYPES {
            cumulative_targets |= opp_piece_bb[pt];
            target_bbs[pt] = cumulative_targets;
        }

        // 特殊规则修正:
        // 1. 兵(0) 可以吃 将(6)
        target_bbs[PieceType::Soldier as usize] |= opp_piece_bb[PieceType::General as usize];
        // 2. 将(6) 不能吃 兵(0)
        target_bbs[PieceType::General as usize] &= !opp_piece_bb[PieceType::Soldier as usize];

        // 移动方向与偏移
        let shifts = [
            -(BOARD_COLS as isize) as i32, // Up
            (BOARD_COLS as i32),           // Down
            -1,                            // Left
            1,                             // Right
        ];

        // 边界掩码 (防止左右移动穿越棋盘边缘)
        let wrap_checks = [BOARD_MASK, BOARD_MASK, NOT_FILE_A, NOT_FILE_H];

        // 遍历己方除炮以外的所有棋子类型
        for pt in 0..NUM_PIECE_TYPES {
            if pt == PieceType::Cannon as usize {
                continue; // 炮的逻辑单独处理
            }

            let from_bb = my_piece_bb[pt];
            if from_bb == 0 {
                continue;
            }

            for (dir_idx, &shift) in shifts.iter().enumerate() {
                let wrap = wrap_checks[dir_idx];

                let temp_from_bb = from_bb & wrap;
                if temp_from_bb == 0 {
                    continue;
                }

                // 计算潜在目标位 (纯位移)
                let potential_to_bb = if shift > 0 {
                    (temp_from_bb << (shift as u32)) & BOARD_MASK
                } else {
                    (temp_from_bb >> ((-shift) as u32)) & BOARD_MASK
                };

                // 过滤有效目标 (必须在允许的 target_bbs 中)
                let mut actual_to_bb = potential_to_bb & target_bbs[pt];

                // 提取结果并设置掩码
                while actual_to_bb != 0 {
                    let to_sq = pop_lsb(&mut actual_to_bb);

                    // 反推 from_sq
                    let from_sq = if shift > 0 {
                        (to_sq as isize - (shift as isize)) as usize
                    } else {
                        (to_sq as isize + ((-shift) as isize)) as usize
                    };

                    if let Some(&idx) = lookup.coords_to_action.get(&vec![from_sq, to_sq]) {
                        mask[idx] = 1;
                    }
                }
            }
        }

        // --------------------------------------------------------
        // 4. 炮的特殊攻击 (Cannon Attacks)
        // 规则：炮必须隔一个子（炮架）才能攻击，目标可以是敌方明子或任何暗子
        // --------------------------------------------------------
        let my_cannons_bb = my_piece_bb[PieceType::Cannon as usize];
        if my_cannons_bb != 0 {
            let all_pieces_bb = BOARD_MASK & !empty_bb; // 所有非空位置

            // 炮的合法目标: 非(己方明子) => 敌方明子 + 任何暗子
            let valid_cannon_targets = BOARD_MASK & (!my_revealed_bb);

            let mut temp_cannons = my_cannons_bb;
            let ray_attacks = ray_attacks();
            while temp_cannons != 0 {
                let from_sq = pop_lsb(&mut temp_cannons);

                // 遍历4个方向
                for dir in 0..NUM_DIRECTIONS {
                    let ray_bb = ray_attacks[dir][from_sq];
                    let blockers = ray_bb & all_pieces_bb;

                    if blockers == 0 {
                        continue; // 无炮架
                    }

                    // 寻找炮架 (Screen): 离 from_sq 最近的阻挡物
                    // Up/Left: 索引减小 -> 离from最近的是最大索引 (MSB)
                    // Down/Right: 索引增大 -> 离from最近是最小索引 (LSB)
                    let screen_sq = match dir {
                        DIRECTION_UP | DIRECTION_LEFT => msb_index(blockers), 
                        _ => Some(trailing_zeros(blockers)), 
                    };

                    if screen_sq.is_none() {
                        continue;
                    }
                    let screen_sq = screen_sq.unwrap();

                    // 寻找目标 (Target): 炮架后的第一个棋子
                    // 同样使用射线表，但起点改为炮架
                    let after_screen_ray = ray_attacks[dir][screen_sq];
                    let targets = after_screen_ray & all_pieces_bb;

                    if targets == 0 {
                        continue; // 炮架后无子
                    }

                    // 寻找最近的目标
                    let target_sq = match dir {
                        DIRECTION_UP | DIRECTION_LEFT => msb_index(targets),
                        _ => Some(trailing_zeros(targets)),
                    };

                    if target_sq.is_none() {
                        continue;
                    }
                    let target_sq = target_sq.unwrap();

                    // 检查目标是否合法 (不能吃己方明子)
                    if ((ull(target_sq)) & valid_cannon_targets) != 0 {
                        if let Some(&idx) = lookup.coords_to_action.get(&vec![from_sq, target_sq]) {
                            mask[idx] = 1;
                        }
                    }
                }
            }
        }
    }

    /// 获取动作的目标位置的 Slot
    /// 用于 MCTS 处理 Chance Node
    pub fn get_target_slot(&self, action: usize) -> Slot {
        let coords = &action_lookup_tables().action_to_coords[action];
        
        if action < REVEAL_ACTIONS_COUNT {
            // 翻棋动作：只有一个坐标，就是要翻开的位置
            let sq = coords[0];
            self.board[sq].clone()
        } else {
            // 移动/攻击动作：有两个坐标，返回目标位置
            let to_sq = coords[1];
            self.board[to_sq].clone()
        }
    }

    /// 调试用：打印棋盘
    pub fn print_board(&self) {
        println!("\n      0         1         2         3");
        println!("   +---------+---------+---------+---------+");
        for r in 0..BOARD_ROWS {
            print!(" {} |", (b'A' + r as u8) as char);
            for c in 0..BOARD_COLS {
                let idx = r * BOARD_COLS + c;
                match &self.board[idx] {
                    Slot::Empty => print!("   .     |"),
                    Slot::Hidden => print!("    ?    |"),
                    Slot::Revealed(p) => print!(" {:^7} |", p.short_name()),
                }
            }
            println!("\n   +---------+---------+---------+---------+");
        }
        println!("当前玩家: {}", self.current_player);
        println!(
            "Total Steps: {}, Move Counter: {}",
            self.total_step_counter, self.move_counter
        );
        println!("Dead (Red): {:?}", self.dead_pieces[&Player::Red]);
        println!("Dead (Black): {:?}", self.dead_pieces[&Player::Black]);
        println!("---------------------------------------------");
    }

    // === 公共访问器方法 (用于 GUI / API) ===

    pub fn get_board_slots(&self) -> &Vec<Slot> {
        &self.board
    }

    pub fn get_current_player(&self) -> Player {
        self.current_player
    }

    pub fn get_move_counter(&self) -> usize {
        self.move_counter
    }

    pub fn get_total_steps(&self) -> usize {
        self.total_step_counter
    }

    pub fn get_score(&self, player: Player) -> i32 {
        *self.scores.get(&player).unwrap_or(&0)
    }

    pub fn get_scores(&self) -> (i32, i32) {
        (self.get_score(Player::Red), self.get_score(Player::Black))
    }

    pub fn get_hp(&self, player: Player) -> i32 {
        self.get_score(player)
    }

    pub fn get_dead_pieces(&self, player: Player) -> &Vec<PieceType> {
        &self.dead_pieces[&player]
    }

    pub fn get_hidden_pieces(&self, player: Player) -> Vec<PieceType> {
        self.hidden_pieces
            .iter()
            .filter(|p| p.player == player)
            .map(|p| p.piece_type)
            .collect()
    }

    pub fn get_action_for_coords(&self, coords: &[usize]) -> Option<usize> {
        action_lookup_tables().coords_to_action.get(coords).copied()
    }

    /// 获取所有 Bitboards (用于 GUI 可视化)
    /// 返回格式: HashMap<标签, Vec<bool>>，其中 Vec 长度为 TOTAL_POSITIONS
    pub fn get_bitboards(&self) -> std::collections::HashMap<String, Vec<bool>> {
        let mut bitboards = std::collections::HashMap::new();

        // 通用闭包：将 bitboard 转换为 Vec<bool>
        let bb_to_vec =
            |bb: u64| -> Vec<bool> { (0..TOTAL_POSITIONS).map(|sq| (bb & ull(sq)) != 0).collect() };

        bitboards.insert("hidden".to_string(), bb_to_vec(self.hidden_bitboard));
        bitboards.insert("empty".to_string(), bb_to_vec(self.empty_bitboard));

        const PIECE_NAMES: [&str; NUM_PIECE_TYPES] = [
            "soldier", "cannon", "horse", "chariot", "elephant", "advisor", "general",
        ];

        for &player in &[Player::Red, Player::Black] {
            let prefix = match player {
                Player::Red => "red",
                Player::Black => "black",
            };

            bitboards.insert(
                format!("{}_revealed", prefix),
                bb_to_vec(self.revealed_bitboards[&player]),
            );

            for (pt, &name) in PIECE_NAMES.iter().enumerate() {
                bitboards.insert(
                    format!("{}_{}", prefix, name),
                    bb_to_vec(self.piece_bitboards[&player][pt]),
                );
            }
        }

        bitboards
    }
}