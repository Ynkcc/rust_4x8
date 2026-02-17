use ndarray::{Array1, Array4};
use rand::seq::SliceRandom;
use rand::thread_rng;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
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
pub const MAX_CONSECUTIVE_MOVES_FOR_DRAW: usize = 24;

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
pub const REVEAL_ACTIONS_COUNT: usize = 32;
pub const REGULAR_MOVE_ACTIONS_COUNT: usize = 104;
pub const CANNON_ATTACK_ACTIONS_COUNT: usize = 216;
pub const ACTION_SPACE_SIZE: usize =
    REVEAL_ACTIONS_COUNT + REGULAR_MOVE_ACTIONS_COUNT + CANNON_ATTACK_ACTIONS_COUNT;

// ==============================================================================
// --- Bitboard 辅助函数 ---
// ==============================================================================

const BOARD_MASK: u64 = (1u64 << TOTAL_POSITIONS) - 1;

#[inline]
const fn ull(x: usize) -> u64 {
    1u64 << x
}

#[inline]
fn trailing_zeros(bb: u64) -> usize {
    bb.trailing_zeros() as usize
}

#[inline]
fn msb_index(bb: u64) -> Option<usize> {
    if bb == 0 {
        None
    } else {
        Some(U64_BITS - 1 - bb.leading_zeros() as usize)
    }
}

#[inline]
fn pop_lsb(bb: &mut u64) -> usize {
    let tz = bb.trailing_zeros() as usize;
    *bb &= *bb - 1;
    tz
}

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

const NOT_FILE_A: u64 = BOARD_MASK & !(file_mask(0)); 
const NOT_FILE_H: u64 = BOARD_MASK & !(file_mask(BOARD_COLS - 1));

// ==============================================================================
// --- 基础数据结构 ---
// ==============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PieceType {
    Soldier = 0,  
    Cannon = 1,   
    Horse = 2,    
    Chariot = 3,  
    Elephant = 4, 
    Advisor = 5,  
    General = 6,  
}

impl Default for PieceType {
    fn default() -> Self {
        PieceType::Soldier // 默认值，仅用于初始化数组占位
    }
}

impl PieceType {
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

    pub fn idx(&self) -> usize {
        match self {
            Player::Red => 0,
            Player::Black => 1,
        }
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
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Piece {
    pub piece_type: PieceType,
    pub player: Player,
}

impl Default for Piece {
    fn default() -> Self {
        // 用于初始化数组的默认值
        Self { piece_type: PieceType::Soldier, player: Player::Red }
    }
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
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum Slot {
    Empty,           // 空位
    Hidden,          // 暗子 (未翻开)
    Revealed(Piece), // 明子 (已翻开)
}

// ==============================================================================
// --- 预计算表 (Lookup Tables) ---
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

fn build_action_lookup_tables() -> ActionLookupTables {
    let mut action_to_coords = Vec::with_capacity(ACTION_SPACE_SIZE);
    let mut coords_to_action = HashMap::with_capacity(ACTION_SPACE_SIZE);
    let mut idx = 0;

    // 1. 翻棋
    for sq in 0..TOTAL_POSITIONS {
        let coords = vec![sq];
        action_to_coords.push(coords.clone());
        coords_to_action.insert(coords, idx);
        idx += 1;
    }

    // 2. 常规移动
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

    // 3. 炮击
    for r1 in 0..BOARD_ROWS {
        for c1 in 0..BOARD_COLS {
            let from_sq = r1 * BOARD_COLS + c1;
            // 水平
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
            // 垂直
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

    ActionLookupTables {
        action_to_coords,
        coords_to_action,
    }
}

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
    /// 棋盘特征张量: (Channels, H, W)
    pub board: Array4<f32>,
    /// 全局标量特征: (Features,)
    pub scalars: Array1<f32>,
}

/// 支持 Copy 的暗棋环境
/// 所有 Vec 已替换为定长数组 + 计数器
#[derive(Clone, Copy, Debug)]
pub struct DarkChessEnv {
    // --- 游戏核心状态 ---
    /// 棋盘格子状态
    board: [Slot; TOTAL_POSITIONS],
    /// 当前玩家
    current_player: Player,
    /// 连续无吃子步数
    move_counter: usize,
    /// 游戏总步数
    total_step_counter: usize,

    // --- 位棋盘 (Bitboards) ---
    piece_bitboards: [[u64; NUM_PIECE_TYPES]; 2],
    revealed_bitboards: [u64; 2],
    hidden_bitboard: u64,
    empty_bitboard: u64,

    // --- 游戏统计与记录 (Copy Refactor) ---
    /// 阵亡棋子池 [PlayerIdx][Idx]
    dead_pieces_pool: [[PieceType; TOTAL_PIECES_PER_PLAYER]; 2],
    /// 阵亡棋子计数 [PlayerIdx]
    dead_pieces_count: [usize; 2],
    
    /// 玩家分数/血量
    scores: [i32; 2],
    /// 上一步动作
    last_action: i32,

    // --- 概率相关 (Bag Model - Copy Refactor) ---
    /// 隐藏棋子池: 使用定长数组代替 Vec
    hidden_pieces_pool: [Piece; TOTAL_POSITIONS],
    /// 当前隐藏棋子数量
    hidden_pieces_count: usize,
    
    /// 翻棋概率表
    reveal_probabilities: [f32; REVEAL_PROBABILITY_SIZE],
}

impl DarkChessEnv {
    pub fn new() -> Self {
        let mut env = Self {
            board: [Slot::Empty; TOTAL_POSITIONS],
            current_player: Player::Red,
            move_counter: 0,
            total_step_counter: 0,

            piece_bitboards: [[0; NUM_PIECE_TYPES]; 2],
            revealed_bitboards: [0; 2],
            hidden_bitboard: 0,
            empty_bitboard: 0,

            // 初始化阵亡列表
            dead_pieces_pool: [[PieceType::default(); TOTAL_PIECES_PER_PLAYER]; 2],
            dead_pieces_count: [0; 2],
            
            scores: [0; 2],
            last_action: -1,

            // 初始化隐藏池
            hidden_pieces_pool: [Piece::default(); TOTAL_POSITIONS],
            hidden_pieces_count: 0,
            
            reveal_probabilities: [0.0; REVEAL_PROBABILITY_SIZE],
        };

        action_lookup_tables();
        ray_attacks();
        
        env.reset();
        env
    }

    pub fn get_coords_for_action(&self, action: usize) -> Option<&Vec<usize>> {
        action_lookup_tables().action_to_coords.get(action)
    }

    fn reset_internal_state(&mut self) {
        self.board = [Slot::Empty; TOTAL_POSITIONS];

        self.piece_bitboards = [[0; NUM_PIECE_TYPES]; 2];
        self.revealed_bitboards = [0; 2];

        self.hidden_bitboard = 0;
        self.empty_bitboard = 0;

        // 重置阵亡计数，无需清空 pool 内容，依靠 count 即可
        self.dead_pieces_count = [0; 2];

        self.scores = [INITIAL_HEALTH_POINTS, INITIAL_HEALTH_POINTS];

        self.current_player = Player::Red;
        self.move_counter = 0;
        self.total_step_counter = 0;
        self.last_action = -1;

        self.hidden_pieces_count = 0;
        self.reveal_probabilities = [0.0; REVEAL_PROBABILITY_SIZE];
    }

    /// 初始化棋盘布局 (Shuffle Bag Model)
    fn initialize_board(&mut self) {
        let mut rng = thread_rng();

        // 1. 生成实际棋子池 (写入 Buffer)
        let mut idx = 0;
        for &player in &[Player::Red, Player::Black] {
            for _ in 0..GENERALS_COUNT { self.hidden_pieces_pool[idx] = Piece::new(PieceType::General, player); idx += 1; }
            for _ in 0..ADVISORS_COUNT { self.hidden_pieces_pool[idx] = Piece::new(PieceType::Advisor, player); idx += 1; }
            for _ in 0..ELEPHANTS_COUNT { self.hidden_pieces_pool[idx] = Piece::new(PieceType::Elephant, player); idx += 1; }
            for _ in 0..CHARIOTS_COUNT { self.hidden_pieces_pool[idx] = Piece::new(PieceType::Chariot, player); idx += 1; }
            for _ in 0..HORSES_COUNT { self.hidden_pieces_pool[idx] = Piece::new(PieceType::Horse, player); idx += 1; }
            for _ in 0..CANNONS_COUNT { self.hidden_pieces_pool[idx] = Piece::new(PieceType::Cannon, player); idx += 1; }
            for _ in 0..SOLDIERS_COUNT { self.hidden_pieces_pool[idx] = Piece::new(PieceType::Soldier, player); idx += 1; }
        }
        self.hidden_pieces_count = idx;

        // 打乱 slice
        self.hidden_pieces_pool[0..self.hidden_pieces_count].shuffle(&mut rng);

        // 2. 填充棋盘
        self.empty_bitboard = 0; 
        self.hidden_bitboard = BOARD_MASK; 

        for sq in 0..TOTAL_POSITIONS {
            self.board[sq] = Slot::Hidden;
        }

        self.update_reveal_probabilities();

        // 3. 随机翻开 N 个 Hidden 位置
        if TOTAL_POSITIONS > 0 {
            let mut hidden_indices: Vec<usize> = (0..TOTAL_POSITIONS).collect();
            hidden_indices.shuffle(&mut rng);
            let reveal_count = std::cmp::min(hidden_indices.len(), INITIAL_REVEALED_PIECES);

            for &idx in hidden_indices.iter().take(reveal_count) {
                self.reveal_piece_at(idx, None);
            }
        }
    }

    pub fn reset(&mut self) -> Observation {
        self.reset_internal_state();
        self.initialize_board();
        self.get_state()
    }

    // --- 翻子逻辑 ---

    /// 翻开指定位置的棋子并更新 Bitboards
    fn reveal_piece_at(&mut self, sq: usize, specified_piece: Option<Piece>) {
        if !matches!(self.board[sq], Slot::Hidden) {
            panic!("尝试翻开非 Hidden 位置: {}", sq);
        }

        if self.hidden_pieces_count == 0 {
            panic!("逻辑错误：棋盘上有 Hidden 位置，但 hidden_pieces 池已空");
        }

        // 获取 slice 视图
        let active_slice = &self.hidden_pieces_pool[0..self.hidden_pieces_count];

        let idx = if let Some(target) = specified_piece {
            active_slice
                .iter()
                .position(|p| *p == target)
                .expect("指定的棋子不在隐藏棋子池中")
        } else {
            let mut rng = thread_rng();
            rng.gen_range(0..self.hidden_pieces_count)
        };

        // Swap Remove 逻辑 (Copy version)
        let last_idx = self.hidden_pieces_count - 1;
        self.hidden_pieces_pool.swap(idx, last_idx); // 将选中的棋子交换到末尾
        let piece = self.hidden_pieces_pool[last_idx]; // 取出
        self.hidden_pieces_count -= 1; // 缩小有效范围

        // 更新 Bitboards
        let mask = ull(sq);
        self.hidden_bitboard &= !mask;

        let p_bb = &mut self.revealed_bitboards[piece.player.idx()];
        *p_bb |= mask;

        let pt_bb =
            &mut self.piece_bitboards[piece.player.idx()][piece.piece_type as usize];
        *pt_bb |= mask;

        self.board[sq] = Slot::Revealed(piece);
        self.update_reveal_probabilities();
    }

    fn update_reveal_probabilities(&mut self) {
        let total_hidden = self.hidden_pieces_count;

        if total_hidden == 0 {
            self.reveal_probabilities = [0.0; REVEAL_PROBABILITY_SIZE];
            return;
        }

        let mut counts = vec![0; REVEAL_PROBABILITY_SIZE];
        for i in 0..total_hidden {
            let piece = self.hidden_pieces_pool[i];
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

    pub fn get_reveal_probabilities(&self) -> &[f32] {
        &self.reveal_probabilities
    }

    // --- 核心 Step 逻辑 ---

    pub fn step(
        &mut self,
        action: usize,
        reveal_piece: Option<Piece>,
    ) -> Result<(Observation, f32, bool, bool, Option<i32>), String> {
        let masks = self.action_masks();
        if masks[action] == 0 {
            return Err(format!("无效动作: {}", action));
        }

        self.last_action = action as i32;
        self.total_step_counter += 1;

        let lookup = action_lookup_tables();

        if action < REVEAL_ACTIONS_COUNT {
            let sq = lookup.action_to_coords[action][0];
            self.reveal_piece_at(sq, reveal_piece);
            self.move_counter = 0;
        } else {
            let coords = &lookup.action_to_coords[action];
            let from_sq = coords[0];
            let to_sq = coords[1];
            self.apply_move_action(from_sq, to_sq, reveal_piece);
        }

        self.current_player = self.current_player.opposite();
        let (terminated, truncated, winner) = self.check_game_over_conditions();
        Ok((self.get_state(), 0.0, terminated, truncated, winner))
    }

    fn apply_move_action(&mut self, from_sq: usize, to_sq: usize, reveal_piece: Option<Piece>) {
        let attacker = match std::mem::replace(&mut self.board[from_sq], Slot::Empty) {
            Slot::Revealed(p) => p,
            _ => panic!("Move action source is not a revealed piece!"),
        };

        if matches!(self.board[to_sq], Slot::Hidden) {
            self.reveal_piece_at(to_sq, reveal_piece);
        }

        let target_slot =
            std::mem::replace(&mut self.board[to_sq], Slot::Revealed(attacker.clone()));

        let attacker_mask = ull(from_sq);
        let defender_mask = ull(to_sq);
        let p = attacker.player;
        let pt = attacker.piece_type as usize;

        let my_revealed_bb = &mut self.revealed_bitboards[p.idx()];
        *my_revealed_bb &= !attacker_mask;

        let my_pt_bb = &mut self.piece_bitboards[p.idx()][pt];
        *my_pt_bb &= !attacker_mask;

        self.empty_bitboard |= attacker_mask;

        *my_revealed_bb |= defender_mask;
        *my_pt_bb |= defender_mask;
        self.empty_bitboard &= !defender_mask;

        match target_slot {
            Slot::Empty => {
                self.move_counter += 1;
            }
            Slot::Revealed(defender) => {
                let opp = defender.player;
                let opp_pt = defender.piece_type as usize;

                let opp_revealed_bb = &mut self.revealed_bitboards[opp.idx()];
                *opp_revealed_bb &= !defender_mask;

                let opp_pt_bb = &mut self.piece_bitboards[opp.idx()][opp_pt];
                *opp_pt_bb &= !defender_mask;

                // 记录被吃子 (使用 Array + Count 模拟 push)
                let opp_idx = defender.player.idx();
                let dead_idx = self.dead_pieces_count[opp_idx];
                if dead_idx < TOTAL_PIECES_PER_PLAYER {
                    self.dead_pieces_pool[opp_idx][dead_idx] = defender.piece_type;
                    self.dead_pieces_count[opp_idx] += 1;
                } else {
                    // 理论上不可能发生，除非逻辑错误
                    panic!("Dead pieces buffer overflow!");
                }

                let score = &mut self.scores[defender.player.idx()];
                *score = score.saturating_sub(defender.piece_type.value());

                self.move_counter = 0;
            }
            Slot::Hidden => {
                panic!("Unexpected Hidden slot after reveal");
            }
        }
    }

    // --- 状态特征提取 ---

    fn get_board_state_tensor(&self) -> Vec<f32> {
        let mut tensor = Vec::with_capacity(BOARD_CHANNELS * TOTAL_POSITIONS);
        let my = self.current_player;
        let opp = my.opposite();

        let mut push_bitboard = |bb: u64| {
            for sq in 0..TOTAL_POSITIONS {
                tensor.push(if (bb & ull(sq)) != 0 { 1.0 } else { 0.0 });
            }
        };

        for pt in 0..NUM_PIECE_TYPES {
            push_bitboard(self.piece_bitboards[my.idx()][pt]);
        }
        for pt in 0..NUM_PIECE_TYPES {
            push_bitboard(self.piece_bitboards[opp.idx()][pt]);
        }
        push_bitboard(self.hidden_bitboard);
        push_bitboard(self.empty_bitboard);

        tensor
    }

    fn get_scalar_state_vector(&self) -> Vec<f32> {
        let mut vec = Vec::with_capacity(SCALAR_FEATURE_COUNT);
        self.get_scalar_state_vector_into(&mut vec);
        vec
    }

    fn get_scalar_state_vector_into(&self, vec: &mut Vec<f32>) {
        vec.clear();
        vec.reserve(SCALAR_FEATURE_COUNT);

        let my = self.current_player;
        let opp = my.opposite();

        vec.push(self.move_counter as f32 / MAX_CONSECUTIVE_MOVES_FOR_DRAW as f32);
        vec.push(self.get_hp(my) as f32 / INITIAL_HEALTH_POINTS as f32);
        vec.push(self.get_hp(opp) as f32 / INITIAL_HEALTH_POINTS as f32);

        for &player in &[my, opp] {
            let bitboards = &self.piece_bitboards[player.idx()];
            for pt in 0..NUM_PIECE_TYPES {
                let count = bitboards[pt].count_ones() as usize;
                let max_count = PIECE_MAX_COUNTS[pt];
                vec.extend(std::iter::repeat(1.0).take(count));
                vec.extend(std::iter::repeat(0.0).take(max_count - count));
            }
        }

        let action_masks = self.action_masks();
        vec.extend(action_masks.iter().map(|&x| x as f32));
    }

    pub fn get_state(&self) -> Observation {
        let board_data = self.get_board_state_tensor();
        let board = Array4::from_shape_vec(
            (1, BOARD_CHANNELS, BOARD_ROWS, BOARD_COLS),
            board_data,
        )
        .expect("Failed to reshape board array");

        let scalars_data = self.get_scalar_state_vector();
        let scalars = Array1::from_vec(scalars_data);

        Observation { board, scalars }
    }

    pub fn get_state_into(&self, board_data: &mut Vec<f32>, scalars_data: &mut Vec<f32>) -> Observation {
        board_data.clear();
        board_data.reserve(BOARD_CHANNELS * BOARD_ROWS * BOARD_COLS);
        board_data.extend_from_slice(&self.get_board_state_tensor());
        
        let board = Array4::from_shape_vec(
            (1, BOARD_CHANNELS, BOARD_ROWS, BOARD_COLS),
            board_data.clone(),
        )
        .expect("Failed to reshape board array");

        scalars_data.clear();
        scalars_data.reserve(SCALAR_FEATURE_COUNT);
        scalars_data.extend_from_slice(&self.get_scalar_state_vector());
        let scalars = Array1::from_vec(scalars_data.clone());

        Observation { board, scalars }
    }

    // --- 游戏规则检查 ---

    pub fn check_game_over_conditions(&self) -> (bool, bool, Option<i32>) {
        if self.scores[Player::Red.idx()] <= 0 {
            return (true, false, Some(Player::Black.val()));
        }
        if self.scores[Player::Black.idx()] <= 0 {
            return (true, false, Some(Player::Red.val()));
        }

        // 全灭判定 (使用 count 判断)
        if self.dead_pieces_count[Player::Red.idx()] == TOTAL_PIECES_PER_PLAYER {
            return (true, false, Some(Player::Black.val()));
        }
        if self.dead_pieces_count[Player::Black.idx()] == TOTAL_PIECES_PER_PLAYER {
            return (true, false, Some(Player::Red.val()));
        }

        let masks = self.get_action_masks_for_player(self.current_player);
        if masks.iter().all(|&x| x == 0) {
            return (true, false, Some(self.current_player.opposite().val()));
        }

        if self.move_counter >= MAX_CONSECUTIVE_MOVES_FOR_DRAW {
            return (true, false, Some(0));
        }

        if self.total_step_counter >= MAX_STEPS_PER_EPISODE {
            return (false, true, Some(0));
        }

        (false, false, None)
    }

    // --- 动作掩码计算 ---

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

    fn get_action_masks_for_player_into(&self, player: Player, mask: &mut [i32]) {
        for m in mask.iter_mut() {
            *m = 0;
        }
        let lookup = action_lookup_tables();

        // 1. 翻棋动作
        let mut temp_hidden = self.hidden_bitboard;
        while temp_hidden != 0 {
            let sq = pop_lsb(&mut temp_hidden);
            if let Some(&idx) = lookup.coords_to_action.get(&vec![sq]) {
                mask[idx] = 1;
            }
        }

        let empty_bb = self.empty_bitboard;
        let my = player;
        let opp = player.opposite();

        let my_revealed_bb = self.revealed_bitboards[my.idx()];
        let my_piece_bb = self.piece_bitboards[my.idx()];
        let opp_piece_bb = self.piece_bitboards[opp.idx()];

        // 2. 常规移动
        let mut target_bbs: [u64; NUM_PIECE_TYPES] = [0; NUM_PIECE_TYPES];
        let mut cumulative_targets: u64 = empty_bb; 

        for pt in 0..NUM_PIECE_TYPES {
            cumulative_targets |= opp_piece_bb[pt];
            target_bbs[pt] = cumulative_targets;
        }

        target_bbs[PieceType::Soldier as usize] |= opp_piece_bb[PieceType::General as usize];
        target_bbs[PieceType::General as usize] &= !opp_piece_bb[PieceType::Soldier as usize];

        let shifts = [
            -(BOARD_COLS as isize) as i32, 
            (BOARD_COLS as i32),           
            -1,                            
            1,                             
        ];
        let wrap_checks = [BOARD_MASK, BOARD_MASK, NOT_FILE_A, NOT_FILE_H];

        for pt in 0..NUM_PIECE_TYPES {
            if pt == PieceType::Cannon as usize {
                continue; 
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

                let potential_to_bb = if shift > 0 {
                    (temp_from_bb << (shift as u32)) & BOARD_MASK
                } else {
                    (temp_from_bb >> ((-shift) as u32)) & BOARD_MASK
                };

                let mut actual_to_bb = potential_to_bb & target_bbs[pt];

                while actual_to_bb != 0 {
                    let to_sq = pop_lsb(&mut actual_to_bb);

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

        // 3. 炮击
        let my_cannons_bb = my_piece_bb[PieceType::Cannon as usize];
        if my_cannons_bb != 0 {
            let all_pieces_bb = BOARD_MASK & !empty_bb; 

            let valid_cannon_targets = BOARD_MASK & (!my_revealed_bb);

            let mut temp_cannons = my_cannons_bb;
            let ray_attacks = ray_attacks();
            while temp_cannons != 0 {
                let from_sq = pop_lsb(&mut temp_cannons);

                for dir in 0..NUM_DIRECTIONS {
                    let ray_bb = ray_attacks[dir][from_sq];
                    let blockers = ray_bb & all_pieces_bb;

                    if blockers == 0 {
                        continue; 
                    }

                    let screen_sq = match dir {
                        DIRECTION_UP | DIRECTION_LEFT => msb_index(blockers), 
                        _ => Some(trailing_zeros(blockers)), 
                    };

                    if screen_sq.is_none() {
                        continue;
                    }
                    let screen_sq = screen_sq.unwrap();

                    let after_screen_ray = ray_attacks[dir][screen_sq];
                    let targets = after_screen_ray & all_pieces_bb;

                    if targets == 0 {
                        continue; 
                    }

                    let target_sq = match dir {
                        DIRECTION_UP | DIRECTION_LEFT => msb_index(targets),
                        _ => Some(trailing_zeros(targets)),
                    };

                    if target_sq.is_none() {
                        continue;
                    }
                    let target_sq = target_sq.unwrap();

                    if ((ull(target_sq)) & valid_cannon_targets) != 0 {
                        if let Some(&idx) = lookup.coords_to_action.get(&vec![from_sq, target_sq]) {
                            mask[idx] = 1;
                        }
                    }
                }
            }
        }
    }

    pub fn get_target_slot(&self, action: usize) -> Slot {
        let coords = &action_lookup_tables().action_to_coords[action];
        
        if action < REVEAL_ACTIONS_COUNT {
            let sq = coords[0];
            self.board[sq].clone()
        } else {
            let to_sq = coords[1];
            self.board[to_sq].clone()
        }
    }

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
        println!("Dead (Red): {:?}", self.get_dead_pieces(Player::Red));
        println!("Dead (Black): {:?}", self.get_dead_pieces(Player::Black));
        println!("---------------------------------------------");
    }

    // === 公共访问器方法 ===

    pub fn get_board_slots(&self) -> &[Slot] {
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
        self.scores[player.idx()]
    }

    pub fn get_scores(&self) -> (i32, i32) {
        (self.get_score(Player::Red), self.get_score(Player::Black))
    }

    pub fn get_hp(&self, player: Player) -> i32 {
        self.get_score(player)
    }

    /// 返回死亡棋子的切片视图（替代 Vec 返回）
    pub fn get_dead_pieces(&self, player: Player) -> &[PieceType] {
        let count = self.dead_pieces_count[player.idx()];
        &self.dead_pieces_pool[player.idx()][0..count]
    }

    /// 返回隐藏棋子的 Vec（此处需要分配内存来收集，或返回迭代器）
    /// 为了保持兼容性返回 Vec
    pub fn get_hidden_pieces(&self, player: Player) -> Vec<PieceType> {
        self.hidden_pieces_pool[0..self.hidden_pieces_count]
            .iter()
            .filter(|p| p.player == player)
            .map(|p| p.piece_type)
            .collect()
    }

    pub fn get_hidden_pieces_raw(&self) -> &[Piece] {
        &self.hidden_pieces_pool[0..self.hidden_pieces_count]
    }

    pub fn get_action_for_coords(&self, coords: &[usize]) -> Option<usize> {
        action_lookup_tables().coords_to_action.get(coords).copied()
    }

    pub fn get_bitboards(&self) -> std::collections::HashMap<String, Vec<bool>> {
        let mut bitboards = std::collections::HashMap::new();

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
                bb_to_vec(self.revealed_bitboards[player.idx()]),
            );

            for (pt, &name) in PIECE_NAMES.iter().enumerate() {
                bitboards.insert(
                    format!("{}_{}", prefix, name),
                    bb_to_vec(self.piece_bitboards[player.idx()][pt]),
                );
            }
        }

        bitboards
    }
}