use ndarray::{Array1, Array4};
use rand::seq::SliceRandom;
use rand::thread_rng;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::fmt;
use std::sync::OnceLock;

// ==============================================================================
// --- 常量定义 (与 Python environment.py 保持一致) ---
// 由于 BOARD_COLS = 8 > 4，因此 ACTION_SPACE_SIZE 需要重新计算
// ==============================================================================
pub const BOARD_ROWS: usize = 4;
pub const BOARD_COLS: usize = 8; // 4x8 棋盘
pub const TOTAL_POSITIONS: usize = BOARD_ROWS * BOARD_COLS; // 32
pub const NUM_PIECE_TYPES: usize = 7;
pub const MAX_CONSECUTIVE_MOVES_FOR_DRAW: usize = 24; // 参照 README.md

pub const STATE_STACK_SIZE: usize = 1; // 禁用状态堆叠，仅使用当前帧
const MAX_STEPS_PER_EPISODE: usize = 100;
const INITIAL_REVEALED_PIECES: usize = 4;

// 初始血量 (减分制，每方从60分开始)
const INITIAL_HEALTH_POINTS: i32 = 60;

// 棋子数量定义（每方）
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

// 预定义的各棋子最大数量
const PIECE_MAX_COUNTS: [usize; NUM_PIECE_TYPES] = [
    SOLDIERS_COUNT,
    CANNONS_COUNT,
    HORSES_COUNT,
    CHARIOTS_COUNT,
    ELEPHANTS_COUNT,
    ADVISORS_COUNT,
    GENERALS_COUNT,
];

// 存活向量大小: 5(兵) + 2(炮) + 2(马) + 2(车) + 2(象) + 2(士) + 1(将) = 16
const SURVIVAL_VECTOR_SIZE: usize = TOTAL_PIECES_PER_PLAYER;

// Scalar 特征数量: 3个全局标量 + 2个存活向量(各16) + 动作掩码长度
pub const SCALAR_FEATURE_COUNT: usize = 3 + 2 * SURVIVAL_VECTOR_SIZE + ACTION_SPACE_SIZE;

// 翻棋概率表大小: 2个玩家 * 7种棋子 = 14
const REVEAL_PROBABILITY_SIZE: usize = 2 * NUM_PIECE_TYPES;

// 方向常量
const DIRECTION_UP: usize = 0;
const DIRECTION_DOWN: usize = 1;
const DIRECTION_LEFT: usize = 2;
const DIRECTION_RIGHT: usize = 3;
const NUM_DIRECTIONS: usize = 4;

// MSB (Most Significant Bit) 位数 (64位整数的最高位索引基数)
const U64_BITS: usize = 64;

// 棋盘状态张量的通道数: 自己棋子(7) + 对手棋子(7) + 暗子(1) + 空位(1) = 16
pub const BOARD_CHANNELS: usize = 2 * NUM_PIECE_TYPES + 2; // 自己 + 对手 + 隐藏 + 空位

// 动作空间 (TOTAL_POSITIONS = 32)
pub const REVEAL_ACTIONS_COUNT: usize = 32;
// 常规移动：4x8 网格共有 52 条无向相邻边，方向化为 104 个动作
pub const REGULAR_MOVE_ACTIONS_COUNT: usize = 104;
// 炮的攻击：水平 (每行21对×2方向=42, 4行=168) + 垂直 (每列3对×2方向=6, 8列=48) = 216
pub const CANNON_ATTACK_ACTIONS_COUNT: usize = 216;
pub const ACTION_SPACE_SIZE: usize =
    REVEAL_ACTIONS_COUNT + REGULAR_MOVE_ACTIONS_COUNT + CANNON_ATTACK_ACTIONS_COUNT; // 32 + 104 + 216 = 352

// ==============================================================================
// --- Bitboard 辅助函数 ---
// ==============================================================================

// 棋盘全掩码 (4x8=32)
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

/// 获取 LSB 索引并清除该位
#[inline]
fn pop_lsb(bb: &mut u64) -> usize {
    let tz = bb.trailing_zeros() as usize;
    *bb &= *bb - 1;
    tz
}

/// 生成列掩码 (File Mask)
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
const NOT_FILE_A: u64 = BOARD_MASK & !(file_mask(0)); // 非第一列
#[allow(dead_code)]
const NOT_FILE_H: u64 = BOARD_MASK & !(file_mask(BOARD_COLS - 1)); // 非最后一列

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PieceType {
    Soldier = 0,  // 兵
    Cannon = 1,   // 炮
    Horse = 2,    // 马
    Chariot = 3,  // 车
    Elephant = 4, // 象
    Advisor = 5,  // 士
    General = 6,  // 将
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

    /// 获取棋子价值
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

// 移除 revealed 属性，只保留棋子身份
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

// 新增 Slot 枚举来管理棋盘格状态
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Slot {
    Empty,           // 空位 (对应 Python 中的 None)
    Hidden,          // 暗子 (知道这里有棋，但不知道是什么)
    Revealed(Piece), // 明子 (已翻开)
}

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

    for sq in 0..TOTAL_POSITIONS {
        let coords = vec![sq];
        action_to_coords.push(coords.clone());
        coords_to_action.insert(coords, idx);
        idx += 1;
    }

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

    for r1 in 0..BOARD_ROWS {
        for c1 in 0..BOARD_COLS {
            let from_sq = r1 * BOARD_COLS + c1;

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

fn build_ray_attacks() -> Vec<Vec<u64>> {
    let mut ray_attacks = vec![vec![0u64; TOTAL_POSITIONS]; NUM_DIRECTIONS];
    for sq in 0..TOTAL_POSITIONS {
        let r = sq / BOARD_COLS;
        let c = sq % BOARD_COLS;

        for i in (0..r).rev() {
            let target_sq = i * BOARD_COLS + c;
            ray_attacks[DIRECTION_UP][sq] |= ull(target_sq);
        }

        for i in (r + 1)..BOARD_ROWS {
            let target_sq = i * BOARD_COLS + c;
            ray_attacks[DIRECTION_DOWN][sq] |= ull(target_sq);
        }

        for i in (0..c).rev() {
            let target_sq = r * BOARD_COLS + i;
            ray_attacks[DIRECTION_LEFT][sq] |= ull(target_sq);
        }

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

/// 观察空间数据结构
#[derive(Debug, Clone)]
pub struct Observation {
    // board shape: (Stack, Channels, H, W) = (2, 8, 3, 4)
    pub board: Array4<f32>,
    // scalar shape: (Stack, Features) = (2, 56)
    pub scalars: Array1<f32>,
}

#[derive(Clone, Debug)]
pub struct DarkChessEnv {
    // 游戏核心状态 (改为存储 Slot)
    board: Vec<Slot>,
    current_player: Player,
    move_counter: usize,
    total_step_counter: usize,

    // 向量化状态 (Bitboard 化)
    // HashMap 存储 Red/Black，内部 Vec 存储 7 种棋子类型 (u64)
    piece_bitboards: HashMap<Player, [u64; NUM_PIECE_TYPES]>,
    revealed_bitboards: HashMap<Player, u64>,
    hidden_bitboard: u64,
    empty_bitboard: u64,

    // 存活与死亡记录
    dead_pieces: HashMap<Player, Vec<PieceType>>,

    // 血量追踪（减分制，初始60分，被吃子扣血）
    scores: HashMap<Player, i32>,

    last_action: i32,

    // 历史堆叠
    board_history: VecDeque<Vec<f32>>,
    scalar_history: VecDeque<Vec<f32>>,

    // 隐藏棋子池 (Bag Model: 翻棋时从此池中随机抽取)
    pub hidden_pieces: Vec<Piece>,

    // 翻子事件概率表
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

        action_lookup_tables();
        ray_attacks();
        env.reset();
        env
    }

    /// 根据动作编号获取对应的坐标列表（便于调试/可视化）
    pub fn get_coords_for_action(&self, action: usize) -> Option<&Vec<usize>> {
        action_lookup_tables().action_to_coords.get(action)
    }

    /// 自定义场景：仅保留红黑士 — 已弃用。为兼容旧调用，提供空实现。
    pub fn setup_two_advisors(&mut self, _current_player: Player) {
        // no-op: 保持现有随机初始局面
    }

    /// 自定义场景：隐藏的威胁 — 已弃用。为兼容旧调用，提供空实现。
    pub fn setup_hidden_threats(&mut self) {
        // no-op: 保持现有随机初始局面
    }

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

        // 初始化血量（减分制）：每方从初始血量值开始
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

    fn initialize_board(&mut self) {
        let mut rng = thread_rng();

        // 1. 生成实际棋子池 (Bag)
        let mut pieces = Vec::new();
        for &player in &[Player::Red, Player::Black] {
            // 每方: GENERALS_COUNT 将, ADVISORS_COUNT 士, ELEPHANTS_COUNT 象, CHARIOTS_COUNT 车,
            // HORSES_COUNT 马, CANNONS_COUNT 炮, SOLDIERS_COUNT 兵
            for _ in 0..GENERALS_COUNT {
                pieces.push(Piece::new(PieceType::General, player));
            }
            for _ in 0..ADVISORS_COUNT {
                pieces.push(Piece::new(PieceType::Advisor, player));
            }
            for _ in 0..ELEPHANTS_COUNT {
                pieces.push(Piece::new(PieceType::Elephant, player));
            }
            for _ in 0..CHARIOTS_COUNT {
                pieces.push(Piece::new(PieceType::Chariot, player));
            }
            for _ in 0..HORSES_COUNT {
                pieces.push(Piece::new(PieceType::Horse, player));
            }
            for _ in 0..CANNONS_COUNT {
                pieces.push(Piece::new(PieceType::Cannon, player));
            }
            for _ in 0..SOLDIERS_COUNT {
                pieces.push(Piece::new(PieceType::Soldier, player));
            }
        }
        // 打乱棋子池
        pieces.shuffle(&mut rng);
        self.hidden_pieces = pieces; // 这些是等待被翻出来的棋子

        // 2. 生成棋盘布局 (Layout): 32个 Hidden, 0个 Empty (总共32个位置)
        // Python逻辑: board_setup = pieces + empty_slots -> shuffle
        // 这里我们用 Hidden 替代具体的 Piece，直到翻开时才决定是哪个 Piece
        let piece_count = self.hidden_pieces.len(); // 32
        let empty_count = TOTAL_POSITIONS - piece_count; // 0

        let mut layout = Vec::with_capacity(TOTAL_POSITIONS);
        for _ in 0..piece_count {
            layout.push(Slot::Hidden);
        }
        for _ in 0..empty_count {
            layout.push(Slot::Empty);
        }

        // 打乱布局
        layout.shuffle(&mut rng);

        // 3. 填充棋盘并初始化向量
        self.empty_bitboard = 0; // 初始棋盘满子 (4x8=32)
        self.hidden_bitboard = BOARD_MASK; // 全部为暗子

        for sq in 0..TOTAL_POSITIONS {
            self.board[sq] = Slot::Hidden;
        }

        // 计算初始概率表
        self.update_reveal_probabilities();

        // 4. 随机翻开 N 个 Hidden 位置
        if TOTAL_POSITIONS > 0 {
            // 确保棋盘非空
            let mut hidden_indices: Vec<usize> = (0..TOTAL_POSITIONS).collect();
            hidden_indices.shuffle(&mut rng);
            let reveal_count = std::cmp::min(hidden_indices.len(), INITIAL_REVEALED_PIECES);

            for &idx in hidden_indices.iter().take(reveal_count) {
                // 这里调用 reveal_piece_at，它会更新 Bitboard
                self.reveal_piece_at(idx, None);
            }
        }
    }

    pub fn reset(&mut self) -> Observation {
        self.reset_internal_state();
        self.initialize_board();

        let initial_board = self.get_board_state_tensor();
        let initial_scalar = self.get_scalar_state_vector();

        for _ in 0..STATE_STACK_SIZE {
            self.board_history.push_back(initial_board.clone());
            self.scalar_history.push_back(initial_scalar.clone());
        }

        self.get_state()
    }

    // --- 翻子相关方法 ---

    /// 翻开指定位置的棋子并更新 Bitboards
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
        self.update_history();

        let (terminated, truncated, winner) = self.check_game_over_conditions();
        Ok((self.get_state(), 0.0, terminated, truncated, winner))
    }

    fn apply_move_action(&mut self, from_sq: usize, to_sq: usize, reveal_piece: Option<Piece>) {
        // 提取源棋子 (必须是 Revealed)
        let attacker = match std::mem::replace(&mut self.board[from_sq], Slot::Empty) {
            Slot::Revealed(p) => p,
            _ => panic!("Move action source is not a revealed piece!"),
        };

        // 如果目标是暗子，先翻开
        if matches!(self.board[to_sq], Slot::Hidden) {
            self.reveal_piece_at(to_sq, reveal_piece);
        }

        // 统一处理移动或吃子（此时目标已经是明子或空位）
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
                self.move_counter += 1;
            }
            Slot::Revealed(defender) => {
                // 3. 处理吃子 (Defender is captured at to_sq)
                let opp = defender.player;
                let opp_pt = defender.piece_type as usize;

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

    // --- 状态获取 ---
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

        // My pieces (Revealed)
        for pt in 0..NUM_PIECE_TYPES {
            push_bitboard(self.piece_bitboards[&my][pt]);
        }
        // Opponent pieces (Revealed)
        for pt in 0..NUM_PIECE_TYPES {
            push_bitboard(self.piece_bitboards[&opp][pt]);
        }
        // Hidden & Empty
        push_bitboard(self.hidden_bitboard);
        push_bitboard(self.empty_bitboard);

        tensor
    }

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

        vec.push(self.move_counter as f32 / MAX_CONSECUTIVE_MOVES_FOR_DRAW as f32);
        vec.push(self.get_hp(my) as f32 / INITIAL_HEALTH_POINTS as f32);
        vec.push(self.get_hp(opp) as f32 / INITIAL_HEALTH_POINTS as f32);

        for &player in &[my, opp] {
            let bitboards = &self.piece_bitboards[&player];
            for pt in 0..NUM_PIECE_TYPES {
                let count = bitboards[pt].count_ones() as usize;
                let max_count = PIECE_MAX_COUNTS[pt];
                vec.extend(std::iter::repeat(1.0).take(count));
                vec.extend(std::iter::repeat(0.0).take(max_count - count));
            }
        }

        // 使用提供的 action_mask 缓冲区
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

    /// 使用提供的缓冲区填充 Observation，避免重复分配内存
    pub fn get_state_into(&self, board_data: &mut Vec<f32>, scalars_data: &mut Vec<f32>) -> Observation {
        board_data.clear();
        board_data.reserve(STATE_STACK_SIZE * BOARD_CHANNELS * BOARD_ROWS * BOARD_COLS);
        for frame in &self.board_history {
            board_data.extend_from_slice(frame);
        }
        let board = Array4::from_shape_vec(
            (STATE_STACK_SIZE, BOARD_CHANNELS, BOARD_ROWS, BOARD_COLS),
            board_data.clone(), // ndarray 需要拥有数据
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

    // --- 规则检查 ---
    fn check_game_over_conditions(&self) -> (bool, bool, Option<i32>) {
        // 1. 血量归零：任一方血量降至0或以下则失败（减分制）
        if self.scores[&Player::Red] <= 0 {
            return (true, false, Some(Player::Black.val()));
        }
        if self.scores[&Player::Black] <= 0 {
            return (true, false, Some(Player::Red.val()));
        }

        // 2. 全灭判定：每方共TOTAL_PIECES_PER_PLAYER个棋子
        if self.dead_pieces[&Player::Red].len() == TOTAL_PIECES_PER_PLAYER {
            return (true, false, Some(Player::Black.val()));
        }
        if self.dead_pieces[&Player::Black].len() == TOTAL_PIECES_PER_PLAYER {
            return (true, false, Some(Player::Red.val()));
        }

        // 3. 无棋可走：对手无任何合法动作时，当前玩家获胜
        let masks = self.get_action_masks_for_player(self.current_player);
        if masks.iter().all(|&x| x == 0) {
            return (true, false, Some(self.current_player.opposite().val()));
        }

        // 4. 平局条件：回合限制 - 连续24个回合没有吃子发生
        if self.move_counter >= MAX_CONSECUTIVE_MOVES_FOR_DRAW {
            return (true, false, Some(0));
        }

        // 5. 平局条件：步数限制 - 游戏总步数达到100步
        if self.total_step_counter >= MAX_STEPS_PER_EPISODE {
            return (false, true, Some(0));
        }

        (false, false, None)
    }
    // --- 动作掩码 ---
    pub fn action_masks(&self) -> Vec<i32> {
        let mut mask = vec![0; ACTION_SPACE_SIZE];
        self.action_masks_into(&mut mask);
        mask
    }

    /// 填充动作掩码到提供的缓冲区，避免重复分配
    pub fn action_masks_into(&self, mask: &mut [i32]) {
        self.get_action_masks_for_player_into(self.current_player, mask);
    }

    fn get_action_masks_for_player(&self, player: Player) -> Vec<i32> {
        let mut mask = vec![0; ACTION_SPACE_SIZE];
        self.get_action_masks_for_player_into(player, &mut mask);
        mask
    }

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
        // 2. 准备位棋盘数据 (Bitboards Preparation)
        // --------------------------------------------------------
        let empty_bb = self.empty_bitboard;
        let my = player;
        let opp = player.opposite();

        let my_revealed_bb = *self.revealed_bitboards.get(&my).unwrap();

        let my_piece_bb = *self.piece_bitboards.get(&my).unwrap();
        let opp_piece_bb = *self.piece_bitboards.get(&opp).unwrap();

        // --------------------------------------------------------
        // 3. 常规移动 (Regular Moves)
        // 对应 env.py: Loop over piece types, shift bits
        // --------------------------------------------------------

        // 计算目标集合 (Target Bitboards)
        // 规则: 目标可以是 空位 OR 比自己弱/同级的敌方棋子
        let mut target_bbs: [u64; NUM_PIECE_TYPES] = [0; NUM_PIECE_TYPES];
        let mut cumulative_targets: u64 = empty_bb; // 初始包含空位

        // 按等级累积敌方棋子: Soldier(0) -> General(6)
        for pt in 0..NUM_PIECE_TYPES {
            cumulative_targets |= opp_piece_bb[pt];
            target_bbs[pt] = cumulative_targets;
        }

        // 特殊规则修正:
        // 1. 兵(0) 可以吃 将(6) -> 将 Soldier 的目标加上敌方 General
        target_bbs[PieceType::Soldier as usize] |= opp_piece_bb[PieceType::General as usize];
        // 2. 将(6) 不能吃 兵(0) -> 将 General 的目标移除敌方 Soldier
        target_bbs[PieceType::General as usize] &= !opp_piece_bb[PieceType::Soldier as usize];

        // 移动偏移量与边界检查
        // env.py: shifts = [-4 (Up), 4 (Down), -1 (Left), 1 (Right)]
        // Rust: 对应BOARD_COLS
        let shifts = [
            -(BOARD_COLS as isize) as i32, // Up
            (BOARD_COLS as i32),           // Down
            -1,                            // Left
            1,                             // Right
        ];

        // 边界掩码 (防止左右移动穿越棋盘边缘)
        let not_file_a = BOARD_MASK & !file_mask(0); // 非第一列
        let not_file_h = BOARD_MASK & !file_mask(BOARD_COLS - 1); // 非最后一列

        // 对应 shifts 的 wrap_check
        // Up/Down 不需要列掩码，Left 需要 not_file_a，Right 需要 not_file_h
        let wrap_checks = [BOARD_MASK, BOARD_MASK, not_file_a, not_file_h];

        for pt in 0..NUM_PIECE_TYPES {
            // 炮(1) 不走常规移动逻辑
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

                // 计算潜在目标位 (纯位移)
                let potential_to_bb = if shift > 0 {
                    (temp_from_bb << (shift as u32)) & BOARD_MASK
                } else {
                    (temp_from_bb >> ((-shift) as u32)) & BOARD_MASK
                };

                // 过滤有效目标 (必须在 target_bbs 中)
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
        // --------------------------------------------------------
        let my_cannons_bb = my_piece_bb[PieceType::Cannon as usize];
        if my_cannons_bb != 0 {
            let all_pieces_bb = BOARD_MASK & !empty_bb;

            // 炮的合法目标: 非(己方明子) => 敌方明子 + 任何暗子
            let valid_cannon_targets = BOARD_MASK & (!my_revealed_bb);

            let mut temp_cannons = my_cannons_bb;
            let ray_attacks = ray_attacks();
            while temp_cannons != 0 {
                let from_sq = pop_lsb(&mut temp_cannons);

                // 遍历 NUM_DIRECTIONS 个方向: DIRECTION_UP, DIRECTION_DOWN, DIRECTION_LEFT, DIRECTION_RIGHT
                for dir in 0..NUM_DIRECTIONS {
                    let ray_bb = ray_attacks[dir][from_sq];
                    let blockers = ray_bb & all_pieces_bb;

                    if blockers == 0 {
                        continue;
                    }

                    // 寻找炮架 (Screen): 离 from_sq 最近的阻挡物
                    let screen_sq = match dir {
                        DIRECTION_UP | DIRECTION_LEFT => msb_index(blockers), // Up/Left: 索引减小 -> 离from最近的是最大索引
                        _ => Some(trailing_zeros(blockers)), // Down/Right: 索引增大 -> 离from最近是最小索引
                    };

                    if screen_sq.is_none() {
                        continue;
                    }
                    let screen_sq = screen_sq.unwrap();

                    // 寻找目标 (Target): 炮架后的第一个棋子
                    let after_screen_ray = ray_attacks[dir][screen_sq];
                    let targets = after_screen_ray & all_pieces_bb;

                    if targets == 0 {
                        continue;
                    }

                    // 寻找目标 (Target) - 同样的最近原则
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

    pub fn is_cannon_attack_on_hidden(&self, action: usize) -> bool {
        if action < REVEAL_ACTIONS_COUNT {
            return false; // 翻棋动作已经处理
        }

        if action < REVEAL_ACTIONS_COUNT + REGULAR_MOVE_ACTIONS_COUNT {
            return false; // 常规移动不是炮攻击
        }

    let coords = &action_lookup_tables().action_to_coords[action];
        let to_sq = coords[1];
        matches!(self.board[to_sq], Slot::Hidden)
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
        println!("Dead (Red): {:?}", self.dead_pieces[&Player::Red]);
        println!("Dead (Black): {:?}", self.dead_pieces[&Player::Black]);
        println!("---------------------------------------------");
    }

    // === 公共访问器方法 (用于 GUI) ===

    /// 获取棋盘状态的字符串表示
    pub fn get_board_slots(&self) -> &Vec<Slot> {
        &self.board
    }

    /// 获取当前玩家
    pub fn get_current_player(&self) -> Player {
        self.current_player
    }

    /// 获取移动计数器
    pub fn get_move_counter(&self) -> usize {
        self.move_counter
    }

    /// 获取总步数
    pub fn get_total_steps(&self) -> usize {
        self.total_step_counter
    }

    /// 获取当前血量（减分制）
    pub fn get_score(&self, player: Player) -> i32 {
        *self.scores.get(&player).unwrap_or(&0)
    }

    /// 获取双方血量（减分制）
    pub fn get_scores(&self) -> (i32, i32) {
        (self.get_score(Player::Red), self.get_score(Player::Black))
    }

    /// 获取单个玩家的血量
    pub fn get_hp(&self, player: Player) -> i32 {
        self.get_score(player)
    }

    /// 获取已阵亡的棋子
    pub fn get_dead_pieces(&self, player: Player) -> &Vec<PieceType> {
        &self.dead_pieces[&player]
    }

    /// 获取隐藏棋子（按玩家分类）
    pub fn get_hidden_pieces(&self, player: Player) -> Vec<PieceType> {
        self.hidden_pieces
            .iter()
            .filter(|p| p.player == player)
            .map(|p| p.piece_type)
            .collect()
    }

    /// 根据坐标获取对应的动作编号
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

        // 隐藏和空位
        bitboards.insert("hidden".to_string(), bb_to_vec(self.hidden_bitboard));
        bitboards.insert("empty".to_string(), bb_to_vec(self.empty_bitboard));

        // 棋子类型名称
        const PIECE_NAMES: [&str; NUM_PIECE_TYPES] = [
            "soldier", "cannon", "horse", "chariot", "elephant", "advisor", "general",
        ];

        // 遍历双方玩家
        for &player in &[Player::Red, Player::Black] {
            let prefix = match player {
                Player::Red => "red",
                Player::Black => "black",
            };

            // Revealed bitboard
            bitboards.insert(
                format!("{}_revealed", prefix),
                bb_to_vec(self.revealed_bitboards[&player]),
            );

            // 各类型棋子
            for (pt, &name) in PIECE_NAMES.iter().enumerate() {
                bitboards.insert(
                    format!("{}_{}", prefix, name),
                    bb_to_vec(self.piece_bitboards[&player][pt]),
                );
            }
        }

        bitboards
    }

    /*
            // ------------------------------------------------------------------
            // 注意: 以下自定义场景函数已过时，需要根据新的4x8棋盘和7种棋子重新设计
            // ------------------------------------------------------------------

        // ------------------------------------------------------------------
        // 自定义场景: 仅剩红方士(R_A)与黑方士(B_A)，其它棋子全部阵亡
        // current_player 指定当前行动方
        // ------------------------------------------------------------------
        pub fn setup_two_advisors(&mut self, current_player: Player) {
            // 复位内部状态再手动布置
            self.reset_internal_state();
            self.current_player = current_player;

            // 放置棋子: R_A at index 0 (row0 col0), B_A at index 9 (row2 col1)
            let red_adv = Piece::new(PieceType::Advisor, Player::Red);
            let black_adv = Piece::new(PieceType::Advisor, Player::Black);
            self.board[0] = Slot::Revealed(red_adv.clone());
            self.board[9] = Slot::Revealed(black_adv.clone());

            // 更新 bitsets
            // 先全部清空 (reset_internal_state 已完成), 填充 empty_vector
            for sq in 0..TOTAL_POSITIONS {
                self.empty_vector.set(sq, true);
            }
            // 设置已占用格
            self.empty_vector.set(0, false);
            self.empty_vector.set(9, false);

            // Revealed vectors
            self.revealed_vectors.get_mut(&Player::Red).unwrap().set(0, true);
            self.revealed_vectors.get_mut(&Player::Black).unwrap().set(9, true);

            // Piece vectors (Advisor index = 1)
            self.piece_vectors.get_mut(&Player::Red).unwrap()[PieceType::Advisor as usize].set(0, true);
            self.piece_vectors.get_mut(&Player::Black).unwrap()[PieceType::Advisor as usize].set(9, true);

            // 其它棋子全部阵亡: 士存活, 两兵+将阵亡
            self.dead_pieces.get_mut(&Player::Red).unwrap().extend([PieceType::Soldier, PieceType::Soldier, PieceType::General]);
            self.dead_pieces.get_mut(&Player::Black).unwrap().extend([PieceType::Soldier, PieceType::Soldier, PieceType::General]);

            // 存活向量: [Sol, Sol, Adv, Gen] 只保留 Adv
            self.survival_vectors.get_mut(&Player::Red).unwrap().clone_from(&vec![0.0, 0.0, 1.0, 0.0]);
            self.survival_vectors.get_mut(&Player::Black).unwrap().clone_from(&vec![0.0, 0.0, 1.0, 0.0]);

            // 无隐藏棋子
            self.hidden_pieces.clear();
            self.hidden_vector = FixedBitSet::with_capacity(TOTAL_POSITIONS); // 全空
            self.update_reveal_probabilities(); // 设置为 0

            // 历史堆叠 (填充两帧相同状态)
            let board_state = self.get_board_state_tensor();
            let scalar_state = self.get_scalar_state_vector();
            for _ in 0..STATE_STACK_SIZE {
                self.board_history.push_back(board_state.clone());
                self.scalar_history.push_back(scalar_state.clone());
            }
        }

        /// 公开动作 -> 坐标查询 (用于调试/验证)
        pub fn get_coords_for_action(&self, action: usize) -> Option<&Vec<usize>> {
            self.action_to_coords.get(&action)
        }


    // ------------------------------------------------------------------
        // 自定义场景 2: 隐藏的威胁
        // 棋子: R_Adv(明), B_Adv(暗), B_Sol(暗)
        // 布局:
        // ___ ___  ___ Hid  (row 0: pos 2)
        // ___ Hid ___ ___  (row 1: pos 5)
        // ___ R_A ___ ___  (row 2: pos 9)
        //
        // 当前轮次: Black (需要做决定翻哪个)
        // ------------------------------------------------------------------
        pub fn setup_hidden_threats(&mut self) {
            // 重置状态
            self.reset_internal_state();
            self.current_player = Player::Black; // 轮到黑方行动

            // 1. 放置明子: R_Adv 在位置 9 (Row 2, Col 1)
            let red_adv = Piece::new(PieceType::Advisor, Player::Red);
            self.board[9] = Slot::Revealed(red_adv);

            // 2. 设置暗子位置: 3 和 5
            self.board[3] = Slot::Hidden;
            self.board[5] = Slot::Hidden;

            // 3. 设置隐藏棋子池 (Bag Model)
            // 包含 B_Adv 和 B_Sol
            let black_adv = Piece::new(PieceType::Advisor, Player::Black);
            let black_sol = Piece::new(PieceType::Soldier, Player::Black);
            self.hidden_pieces = vec![black_adv, black_sol];

            // 4. 更新辅助向量 (Bitsets)
            // Empty: 除了 3, 5, 9 以外全为 true
            for sq in 0..TOTAL_POSITIONS {
                self.empty_vector.set(sq, true);
            }
            self.empty_vector.set(3, false);
            self.empty_vector.set(5, false);
            self.empty_vector.set(9, false);

            // Hidden Vector
            self.hidden_vector.set(3, true);
            self.hidden_vector.set(5, true);

            // Revealed Vectors & Piece Vectors (只有 R_Adv 在 9)
            self.revealed_vectors.get_mut(&Player::Red).unwrap().set(9, true);
            self.piece_vectors.get_mut(&Player::Red).unwrap()[PieceType::Advisor as usize].set(9, true);

            // 5. 设置阵亡名单 (Dead Pieces)
            // 初始每方: 1将, 1士, 2兵
            // 红方存活: 1士. 阵亡: 1将, 2兵
            self.dead_pieces.get_mut(&Player::Red).unwrap().extend([
                PieceType::General,
                PieceType::Soldier,
                PieceType::Soldier
            ]);
            // 黑方存活(在池中): 1士, 1兵. 阵亡: 1将, 1兵
            self.dead_pieces.get_mut(&Player::Black).unwrap().extend([
                PieceType::General,
                PieceType::Soldier
            ]);

            // 6. 更新存活向量 (Survival Vectors) [Sol, Sol, Adv, Gen]
            // Red: 只有 Adv 存活 -> [0, 0, 1, 0]
            self.survival_vectors.get_mut(&Player::Red).unwrap().clone_from(&vec![0.0, 0.0, 1.0, 0.0]);
            // Black: 1 Sol, 1 Adv 存活 -> [1, 0, 1, 0]
            self.survival_vectors.get_mut(&Player::Black).unwrap().clone_from(&vec![1.0, 0.0, 1.0, 0.0]);

            // 7. 计算翻棋概率
            self.update_reveal_probabilities();

            // 8. 填充历史堆叠
            let board_state = self.get_board_state_tensor();
            let scalar_state = self.get_scalar_state_vector();
            for _ in 0..STATE_STACK_SIZE {
                self.board_history.push_back(board_state.clone());
                self.scalar_history.push_back(scalar_state.clone());
            }
        }
        */
}
