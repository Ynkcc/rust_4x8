use rand::seq::SliceRandom;
use rand::thread_rng;
use rand::Rng;
use std::collections::{HashMap, VecDeque};
use std::fmt;
use fixedbitset::FixedBitSet;
use ndarray::{Array1, Array4};
use serde::{Serialize, Deserialize};

// ==============================================================================
// --- 常量定义 (与 Python environment.py 保持一致) ---
// ==============================================================================

pub const STATE_STACK_SIZE: usize = 1; // 禁用状态堆叠，仅使用当前帧
const MAX_CONSECUTIVE_MOVES_FOR_DRAW: usize = 14;
const MAX_STEPS_PER_EPISODE: usize = 100;
pub const BOARD_ROWS: usize = 3;
pub const BOARD_COLS: usize = 4;
pub const TOTAL_POSITIONS: usize = BOARD_ROWS * BOARD_COLS;
pub const NUM_PIECE_TYPES: usize = 3;
const INITIAL_REVEALED_PIECES: usize = 2;

// 存活向量大小: 2(兵) + 1(士) + 1(将) = 4
const SURVIVAL_VECTOR_SIZE: usize = 4;

// 动作空间
pub const REVEAL_ACTIONS_COUNT: usize = 12;
pub const REGULAR_MOVE_ACTIONS_COUNT: usize = 34;
pub const ACTION_SPACE_SIZE: usize = REVEAL_ACTIONS_COUNT + REGULAR_MOVE_ACTIONS_COUNT; // 46

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PieceType {
    Soldier = 0,
    Advisor = 1,
    General = 2,
}

impl PieceType {
    fn from_usize(u: usize) -> Option<Self> {
        match u {
            0 => Some(PieceType::Soldier),
            1 => Some(PieceType::Advisor),
            2 => Some(PieceType::General),
            _ => None,
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
        Self {
            piece_type,
            player,
        }
    }

    fn short_name(&self) -> String {
        let p_char = match self.player {
            Player::Red => "R",
            Player::Black => "B",
        };
        let t_char = match self.piece_type {
            PieceType::General => "Gen",
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
    
    // 向量化状态
    piece_vectors: HashMap<Player, Vec<FixedBitSet>>, 
    revealed_vectors: HashMap<Player, FixedBitSet>,
    hidden_vector: FixedBitSet,
    empty_vector: FixedBitSet,
    
    // 存活与死亡记录
    dead_pieces: HashMap<Player, Vec<PieceType>>,
    survival_vectors: HashMap<Player, Vec<f32>>,
    
    last_action: i32,

    // 历史堆叠
    board_history: VecDeque<Vec<f32>>,
    scalar_history: VecDeque<Vec<f32>>,

    // 查找表
    action_to_coords: HashMap<usize, Vec<usize>>, 
    coords_to_action: HashMap<Vec<usize>, usize>, 
    
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
            
            piece_vectors: HashMap::new(),
            revealed_vectors: HashMap::new(),
            hidden_vector: FixedBitSet::with_capacity(TOTAL_POSITIONS),
            empty_vector: FixedBitSet::with_capacity(TOTAL_POSITIONS),
            
            dead_pieces: HashMap::new(),
            survival_vectors: HashMap::new(),
            
            last_action: -1,
            
            board_history: VecDeque::with_capacity(STATE_STACK_SIZE),
            scalar_history: VecDeque::with_capacity(STATE_STACK_SIZE),
            
            action_to_coords: HashMap::new(),
            coords_to_action: HashMap::new(),
            
            hidden_pieces: Vec::new(),
            reveal_probabilities: vec![0.0; 6],
        };

        env.initialize_lookup_tables();
        env.reset();
        env
    }

    fn initialize_lookup_tables(&mut self) {
        let mut idx = 0;
        
        // 1. 翻棋动作 (0-11)
        for sq in 0..TOTAL_POSITIONS {
            self.action_to_coords.insert(idx, vec![sq]);
            self.coords_to_action.insert(vec![sq], idx);
            idx += 1;
        }

        // 2. 常规移动 (12-45)
        let moves = [(-1, 0), (1, 0), (0, -1), (0, 1)];
        for r1 in 0..BOARD_ROWS {
            for c1 in 0..BOARD_COLS {
                let from_sq = r1 * BOARD_COLS + c1;
                for (dr, dc) in moves.iter() {
                    let r2 = r1 as i32 + dr;
                    let c2 = c1 as i32 + dc;
                    
                    if r2 >= 0 && r2 < BOARD_ROWS as i32 && c2 >= 0 && c2 < BOARD_COLS as i32 {
                        let to_sq = (r2 as usize) * BOARD_COLS + (c2 as usize);
                        
                        self.action_to_coords.insert(idx, vec![from_sq, to_sq]);
                        self.coords_to_action.insert(vec![from_sq, to_sq], idx);
                        idx += 1;
                    }
                }
            }
        }
        
        if idx != ACTION_SPACE_SIZE {
            panic!("动作空间计算错误: 预期 {}, 实际 {}", ACTION_SPACE_SIZE, idx);
        }
    }

    fn reset_internal_state(&mut self) {
        self.board = vec![Slot::Empty; TOTAL_POSITIONS];
        
        // 初始化向量
        self.piece_vectors.insert(Player::Red, vec![FixedBitSet::with_capacity(TOTAL_POSITIONS); NUM_PIECE_TYPES]);
        self.piece_vectors.insert(Player::Black, vec![FixedBitSet::with_capacity(TOTAL_POSITIONS); NUM_PIECE_TYPES]);
        
        self.revealed_vectors.insert(Player::Red, FixedBitSet::with_capacity(TOTAL_POSITIONS));
        self.revealed_vectors.insert(Player::Black, FixedBitSet::with_capacity(TOTAL_POSITIONS));
        
        self.hidden_vector = FixedBitSet::with_capacity(TOTAL_POSITIONS);
        self.empty_vector = FixedBitSet::with_capacity(TOTAL_POSITIONS);
        // empty_vector 和 hidden_vector 将在 initialize_board 中根据布局设置
        
        self.dead_pieces.insert(Player::Red, Vec::new());
        self.dead_pieces.insert(Player::Black, Vec::new());
        
        self.current_player = Player::Red;
        self.move_counter = 0;
        self.total_step_counter = 0;
        
        self.survival_vectors.insert(Player::Red, vec![1.0; SURVIVAL_VECTOR_SIZE]);
        self.survival_vectors.insert(Player::Black, vec![1.0; SURVIVAL_VECTOR_SIZE]);
        
        self.last_action = -1;
        
        self.board_history.clear();
        self.scalar_history.clear();
        
        self.hidden_pieces.clear();
        self.reveal_probabilities = vec![0.0; 6];
    }

    fn initialize_board(&mut self) {
        let mut rng = thread_rng();

        // 1. 生成实际棋子池 (Bag)
        let mut pieces = Vec::new();
        for &player in &[Player::Red, Player::Black] {
            pieces.push(Piece::new(PieceType::General, player));
            pieces.push(Piece::new(PieceType::Advisor, player));
            pieces.push(Piece::new(PieceType::Soldier, player));
            pieces.push(Piece::new(PieceType::Soldier, player));
        }
        // 打乱棋子池
        pieces.shuffle(&mut rng);
        self.hidden_pieces = pieces; // 这些是等待被翻出来的棋子

        // 2. 生成棋盘布局 (Layout): 8个 Hidden, 4个 Empty
        // Python逻辑: board_setup = pieces + empty_slots -> shuffle
        // 这里我们用 Hidden 替代具体的 Piece，直到翻开时才决定是哪个 Piece
        let piece_count = self.hidden_pieces.len(); // 8
        let empty_count = TOTAL_POSITIONS - piece_count; // 4
        
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
        self.empty_vector.clear();
        self.hidden_vector.clear();
        
        for (sq, slot) in layout.into_iter().enumerate() {
            self.board[sq] = slot.clone();
            match slot {
                Slot::Empty => self.empty_vector.set(sq, true),
                Slot::Hidden => self.hidden_vector.set(sq, true),
                _ => {} // 初始没有 Revealed
            }
        }
        
        // 计算初始概率表
        self.update_reveal_probabilities();
        
        // 4. 随机翻开 N 个 Hidden 位置
        if INITIAL_REVEALED_PIECES > 0 {
            // 获取所有 Hidden 的位置索引
            let hidden_indices: Vec<usize> = self.hidden_vector.ones().collect();
            
            // 随机选择 N 个进行翻开
            // 注意：这里选择的是位置，翻开时会从 hidden_pieces 池中消耗棋子
            let reveal_count = std::cmp::min(hidden_indices.len(), INITIAL_REVEALED_PIECES);
            let chosen_indices = hidden_indices
                .choose_multiple(&mut rng, reveal_count)
                .cloned()
                .collect::<Vec<_>>();

            for idx in chosen_indices {
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
    
    /// 翻开指定位置的棋子
    /// 逻辑：检查该位置是否为 Hidden
    /// 如果 specified_piece 为 None，从 hidden_pieces 池中随机取出一个
    /// 如果 specified_piece 为 Some(p)，则强制取出该棋子（如果不在池中则 panic）
    fn reveal_piece_at(&mut self, sq: usize, specified_piece: Option<Piece>) {
        // 确保位置是 Hidden
        match self.board[sq] {
            Slot::Hidden => {},
            _ => panic!("尝试翻开非 Hidden 位置: {}", sq),
        }
        
        if self.hidden_pieces.is_empty() {
            panic!("逻辑错误：棋盘上有 Hidden 位置，但 hidden_pieces 池已空");
        }
        
        let idx = if let Some(target) = specified_piece {
            // 查找指定棋子
            self.hidden_pieces.iter().position(|p| *p == target)
                .expect("指定的棋子不在隐藏棋子池中 (Cheat/Determinization Error)")
        } else {
            // 随机选择
            let mut rng = thread_rng();
            rng.gen_range(0..self.hidden_pieces.len())
        };
        
        // 从池中移除并使用
        let piece = self.hidden_pieces.swap_remove(idx);
        
        // 更新向量
        self.hidden_vector.set(sq, false);
        self.revealed_vectors.get_mut(&piece.player).unwrap().set(sq, true);
        self.piece_vectors.get_mut(&piece.player).unwrap()[piece.piece_type as usize].set(sq, true);
        
        // 更新棋盘状态
        self.board[sq] = Slot::Revealed(piece);
        
        // 更新概率表
        self.update_reveal_probabilities();
        
        // 开启调试模式时验证向量一致性
        #[cfg(debug_assertions)]
        self.verify_vector_consistency();
    }
    
    fn update_reveal_probabilities(&mut self) {
        let total_hidden = self.hidden_pieces.len();
        
        if total_hidden == 0 {
            self.reveal_probabilities = vec![0.0; 6];
            return;
        }
        
        let mut counts = vec![0; 6];
        for piece in &self.hidden_pieces {
            let idx = match (piece.player, piece.piece_type) {
                (Player::Red, PieceType::Soldier) => 0,
                (Player::Red, PieceType::Advisor) => 1,
                (Player::Red, PieceType::General) => 2,
                (Player::Black, PieceType::Soldier) => 3,
                (Player::Black, PieceType::Advisor) => 4,
                (Player::Black, PieceType::General) => 5,
            };
            counts[idx] += 1;
        }
        
        for i in 0..6 {
            self.reveal_probabilities[i] = counts[i] as f32 / total_hidden as f32;
        }
    }
    
    pub fn get_reveal_probabilities(&self) -> &Vec<f32> {
        &self.reveal_probabilities
    }

    // --- 核心 Step 逻辑 ---
    pub fn step(&mut self, action: usize, reveal_piece: Option<Piece>) -> Result<(Observation, f32, bool, bool, Option<i32>), String> {
        let masks = self.action_masks();
        if masks[action] == 0 {
            return Err(format!("无效动作: {}", action));
        }

        self.last_action = action as i32;
        self.total_step_counter += 1;

        if action < REVEAL_ACTIONS_COUNT {
            let sq = self.action_to_coords[&action][0];
            self.reveal_piece_at(sq, reveal_piece);
            self.move_counter = 0;
        } else {
            let coords = &self.action_to_coords[&action];
            let from_sq = coords[0];
            let to_sq = coords[1];
            self.apply_move_action(from_sq, to_sq);
        }

        self.current_player = self.current_player.opposite();
        self.update_history();

        let (terminated, truncated, winner) = self.check_game_over_conditions();
        Ok((self.get_state(), 0.0, terminated, truncated, winner))
    }

    fn apply_move_action(&mut self, from_sq: usize, to_sq: usize) {
        // 先验证吃子规则
        if let Err(e) = self.verify_capture_rules(from_sq, to_sq) {
            panic!("吃子规则验证失败: {}", e);
        }
        
        // 提取源棋子 (必须是 Revealed)
        let attacker = match std::mem::replace(&mut self.board[from_sq], Slot::Empty) {
            Slot::Revealed(p) => p,
            _ => panic!("Move action source is not a revealed piece!"),
        };
        
        // 处理目标位置
        let target_slot = std::mem::replace(&mut self.board[to_sq], Slot::Revealed(attacker.clone()));

        // 更新源位置向量
        let p = attacker.player;
        let pt = attacker.piece_type as usize;
        
        self.piece_vectors.get_mut(&p).unwrap()[pt].set(from_sq, false);
        self.revealed_vectors.get_mut(&p).unwrap().set(from_sq, false);
        self.empty_vector.set(from_sq, true);

        // 更新目标位置向量
        self.piece_vectors.get_mut(&p).unwrap()[pt].set(to_sq, true);
        self.revealed_vectors.get_mut(&p).unwrap().set(to_sq, true);
        self.empty_vector.set(to_sq, false);

        match target_slot {
            Slot::Empty => {
                self.move_counter += 1;
            },
            Slot::Revealed(defender) => {
                // 吃子
                // println!("  [吃子] {} 吃掉 {}", attacker.short_name(), defender.short_name());
                // 被吃掉的棋子是 Revealed，需清理其在向量中的记录
                self.piece_vectors.get_mut(&defender.player).unwrap()[defender.piece_type as usize].set(to_sq, false);
                self.revealed_vectors.get_mut(&defender.player).unwrap().set(to_sq, false);
                
                self.dead_pieces.get_mut(&defender.player).unwrap().push(defender.piece_type);
                self.update_survival_vector_on_capture(&defender);
                self.move_counter = 0;
            },
            Slot::Hidden => {
                panic!("Error: Moved onto a Hidden slot. This should be prevented by action masks.");
            }
        }
        
        // 开启调试模式时验证向量一致性
        #[cfg(debug_assertions)]
        self.verify_vector_consistency();
    }

    fn update_survival_vector_on_capture(&mut self, captured: &Piece) {
        let vec = self.survival_vectors.get_mut(&captured.player).unwrap();
        let (start_idx, count) = match captured.piece_type {
            PieceType::Soldier => (0, 2),
            PieceType::Advisor => (2, 1),
            PieceType::General => (3, 1),
        };
        
        for i in 0..count {
            if vec[start_idx + i] == 1.0 {
                vec[start_idx + i] = 0.0;
                break;
            }
        }
    }
    
    /// 验证向量表示与棋盘状态的一致性
    fn verify_vector_consistency(&self) {
        for sq in 0..TOTAL_POSITIONS {
            match &self.board[sq] {
                Slot::Empty => {
                    if !self.empty_vector.contains(sq) {
                        panic!("位置 {} 是 Empty 但 empty_vector 未设置！", sq);
                    }
                    if self.hidden_vector.contains(sq) {
                        panic!("位置 {} 是 Empty 但 hidden_vector 被设置！", sq);
                    }
                    // 检查没有棋子向量指向这里
                    for player in [Player::Red, Player::Black] {
                        if self.revealed_vectors[&player].contains(sq) {
                            panic!("位置 {} 是 Empty 但 {:?} revealed_vector 被设置！", sq, player);
                        }
                        for pt in 0..NUM_PIECE_TYPES {
                            if self.piece_vectors[&player][pt].contains(sq) {
                                panic!("位置 {} 是 Empty 但 {:?} piece_vector[{}] 被设置！", sq, player, pt);
                            }
                        }
                    }
                },
                Slot::Hidden => {
                    if !self.hidden_vector.contains(sq) {
                        panic!("位置 {} 是 Hidden 但 hidden_vector 未设置！", sq);
                    }
                    if self.empty_vector.contains(sq) {
                        panic!("位置 {} 是 Hidden 但 empty_vector 被设置！", sq);
                    }
                },
                Slot::Revealed(piece) => {
                    if self.empty_vector.contains(sq) {
                        panic!("位置 {} 有棋子 {} 但 empty_vector 被设置！", sq, piece.short_name());
                    }
                    if self.hidden_vector.contains(sq) {
                        panic!("位置 {} 有棋子 {} 但 hidden_vector 被设置！", sq, piece.short_name());
                    }
                    if !self.revealed_vectors[&piece.player].contains(sq) {
                        panic!("位置 {} 有棋子 {} 但 {:?} revealed_vector 未设置！", 
                            sq, piece.short_name(), piece.player);
                    }
                    if !self.piece_vectors[&piece.player][piece.piece_type as usize].contains(sq) {
                        panic!("位置 {} 有棋子 {} 但对应 piece_vector 未设置！", 
                            sq, piece.short_name());
                    }
                    // 检查对手的向量没有指向这里
                    let opponent = piece.player.opposite();
                    if self.revealed_vectors[&opponent].contains(sq) {
                        panic!("位置 {} 有棋子 {} 但对手 {:?} revealed_vector 被设置！", 
                            sq, piece.short_name(), opponent);
                    }
                }
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
        let mut tensor = Vec::with_capacity(8 * TOTAL_POSITIONS);
        
        let my = self.current_player;
        let opp = my.opposite();
        
        // My pieces (Revealed)
        for pt in 0..NUM_PIECE_TYPES {
            for sq in 0..TOTAL_POSITIONS {
                tensor.push(if self.piece_vectors[&my][pt].contains(sq) {1.0} else {0.0});
            }
        }
        // Opponent pieces (Revealed)
        for pt in 0..NUM_PIECE_TYPES {
            for sq in 0..TOTAL_POSITIONS {
                tensor.push(if self.piece_vectors[&opp][pt].contains(sq) {1.0} else {0.0});
            }
        }
        // Hidden
        for sq in 0..TOTAL_POSITIONS {
            tensor.push(if self.hidden_vector.contains(sq) {1.0} else {0.0});
        }
        // Empty
        for sq in 0..TOTAL_POSITIONS {
            tensor.push(if self.empty_vector.contains(sq) {1.0} else {0.0});
        }
        
        tensor
    }

    fn get_scalar_state_vector(&self) -> Vec<f32> {
        let mut vec = Vec::new();
        let my = self.current_player;
        let opp = my.opposite();
        
        vec.push(self.move_counter as f32 / MAX_CONSECUTIVE_MOVES_FOR_DRAW as f32);
        vec.extend_from_slice(&self.survival_vectors[&my]);
        vec.extend_from_slice(&self.survival_vectors[&opp]);
        vec.push(self.total_step_counter as f32 / MAX_STEPS_PER_EPISODE as f32);
        
        // 编码当前玩家的 action_masks (46维)
        let action_masks = self.action_masks();
        let action_masks_float: Vec<f32> = action_masks.iter().map(|&x| x as f32).collect();
        vec.extend(action_masks_float);
        
        vec
    }
    
    pub fn get_state(&self) -> Observation {
        let mut board_data = Vec::with_capacity(STATE_STACK_SIZE * 8 * BOARD_ROWS * BOARD_COLS);
        for frame in &self.board_history {
            board_data.extend_from_slice(frame);
        }
        let board = Array4::from_shape_vec(
            (STATE_STACK_SIZE, 8, BOARD_ROWS, BOARD_COLS),
            board_data
        ).expect("Failed to reshape board array");
        
        let mut scalars_data = Vec::with_capacity(STATE_STACK_SIZE * 56);
        for frame in &self.scalar_history {
            scalars_data.extend_from_slice(frame);
        }
        let scalars = Array1::from_vec(scalars_data);
        
        Observation { board, scalars }
    }

    // --- 规则检查 ---
    fn check_game_over_conditions(&self) -> (bool, bool, Option<i32>) {
        if self.dead_pieces[&Player::Red].len() == 4 {
            return (true, false, Some(Player::Black.val()));
        }
        if self.dead_pieces[&Player::Black].len() == 4 {
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

    // --- 动作掩码 ---
    pub fn action_masks(&self) -> Vec<i32> {
        self.get_action_masks_for_player(self.current_player)
    }

    fn get_action_masks_for_player(&self, player: Player) -> Vec<i32> {
        let mut mask = vec![0; ACTION_SPACE_SIZE];
        
        // 1. 翻棋 (只要是 Hidden Slot 就可以翻)
        for sq in self.hidden_vector.ones() {
            let idx = self.coords_to_action[&vec![sq]];
            mask[idx] = 1;
        }
        
        // 2. 移动和吃子
        self.add_regular_move_masks(&mut mask, player);
        
        mask
    }
    
    fn add_regular_move_masks(&self, mask: &mut Vec<i32>, player: Player) {
        let target_vectors = self.get_valid_target_vectors(player);
        
        for pt_val in 0..NUM_PIECE_TYPES {
            let pt = PieceType::from_usize(pt_val).unwrap();
            
            let piece_locs: Vec<usize> = self.piece_vectors[&player][pt_val].ones().collect();
            let valid_targets = &target_vectors[&pt];
            
            for from_sq in piece_locs {
                let r = from_sq / BOARD_COLS;
                let c = from_sq % BOARD_COLS;
                let moves = [(-1, 0), (1, 0), (0, -1), (0, 1)];
                
                for (dr, dc) in moves.iter() {
                    let nr = r as i32 + dr;
                    let nc = c as i32 + dc;
                    
                    if nr >= 0 && nr < BOARD_ROWS as i32 && nc >= 0 && nc < BOARD_COLS as i32 {
                        let to_sq = (nr as usize) * BOARD_COLS + (nc as usize);
                        
                        // 目标必须是有效吃子对象 或者 是空位
                        if valid_targets.contains(to_sq) || self.empty_vector.contains(to_sq) {
                            if let Some(&idx) = self.coords_to_action.get(&vec![from_sq, to_sq]) {
                                mask[idx] = 1;
                            }
                        }
                    }
                }
            }
        }
    }
    
    /// 验证吃子规则的正确性
    fn verify_capture_rules(&self, from_sq: usize, to_sq: usize) -> Result<(), String> {
        let attacker = match &self.board[from_sq] {
            Slot::Revealed(p) => p,
            _ => return Err(format!("源位置 {} 不是明子", from_sq)),
        };
        
        let defender = match &self.board[to_sq] {
            Slot::Revealed(p) => p,
            Slot::Empty => return Ok(()), // 移动到空位，无需检查吃子规则
            Slot::Hidden => return Err(format!("不能移动到暗子位置 {}", to_sq)),
        };
        
        // 不能吃自己的棋子
        if attacker.player == defender.player {
            return Err(format!("{} 不能吃自己的 {}", attacker.short_name(), defender.short_name()));
        }
        
        // 检查吃子规则
        let can_capture = match (attacker.piece_type, defender.piece_type) {
            // 兵可以吃兵和将
            (PieceType::Soldier, PieceType::Soldier) => true,
            (PieceType::Soldier, PieceType::General) => true,
            (PieceType::Soldier, PieceType::Advisor) => false,
            
            // 士可以吃兵和士
            (PieceType::Advisor, PieceType::Soldier) => true,
            (PieceType::Advisor, PieceType::Advisor) => true,
            (PieceType::Advisor, PieceType::General) => false,
            
            // 将可以吃士和将，不能吃兵
            (PieceType::General, PieceType::Soldier) => false,
            (PieceType::General, PieceType::Advisor) => true,
            (PieceType::General, PieceType::General) => true,
        };
        
        if !can_capture {
            return Err(format!("{} 不能吃 {} (规则不允许)", 
                attacker.short_name(), defender.short_name()));
        }
        
        Ok(())
    }

    fn get_valid_target_vectors(&self, player: Player) -> HashMap<PieceType, FixedBitSet> {
        let opponent = player.opposite();
        let mut target_vectors = HashMap::new();
        
        let mut cumulative_targets = FixedBitSet::with_capacity(TOTAL_POSITIONS);
        
        // Loop 0 (Soldier) -> Can eat Soldier
        cumulative_targets.union_with(&self.piece_vectors[&opponent][0]);
        target_vectors.insert(PieceType::Soldier, cumulative_targets.clone());
        
        // Loop 1 (Advisor) -> Can eat Soldier + Advisor
        cumulative_targets.union_with(&self.piece_vectors[&opponent][1]);
        target_vectors.insert(PieceType::Advisor, cumulative_targets.clone());
        
        // Loop 2 (General) -> Can eat Soldier + Advisor + General
        cumulative_targets.union_with(&self.piece_vectors[&opponent][2]);
        target_vectors.insert(PieceType::General, cumulative_targets.clone());
        
        // 特殊规则修正：
        // 1. 兵可以吃将
        let opp_gen = &self.piece_vectors[&opponent][PieceType::General as usize];
        target_vectors.get_mut(&PieceType::Soldier).unwrap().union_with(opp_gen);
        
        // 2. 将不能吃兵
        let opp_sol = &self.piece_vectors[&opponent][PieceType::Soldier as usize];
        target_vectors.get_mut(&PieceType::General).unwrap().difference_with(opp_sol);
        
        target_vectors
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
        println!("Total Steps: {}, Move Counter: {}", self.total_step_counter, self.move_counter);
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
    
    /// 获取已阵亡的棋子
    pub fn get_dead_pieces(&self, player: Player) -> &Vec<PieceType> {
        &self.dead_pieces[&player]
    }
    
    /// 根据坐标获取对应的动作编号
    pub fn get_action_for_coords(&self, coords: &[usize]) -> Option<usize> {
        self.coords_to_action.get(coords).copied()
    }
    
    /// 获取所有 Bitboards (用于 GUI 可视化)
    /// 返回格式: HashMap<标签, Vec<bool>>，其中 Vec 长度为 TOTAL_POSITIONS
    pub fn get_bitboards(&self) -> std::collections::HashMap<String, Vec<bool>> {
        let mut bitboards = std::collections::HashMap::new();
        
        // Hidden 和 Empty
        bitboards.insert("hidden".to_string(), 
            (0..TOTAL_POSITIONS).map(|sq| self.hidden_vector.contains(sq)).collect());
        bitboards.insert("empty".to_string(), 
            (0..TOTAL_POSITIONS).map(|sq| self.empty_vector.contains(sq)).collect());
        
        // 红方明子
        bitboards.insert("red_revealed".to_string(), 
            (0..TOTAL_POSITIONS).map(|sq| self.revealed_vectors[&Player::Red].contains(sq)).collect());
        bitboards.insert("red_soldier".to_string(), 
            (0..TOTAL_POSITIONS).map(|sq| self.piece_vectors[&Player::Red][PieceType::Soldier as usize].contains(sq)).collect());
        bitboards.insert("red_advisor".to_string(), 
            (0..TOTAL_POSITIONS).map(|sq| self.piece_vectors[&Player::Red][PieceType::Advisor as usize].contains(sq)).collect());
        bitboards.insert("red_general".to_string(), 
            (0..TOTAL_POSITIONS).map(|sq| self.piece_vectors[&Player::Red][PieceType::General as usize].contains(sq)).collect());
        
        // 黑方明子
        bitboards.insert("black_revealed".to_string(), 
            (0..TOTAL_POSITIONS).map(|sq| self.revealed_vectors[&Player::Black].contains(sq)).collect());
        bitboards.insert("black_soldier".to_string(), 
            (0..TOTAL_POSITIONS).map(|sq| self.piece_vectors[&Player::Black][PieceType::Soldier as usize].contains(sq)).collect());
        bitboards.insert("black_advisor".to_string(), 
            (0..TOTAL_POSITIONS).map(|sq| self.piece_vectors[&Player::Black][PieceType::Advisor as usize].contains(sq)).collect());
        bitboards.insert("black_general".to_string(), 
            (0..TOTAL_POSITIONS).map(|sq| self.piece_vectors[&Player::Black][PieceType::General as usize].contains(sq)).collect());
        
        bitboards
    }

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
}