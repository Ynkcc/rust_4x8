use rand::Rng;
use rand::seq::SliceRandom;
use rand::thread_rng;
use rand::SeedableRng;

use super::actions::{action_lookup_tables, pack_coords};
use super::bitboard::{BOARD_MASK, ray_attacks, ull};
use super::constants::*;
use super::traits::get_outcome_id;
use super::types::*;

// ==============================================================================
// --- 环境结构体 (DarkChessEnv) ---
// ==============================================================================

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
    /// 按棋子类型统计的阵亡计数 [PlayerIdx][PieceType]，供存活向量/棋谱推导使用
    dead_piece_counts_by_type: [[u8; NUM_PIECE_TYPES]; 2],

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

    /// 随机环境种子
    pub seed: Option<u64>,
    /// 真实棋子布局 (如果配置了seed，初始化时固定布局)
    pub true_board: Option<[Piece; TOTAL_POSITIONS]>,

    /// 最近一次翻出的棋子（供机会节点子树复用匹配 outcome_id）。
    /// 翻棋动作与吃暗子动作都会更新；普通动作保持旧值（此时 step_outcome_id 不会被调用）。
    last_revealed_piece: Option<Piece>,
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
            dead_piece_counts_by_type: [[0; NUM_PIECE_TYPES]; 2],

            scores: [0; 2],
            last_action: -1,

            // 初始化隐藏池
            hidden_pieces_pool: [Piece::default(); TOTAL_POSITIONS],
            hidden_pieces_count: 0,

            reveal_probabilities: [0.0; REVEAL_PROBABILITY_SIZE],
            seed: None,
            true_board: None,
            last_revealed_piece: None,
        };

        action_lookup_tables();
        ray_attacks();

        env.reset();
        env
    }

    /// 从已解码的棋盘槽位与当前玩家重建环境（供对局记录还原 / 棋谱文字解析使用）。
    ///
    /// 与 `new()` 不同，此构造器不随机生成隐藏棋子，而是直接采用给定的槽位布局，
    /// 因此只能用于校验类操作（如重新生成 action_masks / 展示棋盘），不能用于正常对局。
    pub fn from_board(board: [Slot; TOTAL_POSITIONS], current_player: Player) -> Self {
        let mut env = Self {
            board,
            current_player,
            move_counter: 0,
            total_step_counter: 0,
            piece_bitboards: [[0; NUM_PIECE_TYPES]; 2],
            revealed_bitboards: [0; 2],
            hidden_bitboard: 0,
            empty_bitboard: 0,
            dead_pieces_pool: [[PieceType::default(); TOTAL_PIECES_PER_PLAYER]; 2],
            dead_pieces_count: [0; 2],
            dead_piece_counts_by_type: [[0; NUM_PIECE_TYPES]; 2],
            scores: [INITIAL_HEALTH_POINTS, INITIAL_HEALTH_POINTS],
            last_action: -1,
            hidden_pieces_pool: [Piece::default(); TOTAL_POSITIONS],
            hidden_pieces_count: 0,
            reveal_probabilities: [0.0; REVEAL_PROBABILITY_SIZE],
            seed: None,
            true_board: None,
            last_revealed_piece: None,
        };
        for (sq, slot) in board.iter().enumerate() {
            match slot {
                Slot::Empty => env.empty_bitboard |= ull(sq),
                Slot::Hidden => env.hidden_bitboard |= ull(sq),
                Slot::Revealed(p) => {
                    env.revealed_bitboards[p.player.idx()] |= ull(sq);
                    env.piece_bitboards[p.player.idx()][p.piece_type as usize] |= ull(sq);
                }
            }
        }
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
        self.dead_piece_counts_by_type = [[0; NUM_PIECE_TYPES]; 2];

        self.scores = [INITIAL_HEALTH_POINTS, INITIAL_HEALTH_POINTS];

        self.current_player = Player::Red;
        self.move_counter = 0;
        self.total_step_counter = 0;
        self.last_action = -1;

        self.hidden_pieces_count = 0;
        self.reveal_probabilities = [0.0; REVEAL_PROBABILITY_SIZE];
        self.last_revealed_piece = None;
    }

    /// 初始化棋盘布局 (Shuffle Bag Model)
    fn initialize_board(&mut self) {
        let mut rng_std = self.seed.map(rand::rngs::StdRng::seed_from_u64);

        // 1. 生成实际棋子池 (写入 Buffer)
        let mut idx = 0;
        for &player in &[Player::Red, Player::Black] {
            for _ in 0..GENERALS_COUNT {
                self.hidden_pieces_pool[idx] = Piece::new(PieceType::General, player);
                idx += 1;
            }
            for _ in 0..ADVISORS_COUNT {
                self.hidden_pieces_pool[idx] = Piece::new(PieceType::Advisor, player);
                idx += 1;
            }
            for _ in 0..ELEPHANTS_COUNT {
                self.hidden_pieces_pool[idx] = Piece::new(PieceType::Elephant, player);
                idx += 1;
            }
            for _ in 0..CHARIOTS_COUNT {
                self.hidden_pieces_pool[idx] = Piece::new(PieceType::Chariot, player);
                idx += 1;
            }
            for _ in 0..HORSES_COUNT {
                self.hidden_pieces_pool[idx] = Piece::new(PieceType::Horse, player);
                idx += 1;
            }
            for _ in 0..CANNONS_COUNT {
                self.hidden_pieces_pool[idx] = Piece::new(PieceType::Cannon, player);
                idx += 1;
            }
            for _ in 0..SOLDIERS_COUNT {
                self.hidden_pieces_pool[idx] = Piece::new(PieceType::Soldier, player);
                idx += 1;
            }
        }
        self.hidden_pieces_count = idx;

        // 打乱 slice
        if let Some(ref mut rng) = rng_std {
            self.hidden_pieces_pool[0..self.hidden_pieces_count].shuffle(rng);
            let mut tb = [Piece::default(); TOTAL_POSITIONS];
            let count = self.hidden_pieces_count;
            tb[..count].copy_from_slice(&self.hidden_pieces_pool[0..count]);
            self.true_board = Some(tb);
        } else {
            let mut rng = thread_rng();
            self.hidden_pieces_pool[0..self.hidden_pieces_count].shuffle(&mut rng);
            self.true_board = None;
        }

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
            if let Some(ref mut rng) = rng_std {
                hidden_indices.shuffle(rng);
            } else {
                let mut rng = thread_rng();
                hidden_indices.shuffle(&mut rng);
            }
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
        } else if let Some(tb) = self.true_board {
            let target = tb[sq];
            active_slice
                .iter()
                .position(|p| *p == target)
                .unwrap_or_else(|| panic!("真实棋盘指定棋子不在隐藏池: sq={}, tb[sq]={:?}", sq, target))
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

        let pt_bb = &mut self.piece_bitboards[piece.player.idx()][piece.piece_type as usize];
        *pt_bb |= mask;

        self.board[sq] = Slot::Revealed(piece);
        self.last_revealed_piece = Some(piece);
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
        // 使用栈数组避免每次 step 堆分配掩码（热路径）
        let mut masks = [0i32; ACTION_SPACE_SIZE];
        self.action_masks_into(&mut masks);
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
            std::mem::replace(&mut self.board[to_sq], Slot::Revealed(attacker));

        let attacker_mask = ull(from_sq);
        let defender_mask = ull(to_sq);
        let p = attacker.player;
        let pt = attacker.piece_type as usize;

        // --- 1. 清除攻击方在源格 from_sq 的 bitboard，源格标为空位 ---
        self.revealed_bitboards[p.idx()] &= !attacker_mask;
        self.piece_bitboards[p.idx()][pt] &= !attacker_mask;
        self.empty_bitboard |= attacker_mask;

        // --- 2. 目标格 to_sq：彻底清除所有既有归属，再写入攻击方 ---
        // 这样保证 bitboard 与 board 数组严格一致，且不依赖"被吃子必为对方"的假设。
        // （关键修复：炮隔子打己方暗子翻开己方棋子时，被翻开子与攻击方同阵营，
        //   旧的增量清除逻辑会把攻击方自身从目标格误清，导致该格在观测中"全 0"。）
        self.hidden_bitboard &= !defender_mask;
        self.empty_bitboard &= !defender_mask;
        for player_idx in 0..2 {
            self.revealed_bitboards[player_idx] &= !defender_mask;
            for t in 0..NUM_PIECE_TYPES {
                self.piece_bitboards[player_idx][t] &= !defender_mask;
            }
        }
        self.revealed_bitboards[p.idx()] |= defender_mask;
        self.piece_bitboards[p.idx()][pt] |= defender_mask;

        match target_slot {
            Slot::Empty => {
                self.move_counter += 1;
            }
            Slot::Revealed(defender) => {
                // 吃子：移除被攻击棋子并扣其所属方血量。
                // 注意：炮可以攻击同阵营暗子（翻开确认身份），defender 可能为己方棋子，
                // 此时同样移除它并扣己方血量——规则允许炮吃己方暗子。
                let victim_idx = defender.player.idx();
                let dead_idx = self.dead_pieces_count[victim_idx];
                if dead_idx < TOTAL_PIECES_PER_PLAYER {
                    self.dead_pieces_pool[victim_idx][dead_idx] = defender.piece_type;
                    self.dead_pieces_count[victim_idx] += 1;
                    // 按棋子类型累计阵亡计数，供存活向量与棋谱推导
                    self.dead_piece_counts_by_type[victim_idx][defender.piece_type as usize] += 1;
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

    pub fn get_target_slot(&self, action: usize) -> Slot {
        let coords = &action_lookup_tables().action_to_coords[action];

        if action < REVEAL_ACTIONS_COUNT {
            let sq = coords[0];
            self.board[sq]
        } else {
            let to_sq = coords[1];
            self.board[to_sq]
        }
    }

    // --- 机会节点扩展（GameEnv trait 的实现下沉点） ---

    /// 该动作是否会产生机会节点（翻棋动作或吃暗子动作，目标格为 Hidden 即随机翻出）。
    pub fn is_chance_action(&self, action: usize) -> bool {
        matches!(self.get_target_slot(action), Slot::Hidden)
    }

    /// 枚举机会动作的所有可能结果：`(outcome_id, 概率, 结果环境)`。
    ///
    /// 在「执行该动作之前」的环境上调用。按隐藏棋子池逐类统计概率，
    /// 并为每类棋子构造一个「翻出该棋子的后继环境」。
    ///
    /// 注意：不要修改此处的全量展开逻辑（等价于既有 mcts::expand_chance_node）。
    pub fn chance_outcomes(&self, action: usize) -> Vec<(usize, f32, Self)> {
        let mut counts = [0usize; 14];
        for p in self.get_hidden_pieces_raw() {
            counts[get_outcome_id(p)] += 1;
        }
        let total_hidden = self.get_hidden_pieces_raw().len() as f32;
        if total_hidden == 0.0 {
            return Vec::new();
        }
        let mut outcomes = Vec::new();
        for outcome_id in 0..14 {
            if counts[outcome_id] > 0 {
                let prob = counts[outcome_id] as f32 / total_hidden;
                let mut next_env = *self;
                let specific_piece = *self
                    .get_hidden_pieces_raw()
                    .iter()
                    .find(|p| get_outcome_id(p) == outcome_id)
                    .expect("Piece not found");
                let _ = next_env.step(action, Some(specific_piece));
                outcomes.push((outcome_id, prob, next_env));
            }
        }
        outcomes
    }

    /// 执行动作后，若该动作产生了机会结果，返回其 `outcome_id`
    /// （用于 MCTS 子树复用匹配）。普通动作返回 `None`。
    ///
    /// 优先读取「最近一次翻出的棋子」：吃暗子动作（移动/炮击翻开 Hidden 目标格）
    /// 中，被翻开的守方棋子会立即被移除、目标格被攻击方占用，无法再从棋盘读出，
    /// 只有 `last_revealed_piece` 能还原其身份；普通翻棋动作两种方式结果一致。
    pub fn step_outcome_id(&self, action: usize) -> Option<usize> {
        if let Some(piece) = self.last_revealed_piece {
            return Some(get_outcome_id(&piece));
        }
        None
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

    /// 切换当前玩家（视角反转验证专用）。
    ///
    /// 仅改变 `current_player` 字段，**不改变**棋盘棋子归属 / hp / 死子计数等
    /// 绝对状态，因此 `get_state()` 返回同一绝对局面从另一玩家视角的观测
    /// （my/opp 通道与存活向量互换）。
    ///
    /// 注意：这会改变游戏规则语义（合法动作 / 胜负判定按新的当前玩家），
    /// 仅供视角反转验证获取不同视角观测，**不可用于继续对局**。
    pub fn flip_player(&mut self) {
        self.current_player = self.current_player.opposite();
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
        action_lookup_tables()
            .coords_to_action
            .get(&pack_coords(coords))
            .copied()
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

    // === 内部辅助方法 (供其他模块调用) ===

    /// 获取内部 bitboards (供 rules.rs 使用)
    pub(super) fn get_piece_bitboards(&self) -> &[[u64; NUM_PIECE_TYPES]; 2] {
        &self.piece_bitboards
    }

    /// 获取按棋子类型统计的阵亡计数 (供 features.rs 计算存活向量)
    pub(super) fn get_dead_piece_counts_by_type(&self) -> &[[u8; NUM_PIECE_TYPES]; 2] {
        &self.dead_piece_counts_by_type
    }

    pub(super) fn get_revealed_bitboards(&self) -> &[u64; 2] {
        &self.revealed_bitboards
    }

    pub(super) fn get_hidden_bitboard(&self) -> u64 {
        self.hidden_bitboard
    }

    pub(super) fn get_empty_bitboard(&self) -> u64 {
        self.empty_bitboard
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 随机走子对局，持续检查每个观测的 bitboard 一致性。
    ///
    /// 背景：修复前，炮隔子攻击己方暗子（翻开己方棋子）时，增量式 bitboard
    /// 清除会把攻击方自身从目标格误清，导致该格在观测中"全 0"（16 通道均为 0）。
    /// 本测试即回归测试，确保每个格子始终恰好属于一个状态。
    #[test]
    fn random_game_keeps_board_consistent() {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let max_games = 200;
        for game in 0..max_games {
            let mut env = DarkChessEnv::new();
            let mut steps = 0;
            let mut action_history: Vec<usize> = Vec::new();
            let mut consistent = true;
            let mut fail_step = 0;
            let mut fail_active = 0usize;
            let mut fail_sq = 0usize;
            // 诊断：失败最后一步的 from/to 槽位
            let mut last_from_slot: Option<Slot> = None;
            let mut last_to_slot: Option<Slot> = None;
            while steps < 200 {
                let masks = env.action_masks();
                let legal: Vec<usize> = masks
                    .iter()
                    .enumerate()
                    .filter(|&(_, &m)| m == 1)
                    .map(|(i, _)| i)
                    .collect();
                if legal.is_empty() {
                    break;
                }
                let action = legal[rng.gen_range(0..legal.len())];
                // 记录 step 前的源/目标格状态（诊断用）
                let coords = action_lookup_tables().action_to_coords[action].clone();
                last_to_slot = if coords.len() == 2 {
                    Some(env.board[coords[1]])
                } else {
                    None
                };
                last_from_slot = if coords.len() == 2 {
                    Some(env.board[coords[0]])
                } else {
                    None
                };
                match env.step(action, None) {
                    Ok((obs, _, terminated, truncated, _)) => {
                        action_history.push(action);
                        // 检查一致性
                        let b = obs.board.as_slice().unwrap();
                        for sq in 0..TOTAL_POSITIONS {
                            let active = (0..BOARD_CHANNELS)
                                .filter(|&pt| b[pt * TOTAL_POSITIONS + sq] > 0.5)
                                .count();
                            if active != 1 {
                                consistent = false;
                                fail_step = steps + 1;
                                fail_active = active;
                                fail_sq = sq;
                                break;
                            }
                        }
                        if !consistent {
                            break;
                        }
                        if terminated || truncated {
                            break;
                        }
                        steps += 1;
                    }
                    Err(e) => {
                        panic!("第{steps}步: env.step 返回 Err: {e}");
                    }
                }
            }
            if !consistent {
                // 打印动作序列，便于精确复现
                let coords: Vec<String> = action_history
                    .iter()
                    .map(|&a| match action_lookup_tables().action_to_coords[a].len() {
                        1 => format!("翻({})", action_lookup_tables().action_to_coords[a][0]),
                        _ => format!(
                            "({}->{})",
                            action_lookup_tables().action_to_coords[a][0],
                            action_lookup_tables().action_to_coords[a][1]
                        ),
                    })
                    .collect();
                panic!(
                    "第{game}局 第{fail_step}步: 第{fail_sq}格 归属通道数={fail_active} (应为1)\n\
                     最后动作 from_slot={:?} to_slot={:?}\n动作序列: {coords:?}",
                    last_from_slot, last_to_slot
                );
            }
        }
    }
}
