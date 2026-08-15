use rand::Rng;
use rand::seq::SliceRandom;
use rand::thread_rng;
use rand::SeedableRng;

use super::actions::{action_lookup_tables, pack_coords};
use super::bitboard::{board_mask, ray_attacks, ull};
use super::config::{
    GameConfig, MAX_PIECES_PER_PLAYER, MAX_POSITIONS, MAX_REVEAL_PROBABILITY_SIZE,
    NUM_PIECE_TYPES_MAX, darkchess_config, game_4x4_config, mini_config,
};
use super::traits::get_outcome_id;
use super::types::*;

// ==============================================================================
// --- 环境结构体 (DarkChessEnv) ---
//
// 支持 Copy 的暗棋环境，config 驱动。
// 所有数组保持最大尺寸（MAX_POSITIONS=32 格 / 7 种 / 16 子 / 14 概率），
// 由 `config` 决定活跃范围，从而保证 env 仍为 Copy（MCTS 值语义快照）。
// ==============================================================================
#[derive(Clone, Copy, Debug)]
pub struct DarkChessEnv {
    /// 本局配置（棋盘尺寸 / 子力 / 血量 / 动作空间 / 特征维度）
    pub config: GameConfig,
    // --- 游戏核心状态 ---
    /// 棋盘格子状态
    board: [Slot; MAX_POSITIONS],
    /// 当前玩家
    current_player: Player,
    /// 连续无吃子步数
    move_counter: usize,
    /// 游戏总步数
    total_step_counter: usize,

    // --- 位棋盘 (Bitboards) ---
    piece_bitboards: [[u64; NUM_PIECE_TYPES_MAX]; 2],
    revealed_bitboards: [u64; 2],
    hidden_bitboard: u64,
    empty_bitboard: u64,

    // --- 游戏统计与记录 (Copy Refactor) ---
    /// 阵亡棋子池 [PlayerIdx][Idx]
    dead_pieces_pool: [[PieceType; MAX_PIECES_PER_PLAYER]; 2],
    /// 阵亡棋子计数 [PlayerIdx]
    dead_pieces_count: [usize; 2],
    /// 按棋子类型统计的阵亡计数 [PlayerIdx][PieceType]，供存活向量/棋谱推导使用
    dead_piece_counts_by_type: [[u8; NUM_PIECE_TYPES_MAX]; 2],

    /// 玩家分数/血量
    scores: [i32; 2],
    /// 上一步动作
    last_action: i32,

    // --- 概率相关 (Bag Model - Copy Refactor) ---
    /// 隐藏棋子池: 使用定长数组代替 Vec
    hidden_pieces_pool: [Piece; MAX_POSITIONS],
    /// 当前隐藏棋子数量
    hidden_pieces_count: usize,

    /// 翻棋概率表
    reveal_probabilities: [f32; MAX_REVEAL_PROBABILITY_SIZE],

    /// 随机环境种子
    pub seed: Option<u64>,
    /// 真实棋子布局 (如果配置了seed，初始化时固定布局)
    pub true_board: Option<[Piece; MAX_POSITIONS]>,

    /// 最近一次翻出的棋子（供机会节点子树复用匹配 outcome_id）。
    /// 翻棋动作与吃暗子动作都会更新；普通动作保持旧值（此时 step_outcome_id 不会被调用）。
    last_revealed_piece: Option<Piece>,
}

impl DarkChessEnv {
    /// 创建 4x8 标准暗棋环境。
    pub fn new() -> Self {
        Self::with_config(darkchess_config())
    }

    /// 创建 4x2 迷你暗棋环境（仅兵/将/士/炮，血量上限=47）。
    pub fn new_mini() -> Self {
        Self::with_config(mini_config())
    }

    /// 创建 4x4 暗棋环境（7 类棋子全激活，每方 8 子，血量上限=60）。
    pub fn new_4x4() -> Self {
        Self::with_config(game_4x4_config())
    }

    /// 以指定配置创建环境（初始化并复位）。
    pub fn with_config(config: GameConfig) -> Self {
        let mut env = Self {
            config,
            board: [Slot::Empty; MAX_POSITIONS],
            current_player: Player::Red,
            move_counter: 0,
            total_step_counter: 0,

            piece_bitboards: [[0; NUM_PIECE_TYPES_MAX]; 2],
            revealed_bitboards: [0; 2],
            hidden_bitboard: 0,
            empty_bitboard: 0,

            dead_pieces_pool: [[PieceType::default(); MAX_PIECES_PER_PLAYER]; 2],
            dead_pieces_count: [0; 2],
            dead_piece_counts_by_type: [[0; NUM_PIECE_TYPES_MAX]; 2],

            scores: [0; 2],
            last_action: -1,

            hidden_pieces_pool: [Piece::default(); MAX_POSITIONS],
            hidden_pieces_count: 0,

            reveal_probabilities: [0.0; MAX_REVEAL_PROBABILITY_SIZE],
            seed: None,
            true_board: None,
            last_revealed_piece: None,
        };

        // 预热动作表与射线表（按 config 分键缓存）
        action_lookup_tables(&config);
        ray_attacks(&config);

        env.reset();
        env
    }

    /// 从已解码的棋盘槽位与当前玩家重建环境（供对局记录还原 / 棋谱文字解析使用）。
    ///
    /// 与 `new()` 不同，此构造器不随机生成隐藏棋子，而是直接采用给定的槽位布局，
    /// 因此只能用于校验类操作（如重新生成 action_masks / 展示棋盘），不能用于正常对局。
    /// 默认使用 4x8 暗棋配置（棋谱还原为 4x8 专用）。
    pub fn from_board(board: [Slot; MAX_POSITIONS], current_player: Player) -> Self {
        Self::from_board_with_config(board, current_player, darkchess_config())
    }

    /// 指定配置的棋盘重建。
    pub fn from_board_with_config(
        board: [Slot; MAX_POSITIONS],
        current_player: Player,
        config: GameConfig,
    ) -> Self {
        let mut env = Self {
            config,
            board,
            current_player,
            move_counter: 0,
            total_step_counter: 0,
            piece_bitboards: [[0; NUM_PIECE_TYPES_MAX]; 2],
            revealed_bitboards: [0; 2],
            hidden_bitboard: 0,
            empty_bitboard: 0,
            dead_pieces_pool: [[PieceType::default(); MAX_PIECES_PER_PLAYER]; 2],
            dead_pieces_count: [0; 2],
            dead_piece_counts_by_type: [[0; NUM_PIECE_TYPES_MAX]; 2],
            scores: [config.initial_health, config.initial_health],
            last_action: -1,
            hidden_pieces_pool: [Piece::default(); MAX_POSITIONS],
            hidden_pieces_count: 0,
            reveal_probabilities: [0.0; MAX_REVEAL_PROBABILITY_SIZE],
            seed: None,
            true_board: None,
            last_revealed_piece: None,
        };
        for (sq, slot) in board.iter().enumerate().take(config.total_positions) {
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

    pub fn get_coords_for_action(&self, action: usize) -> Option<Vec<usize>> {
        action_lookup_tables(&self.config)
            .action_to_coords
            .get(action)
            .cloned()
    }

    fn reset_internal_state(&mut self) {
        self.board = [Slot::Empty; MAX_POSITIONS];

        self.piece_bitboards = [[0; NUM_PIECE_TYPES_MAX]; 2];
        self.revealed_bitboards = [0; 2];

        self.hidden_bitboard = 0;
        self.empty_bitboard = 0;

        // 重置阵亡计数，无需清空 pool 内容，依靠 count 即可
        self.dead_pieces_count = [0; 2];
        self.dead_piece_counts_by_type = [[0; NUM_PIECE_TYPES_MAX]; 2];

        self.scores = [self.config.initial_health, self.config.initial_health];

        self.current_player = Player::Red;
        self.move_counter = 0;
        self.total_step_counter = 0;
        self.last_action = -1;

        self.hidden_pieces_count = 0;
        self.reveal_probabilities = [0.0; MAX_REVEAL_PROBABILITY_SIZE];
        self.last_revealed_piece = None;
    }

    /// 初始化棋盘布局 (Shuffle Bag Model)
    fn initialize_board(&mut self) {
        let cfg = self.config; // Copy，避免对 self 的长期借用
        let mut rng_std = self.seed.map(rand::rngs::StdRng::seed_from_u64);

        // 1. 生成实际棋子池 (写入 Buffer)：按激活类型、每类数量
        let mut idx = 0;
        for &player in &[Player::Red, Player::Black] {
            for &pt_idx in cfg.active_types.iter().take(cfg.num_active) {
                let piece_type = PieceType::from_index(pt_idx);
                for _ in 0..cfg.piece_counts[pt_idx] {
                    self.hidden_pieces_pool[idx] = Piece::new(piece_type, player);
                    idx += 1;
                }
            }
        }
        self.hidden_pieces_count = idx;

        // 打乱 slice
        if let Some(ref mut rng) = rng_std {
            self.hidden_pieces_pool[0..self.hidden_pieces_count].shuffle(rng);
            let mut tb = [Piece::default(); MAX_POSITIONS];
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
        let bmask = board_mask(&cfg);
        self.hidden_bitboard = bmask;

        for sq in 0..cfg.total_positions {
            self.board[sq] = Slot::Hidden;
        }

        self.update_reveal_probabilities();

        // 3. 随机翻开 N 个 Hidden 位置
        if cfg.total_positions > 0 {
            let mut hidden_indices: Vec<usize> = (0..cfg.total_positions).collect();
            if let Some(ref mut rng) = rng_std {
                hidden_indices.shuffle(rng);
            } else {
                let mut rng = thread_rng();
                hidden_indices.shuffle(&mut rng);
            }
            let reveal_count = std::cmp::min(hidden_indices.len(), cfg.initial_revealed_pieces);

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
        let cfg = &self.config;
        let total_hidden = self.hidden_pieces_count;

        if total_hidden == 0 {
            self.reveal_probabilities = [0.0; MAX_REVEAL_PROBABILITY_SIZE];
            return;
        }

        let mut counts = vec![0; cfg.reveal_probability_size];
        for i in 0..total_hidden {
            let piece = self.hidden_pieces_pool[i];
            let id = cfg.outcome_id_for(piece.piece_type, piece.player == Player::Black);
            counts[id] += 1;
        }

        for i in 0..cfg.reveal_probability_size {
            self.reveal_probabilities[i] = counts[i] as f32 / total_hidden as f32;
        }
    }

    pub fn get_reveal_probabilities(&self) -> &[f32] {
        &self.reveal_probabilities[0..self.config.reveal_probability_size]
    }

    // --- 核心 Step 逻辑 ---

    pub fn step(
        &mut self,
        action: usize,
        reveal_piece: Option<Piece>,
    ) -> Result<(Observation, f32, bool, bool, Option<i32>), String> {
        // 动作空间大小随 config 变化，使用 Vec（无法用编译期定长栈数组）
        let mut masks = vec![0i32; self.config.action_space_size];
        self.action_masks_into(&mut masks);
        if masks[action] == 0 {
            return Err(format!("无效动作: {}", action));
        }

        self.last_action = action as i32;
        self.total_step_counter += 1;

        let lookup = action_lookup_tables(&self.config);

        if action < self.config.reveal_actions_count {
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

        let target_slot = std::mem::replace(&mut self.board[to_sq], Slot::Revealed(attacker));

        let attacker_mask = ull(from_sq);
        let defender_mask = ull(to_sq);
        let p = attacker.player;
        let pt = attacker.piece_type as usize;

        // --- 1. 清除攻击方在源格 from_sq 的 bitboard，源格标为空位 ---
        self.revealed_bitboards[p.idx()] &= !attacker_mask;
        self.piece_bitboards[p.idx()][pt] &= !attacker_mask;
        self.empty_bitboard |= attacker_mask;

        // --- 2. 目标格 to_sq：彻底清除所有既有归属，再写入攻击方 ---
        self.hidden_bitboard &= !defender_mask;
        self.empty_bitboard &= !defender_mask;
        for player_idx in 0..2 {
            self.revealed_bitboards[player_idx] &= !defender_mask;
            for t in 0..NUM_PIECE_TYPES_MAX {
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
                let victim_idx = defender.player.idx();
                let dead_idx = self.dead_pieces_count[victim_idx];
                if dead_idx < MAX_PIECES_PER_PLAYER {
                    self.dead_pieces_pool[victim_idx][dead_idx] = defender.piece_type;
                    self.dead_pieces_count[victim_idx] += 1;
                    self.dead_piece_counts_by_type[victim_idx][defender.piece_type as usize] += 1;
                } else {
                    panic!("Dead pieces buffer overflow!");
                }
                let score = &mut self.scores[defender.player.idx()];
                // 吃子扣血：分值为变体可配置（config.piece_values），不再用硬编码 value()。
                *score = score.saturating_sub(self.config.piece_values[defender.piece_type as usize]);
                self.move_counter = 0;
            }
            Slot::Hidden => {
                panic!("Unexpected Hidden slot after reveal");
            }
        }
    }

    pub fn get_target_slot(&self, action: usize) -> Slot {
        let coords = &action_lookup_tables(&self.config).action_to_coords[action];

        if action < self.config.reveal_actions_count {
            let sq = coords[0];
            self.board[sq]
        } else {
            let to_sq = coords[1];
            self.board[to_sq]
        }
    }

    // --- 机会节点扩展 ---

    /// 该动作是否会产生机会节点（翻棋动作或吃暗子动作，目标格为 Hidden 即随机翻出）。
    pub fn is_chance_action(&self, action: usize) -> bool {
        matches!(self.get_target_slot(action), Slot::Hidden)
    }

    /// 枚举机会动作的所有可能结果：`(outcome_id, 概率, 结果环境)`。
    pub fn chance_outcomes(&self, action: usize) -> Vec<(usize, f32, Self)> {
        let cfg = &self.config;
        let mut counts = vec![0usize; cfg.reveal_probability_size];
        for p in self.get_hidden_pieces_raw() {
            let id = cfg.outcome_id_for(p.piece_type, p.player == Player::Black);
            counts[id] += 1;
        }
        let total_hidden = self.get_hidden_pieces_raw().len() as f32;
        if total_hidden == 0.0 {
            return Vec::new();
        }
        let mut outcomes = Vec::new();
        for outcome_id in 0..cfg.reveal_probability_size {
            if counts[outcome_id] > 0 {
                let prob = counts[outcome_id] as f32 / total_hidden;
                let mut next_env = *self;
                let specific_piece = *self
                    .get_hidden_pieces_raw()
                    .iter()
                    .find(|p| cfg.outcome_id_for(p.piece_type, p.player == Player::Black) == outcome_id)
                    .expect("Piece not found");
                let _ = next_env.step(action, Some(specific_piece));
                outcomes.push((outcome_id, prob, next_env));
            }
        }
        outcomes
    }

    /// 执行动作后，若该动作产生了机会结果，返回其 `outcome_id`。
    pub fn step_outcome_id(&self, _action: usize) -> Option<usize> {
        if let Some(piece) = self.last_revealed_piece {
            return Some(get_outcome_id(&self.config, &piece));
        }
        None
    }

    pub fn print_board(&self) {
        let cfg = &self.config;
        println!("\n      {}", (0..cfg.cols).map(|c| format!("{:^9}", c)).collect::<Vec<_>>().join(""));
        println!("   +{}+", "---------+".repeat(cfg.cols));
        for r in 0..cfg.rows {
            print!(" {} |", (b'A' + r as u8) as char);
            for c in 0..cfg.cols {
                let idx = r * cfg.cols + c;
                match &self.board[idx] {
                    Slot::Empty => print!("   .     |"),
                    Slot::Hidden => print!("    ?    |"),
                    Slot::Revealed(p) => print!(" {:^7} |", p.short_name()),
                }
            }
            println!("\n   +{}+", "---------+".repeat(cfg.cols));
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
        &self.board[0..self.config.total_positions]
    }

    pub fn get_current_player(&self) -> Player {
        self.current_player
    }

    /// 切换当前玩家（视角反转验证专用）。
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

    /// 返回死亡棋子的切片视图。
    pub fn get_dead_pieces(&self, player: Player) -> &[PieceType] {
        let count = self.dead_pieces_count[player.idx()];
        &self.dead_pieces_pool[player.idx()][0..count]
    }

    /// 返回隐藏棋子中属于指定玩家的类型列表。
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
        action_lookup_tables(&self.config)
            .coords_to_action
            .get(&pack_coords(coords))
            .copied()
    }

    pub fn get_bitboards(&self) -> std::collections::HashMap<String, Vec<bool>> {
        let cfg = &self.config;
        let mut bitboards = std::collections::HashMap::new();

        let bb_to_vec =
            |bb: u64| -> Vec<bool> { (0..cfg.total_positions).map(|sq| (bb & ull(sq)) != 0).collect() };

        bitboards.insert("hidden".to_string(), bb_to_vec(self.hidden_bitboard));
        bitboards.insert("empty".to_string(), bb_to_vec(self.empty_bitboard));

        const PIECE_NAMES: [&str; NUM_PIECE_TYPES_MAX] = [
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

            for &pt in cfg.active_types.iter().take(cfg.num_active) {
                bitboards.insert(
                    format!("{}_{}", prefix, PIECE_NAMES[pt]),
                    bb_to_vec(self.piece_bitboards[player.idx()][pt]),
                );
            }
        }

        bitboards
    }

    // === 内部辅助方法 (供其他模块调用) ===

    pub(super) fn get_piece_bitboards(&self) -> &[[u64; NUM_PIECE_TYPES_MAX]; 2] {
        &self.piece_bitboards
    }

    pub(super) fn get_dead_piece_counts_by_type(&self) -> &[[u8; NUM_PIECE_TYPES_MAX]; 2] {
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
                let coords = action_lookup_tables(&env.config).action_to_coords[action].clone();
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
                        let b = obs.board.as_slice().unwrap();
                        for sq in 0..env.config.total_positions {
                            let active = (0..env.config.board_channels)
                                .filter(|&pt| b[pt * env.config.total_positions + sq] > 0.5)
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
                let coords: Vec<String> = action_history
                    .iter()
                    .map(|&a| {
                        let t = action_lookup_tables(&env.config);
                        match t.action_to_coords[a].len() {
                            1 => format!("翻({})", t.action_to_coords[a][0]),
                            _ => format!(
                                "({}->{})",
                                t.action_to_coords[a][0], t.action_to_coords[a][1]
                            ),
                        }
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
