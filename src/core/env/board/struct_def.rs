// src/game_env/board/struct_def.rs
// DarkChessEnv 结构体定义与构造器。

use super::*;

/// 支持 Copy 的暗棋环境，config 驱动。
/// 所有数组保持最大尺寸（MAX_POSITIONS=32 格 / 7 种 / 16 子 / 14 概率），
/// 由 `config` 决定活跃范围，从而保证 env 仍为 Copy（MCTS 值语义快照）。
#[derive(Clone, Copy, Debug)]
pub struct DarkChessEnv {
    /// 本局配置（棋盘尺寸 / 子力 / 血量 / 动作空间 / 特征维度）
    pub config: GameConfig,
    // --- 游戏核心状态 ---
    /// 棋盘格子状态
    pub(crate) board: [Slot; MAX_POSITIONS],
    /// 当前玩家
    pub(crate) current_player: Player,
    /// 连续无吃子步数
    pub(crate) move_counter: usize,
    /// 游戏总步数
    pub(crate) total_step_counter: usize,

    // --- 位棋盘 (Bitboards) ---
    pub(crate) piece_bitboards: [[u64; NUM_PIECE_TYPES_MAX]; 2],
    pub(crate) revealed_bitboards: [u64; 2],
    pub(crate) hidden_bitboard: u64,
    pub(crate) empty_bitboard: u64,

    // --- 游戏统计与记录 (Copy Refactor) ---
    /// 阵亡棋子池 [PlayerIdx][Idx]
    pub(crate) dead_pieces_pool: [[PieceType; MAX_PIECES_PER_PLAYER]; 2],
    /// 阵亡棋子计数 [PlayerIdx]
    pub(crate) dead_pieces_count: [usize; 2],
    /// 按棋子类型统计的阵亡计数 [PlayerIdx][PieceType]，供存活向量/棋谱推导使用
    pub(crate) dead_piece_counts_by_type: [[u8; NUM_PIECE_TYPES_MAX]; 2],

    /// 玩家分数/血量
    pub(crate) scores: [i32; 2],
    /// 上一步动作
    pub(crate) last_action: i32,

    // --- 概率相关 (Bag Model - Copy Refactor) ---
    /// 隐藏棋子池: 使用定长数组代替 Vec
    pub(crate) hidden_pieces_pool: [Piece; MAX_POSITIONS],
    /// 当前隐藏棋子数量
    pub(crate) hidden_pieces_count: usize,

    /// 翻棋概率表
    pub(crate) reveal_probabilities: [f32; MAX_REVEAL_PROBABILITY_SIZE],

    /// 随机环境种子
    pub seed: Option<u64>,
    /// 真实棋子布局 (如果配置了seed，初始化时固定布局)
    pub true_board: Option<[Piece; MAX_POSITIONS]>,

    /// 最近一次翻出的棋子（供机会节点子树复用匹配 outcome_id）。
    /// 翻棋动作与吃暗子动作都会更新；普通动作保持旧值（此时 step_outcome_id 不会被调用）。
    pub(crate) last_revealed_piece: Option<Piece>,
}

impl Default for DarkChessEnv {
    fn default() -> Self {
        Self::new()
    }
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
}
