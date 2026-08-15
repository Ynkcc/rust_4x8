// ==============================================================================
// --- 游戏配置 (GameConfig) ---
//
// 将原本散落在 constants.rs 的编译期常量抽取为可携带的配置结构，使同一套
// 棋盘/规则/特征/动作表逻辑可被多个棋盘变体复用（如 4x8 暗棋 与 4x2 迷你暗棋）。
//
// 设计要点：
// - `GameConfig` 是 `Copy` 纯数据，随环境携带，决定"活跃范围"。
// - 环境内部的定长数组保持最大尺寸（7 种棋子 / 32 格 / 16 子 / 14 概率），
//   由 config 决定实际使用的部分，从而保证环境仍为 `Copy`（MCTS 值语义快照）。
// - 特征通道数/标量数按"激活棋子种数 num_active"计算，而非固定 7。
// - 动作空间（翻棋/常规移动/炮击）由共享构建器按棋盘尺寸生成，配置需与之一致。
// ==============================================================================

use super::types::PieceType;

/// 每种棋子类型可拥有数量的上限（固定为 7 种，含迷你中不使用的类型，计数为 0）。
pub const NUM_PIECE_TYPES_MAX: usize = 7;
/// 单方可放置棋子的最大数量（4x8 暗棋 16；迷你 4）。数组按此上界分配。
pub const MAX_PIECES_PER_PLAYER: usize = 16;
/// 棋盘最大格数（4x8 为 32；迷你为 8）。数组按此上界分配。
pub const MAX_POSITIONS: usize = 32;
/// 翻棋概率表最大大小（2 * 7 = 14）。
pub const MAX_REVEAL_PROBABILITY_SIZE: usize = 14;

/// 一局游戏的动作空间计数（翻棋 / 常规移动 / 炮击），由棋盘尺寸决定。
/// 逻辑必须与 `actions.rs::build_action_lookup_tables` 完全一致。
pub fn compute_action_counts(rows: usize, cols: usize) -> (usize, usize, usize) {
    let reveal = rows * cols;

    // 常规移动：每格向上下左右四个方向的有效出边数之和
    let mut regular = 0usize;
    let moves = [(-1i32, 0i32), (1, 0), (0, -1), (0, 1)];
    for r1 in 0..rows {
        for c1 in 0..cols {
            for (dr, dc) in moves.iter() {
                let r2 = r1 as i32 + dr;
                let c2 = c1 as i32 + dc;
                if r2 >= 0 && r2 < rows as i32 && c2 >= 0 && c2 < cols as i32 {
                    regular += 1;
                }
            }
        }
    }

    // 炮击：同列/同行且距离 > 1 的 (from, to) 有序对
    let mut cannon = 0usize;
    for r1 in 0..rows {
        for c1 in 0..cols {
            for c2 in 0..cols {
                if (c1 as i32 - c2 as i32).abs() > 1 {
                    cannon += 1;
                }
            }
            for r2 in 0..rows {
                if (r1 as i32 - r2 as i32).abs() > 1 {
                    cannon += 1;
                }
            }
        }
    }

    (reveal, regular, cannon)
}

/// 一局游戏的完整配置（Copy 纯数据）。
#[derive(Clone, Copy, Debug)]
pub struct GameConfig {
    // --- 棋盘尺寸 ---
    pub rows: usize,
    pub cols: usize,
    pub total_positions: usize,

    // --- 子力 ---
    /// 激活的棋子类型数（决定特征通道数与翻棋概率表大小）。
    pub num_active: usize,
    /// 激活的棋子类型（实际 PieceType 索引），按顺序排列，位置即紧凑索引。
    pub active_types: [usize; NUM_PIECE_TYPES_MAX],
    /// 每方每种棋子的数量（按实际 PieceType 索引）。
    pub piece_counts: [usize; NUM_PIECE_TYPES_MAX],
    /// 每方棋子总数（= sum(piece_counts)）。
    pub total_pieces_per_player: usize,
    /// 每种棋子的分值（按实际 PieceType 索引；吃子扣血 / 启发式评估使用）。
    /// 由变体指定（如 4x8: 2/5/5/5/5/10/30；4x2: 2/5/0/0/0/10/30；4x4: 4/10/10/10/10/20/30）。
    pub piece_values: [i32; NUM_PIECE_TYPES_MAX],

    // --- 血量与步数 ---
    /// 初始血量上限（可由变体指定，不要求等于棋子价值总和）。
    pub initial_health: i32,
    /// 初始预翻棋子数量。
    pub initial_revealed_pieces: usize,
    /// 连续无吃子判和步数。
    pub max_consecutive_moves_for_draw: usize,
    /// 每局最大总步数。
    pub max_steps_per_episode: usize,

    // --- 动作空间 ---
    pub reveal_actions_count: usize,
    pub regular_move_actions_count: usize,
    pub cannon_attack_actions_count: usize,
    pub action_space_size: usize,

    // --- 特征维度 ---
    pub board_channels: usize,
    pub scalar_feature_count: usize,
    pub reveal_probability_size: usize,
}

impl GameConfig {
    /// 给定实际 PieceType 索引，返回其紧凑索引（0..num_active）。
    pub fn compact_index(&self, piece_type: usize) -> usize {
        for (i, &t) in self.active_types.iter().enumerate().take(self.num_active) {
            if t == piece_type {
                return i;
            }
        }
        // 未激活类型不应出现
        panic!("未找到激活的 piece_type: {piece_type}");
    }

    /// 给定棋子，返回其翻棋结果 ID（0..reveal_probability_size）。
    /// 红方紧凑索引 = compact_index，黑方偏移 num_active。
    pub fn outcome_id_for(&self, piece: PieceType, is_black: bool) -> usize {
        let base = self.compact_index(piece as usize);
        if is_black {
            base + self.num_active
        } else {
            base
        }
    }
}

/// 4x8 暗棋配置（回归基准，行为与原 constants.rs 完全一致）。
pub fn darkchess_config() -> GameConfig {
    let rows = 4usize;
    let cols = 8usize;
    let (reveal, regular, cannon) = compute_action_counts(rows, cols);

    let active_types = [0, 1, 2, 3, 4, 5, 6]; // Soldier..General
    let piece_counts = [5, 2, 2, 2, 2, 2, 1]; // 兵/炮/马/车/象/士/将
    let piece_values = [2, 5, 5, 5, 5, 10, 30]; // 与旧硬编码 value() 一致，回归安全
    let num_active = 7;
    let total_pieces: usize = piece_counts.iter().sum();

    GameConfig {
        rows,
        cols,
        total_positions: rows * cols,
        num_active,
        active_types,
        piece_counts,
        total_pieces_per_player: total_pieces,
        piece_values,
        initial_health: 60,
        initial_revealed_pieces: 4,
        max_consecutive_moves_for_draw: 24,
        max_steps_per_episode: 100,
        reveal_actions_count: reveal,
        regular_move_actions_count: regular,
        cannon_attack_actions_count: cannon,
        action_space_size: reveal + regular + cannon,
        board_channels: 2 * num_active + 2,
        scalar_feature_count: 3 + 2 * total_pieces,
        reveal_probability_size: 2 * num_active,
    }
}

/// 4x4 暗棋配置：7 类棋子全激活，每方 8 子，血量上限 = 60（由变体指定）。
///
/// | 类型 | 分值 | 数量/方 |
/// |------|------|--------|
/// | 兵   | 4    | 2      |
/// | 炮   | 10   | 1      |
/// | 马   | 10   | 1      |
/// | 车   | 10   | 1      |
/// | 象   | 10   | 1      |
/// | 士   | 20   | 1      |
/// | 将   | 30   | 1      |
pub fn game_4x4_config() -> GameConfig {
    let rows = 4usize;
    let cols = 4usize;
    let (reveal, regular, cannon) = compute_action_counts(rows, cols);

    let active_types = [0, 1, 2, 3, 4, 5, 6]; // Soldier..General
    let piece_counts = [2usize, 1, 1, 1, 1, 1, 1]; // 兵2 炮1 马1 车1 象1 士1 将1
    let piece_values = [4, 10, 10, 10, 10, 20, 30]; // 变体自定义分值
    let num_active = 7;
    let total_pieces: usize = piece_counts.iter().sum();

    GameConfig {
        rows,
        cols,
        total_positions: rows * cols,
        num_active,
        active_types,
        piece_counts,
        total_pieces_per_player: total_pieces,
        piece_values,
        initial_health: 60,
        initial_revealed_pieces: 4,
        max_consecutive_moves_for_draw: 24,
        max_steps_per_episode: 100,
        reveal_actions_count: reveal,
        regular_move_actions_count: regular,
        cannon_attack_actions_count: cannon,
        action_space_size: reveal + regular + cannon,
        board_channels: 2 * num_active + 2,
        scalar_feature_count: 3 + 2 * total_pieces,
        reveal_probability_size: 2 * num_active,
    }
}

/// 4x2 迷你暗棋配置：仅 兵/炮/士/将，每方各 1 子，血量上限 = 2+5+10+30 = 47。
pub fn mini_config() -> GameConfig {
    let rows = 4usize;
    let cols = 2usize;
    let (reveal, regular, cannon) = compute_action_counts(rows, cols);

    // 激活类型：Soldier(0), Cannon(1), Advisor(5), General(6)，紧凑索引 0..4
    let active_types = [0usize, 1, 5, 6, 0, 0, 0];
    let num_active = 4;
    // 每方：兵1 炮1 马0 车0 象0 士1 将1 = 4
    let piece_counts = [1usize, 1, 0, 0, 0, 1, 1];
    let piece_values = [2, 5, 0, 0, 0, 10, 30]; // 与旧硬编码 value() 一致，回归安全
    let total_pieces: usize = piece_counts.iter().sum();

    GameConfig {
        rows,
        cols,
        total_positions: rows * cols,
        num_active,
        active_types,
        piece_counts,
        total_pieces_per_player: total_pieces,
        piece_values,
        initial_health: 47,
        initial_revealed_pieces: 2,
        max_consecutive_moves_for_draw: 24,
        max_steps_per_episode: 60,
        reveal_actions_count: reveal,
        regular_move_actions_count: regular,
        cannon_attack_actions_count: cannon,
        action_space_size: reveal + regular + cannon,
        board_channels: 2 * num_active + 2,
        scalar_feature_count: 3 + 2 * total_pieces,
        reveal_probability_size: 2 * num_active,
    }
}
