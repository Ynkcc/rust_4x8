// src/replay/scalar.rs - 标量特征解码（供 MongoDB 数据诊断 / Python 侧解析）

use crate::core::env::config::GameConfig;
use crate::core::env::types::{Piece, PieceType, Player};

use super::util::{piece_name, piece_type_from_idx};

/// 从 scalars 的存活向量中解析出某一方的存活棋子数量。
///
/// 存活编码（见 features.rs）：对每种棋子按 `piece_counts[pt]` 分块，
/// 每块用 `count` 个 1 + `(max - count)` 个 0 表示该种棋子存活 count 个。
/// `start` 为存活向量在 scalars 中的起始偏移。
/// 返回按 `cfg.active_types[0..num_active]` 顺序的存活数数组。
pub(super) fn parse_survival(scalars: &[f32], start: usize, cfg: &GameConfig) -> Vec<u8> {
    let mut counts = vec![0u8; cfg.num_active];
    let mut offset = start;
    for (ci, &pt) in cfg.active_types.iter().enumerate().take(cfg.num_active) {
        let max = cfg.piece_counts[pt];
        let mut c = 0u8;
        for k in 0..max {
            if scalars[offset + k] > 0.5 {
                c += 1;
            }
        }
        counts[ci] = c;
        offset += max;
    }
    counts
}

/// 从一手的 scalars 解析出双方存活棋子数，返回 (红方存活, 黑方存活)。
///
/// scalars 布局：`[0]` 步数、`[1]` 当前行棋方 HP、`[2]` 对方 HP、
/// `[3..3+total_pieces]` 当前行棋方存活、`[3+total_pieces..]` 对方存活。
/// 存活向量按 active_types 顺序、每类 piece_counts 个槽位编码。
pub(super) fn survival_from_scalars(
    scalars: &[f32],
    cur_player: Player,
    cfg: &GameConfig,
) -> (Vec<u8>, Vec<u8>) {
    let my_counts = parse_survival(scalars, 3, cfg);
    let opp_counts = parse_survival(scalars, 3 + cfg.total_pieces_per_player, cfg);
    match cur_player {
        Player::Red => (my_counts, opp_counts),
        Player::Black => (opp_counts, my_counts),
    }
}

/// 由某一方的存活棋子数推导该方已阵亡棋子列表。
///
/// 阵亡数 = 该类棋子总数 - 存活数。存活数直接从 scalars 得出，天然包含吃暗子
/// 与炮吃己方暗子的情况（只要被吃，存活数就会减少），无需差分、无需特殊处理。
pub(super) fn survival_to_dead(survival: &[u8], cfg: &GameConfig) -> Vec<PieceType> {
    let mut dead = Vec::with_capacity(cfg.total_pieces_per_player);
    for (ci, &pt) in cfg.active_types.iter().enumerate().take(cfg.num_active) {
        let total = cfg.piece_counts[pt];
        let alive = survival[ci] as usize;
        for _ in 0..total.saturating_sub(alive) {
            dead.push(piece_type_from_idx(pt));
        }
    }
    dead
}

/// 单手标量特征解码结果。
#[derive(Debug, Clone)]
pub struct ScalarDecodeResult {
    /// 连续无吃子步数（未归一化，原始步数）。
    pub move_counter: usize,
    /// 当前行棋方 HP。
    pub my_hp: i32,
    /// 对方 HP。
    pub opp_hp: i32,
    /// 当前行棋方存活数（按 active_types 顺序）。
    pub my_survival: Vec<u8>,
    /// 对方存活数（按 active_types 顺序）。
    pub opp_survival: Vec<u8>,
}

/// 由某一方存活数推导该方已阵亡棋子列表（按 active_types 顺序）。
pub fn survival_to_dead_vec(survival: &[u8], cfg: &GameConfig) -> Vec<PieceType> {
    survival_to_dead(survival, cfg)
}

/// 解码单手标量特征（config 驱动，支持 4x8/4x2/4x4）。
///
/// scalars 布局（见 features.rs `get_scalar_state_vector_into`）：
/// `[0]` = move_counter / max_consecutive_moves_for_draw
/// `[1]` = 当前行棋方 HP / initial_health
/// `[2]` = 对方 HP / initial_health
/// `[3..3+total_pieces]` = 当前行棋方存活 one-hot（按 active_types / piece_counts 分块）
/// `[3+total_pieces..]`   = 对方存活 one-hot
///
/// 返回结构化结果；`cur_player` 用于标注红/黑视角（仅影响 HP 归属，不影响数值）。
pub fn decode_scalar_state(scalars: &[f32], cfg: &GameConfig) -> ScalarDecodeResult {
    assert!(
        scalars.len() >= 3 + 2 * cfg.total_pieces_per_player,
        "scalars 长度不足: 期望 ≥{}，实际 {}",
        3 + 2 * cfg.total_pieces_per_player,
        scalars.len()
    );
    let move_counter =
        (scalars[0] * cfg.max_consecutive_moves_for_draw as f32).round() as usize;
    let my_hp = (scalars[1] * cfg.initial_health as f32).round() as i32;
    let opp_hp = (scalars[2] * cfg.initial_health as f32).round() as i32;
    let my_survival = parse_survival(scalars, 3, cfg);
    let opp_survival = parse_survival(scalars, 3 + cfg.total_pieces_per_player, cfg);
    ScalarDecodeResult {
        move_counter,
        my_hp,
        opp_hp,
        my_survival,
        opp_survival,
    }
}

/// 生成人类可读的单手标量描述（供诊断输出）。
pub fn format_scalar_state(
    scalars: &[f32],
    cfg: &GameConfig,
    cur_player: Player,
) -> String {
    let r = decode_scalar_state(scalars, cfg);
    let (my_name, opp_name) = match cur_player {
        Player::Red => ("红", "黑"),
        Player::Black => ("黑", "红"),
    };
    let mut my_pieces = Vec::new();
    let mut opp_pieces = Vec::new();
    for (ci, &pt) in cfg.active_types.iter().enumerate().take(cfg.num_active) {
        if r.my_survival[ci] > 0 {
            my_pieces.push(format!("{}x{}", piece_name(Piece::new(piece_type_from_idx(pt), cur_player)), r.my_survival[ci]));
        }
        if r.opp_survival[ci] > 0 {
            opp_pieces.push(format!("{}x{}", piece_name(Piece::new(piece_type_from_idx(pt), cur_player.opposite())), r.opp_survival[ci]));
        }
    }
    format!(
        "{}方回合 | 连续无吃子步数 {} | HP {}({}) vs {}({}) | {}存活: {} | {}存活: {}",
        my_name,
        r.move_counter,
        my_name,
        r.my_hp,
        opp_name,
        r.opp_hp,
        my_name,
        if my_pieces.is_empty() { "无".to_string() } else { my_pieces.join(" ") },
        opp_name,
        if opp_pieces.is_empty() { "无".to_string() } else { opp_pieces.join(" ") },
    )
}
