// src/replay/util.rs - 中文棋谱输出的通用辅助函数

use crate::game_env::config::GameConfig;
use crate::game_env::types::{Piece, PieceType, Player, Slot};

/// 根据手数（0 基）确定当前行棋方：偶数为红方，奇数为黑方。
pub(super) fn player_at(step: usize) -> Player {
    if step % 2 == 0 {
        Player::Red
    } else {
        Player::Black
    }
}

pub(super) fn piece_type_from_idx(idx: usize) -> PieceType {
    match idx {
        0 => PieceType::Soldier,
        1 => PieceType::Cannon,
        2 => PieceType::Horse,
        3 => PieceType::Chariot,
        4 => PieceType::Elephant,
        5 => PieceType::Advisor,
        6 => PieceType::General,
        _ => panic!("非法棋子类型索引: {}", idx),
    }
}

fn color_prefix(player: Player) -> &'static str {
    match player {
        Player::Red => "红",
        Player::Black => "黑",
    }
}

/// 棋子的中文名称（含颜色前缀），如 红马 / 黑兵 / 红帅 / 黑将。
pub(super) fn piece_name(piece: Piece) -> String {
    let name = match piece.piece_type {
        PieceType::General => match piece.player {
            Player::Red => "帅",
            Player::Black => "将",
        },
        PieceType::Cannon => "炮",
        PieceType::Horse => "马",
        PieceType::Chariot => "车",
        PieceType::Elephant => "象",
        PieceType::Advisor => "士",
        PieceType::Soldier => "兵",
    };
    format!("{}{}", color_prefix(piece.player), name)
}

/// 坐标文本：行用数字、列用字母，如 (0,a)。
pub(super) fn coord_str(sq: usize, cols: usize) -> String {
    let r = sq / cols;
    let c = sq % cols;
    format!("({},{})", r, (b'a' + c as u8) as char)
}

fn pad_cell(s: &str) -> String {
    // 中文字符按 2 列显示宽度计算，保证棋盘对齐
    let display_w = if s.is_ascii() { s.len() } else { s.chars().count() * 2 };
    let pad = 4usize.saturating_sub(display_w);
    format!("{}{}", s, " ".repeat(pad))
}

pub(super) fn format_board(slots: &[Slot], cfg: &GameConfig) -> String {
    let mut s = String::new();
    s.push_str("    ");
    for c in 0..cfg.cols {
        s.push_str(&pad_cell(&((b'a' + c as u8) as char).to_string()));
        s.push_str("  ");
    }
    s.push('\n');
    for r in 0..cfg.rows {
        s.push_str(&format!("{}   ", r));
        for c in 0..cfg.cols {
            let cell = match slots[r * cfg.cols + c] {
                Slot::Hidden => "?".to_string(),
                Slot::Empty => ".".to_string(),
                Slot::Revealed(p) => piece_name(p),
            };
            s.push_str(&pad_cell(&cell));
            s.push_str("  ");
        }
        s.push('\n');
    }
    s
}

pub(super) fn dead_str(player: Player, dead_list: &[PieceType]) -> String {
    if dead_list.is_empty() {
        "（无）".to_string()
    } else {
        dead_list
            .iter()
            .map(|&pt| piece_name(Piece::new(pt, player)))
            .collect::<Vec<_>>()
            .join("、")
    }
}
