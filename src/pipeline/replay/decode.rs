// src/replay/decode.rs - 将观测张量解码为绝对棋盘槽位

use crate::core::env::config::{darkchess_config, GameConfig};
use crate::core::env::types::{Piece, Player, Slot};

use super::util::piece_type_from_idx;

/// 将观测张量解码为绝对棋盘的槽位数组。
///
/// 通道布局见 features.rs：前 `num_active` 通道为己方棋子（按 active_types 顺序）、
/// 接下来 `num_active` 通道为敌方棋子、再接下来为暗子、最后为空位。
/// "己方/敌方"以当前行棋方为视角。
pub fn decode_board_with_config(board: &[f32], current_player: Player, cfg: &GameConfig) -> Vec<Slot> {
    assert_eq!(
        board.len(),
        cfg.resnet_board_channels * cfg.total_positions,
        "board 张量长度错误: 期望 {}，实际 {}",
        cfg.resnet_board_channels * cfg.total_positions,
        board.len()
    );
    let opp = current_player.opposite();
    let mut slots = Vec::with_capacity(cfg.total_positions);
    for sq in 0..cfg.total_positions {
        let hidden = board[(cfg.resnet_board_channels - 2) * cfg.total_positions + sq];
        let empty = board[(cfg.resnet_board_channels - 1) * cfg.total_positions + sq];
        let slot = if hidden > 0.5 {
            Slot::Hidden
        } else if empty > 0.5 {
            Slot::Empty
        } else {
            let mut piece: Option<Piece> = None;
            for ci in 0..cfg.num_active {
                if board[ci * cfg.total_positions + sq] > 0.5 {
                    let pt = cfg.active_types[ci];
                    piece = Some(Piece::new(piece_type_from_idx(pt), current_player));
                    break;
                }
            }
            if piece.is_none() {
                for ci in 0..cfg.num_active {
                    if board[(cfg.num_active + ci) * cfg.total_positions + sq] > 0.5 {
                        let pt = cfg.active_types[ci];
                        piece = Some(Piece::new(piece_type_from_idx(pt), opp));
                        break;
                    }
                }
            }
            match piece {
                Some(p) => Slot::Revealed(p),
                None => panic!("第 {} 格无法解码为任何棋格状态", sq),
            }
        };
        slots.push(slot);
    }
    slots
}

/// 4x8 兼容入口。
pub fn decode_board(board: &[f32], current_player: Player) -> Vec<Slot> {
    decode_board_with_config(board, current_player, &darkchess_config())
}
