use ndarray::{Array1, Array3};
use serde::{Deserialize, Serialize};
use std::fmt;

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
    /// 根据类型索引（0..7，对应 enum 顺序）构造棋子类型。
    /// 用于从配置的 active_types 索引还原具体棋子。
    pub fn from_index(idx: usize) -> Self {
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

    /// 默认分值（4x8 基准）。运行时吃子扣血应使用 `GameConfig::piece_values`
    /// （变体可配置，如 4x4 兵4/炮10/马10/车10/象10/士20/将30）。
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
        Self {
            piece_type: PieceType::Soldier,
            player: Player::Red,
        }
    }
}

impl Piece {
    pub fn new(piece_type: PieceType, player: Player) -> Self {
        Self { piece_type, player }
    }

    pub fn short_name(&self) -> String {
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

/// 观察空间数据结构 (Neural Network Input)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResNetObservation {
    /// 棋盘特征张量: (Channels, H, W)
    pub board: Array3<f32>,
    /// 全局标量特征: (Features,)
    pub scalars: Array1<f32>,
}
