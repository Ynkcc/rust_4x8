use super::constants::*;
use std::collections::HashMap;
use std::sync::OnceLock;

// ==============================================================================
// --- 动作预计算表 ---
// ==============================================================================

/// 将坐标序列编码为 u64 键，避免每次查询堆分配 Vec。
///
/// 编码格式：`(len << 40) | (c0 << 16) | c1`。
/// 坐标编号 < 2^16（棋盘 32 格远小于 65536），len ∈ {1, 2} 存在高位，
/// 翻棋（len=1）与移动/炮击（len=2）不会冲突。
pub fn pack_coords(coords: &[usize]) -> u64 {
    let len = coords.len();
    let mut key = 0u64;
    for (i, &c) in coords.iter().enumerate() {
        key |= (c as u64) << (16 * i);
    }
    key | ((len as u64) << 40)
}

pub struct ActionLookupTables {
    pub action_to_coords: Vec<Vec<usize>>,
    pub coords_to_action: HashMap<u64, usize>,
}

static ACTION_LOOKUP_TABLES: OnceLock<ActionLookupTables> = OnceLock::new();

pub fn action_lookup_tables() -> &'static ActionLookupTables {
    ACTION_LOOKUP_TABLES.get_or_init(build_action_lookup_tables)
}

fn build_action_lookup_tables() -> ActionLookupTables {
    let mut action_to_coords = Vec::with_capacity(ACTION_SPACE_SIZE);
    let mut coords_to_action = HashMap::with_capacity(ACTION_SPACE_SIZE);
    let mut idx = 0;

    // 1. 翻棋
    for sq in 0..TOTAL_POSITIONS {
        let coords = vec![sq];
        action_to_coords.push(coords.clone());
        coords_to_action.insert(pack_coords(&coords), idx);
        idx += 1;
    }

    // 2. 常规移动
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
                    coords_to_action.insert(pack_coords(&coords), idx);
                    idx += 1;
                }
            }
        }
    }

    // 3. 炮击
    for r1 in 0..BOARD_ROWS {
        for c1 in 0..BOARD_COLS {
            let from_sq = r1 * BOARD_COLS + c1;
            // 水平
            for c2 in 0..BOARD_COLS {
                if (c1 as i32 - c2 as i32).abs() > 1 {
                    let to_sq = r1 * BOARD_COLS + c2;
                    let coords = vec![from_sq, to_sq];
                    if !coords_to_action.contains_key(&pack_coords(&coords)) {
                        action_to_coords.push(coords.clone());
                        coords_to_action.insert(pack_coords(&coords), idx);
                        idx += 1;
                    }
                }
            }
            // 垂直
            for r2 in 0..BOARD_ROWS {
                if (r1 as i32 - r2 as i32).abs() > 1 {
                    let to_sq = r2 * BOARD_COLS + c1;
                    let coords = vec![from_sq, to_sq];
                    if !coords_to_action.contains_key(&pack_coords(&coords)) {
                        action_to_coords.push(coords.clone());
                        coords_to_action.insert(pack_coords(&coords), idx);
                        idx += 1;
                    }
                }
            }
        }
    }

    ActionLookupTables {
        action_to_coords,
        coords_to_action,
    }
}
