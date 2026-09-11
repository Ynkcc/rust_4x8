use super::config::GameConfig;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

// ==============================================================================
// --- 动作预计算表（config 驱动） ---
// ==============================================================================

/// 将坐标序列编码为 u64 键，避免每次查询堆分配 Vec。
///
/// 编码格式：`(len << 40) | (c0 << 16) | c1`。
/// 坐标编号 < 2^16，len ∈ {1, 2} 存在高位，翻棋（len=1）与移动/炮击（len=2）不会冲突。
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

/// 缓存键：rows 与 cols 的组合即可唯一确定一张动作表。
fn table_key(cfg: &GameConfig) -> u64 {
    ((cfg.rows as u64) << 16) | (cfg.cols as u64)
}

static ACTION_TABLE_CACHE: OnceLock<Mutex<HashMap<u64, Arc<ActionLookupTables>>>> = OnceLock::new();

fn table_cache() -> &'static Mutex<HashMap<u64, Arc<ActionLookupTables>>> {
    ACTION_TABLE_CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

pub fn action_lookup_tables(cfg: &GameConfig) -> Arc<ActionLookupTables> {
    let key = table_key(cfg);
    {
        let cache = table_cache().lock().unwrap();
        if let Some(t) = cache.get(&key) {
            return Arc::clone(t);
        }
    }
    let tables = build_action_lookup_tables(cfg);
    let mut cache = table_cache().lock().unwrap();
    cache.entry(key).or_insert_with(|| Arc::new(tables)).clone()
}

fn build_action_lookup_tables(cfg: &GameConfig) -> ActionLookupTables {
    let (rows, cols) = (cfg.rows, cfg.cols);
    let capacity = cfg.action_space_size;
    let mut action_to_coords = Vec::with_capacity(capacity);
    let mut coords_to_action = HashMap::with_capacity(capacity);
    let mut idx = 0;

    // 1. 翻棋
    for sq in 0..cfg.total_positions {
        let coords = vec![sq];
        action_to_coords.push(coords.clone());
        coords_to_action.insert(pack_coords(&coords), idx);
        idx += 1;
    }

    // 2. 常规移动
    let moves = [(-1, 0), (1, 0), (0, -1), (0, 1)];
    for r1 in 0..rows {
        for c1 in 0..cols {
            let from_sq = r1 * cols + c1;
            for (dr, dc) in moves.iter() {
                let r2 = r1 as i32 + dr;
                let c2 = c1 as i32 + dc;
                if r2 >= 0 && r2 < rows as i32 && c2 >= 0 && c2 < cols as i32 {
                    let to_sq = (r2 as usize) * cols + (c2 as usize);
                    let coords = vec![from_sq, to_sq];
                    action_to_coords.push(coords.clone());
                    coords_to_action.insert(pack_coords(&coords), idx);
                    idx += 1;
                }
            }
        }
    }

    // 3. 炮击
    for r1 in 0..rows {
        for c1 in 0..cols {
            let from_sq = r1 * cols + c1;
            // 水平
            for c2 in 0..cols {
                if (c1 as i32 - c2 as i32).abs() > 1 {
                    let to_sq = r1 * cols + c2;
                    let coords = vec![from_sq, to_sq];
                    if !coords_to_action.contains_key(&pack_coords(&coords)) {
                        action_to_coords.push(coords.clone());
                        coords_to_action.insert(pack_coords(&coords), idx);
                        idx += 1;
                    }
                }
            }
            // 垂直
            for r2 in 0..rows {
                if (r1 as i32 - r2 as i32).abs() > 1 {
                    let to_sq = r2 * cols + c1;
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
