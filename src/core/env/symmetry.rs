// ==============================================================================
// --- 空间对称增强模块 (Symmetry) ---
//
// 在 `action_lookup_tables`（动作序唯一来源）之上派生 D4 空间对称变换：
//   - `sq_map`：棋盘格子重排表（map[i] = 变换后位置 i 对应的原格子索引）。
//   - `action_permutation`：动作置换表 perm（new_policy = old_policy[perm]）。
//   - `transform_board`：把扁平特征张量 (channels, rows, cols) 沿空间轴重排。
//
// 语义必须与旧 Python `data_augmentation.py` 的 `_sq_map` / `action_permutation`
// 完全一致，避免跨语言动作序脱节。scalar / 血量 / 存活等全局量保持不变。
//
// 变换集（字符串与 Python 侧 `ALL_SYMMETRY_TRANSFORMS` 对齐）：
//   identity / rot90 / rot180 / rot270 / hflip / vflip / diag / anti_diag
// 各变体可用集由 Python `variant.py::symmetries` 限定（4x8/4x2 用 4 个，
// 4x4 方盘用全部 8 个 D4 对称）。本模块不限定可用集，只提供几何变换。
// ==============================================================================

use super::actions::{action_lookup_tables, pack_coords, ActionLookupTables};
use super::config::GameConfig;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

/// 空间对称变换枚举。
///
/// 字符串名与 Python `data_augmentation.py` 的变换名一一对应，用于 PyO3 绑定传参。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Symmetry {
    Identity,
    Rot90,    // 顺时针 90°
    Rot180,
    Rot270,   // 逆时针 90°
    HFlip,
    VFlip,
    Diag,     // 主对角线镜像（转置）
    AntiDiag, // 反对角线镜像
}

impl Symmetry {
    /// 解析变换名字符串（大小写不敏感，兼容 Python 传参）。
    pub fn from_name(name: &str) -> Option<Symmetry> {
        match name.trim().to_lowercase().as_str() {
            "identity" => Some(Symmetry::Identity),
            "rot90" => Some(Symmetry::Rot90),
            "rot180" => Some(Symmetry::Rot180),
            "rot270" => Some(Symmetry::Rot270),
            "hflip" => Some(Symmetry::HFlip),
            "vflip" => Some(Symmetry::VFlip),
            "diag" => Some(Symmetry::Diag),
            "anti_diag" => Some(Symmetry::AntiDiag),
            _ => None,
        }
    }

    /// 是否为对合变换（两次作用恒等；rot90/rot270 需 4 次还原）。
    pub fn is_involution(self) -> bool {
        matches!(
            self,
            Symmetry::Identity | Symmetry::Rot180 | Symmetry::HFlip | Symmetry::VFlip
                | Symmetry::Diag | Symmetry::AntiDiag
        )
    }
}

/// 缓存键：棋盘尺寸 (rows, cols) 与变换。
fn cache_key(rows: usize, cols: usize, sym: Symmetry) -> u64 {
    ((rows as u64) << 32) | ((cols as u64) << 8) | (sym as u64)
}

/// 动作置换表缓存：key -> Arc<Vec<perm>>（perm: new_policy = old_policy[perm]）。
static PERM_CACHE: OnceLock<Mutex<HashMap<u64, Arc<Vec<usize>>>>> = OnceLock::new();
/// 格子重排表缓存：key -> Arc<Vec<map>>（map[i] = 变换后位置 i 的原格子索引）。
static SQMAP_CACHE: OnceLock<Mutex<HashMap<u64, Arc<Vec<usize>>>>> = OnceLock::new();

fn perm_cache() -> &'static Mutex<HashMap<u64, Arc<Vec<usize>>>> {
    PERM_CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

fn sqmap_cache() -> &'static Mutex<HashMap<u64, Arc<Vec<usize>>>> {
    SQMAP_CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// 生成格子重排表：`map[i] = 变换后位置 i 对应的原格子索引`。
/// 与 Python `data_augmentation.py::_sq_map` 逐条一致。
pub fn sq_map(rows: usize, cols: usize, sym: Symmetry) -> Arc<Vec<usize>> {
    let key = cache_key(rows, cols, sym);
    {
        let cache = sqmap_cache().lock().unwrap();
        if let Some(t) = cache.get(&key) {
            return Arc::clone(t);
        }
    }
    let total = rows * cols;
    let mut map = vec![0usize; total];
    let r = rows as i64;
    let c = cols as i64;
    for rr in 0..rows {
        for cc in 0..cols {
            let (pr, pc) = match sym {
                Symmetry::Identity => (rr as i64, cc as i64),
                // 顺时针 90°：pr = cols-1-c, pc = r
                Symmetry::Rot90 => (c - 1 - cc as i64, rr as i64),
                Symmetry::Rot180 => (r - 1 - rr as i64, c - 1 - cc as i64),
                // 逆时针 90°：pr = c, pc = rows-1-r
                Symmetry::Rot270 => (cc as i64, r - 1 - rr as i64),
                Symmetry::HFlip => (rr as i64, c - 1 - cc as i64),
                Symmetry::VFlip => (r - 1 - rr as i64, cc as i64),
                Symmetry::Diag => (cc as i64, rr as i64),
                // 反对角线镜像：pr = cols-1-c, pc = rows-1-r
                Symmetry::AntiDiag => (c - 1 - cc as i64, r - 1 - rr as i64),
            };
            map[rr * cols + cc] = (pr * cols as i64 + pc) as usize;
        }
    }
    let arc = Arc::new(map);
    sqmap_cache().lock().unwrap().insert(key, Arc::clone(&arc));
    arc
}

/// 生成动作置换表：`perm` 满足 `new_policy = old_policy[perm]`。
///
/// 与 Python `data_augmentation.py::action_permutation` 逻辑一致：
/// 对每个动作 a，取其坐标序列，经格子重排映射后查 `coords_to_action`，
/// 得到目标动作 dest，置 `perm[dest] = a`。
pub fn action_permutation(cfg: &GameConfig, sym: Symmetry) -> Arc<Vec<usize>> {
    let key = cache_key(cfg.rows, cfg.cols, sym);
    {
        let cache = perm_cache().lock().unwrap();
        if let Some(t) = cache.get(&key) {
            return Arc::clone(t);
        }
    }
    let tables: Arc<ActionLookupTables> = action_lookup_tables(cfg);
    let map = sq_map(cfg.rows, cfg.cols, sym);
    let mut perm = vec![0usize; cfg.action_space_size];
    for (a, coords) in tables.action_to_coords.iter().enumerate() {
        let mapped: Vec<usize> = coords.iter().map(|&sq| map[sq]).collect();
        let dest = tables
            .coords_to_action
            .get(&pack_coords(&mapped))
            .copied()
            .unwrap_or(a);
        perm[dest] = a;
    }
    let arc = Arc::new(perm);
    perm_cache().lock().unwrap().insert(key, Arc::clone(&arc));
    arc
}

/// 对扁平特征张量沿空间轴重排。
///
/// 输入/输出均为一维扁平数组，逻辑形状 `(channels, rows, cols)`，通道序不变。
/// 即 `out[c, i, j] = in[c, pr(i,j), pc(i,j)]`，其中 (pr,pc) 是变换后位置 (i,j)
/// 对应的原格子坐标。
pub fn transform_board_flat(
    board: &[f32],
    rows: usize,
    cols: usize,
    channels: usize,
    sym: Symmetry,
) -> Vec<f32> {
    debug_assert_eq!(board.len(), channels * rows * cols, "board 形状与 (ch,rows,cols) 不符");
    if sym == Symmetry::Identity {
        return board.to_vec();
    }
    let map = sq_map(rows, cols, sym);
    let mut out = vec![0f32; board.len()];
    for ch in 0..channels {
        let in_base = ch * rows * cols;
        let out_base = ch * rows * cols;
        for sq in 0..(rows * cols) {
            out[out_base + sq] = board[in_base + map[sq]];
        }
    }
    out
}

/// 对 action 做置换：返回 `perm[action]`。
pub fn transform_action(cfg: &GameConfig, sym: Symmetry, action: usize) -> usize {
    if sym == Symmetry::Identity {
        return action;
    }
    action_permutation(cfg, sym)[action]
}
