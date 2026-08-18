// src/bridge/python/augment.rs
// Python 绑定层：数据空间对称增强（Data Augmentation）。
//
// 把训练侧的空间对称增强下沉到 Rust（`core::env::symmetry`），PyO3 导出：
//   - get_action_symmetry_table(rows, cols, transform) -> List[int]
//     动作置换表 perm（new_policy = old_policy[perm]），与 Python 旧推导一致。
//   - transform_board(flat, rows, cols, channels, transform) -> List[float]
//     扁平特征张量沿空间轴重排。
//   - transform_policy / transform_action
//     按置换表对 policy / action 做 gather。
//   - validate_symmetry(rows, cols, transforms) -> bool
//     一次性校验置换合法性（排列 / 对合或 4 次还原），供 Python self_check 复用。
//
// 动作序与置换表均以 `action_lookup_tables`（Rust 动作表唯一来源）为基准，
// 避免 `python/banqi/actions.py` 与 Rust 动作序不同步。

#[cfg(feature = "pyo3")]
use pyo3::prelude::*;

#[cfg(feature = "pyo3")]
use crate::core::env::config::compute_action_counts;
#[cfg(feature = "pyo3")]
use crate::core::env::{GameConfig, Symmetry, action_permutation, transform_board_flat};

/// 由棋盘尺寸构建一个最小 `GameConfig`（仅含增强所需的维度字段）。
/// 动作序与 `action_lookup_tables` 一致，其余子力/血量字段不影响增强。
#[cfg(feature = "pyo3")]
fn config_for_size(rows: usize, cols: usize) -> GameConfig {
    let (reveal, regular, cannon) = compute_action_counts(rows, cols);
    GameConfig {
        rows,
        cols,
        total_positions: rows * cols,
        num_active: 0,
        active_types: [0; crate::core::env::config::NUM_PIECE_TYPES_MAX],
        piece_counts: [0; crate::core::env::config::NUM_PIECE_TYPES_MAX],
        total_pieces_per_player: 0,
        piece_values: [0; crate::core::env::config::NUM_PIECE_TYPES_MAX],
        initial_health: 0,
        initial_revealed_pieces: 0,
        max_consecutive_moves_for_draw: 0,
        max_steps_per_episode: 0,
        reveal_actions_count: reveal,
        regular_move_actions_count: regular,
        cannon_attack_actions_count: cannon,
        action_space_size: reveal + regular + cannon,
        board_channels: 0,
        scalar_feature_count: 0,
        reveal_probability_size: 0,
    }
}

#[cfg(feature = "pyo3")]
fn parse_sym(transform: &str) -> PyResult<Symmetry> {
    Symmetry::from_name(transform)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err(format!("未知对称变换 {transform:?}")))
}

/// 获取单个对称变换的动作置换表（`perm` 满足 `new_policy = old_policy[perm]`）。
#[cfg(feature = "pyo3")]
#[pyfunction]
#[pyo3(signature = (rows, cols, transform))]
fn get_action_symmetry_table(rows: usize, cols: usize, transform: &str) -> PyResult<Vec<usize>> {
    let cfg = config_for_size(rows, cols);
    let sym = parse_sym(transform)?;
    Ok(action_permutation(&cfg, sym).to_vec())
}

/// 对扁平特征张量沿空间轴重排（逻辑形状 (channels, rows, cols)，通道序不变）。
#[cfg(feature = "pyo3")]
#[pyfunction]
#[pyo3(signature = (board, rows, cols, channels, transform))]
fn transform_board(
    board: Vec<f32>,
    rows: usize,
    cols: usize,
    channels: usize,
    transform: &str,
) -> PyResult<Vec<f32>> {
    if board.len() != channels * rows * cols {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "board 长度 {} != channels({channels})*rows({rows})*cols({cols}) = {}",
            board.len(),
            channels * rows * cols
        )));
    }
    let sym = parse_sym(transform)?;
    Ok(transform_board_flat(&board, rows, cols, channels, sym))
}

/// 按置换表对 policy（或 action_mask）做 gather：`out[a] = policy[perm[a]]`。
#[cfg(feature = "pyo3")]
#[pyfunction]
#[pyo3(signature = (policy, perm))]
fn transform_policy(policy: Vec<f32>, perm: Vec<usize>) -> PyResult<Vec<f32>> {
    if policy.len() != perm.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "policy 长度 {} != 置换表长度 {}",
            policy.len(),
            perm.len()
        )));
    }
    Ok(perm.iter().map(|&a| policy[a]).collect())
}

/// 对单个 action 做置换：返回 `perm[action]`。
#[cfg(feature = "pyo3")]
#[pyfunction]
#[pyo3(signature = (action, perm))]
fn permute_action(action: usize, perm: Vec<usize>) -> PyResult<usize> {
    if action >= perm.len() {
        return Err(pyo3::exceptions::PyIndexError::new_err(format!(
            "action {action} 超出置换表长度 {}",
            perm.len()
        )));
    }
    Ok(perm[action])
}

/// 批量校验若干对称变换的置换合法性：
///   - 是 {0..A} 的一个排列
///   - 对合变换满足 perm[perm[i]]==i；rot90/rot270 满足 4 次还原
/// 供 Python `data_augmentation.self_check` 复用，避免跨语言实现不一致。
#[cfg(feature = "pyo3")]
#[pyfunction]
#[pyo3(signature = (rows, cols, transforms))]
fn validate_symmetry(rows: usize, cols: usize, transforms: Vec<String>) -> PyResult<bool> {
    let cfg = config_for_size(rows, cols);
    let a = cfg.action_space_size;
    for name in &transforms {
        let sym = parse_sym(name)?;
        let perm = action_permutation(&cfg, sym);
        let mut seen = vec![false; a];
        let mut ok = true;
        for &p in perm.iter() {
            if p >= a || seen[p] {
                ok = false;
                break;
            }
            seen[p] = true;
        }
        if !ok {
            return Ok(false);
        }
        if sym.is_involution() {
            for i in 0..a {
                if perm[perm[i]] != i {
                    return Ok(false);
                }
            }
        } else {
            // 4 次还原
            for i in 0..a {
                if perm[perm[perm[perm[i]]]] != i {
                    return Ok(false);
                }
            }
        }
    }
    Ok(true)
}

/// 注册到 `banqi_4x8` pymodule。
#[cfg(feature = "pyo3")]
pub fn register_augment_functions(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(get_action_symmetry_table, m)?)?;
    m.add_function(wrap_pyfunction!(transform_board, m)?)?;
    m.add_function(wrap_pyfunction!(transform_policy, m)?)?;
    m.add_function(wrap_pyfunction!(permute_action, m)?)?;
    m.add_function(wrap_pyfunction!(validate_symmetry, m)?)?;
    // 以 Python 侧习惯名 `transform_action` 暴露（内部函数名 permute_action 避免冲突）
    m.add("transform_action", m.getattr("permute_action")?)?;
    Ok(())
}
