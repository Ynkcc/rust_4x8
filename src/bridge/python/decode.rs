//! 对局记录解码与可读化（Python 侧依赖的辅助 pyfunction）。

#[cfg(feature = "pyo3")]
use pyo3::prelude::*;
#[cfg(feature = "pyo3")]
use pyo3::types::PyDict;

#[cfg(feature = "pyo3")]
use crate::core::env::config::{GameConfig, darkchess_config, game_4x4_config, mini_config};
#[cfg(feature = "pyo3")]
use crate::core::env::types::{PieceType, Player};

/// 按 variant 返回游戏配置：0=4x8 暗棋、1=4x2 迷你、2=4x4。
#[cfg(feature = "pyo3")]
pub fn config_for_variant(variant: u8) -> PyResult<GameConfig> {
    match variant {
        0 => Ok(darkchess_config()),
        1 => Ok(mini_config()),
        2 => Ok(game_4x4_config()),
        _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "未知 variant: {}（应为 0=4x8、1=4x2、2=4x4）",
            variant
        ))),
    }
}

/// 从对局记录 dict（`GameEpisode::to_dict()` 的输出）解析人类可读的中文棋谱描述。
#[cfg(feature = "pyo3")]
#[pyfunction]
#[pyo3(signature = (record, variant = 0))]
pub fn describe_record(record: &Bound<'_, PyDict>, variant: u8) -> PyResult<String> {
    let boards: Vec<Vec<f32>> = record
        .get_item("boards")?
        .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("记录缺少 boards"))?
        .extract()?;
    let scalars: Vec<Vec<f32>> = record
        .get_item("scalars")?
        .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("记录缺少 scalars"))?
        .extract()?;
    let action_masks: Vec<Vec<i32>> = record
        .get_item("action_masks")?
        .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("记录缺少 action_masks"))?
        .extract()?;
    let actions: Vec<usize> = record
        .get_item("actions")?
        .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("记录缺少 actions"))?
        .extract()?;

    let cfg = config_for_variant(variant)?;
    Ok(crate::pipeline::replay::describe_record_with_config(
        &boards,
        &scalars,
        &action_masks,
        &actions,
        &cfg,
    ))
}

/// 解码单手标量特征（MongoDB sample.scalar_state）为结构化/人类可读信息。
#[cfg(feature = "pyo3")]
#[pyfunction]
#[pyo3(signature = (scalars, variant = 2, current_player = 1))]
pub fn decode_scalar_state(
    py: Python<'_>,
    scalars: Vec<f32>,
    variant: u8,
    current_player: i32,
) -> PyResult<Bound<'_, PyDict>> {
    use crate::pipeline::replay::{decode_scalar_state as decode_rs, survival_to_dead_vec};

    let cfg = config_for_variant(variant)?;
    let r = decode_rs(&scalars, &cfg);
    let cur = if current_player == -1 { Player::Black } else { Player::Red };
    let opp = cur.opposite();

    let piece_name = |pt: PieceType, player: Player| -> String {
        let name = match pt {
            PieceType::General => match player {
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
        format!("{}{}", if player == Player::Red { "红" } else { "黑" }, name)
    };

    let my_survival: Vec<i32> = r.my_survival.iter().map(|&v| v as i32).collect();
    let opp_survival: Vec<i32> = r.opp_survival.iter().map(|&v| v as i32).collect();

    let my_dead: Vec<String> = survival_to_dead_vec(&r.my_survival, &cfg)
        .into_iter()
        .map(|pt| piece_name(pt, cur))
        .collect();
    let opp_dead: Vec<String> = survival_to_dead_vec(&r.opp_survival, &cfg)
        .into_iter()
        .map(|pt| piece_name(pt, opp))
        .collect();

    // 存活摘要：仅显示存活数 > 0 的棋子
    let mut my_alive = Vec::new();
    let mut opp_alive = Vec::new();
    for (ci, &pt) in cfg.active_types.iter().enumerate().take(cfg.num_active) {
        let pt = PieceType::from_index(pt);
        if r.my_survival[ci] > 0 {
            my_alive.push(format!("{}x{}", piece_name(pt, cur), r.my_survival[ci]));
        }
        if r.opp_survival[ci] > 0 {
            opp_alive.push(format!("{}x{}", piece_name(pt, opp), r.opp_survival[ci]));
        }
    }

    let text = format!(
        "{}方回合 连续无吃子步数={} HP {}={} vs {}={} | {}存活: [{}] | {}存活: [{}] | {}阵亡: [{}] | {}阵亡: [{}]",
        if cur == Player::Red { "红" } else { "黑" },
        r.move_counter,
        if cur == Player::Red { "红" } else { "黑" },
        r.my_hp,
        if opp == Player::Red { "红" } else { "黑" },
        r.opp_hp,
        if cur == Player::Red { "红" } else { "黑" },
        my_alive.join(" "),
        if opp == Player::Red { "红" } else { "黑" },
        opp_alive.join(" "),
        if cur == Player::Red { "红" } else { "黑" },
        if my_dead.is_empty() { "无".to_string() } else { my_dead.join("、") },
        if opp == Player::Red { "红" } else { "黑" },
        if opp_dead.is_empty() { "无".to_string() } else { opp_dead.join("、") },
    );

    let dict = PyDict::new(py);
    dict.set_item("move_counter", r.move_counter)?;
    dict.set_item("my_hp", r.my_hp)?;
    dict.set_item("opp_hp", r.opp_hp)?;
    dict.set_item("my_survival", my_survival)?;
    dict.set_item("opp_survival", opp_survival)?;
    dict.set_item("my_dead", my_dead)?;
    dict.set_item("opp_dead", opp_dead)?;
    dict.set_item("text", text)?;
    dict.set_item("variant", variant)?;
    Ok(dict)
}
