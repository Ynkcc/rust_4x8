// src/py/ttt.rs
// 井字棋（Tic-Tac-Toe）Python 绑定：
// - `TicTacToe`：环境类（构造 / 合法动作 / 落子 / 胜负 / 特征编码）
// - `ttt_mcts_search`：单步 Gumbel MCTS 搜索（Python 提供 predict_fn 回调）
// - `run_ttt_self_play_with_predictor`：整局自对弈（复用 Rust 泛型 self_play）
//
// 设计约定（与暗棋 py 绑定保持一致）：
// - `predict_fn(boards_np, scalars_np) -> (policy_logits, values)`
//   boards shape [batch, 2, 3, 3]，scalars shape [batch, 0]，policy_logits shape [batch, 9]
// - 玩家编码：1=Red(先手 X)，-1=Black(后手 O)

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::game_env::{GameEnv, Player, TicTacToeEnv, TTT_ACTION_SPACE_SIZE};
use crate::mcts::{GumbelConfig, GumbelMCTS};
use crate::self_play::{GameEpisode, SelfPlayConfig};

use super::py_evaluator::PyEvaluator;

/// 井字棋环境（Python 可见）
#[pyclass(name = "TicTacToe", skip_from_py_object)]
#[derive(Clone)]
pub struct PyTicTacToe {
    pub inner: TicTacToeEnv,
}

#[pymethods]
impl PyTicTacToe {
    #[new]
    #[pyo3(signature = (cells=None, player=1))]
    fn new(cells: Option<Vec<i8>>, player: i32) -> PyResult<Self> {
        let player = if player == 1 {
            Player::Red
        } else if player == -1 {
            Player::Black
        } else {
            return Err(PyValueError::new_err("player 必须是 1 (先手 X) 或 -1 (后手 O)"));
        };
        let inner = match cells {
            Some(cells) => {
                if cells.len() != TTT_ACTION_SPACE_SIZE {
                    return Err(PyValueError::new_err(format!(
                        "cells 长度必须为 {}，实际 {}",
                        TTT_ACTION_SPACE_SIZE,
                        cells.len()
                    )));
                }
                let mut arr = [0i8; TTT_ACTION_SPACE_SIZE];
                arr.copy_from_slice(&cells);
                TicTacToeEnv::from_cells(arr, player)
            }
            None => TicTacToeEnv::new(),
        };
        Ok(Self { inner })
    }

    /// 当前格子（0=空，1=X，-1=O）
    #[getter]
    fn cells(slf: PyRef<'_, Self>) -> Vec<i8> {
        slf.inner.cells().to_vec()
    }

    /// 当前玩家（1=先手 X，-1=后手 O）
    #[getter]
    fn to_play(slf: PyRef<'_, Self>) -> i32 {
        slf.inner.get_current_player().val()
    }

    /// 合法动作列表（空格子索引）
    fn legal_moves(slf: PyRef<'_, Self>) -> Vec<usize> {
        let mut masks = [0i32; TTT_ACTION_SPACE_SIZE];
        slf.inner.action_masks_into(&mut masks);
        (0..TTT_ACTION_SPACE_SIZE)
            .filter(|&i| masks[i] == 1)
            .collect()
    }

    /// 胜者：Some(1)=X 胜，Some(-1)=O 胜，Some(0)=平局，None=未结束
    fn winner(slf: PyRef<'_, Self>) -> Option<i32> {
        slf.inner.check_game_over_conditions().2
    }

    /// 当前局面编码（2 通道 x 3 x 3 扁平，通道0=当前方，通道1=对手）
    fn encode(slf: PyRef<'_, Self>) -> Vec<f32> {
        let mut board = Vec::new();
        let mut scalars = Vec::new();
        slf.inner.encode_features_flat_into(&mut board, &mut scalars);
        board
    }

    /// 落子并返回 (terminated, truncated, winner)
    fn step(mut slf: PyRefMut<'_, Self>, action: usize) -> PyResult<(bool, bool, Option<i32>)> {
        let (_, _, term, trunc, winner) = slf
            .inner
            .step(action)
            .map_err(|e| PyValueError::new_err(e))?;
        Ok((term, trunc, winner))
    }

    /// 从环境快照重建（供验证脚本在 Python 侧复制局面后跨进程使用）
    fn copy(slf: PyRef<'_, Self>) -> Self {
        Self { inner: slf.inner }
    }
}

/// 单步 Gumbel MCTS 搜索（每次调用构建一棵新搜索树）。
///
/// 参数：
/// - `predict_fn`：`(boards_np, scalars_np) -> (policy_logits, values)` 回调
/// - `cells`：9 个格子值（0/1/-1）
/// - `player`：当前玩家（1 或 -1）
/// - `num_simulations`：模拟次数
/// - `max_considered_actions`：Gumbel Top-K 候选动作数
///
/// 返回 dict：
/// `action / policy / mcts_value / completed_q / root_visit_count / player / action_mask / board`
#[pyfunction]
#[pyo3(signature = (
    predict_fn,
    cells,
    player=1,
    num_simulations=64,
    max_considered_actions=9,
))]
pub fn ttt_mcts_search(
    predict_fn: Py<PyAny>,
    cells: Vec<i8>,
    player: i32,
    num_simulations: usize,
    max_considered_actions: usize,
) -> PyResult<Py<PyDict>> {
    if cells.len() != TTT_ACTION_SPACE_SIZE {
        return Err(PyValueError::new_err(format!(
            "cells 长度必须为 {}，实际 {}",
            TTT_ACTION_SPACE_SIZE,
            cells.len()
        )));
    }
    let mut arr = [0i8; TTT_ACTION_SPACE_SIZE];
    arr.copy_from_slice(&cells);
    let player = if player == 1 {
        Player::Red
    } else if player == -1 {
        Player::Black
    } else {
        return Err(PyValueError::new_err("player 必须是 1 或 -1"));
    };
    let env = TicTacToeEnv::from_cells(arr, player);

    let evaluator = PyEvaluator::<TicTacToeEnv>::new(predict_fn);
    let config = GumbelConfig {
        num_simulations,
        max_considered_actions,
        c_scale: 1.0,
        gumbel_scale: 1.0,
    };
    let mut mcts = GumbelMCTS::new(&env, &evaluator, config);
    let result = mcts.run();

    Python::attach(|py| {
        let dict = PyDict::new(py);
        match result {
            Some(r) => {
                dict.set_item("action", r.action)?;
                dict.set_item("policy", r.improved_policy)?;
                dict.set_item("mcts_value", r.mcts_value)?;
                dict.set_item("completed_q", r.completed_q)?;
                dict.set_item("root_visit_count", r.root_visit_count)?;
                dict.set_item("player", r.player.val())?;
                dict.set_item("action_mask", r.action_mask)?;
                let mut board = Vec::new();
                let mut scalars = Vec::new();
                env.encode_features_flat_into(&mut board, &mut scalars);
                dict.set_item("board", board)?;
                dict.set_item("scalars", scalars)?;
                dict.set_item("game_over", false)?;
            }
            None => {
                // 无合法动作：视为终局
                dict.set_item("game_over", true)?;
                dict.set_item("action", None::<i32>)?;
                dict.set_item("policy", vec![0.0f32; TTT_ACTION_SPACE_SIZE])?;
                dict.set_item("mcts_value", 0.0f32)?;
                dict.set_item("completed_q", 0.0f32)?;
                dict.set_item("root_visit_count", 0u32)?;
                dict.set_item("player", player.val())?;
                dict.set_item("action_mask", vec![0i32; TTT_ACTION_SPACE_SIZE])?;
                let mut board = Vec::new();
                let mut scalars = Vec::new();
                env.encode_features_flat_into(&mut board, &mut scalars);
                dict.set_item("board", board)?;
                dict.set_item("scalars", scalars)?;
            }
        }
        Ok(dict.unbind())
    })
}

/// 井字棋自对弈：复用 Rust 泛型 self_play 生成 `num_games` 局完整训练数据。
///
/// 返回 `GameEpisode` 列表（与暗棋相同的样本结构：
/// boards/scalars/policies/mcts_values/completed_qs/root_visits/game_results/action_masks/actions）。
#[pyfunction]
#[pyo3(signature = (
    predict_fn,
    mcts_sims=64,
    max_considered_actions=9,
    temperature_steps=6,
    num_games=1,
))]
pub fn run_ttt_self_play_with_predictor(
    predict_fn: Py<PyAny>,
    mcts_sims: usize,
    max_considered_actions: usize,
    temperature_steps: usize,
    num_games: usize,
) -> PyResult<Vec<Py<PyDict>>> {
    let evaluator = PyEvaluator::<TicTacToeEnv>::new(predict_fn);
    let cfg = SelfPlayConfig {
        mcts_sims,
        max_considered_actions,
        temperature_steps,
        ..Default::default()
    };

    let episodes: Vec<GameEpisode> =
        crate::self_play::run_batch_self_play(&evaluator, &cfg, num_games, TicTacToeEnv::new);

    Python::attach(|py| {
        let mut out = Vec::with_capacity(episodes.len());
        for ep in episodes {
            let dict = super::episode_to_dict_darkchess(py, &ep)?;
            out.push(dict.unbind());
        }
        Ok(out)
    })
}
