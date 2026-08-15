// src/py/game4x4_env.rs
// 4x4 暗棋环境 Python 绑定。
//
// 复用与 DarkChess / MiniDarkChess 相同的接口，但作用于 Game4x4Env：
// 供 python/game_4x4 进行自对弈训练 / 验证 / 对局测试。
//
// 4x4 规格：7 类棋子全激活（兵2/炮1/马1/车1/象1/士1/将1，每方 8 子），
// 分值 4/10/10/10/10/20/30，HP=60，动作空间 112。

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

use crate::game_env::{GAME4X4_ACTION_SPACE_SIZE, Game4x4Env, GameEnv};
use crate::mcts::{Evaluator, GumbelConfig, GumbelMCTS};
use rand::prelude::*;

use super::py_evaluator::PyEvaluator;

/// 4x4 暗棋环境（Python 可见）
#[pyclass(name = "Game4x4")]
pub struct PyGame4x4 {
    pub inner: Game4x4Env,
}

#[pymethods]
impl PyGame4x4 {
    #[new]
    fn new() -> Self {
        Self {
            inner: Game4x4Env::new(),
        }
    }

    /// 当前玩家视角观测：(board 扁平 16×4×4=256, scalars 扁平 19)。
    fn observation(&self) -> (Vec<f32>, Vec<f32>) {
        let obs = self.inner.get_state();
        (
            obs.board.iter().copied().collect(),
            obs.scalars.iter().copied().collect(),
        )
    }

    /// 随机走 `n` 步合法动作（遇终局提前停止），返回实际步数。
    fn random_steps(&mut self, n: usize) -> PyResult<usize> {
        let mut rng = thread_rng();
        let mut done = 0;
        for _ in 0..n {
            let mut masks = vec![0i32; GAME4X4_ACTION_SPACE_SIZE];
            self.inner.action_masks_into(&mut masks);
            let legal: Vec<usize> = (0..GAME4X4_ACTION_SPACE_SIZE)
                .filter(|&i| masks[i] == 1)
                .collect();
            if legal.is_empty() {
                break;
            }
            let action = legal[rng.gen_range(0..legal.len())];
            self.inner
                .step(action)
                .map_err(|e| PyRuntimeError::new_err(format!("step 失败: {}", e)))?;
            done += 1;
            let (terminated, _, _) = self.inner.check_game_over_conditions();
            if terminated {
                break;
            }
        }
        Ok(done)
    }

    /// 当前玩家（1=Red，-1=Black）
    fn current_player(&self) -> i32 {
        self.inner.get_current_player().val()
    }

    /// 切换当前玩家（flip_player：不改变棋盘/棋子归属，仅改变编码视角）。
    fn switch_player(&mut self) {
        self.inner.flip_player();
    }

    /// 当前玩家合法动作索引列表
    fn legal_moves(&self) -> Vec<usize> {
        let mut masks = vec![0i32; GAME4X4_ACTION_SPACE_SIZE];
        self.inner.action_masks_into(&mut masks);
        (0..GAME4X4_ACTION_SPACE_SIZE)
            .filter(|&i| masks[i] == 1)
            .collect()
    }

    /// 终局胜者：None=未结束，Some(0)=平局，Some(1)=红胜，Some(-1)=黑胜
    fn winner(&self) -> Option<i32> {
        self.inner.check_game_over_conditions().2
    }

    /// 是否终局
    fn terminated(&self) -> bool {
        self.inner.check_game_over_conditions().0
    }

    /// 公开落子接口。返回 (terminated, truncated, winner)。
    fn step(&mut self, action: usize) -> PyResult<(bool, bool, Option<i32>)> {
        let (_, _, terminated, truncated, winner) = self
            .inner
            .step(action)
            .map_err(|e| PyRuntimeError::new_err(format!("step 失败: {}", e)))?;
        Ok((terminated, truncated, winner))
    }

    /// 用当前局面做 Gumbel MCTS 搜索（调用 Python 网络回调评估），返回选中的动作。
    fn mcts_search_action(
        &self,
        predict_fn: PyObject,
        num_simulations: usize,
        max_considered_actions: usize,
        c_visit: f64,
        c_scale: f64,
    ) -> PyResult<Option<usize>> {
        let evaluator = PyEvaluator::<Game4x4Env>::new(predict_fn);
        let config = GumbelConfig {
            num_simulations,
            max_considered_actions,
            c_visit: c_visit as f32,
            c_scale: c_scale as f32,
        };
        let mut mcts = GumbelMCTS::new(&self.inner, &evaluator, config);
        Ok(mcts.run().map(|r| r.action))
    }

    /// 用 expectiminimax + alpha-beta 搜索选动作（不依赖网络，纯规则搜索）。
    ///
    /// `max_depth` 为搜索深度；返回 None 表示无合法动作（终局）。
    fn minimax_action(&self, max_depth: usize) -> PyResult<Option<usize>> {
        let result = crate::ai::minimax::minimax_best_action(&self.inner.inner, max_depth);
        Ok(result.map(|r| r.action))
    }

    /// 纯网络贪婪动作：对当前局面一次前向，取合法动作中 logit 最大者（无搜索）。
    fn greedy_action(&self, predict_fn: PyObject) -> PyResult<Option<usize>> {
        let evaluator = PyEvaluator::<Game4x4Env>::new(predict_fn);
        let (logits_batch, _values) = evaluator.evaluate(std::slice::from_ref(&self.inner));
        let mut masks = vec![0i32; GAME4X4_ACTION_SPACE_SIZE];
        self.inner.action_masks_into(&mut masks);
        let mut best: Option<usize> = None;
        let mut best_v = f32::NEG_INFINITY;
        for a in 0..GAME4X4_ACTION_SPACE_SIZE {
            if masks[a] == 1 && logits_batch[0][a] > best_v {
                best_v = logits_batch[0][a];
                best = Some(a);
            }
        }
        Ok(best)
    }
}
