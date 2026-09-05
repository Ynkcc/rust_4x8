// src/py/chess_env.rs
// 统一暗棋环境 Python 绑定（唯一入口）。
//
// 重构说明：原 darkchess_env.rs / game4x4_env.rs / mini_darkchess_env.rs 三份
// 副本逻辑一致度 >95%，仅差异为：具体环境类型、动作空间大小、pyclass 名、以及
// minimax / 启发式搜索底层访问的 DarkChessEnv。本文件用一份公共实现 + 一个
// 声明宏（`define_chess_env!`）为三个变体各生成一个 pyclass：
//   - `DarkChess`（4x8，352 动作空间）
//   - `Game4x4`（4x4，112 动作空间）
//   - `MiniDarkChess`（4x2，40 动作空间）
//
// 对外暴露的统一接口（三类完全一致）：
//   observation() -> (board, scalars)
//   random_steps(n) -> int
//   current_player() -> int
//   switch_player()
//   legal_moves() -> list[int]
//   winner() -> int | None
//   terminated() -> bool
//   step(action) -> (terminated, truncated, winner)
//   mcts_search_action(predict_fn, sims, max_acts, c_scale, gumbel_scale=1.0)
//   greedy_action(predict_fn)
//   minimax_action(max_depth)          # 纯规则搜索
//   heuristic_mcts_action(sims)        # 纯计算启发式
//
// 注：4x8 的 `DarkChess` 也补上了 minimax_action / heuristic_mcts_action（与
// 4x4/4x2 对齐），使 Python 侧统一评估协议（banqi/eval.py）对所有变体一致可用。

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use rand::prelude::*;

use crate::core::env::{DarkChessEnv, Game4x4Env, GameEnv, MiniDarkChessEnv};
use crate::core::mcts::{Evaluator, GumbelConfig, GumbelMCTS};

use super::py_evaluator::PyEvaluator;

// ============================================================================
// 统一底层访问：三类暗棋环境 -> 底层 DarkChessEnv（minimax / 启发式搜索需要）
// ============================================================================

/// 供三类暗棋环境统一访问底层 `DarkChessEnv` 的能力。
///
/// - `DarkChessEnv`（4x8）即自身；
/// - `Game4x4Env` / `MiniDarkChessEnv` 各自包一层 `inner: DarkChessEnv`。
///
/// `GameEnv` 已统一了绝大部分接口（动作掩码、落子、观测、终局、动作空间等），
/// 这里仅补充 minimax / 启发式搜索所需的底层棋盘访问。
pub trait PyChessEnvCore: GameEnv {
    /// 底层棋盘（用于 `minimax_best_action` / `HeuristicMctsPolicy`）。
    fn as_darkchess(&self) -> &DarkChessEnv;
}

impl PyChessEnvCore for DarkChessEnv {
    fn as_darkchess(&self) -> &DarkChessEnv {
        self
    }
}

impl PyChessEnvCore for Game4x4Env {
    fn as_darkchess(&self) -> &DarkChessEnv {
        &self.inner
    }
}

impl PyChessEnvCore for MiniDarkChessEnv {
    fn as_darkchess(&self) -> &DarkChessEnv {
        &self.inner
    }
}

// ============================================================================
// 单一实现：为某变体生成一个 pyclass 的声明宏
// ============================================================================

/// 为变体生成统一的暗棋环境 pyclass。
///
/// `$rust_name`：生成的 Rust 类型名；`$py_name`：Python 侧可见类名；
/// `$env`：具体环境类型（须实现 `PyChessEnvCore`）。
macro_rules! define_chess_env {
    ($rust_name:ident, $py_name:literal, $env:ty) => {
        /// 暗棋环境（Python 可见）。方法见模块级文档。
        #[pyclass(name = $py_name, skip_from_py_object)]
        #[derive(Clone)]
        pub struct $rust_name {
            pub inner: $env,
        }

        #[pymethods]
        impl $rust_name {
            /// 标准开局。
            #[new]
            fn new() -> Self {
                Self {
                    inner: <$env>::new(),
                }
            }

            /// 当前玩家视角观测：返回 (board 扁平, scalars 扁平)。
            /// board 通道 0~N-1=当前方棋子、N~2N-1=对手、隐藏、空（由变体决定）。
            fn observation(&self) -> (Vec<f32>, Vec<f32>) {
                let obs = self.inner.get_resnet_state();
                (
                    obs.board.iter().copied().collect(),
                    obs.scalars.iter().copied().collect(),
                )
            }

            /// 随机走 `n` 步合法动作（遇终局提前停止），返回实际步数。
            fn random_steps(&mut self, n: usize) -> PyResult<usize> {
                let mut rng = thread_rng();
                let mut done = 0;
                let n_actions = <$env>::action_space_size();
                for _ in 0..n {
                    let mut masks = vec![0i32; n_actions];
                    self.inner.action_masks_into(&mut masks);
                    let legal: Vec<usize> = (0..n_actions)
                        .filter(|&i| masks[i] == 1)
                        .collect();
                    if legal.is_empty() {
                        break;
                    }
                    let action = legal[rng.gen_range(0..legal.len())];
                    GameEnv::step(&mut self.inner, action)
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
                let mut masks = vec![0i32; <$env>::action_space_size()];
                self.inner.action_masks_into(&mut masks);
                (0..<$env>::action_space_size())
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
                let (_, terminated, truncated, winner) = GameEnv::step(&mut self.inner, action)
                    .map_err(|e| PyRuntimeError::new_err(format!("step 失败: {}", e)))?;
                Ok((terminated, truncated, winner))
            }

            /// 用当前局面做 Gumbel MCTS 搜索（调用 Python 网络回调评估），返回选中的动作。
            ///
            /// 返回 `None` 表示无合法动作（终局）。
            /// `c_scale`：PUCT 探索系数（必需）。`gumbel_scale`：Gumbel 噪声尺度（默认 1.0）。
            #[pyo3(signature = (predict_fn, num_simulations, max_considered_actions, c_scale, gumbel_scale = 1.0))]
            fn mcts_search_action(
                &self,
                predict_fn: Py<PyAny>,
                num_simulations: usize,
                max_considered_actions: usize,
                c_scale: f64,
                gumbel_scale: f64,
            ) -> PyResult<Option<usize>> {
                let evaluator = PyEvaluator::<$env>::new(predict_fn);
                let config = GumbelConfig {
                    num_simulations,
                    max_considered_actions,
                    c_scale: c_scale as f32,
                    gumbel_scale: gumbel_scale as f32,
                    ..Default::default()
                };
                let mut mcts = GumbelMCTS::new(&self.inner, &evaluator, config);
                Ok(mcts.run().map(|r| r.action))
            }

            /// 用 expectiminimax + alpha-beta 搜索选动作（不依赖网络，纯规则搜索）。
            ///
            /// `max_depth` 为搜索深度；返回 None 表示无合法动作（终局）。
            fn minimax_action(&self, max_depth: usize) -> PyResult<Option<usize>> {
                let result =
                    crate::engine::minimax::minimax_best_action(self.inner.as_darkchess(), max_depth);
                Ok(result.map(|r| r.action))
            }

            /// 用纯计算启发式 Gumbel MCTS 选动作（不依赖网络，规则先验 + 多特征评估）。
            ///
            /// `sims` 为模拟次数；返回 None 表示无合法动作（终局）。
            fn heuristic_mcts_action(&self, sims: usize) -> PyResult<Option<usize>> {
                let policy = crate::engine::HeuristicMctsPolicy::new(sims);
                Ok(policy.choose_action(self.inner.as_darkchess()))
            }

            /// 纯网络贪婪动作：对当前局面一次前向，取合法动作中 logit 最大者（无搜索）。
            ///
            /// 返回 `None` 表示无合法动作（终局）。
            fn greedy_action(&self, predict_fn: Py<PyAny>) -> PyResult<Option<usize>> {
                let evaluator = PyEvaluator::<$env>::new(predict_fn);
                let out = evaluator.evaluate(std::slice::from_ref(&self.inner));
                let logits_batch = &out.logits;
                let mut masks = vec![0i32; <$env>::action_space_size()];
                self.inner.action_masks_into(&mut masks);
                let mut best: Option<usize> = None;
                let mut best_v = f32::NEG_INFINITY;
                for a in 0..<$env>::action_space_size() {
                    if masks[a] == 1 && logits_batch[0][a] > best_v {
                        best_v = logits_batch[0][a];
                        best = Some(a);
                    }
                }
                Ok(best)
            }
        }
    };
}

// --- 三个变体：仅声明 Rust 类型名 / Python 类名 / 环境类型 ---

define_chess_env!(PyDarkChess, "DarkChess", DarkChessEnv);
define_chess_env!(PyGame4x4, "Game4x4", Game4x4Env);
define_chess_env!(PyMiniDarkChess, "MiniDarkChess", MiniDarkChessEnv);
