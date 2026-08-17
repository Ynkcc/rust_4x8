// src/py_data_collector.rs
//
// PyO3 数据收集器 - 基于 Python 回调的自对弈数据生成
// 与 data_collector.rs 的区别：
// 1. 不加载 TorchScript 模型，而是调用 Python 侧提供的预测函数
// 2. 不写入 MongoDB，而是把完整 GameEpisode 通过 Python 回调交给上层
// 3. 依然支持多局循环 + 迭代号统计

use anyhow::{Context, Result};
use banqi_4x8::py::PyEvaluator;
use banqi_4x8::self_play::{run_self_play, ScenarioType, SelfPlayConfig};
use banqi_4x8::DarkChessEnv;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use std::env;
use std::mem::size_of;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

// ============================================================================
// 内存估算模块：估计挂起单局游戏 (游戏状态 + MCTS 树) 需要的内存
// ============================================================================

pub mod memory_estimator {
    use super::*;

    // 核心常量 (与 game_env/constants.rs 保持一致)
    const BOARD_ROWS: usize = 4;
    const BOARD_COLS: usize = 8;
    const TOTAL_POSITIONS: usize = BOARD_ROWS * BOARD_COLS;
    const NUM_PIECE_TYPES: usize = 7;
    const TOTAL_PIECES_PER_PLAYER: usize = 16;
    const BOARD_CHANNELS: usize = 2 * NUM_PIECE_TYPES + 2;
    const SCALAR_FEATURE_COUNT: usize = 3 + 2 * TOTAL_PIECES_PER_PLAYER;
    const ACTION_SPACE_SIZE: usize = 32 + 104 + 216;
    const REVEAL_PROBABILITY_SIZE: usize = 2 * NUM_PIECE_TYPES;
    const MAX_STEPS_PER_EPISODE: usize = 100;

    #[derive(Debug, Clone)]
    pub struct MemoryBreakdown {
        pub item: String,
        pub size_bytes: usize,
        pub note: String,
    }

    #[derive(Debug, Clone)]
    pub struct MemoryEstimate {
        pub breakdown: Vec<MemoryBreakdown>,
        pub total_bytes: usize,
        pub total_kb: f64,
        pub total_mb: f64,
    }

    impl MemoryEstimate {
        pub fn new() -> Self {
            Self {
                breakdown: Vec::new(),
                total_bytes: 0,
                total_kb: 0.0,
                total_mb: 0.0,
            }
        }

        pub fn add(&mut self, item: &str, size_bytes: usize, note: &str) {
            self.breakdown.push(MemoryBreakdown {
                item: item.to_string(),
                size_bytes,
                note: note.to_string(),
            });
            self.total_bytes += size_bytes;
            self.total_kb = self.total_bytes as f64 / 1024.0;
            self.total_mb = self.total_kb / 1024.0;
        }

        pub fn add_subtotal(&mut self, item: &str, size_bytes: usize) {
            self.breakdown.push(MemoryBreakdown {
                item: format!("  ╚ {}", item),
                size_bytes,
                note: "subtotal".to_string(),
            });
            self.total_bytes += size_bytes;
            self.total_kb = self.total_bytes as f64 / 1024.0;
            self.total_mb = self.total_kb / 1024.0;
        }

        /// 合并另一个 MemoryEstimate 的所有条目和总计到当前实例。
        pub fn merge(&mut self, other: MemoryEstimate) {
            for b in other.breakdown {
                self.breakdown.push(b);
            }
            self.total_bytes += other.total_bytes;
            self.total_kb = self.total_bytes as f64 / 1024.0;
            self.total_mb = self.total_kb / 1024.0;
        }

        pub fn print_report(&self, title: &str) {
            println!("\n{:=<80}", "");
            println!("  {}", title);
            println!("{:=<80}", "");
            println!(
                "  {:<40} {:>14} {:>10}  {}",
                "项目", "字节 (B)", "KB", "说明"
            );
            println!("  {:-<80}", "");
            for b in &self.breakdown {
                let kb = b.size_bytes as f64 / 1024.0;
                println!(
                    "  {:<40} {:>14} {:>10.2}  {}",
                    b.item,
                    b.size_bytes,
                    kb,
                    b.note
                );
            }
            println!("  {:-<80}", "");
            println!(
                "  {:<40} {:>14} {:>10.2}  ({:.2} MB)",
                "TOTAL",
                self.total_bytes,
                self.total_kb,
                self.total_mb
            );
            println!("{:=<80}\n", "");
        }
    }

    fn vec_overhead() -> usize {
        size_of::<usize>() * 3
    }

    fn box_overhead() -> usize {
        size_of::<usize>()
    }

    fn option_overhead() -> usize {
        size_of::<u8>()
    }

    fn estimate_dark_chess_env() -> usize {
        // [Slot; 32] - Slot 是枚举，每个约 2 字节 (tag + Piece {type, player})
        let board_size = TOTAL_POSITIONS * 2;
        // Player enum = 1 byte
        let current_player = 1;
        // 2 x usize
        let counters = size_of::<usize>() * 2;
        // [[u64; 7]; 2] 位棋盘
        let piece_bbs = 2 * NUM_PIECE_TYPES * size_of::<u64>();
        // [u64; 2] revealed
        let revealed_bbs = 2 * size_of::<u64>();
        // u64 hidden + u64 empty
        let hidden_empty_bbs = 2 * size_of::<u64>();
        // [[PieceType; 16]; 2] - PieceType enum = 1 byte
        let dead_pool = 2 * TOTAL_PIECES_PER_PLAYER * 1;
        // [usize; 2]
        let dead_count = 2 * size_of::<usize>();
        // [i32; 2]
        let scores = 2 * size_of::<i32>();
        // i32
        let last_action = size_of::<i32>();
        // [Piece; 32] - Piece = 2 bytes
        let hidden_pool = TOTAL_POSITIONS * 2;
        // usize
        let hidden_count = size_of::<usize>();
        // [f32; 14]
        let reveal_probs = REVEAL_PROBABILITY_SIZE * size_of::<f32>();
        // Option<u64> + Option<[Piece; 32]>
        let opts = (option_overhead() + size_of::<u64>())
            + (option_overhead() + TOTAL_POSITIONS * 2);

        board_size
            + current_player
            + counters
            + piece_bbs
            + revealed_bbs
            + hidden_empty_bbs
            + dead_pool
            + dead_count
            + scores
            + last_action
            + hidden_pool
            + hidden_count
            + reveal_probs
            + opts
    }

    fn estimate_observation() -> usize {
        // Array3<f32> (16, 4, 8) = 16*4*8 * 4 bytes + ndarray overhead
        let board_data = BOARD_CHANNELS * BOARD_ROWS * BOARD_COLS * size_of::<f32>();
        let ndarray_overhead = size_of::<usize>() * 6;
        let board = board_data + ndarray_overhead;
        // Array1<f32> (35,) = 35 * 4 + overhead
        let scalar_data = SCALAR_FEATURE_COUNT * size_of::<f32>();
        let scalar = scalar_data + ndarray_overhead;
        board + scalar
    }

    pub fn estimate_game_state_suspended() -> MemoryEstimate {
        let mut est = MemoryEstimate::new();

        let env_size = estimate_dark_chess_env();
        est.add("DarkChessEnv (游戏环境)", env_size, "4x8棋盘+位棋盘+棋子池+概率表");
        est.add("Box<DarkChessEnv> 指针", box_overhead(), "Box指针占用");

        let obs_size = estimate_observation();
        est.add("Observation (NN输入)", obs_size, "board(16,4,8)f32 + scalars(35,)f32");
        est.add("Option<Observation> tag", option_overhead(), "Option判别位");

        let action_mask = ACTION_SPACE_SIZE * size_of::<i32>();
        est.add("Action Mask (动作掩码)", action_mask, "352个动作的i32掩码");

        let policy = ACTION_SPACE_SIZE * size_of::<f32>();
        est.add("Policy π (策略分布)", policy, "352个动作的f32概率");

        let misc_scalars = 4 * size_of::<f32>() + 2 * size_of::<u32>();
        est.add(
            "杂项标量 (value/Q/N等)",
            misc_scalars,
            "MCTS value, Q, visit count等",
        );

        est
    }

    pub fn estimate_mcts_node(
        avg_children: usize,
        avg_possible_states: usize,
        has_env: bool,
        has_state: bool,
    ) -> MemoryEstimate {
        let mut est = MemoryEstimate::new();

        let fixed_size = size_of::<u32>()      // visit_count
            + size_of::<f32>() * 4             // value_sum, prior, logit, initial_value
            + vec_overhead()                   // children
            + size_of::<bool>() * 4            // is_expanded, chance, root, terminal
            + vec_overhead()                   // possible_states
            + box_overhead() + option_overhead() // Option<Box<Env>>
            + size_of::<u8>()                  // Player enum
            + size_of::<usize>();              // Option<Observation> (ptr-ish) + tag approx

        est.add("MctsNode 固定字段", fixed_size, "不含Vec/Box/Option的内部数据");

        let children_size = avg_children * (size_of::<usize>() * 2);
        est.add(
            &format!("children Vec (avg {} entries)", avg_children),
            children_size + vec_overhead(),
            "(action_idx, node_idx) 对",
        );

        let possible_size = avg_possible_states
            * (size_of::<usize>() + size_of::<f32>() + size_of::<usize>());
        est.add(
            &format!("possible_states Vec (avg {})", avg_possible_states),
            possible_size + vec_overhead(),
            "机会节点的 (outcome, prob, node) 三元组",
        );

        if has_env {
            let env_size = estimate_dark_chess_env();
            est.add(
                "Option<Box<DarkChessEnv>>",
                box_overhead() + option_overhead() + env_size,
                "部分节点保存完整环境副本",
            );
        }

        if has_state {
            let obs_size = estimate_observation();
            est.add(
                "Option<Observation>",
                obs_size + option_overhead(),
                "部分节点保存NN输入缓存",
            );
        }

        est
    }

    pub fn estimate_mcts_tree(mcts_sims: usize) -> MemoryEstimate {
        let mut est = MemoryEstimate::new();

        let avg_branching_factor: f64 = 16.0;
        let total_nodes = (mcts_sims as f64 * avg_branching_factor * 0.6) as usize;
        let env_coverage: f64 = 0.15;
        let state_coverage: f64 = 0.35;
        let chance_node_ratio: f64 = 0.20;

        est.add(
            &format!("MCTS 总节点数 (估算)"),
            total_nodes,
            &format!(
                "sims={}, avg_branch={:.1}, factor=0.6",
                mcts_sims, avg_branching_factor
            ),
        );

        let avg_children_regular = 8;
        let avg_possible_regular = 0;
        let regular_node_est = estimate_mcts_node(
            avg_children_regular,
            avg_possible_regular,
            false,
            false,
        );
        let regular_node_size = regular_node_est.total_bytes;
        let regular_count = (total_nodes as f64 * (1.0 - chance_node_ratio)) as usize;
        est.add(
            &format!("普通决策节点 × {}", regular_count),
            regular_node_size * regular_count,
            &format!("avg {} children, 无env/state", avg_children_regular),
        );

        let chance_node_est = estimate_mcts_node(
            2,
            REVEAL_PROBABILITY_SIZE,
            false,
            false,
        );
        let chance_node_size = chance_node_est.total_bytes;
        let chance_count = total_nodes - regular_count;
        est.add(
            &format!("机会节点 × {}", chance_count),
            chance_node_size * chance_count,
            &format!("possible_states ≈ {} outcomes", REVEAL_PROBABILITY_SIZE),
        );

        let env_only_size = estimate_dark_chess_env() + box_overhead() + option_overhead();
        let env_nodes = (total_nodes as f64 * env_coverage) as usize;
        est.add(
            &format!("带Env副本的节点 × {}", env_nodes),
            env_only_size * env_nodes,
            &format!("{:.0}% 节点保存完整游戏环境", env_coverage * 100.0),
        );

        let state_only_size = estimate_observation() + option_overhead();
        let state_nodes = (total_nodes as f64 * state_coverage) as usize;
        est.add(
            &format!("带Observation缓存的节点 × {}", state_nodes),
            state_only_size * state_nodes,
            &format!("{:.0}% 节点缓存NN输入特征", state_coverage * 100.0),
        );

        let slab_overhead = size_of::<usize>() * 4 + total_nodes;
        est.add(
            "Slab<MctsNode> 内存池开销",
            slab_overhead,
            "Slab 元数据 + 可能的空洞 (估算 1B/节点)",
        );

        est
    }

    pub fn estimate_episode_storage(game_length: usize) -> MemoryEstimate {
        let mut est = MemoryEstimate::new();

        est.add(
            &format!("GameEpisode: {} 步样本", game_length),
            0,
            "单局游戏完整训练数据",
        );

        let step_board = BOARD_CHANNELS * BOARD_ROWS * BOARD_COLS * size_of::<f32>();
        let step_scalars = SCALAR_FEATURE_COUNT * size_of::<f32>();
        let step_policy = ACTION_SPACE_SIZE * size_of::<f32>();
        let step_mask = ACTION_SPACE_SIZE * size_of::<i32>();
        let step_scalar_fields = 3 * size_of::<f32>() + size_of::<u32>() + size_of::<f32>();
        let per_step = step_board + step_scalars + step_policy + step_mask + step_scalar_fields;

        est.add(
            "  每步 Observation+Policy+Mask+Scalars",
            per_step,
            &format!(
                "board(16,4,8) + scalars(35) + policy(352) + mask(352) + value/Q/N/result"
            ),
        );
        est.add(
            &format!("  {} 步样本数据小计", game_length),
            per_step * game_length,
            "samples Vec 数据内容",
        );

        let vec_meta = game_length * (
            size_of::<usize>() + vec_overhead() +
            vec_overhead() + vec_overhead()
        );
        est.add(
            "  samples Vec元数据 (tuple+Vec开销)",
            vec_meta,
            "嵌套Vec结构的分配器开销",
        );

        est.add(
            "  Episode 头部字段",
            size_of::<usize>() + option_overhead() + vec_overhead(),
            "game_length + winner + samples Vec元",
        );

        est
    }

    pub fn estimate_single_game_suspended(
        mcts_sims: usize,
        current_game_step: usize,
        expected_total_steps: usize,
    ) -> MemoryEstimate {
        let mut est = MemoryEstimate::new();

        est.add(
            "=== 挂起单局游戏: 运行时内存 ===",
            0,
            "(实际MCTS搜索时的峰值内存)",
        );

        let state_est = estimate_game_state_suspended();
        let state_total = state_est.total_bytes;
        est.merge(state_est);

        let tree_est = estimate_mcts_tree(mcts_sims);
        let tree_total = tree_est.total_bytes;
        est.merge(tree_est);

        est.add(
            "--- 子项小计: 游戏状态 + MCTS树 (运行时) ---",
            state_total + tree_total,
            "当次 MCTS 决策时的峰值占用",
        );

        est.add(
            "\n=== 挂起单局游戏: 训练数据存储 ===".into(),
            0,
            "(整局结束后保存的 GameEpisode)",
        );

        let ep_est = estimate_episode_storage(expected_total_steps);
        est.merge(ep_est);

        let _ = current_game_step;

        est
    }

    pub fn print_full_memory_report(mcts_sims: usize, games_per_iter: usize) {
        println!(
            "\n============================================================"
        );
        println!(
            "  🧮 内存估算报告  |  mcts_sims={}, games_per_iter={}",
            mcts_sims, games_per_iter
        );
        println!(
            "============================================================"
        );

        let env_sz = estimate_dark_chess_env();
        let obs_sz = estimate_observation();
        println!(
            "\n  [基础结构大小]  DarkChessEnv = {} B ({:.1} KB)  |  \
             Observation = {} B ({:.1} KB)",
            env_sz,
            env_sz as f64 / 1024.0,
            obs_sz,
            obs_sz as f64 / 1024.0,
        );

        let node_est = estimate_mcts_node(8, 0, false, false);
        let node_sz = node_est.total_bytes;
        println!(
            "  [基础结构大小]  MctsNode(普通,无env/state) = {} B",
            node_sz
        );

        estimate_game_state_suspended().print_report("① 单个游戏状态 (Suspended Game State)");

        estimate_mcts_tree(mcts_sims)
            .print_report(&format!("② 单次MCTS搜索树 (sims={})", mcts_sims));

        estimate_episode_storage(MAX_STEPS_PER_EPISODE).print_report(
            &format!("③ 单局训练数据 GameEpisode (max {} 步)", MAX_STEPS_PER_EPISODE),
        );

        let single = estimate_single_game_suspended(
            mcts_sims,
            MAX_STEPS_PER_EPISODE / 2,
            MAX_STEPS_PER_EPISODE,
        );
        single.print_report(&format!(
            "④ 挂起单局游戏 = 状态 + MCTS树 (sims={})",
            mcts_sims
        ));

        println!(
            "\n  ⚠️  安全余量建议: 以上为理论估算，实际内存预留建议 × (1.5 ~ 2.0)"
        );
        println!(
            "     - 单局挂起 (mcts_sims={}) 建议预留: {:.0} ~ {:.0} MB",
            mcts_sims,
            single.total_mb * 1.5,
            single.total_mb * 2.0
        );
        println!(
            "     - {} 局并行 (worker × games_per_iter) 建议预留: {:.0} ~ {:.0} MB",
            games_per_iter,
            single.total_mb * 1.5 * games_per_iter as f64,
            single.total_mb * 2.0 * games_per_iter as f64
        );
        println!(
            "============================================================\n"
        );
    }
}

fn load_python_predictor(py: Python<'_>, module_path: &str, func_name: &str) -> Result<Py<PyAny>> {
    let module_name = Path::new(module_path)
        .file_stem()
        .and_then(|s| s.to_str())
        .ok_or_else(|| anyhow::anyhow!("Invalid python module path: {}", module_path))?;

    let dir = Path::new(module_path)
        .parent()
        .and_then(|p| p.to_str())
        .unwrap_or(".");

    let sys = py.import("sys")?;
    let path_list = sys.getattr("path")?;
    let _ = path_list.call_method1("insert", (0, dir));
    let _ = path_list.call_method1("insert", (0, "."));

    let module = py
        .import(module_name)
        .map_err(|e| anyhow::anyhow!("Failed to import python module {}: {}", module_name, e))?;

    let predictor: Py<PyAny> = module
        .getattr(func_name)
        .map_err(|e| anyhow::anyhow!("Python module has no '{}': {}", func_name, e))?
        .unbind();

    Ok(predictor)
}

fn load_python_saver(py: Python<'_>, module_path: &str, func_name: &str) -> Result<Option<Py<PyAny>>> {
    let module_name = match Path::new(module_path).file_stem().and_then(|s| s.to_str()) {
        Some(name) => name,
        None => return Ok(None),
    };

    // 与 load_python_predictor 保持一致：将模块父目录加入 sys.path，
    // 否则当模块不在默认搜索路径时 import 会静默失败。
    let dir = Path::new(module_path)
        .parent()
        .and_then(|p| p.to_str())
        .unwrap_or(".");
    let sys = py.import("sys")?;
    let path_list = sys.getattr("path")?;
    let _ = path_list.call_method1("insert", (0, dir));
    let _ = path_list.call_method1("insert", (0, "."));

    let module = match py.import(module_name) {
        Ok(m) => m,
        Err(e) => {
            eprintln!(
                "[Saver] ⚠️ 无法导入 Python 模块 '{}' (函数 '{}'): {}",
                module_name, func_name, e
            );
            return Ok(None);
        }
    };

    let attr = match module.getattr(func_name) {
        Ok(a) => a,
        Err(e) => {
            eprintln!(
                "[Saver] ⚠️ 模块 '{}' 中未找到函数 '{}': {}",
                module_name, func_name, e
            );
            return Ok(None);
        }
    };

    Ok(Some(attr.unbind()))
}

fn build_episode_dict<'py>(
    py: Python<'py>,
    episode: &banqi_4x8::self_play::GameEpisode,
    iteration: usize,
    worker_id: usize,
) -> PyResult<Bound<'py, PyDict>> {
    // 复用 py::episode_to_dict，消除重复的样本序列化逻辑
    let d = banqi_4x8::py::episode_to_dict_darkchess(py, episode)?;
    // episode_to_dict 不包含 iteration / worker_id，这里补充
    d.set_item("iteration", iteration)?;
    d.set_item("worker_id", worker_id)?;
    Ok(d)
}

fn main() -> Result<()> {
    let python_module = env::var("PY_PREDICTOR_MODULE")
        .unwrap_or_else(|_| "./python/banqi/predictor.py".to_string());
    let predict_func = env::var("PY_PREDICT_FUNC").unwrap_or_else(|_| "predict".to_string());
    let save_func = env::var("PY_SAVE_FUNC").unwrap_or_else(|_| "save_episodes".to_string());

    let args: Vec<String> = env::args().collect();
    let worker_id = if args.len() > 1 {
        args[1].parse::<usize>().unwrap_or(0)
    } else {
        0
    };

    let mcts_sims: usize = env::var("MCTS_SIMS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(64);

    let games_per_iteration: usize = env::var("GAMES_PER_ITERATION")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(100);

    println!("=== PyO3 数据收集器-{} 启动 ===", worker_id);
    println!("Python predictor: {}::{}", python_module, predict_func);
    println!("Python saver func: {}", save_func);
    println!("MCTS Sims: {}", mcts_sims);
    println!("Games per iteration: {}", games_per_iteration);

    // 注册 Ctrl-C 信号处理器，实现优雅退出
    let should_exit = Arc::new(AtomicBool::new(false));
    {
        let flag = Arc::clone(&should_exit);
        ctrlc::set_handler(move || {
            flag.store(true, Ordering::SeqCst);
            eprintln!("\n[Worker-{}] 收到 Ctrl-C 信号，将在当前局结束后优雅退出...", worker_id);
        })
        .context("Failed to set Ctrl-C handler")?;
    }

    memory_estimator::print_full_memory_report(mcts_sims, games_per_iteration);

    Python::attach(|py| -> Result<()> {
        let predictor = load_python_predictor(py, &python_module, &predict_func)
            .context("Loading python predictor")?;
        let saver = load_python_saver(py, &python_module, &save_func)?;

        let evaluator = PyEvaluator::new(predictor);

        let config = SelfPlayConfig {
            mcts_sims,
            max_considered_actions: 16,
            temperature_steps: 12,
            scenario: ScenarioType::Standard,
            c_scale: 1.0,
            gumbel_scale: 1.0,
        };

        let mut game_count: usize = 0;
        let mut iteration: usize = 0;

        loop {
            // 检查是否收到退出信号
            if should_exit.load(Ordering::SeqCst) {
                eprintln!(
                    "[Worker-{}] 优雅退出: iter={}, game_count={}",
                    worker_id, iteration, game_count
                );
                break;
            }

            let start_time = Instant::now();
            let episode = run_self_play(&evaluator, &config, DarkChessEnv::new);
            let duration = start_time.elapsed();

            // 再次检查信号（run_self_play 可能耗时较长）
            if should_exit.load(Ordering::SeqCst) {
                eprintln!(
                    "[Worker-{}] 优雅退出（局后检查）: iter={}, game_count={}",
                    worker_id, iteration, game_count
                );
                break;
            }

            if episode.samples.is_empty() {
                eprintln!(
                    "[Worker-{}] ⚠️ 生成了空游戏数据，跳过保存",
                    worker_id
                );
                // 空局不占用迭代配额，与 data_collector.rs 语义一致
                continue;
            }

            let winner_str = match episode.winner {
                Some(1) => "红胜",
                Some(-1) => "黑胜",
                _ => "平局",
            };
            println!(
                "[Worker-{}] Game #{} (iter={}): 步数={}, 结果={}, 耗时={:.1}s ({:.1} steps/s)",
                worker_id,
                game_count + 1,
                iteration,
                episode.game_length,
                winner_str,
                duration.as_secs_f64(),
                episode.game_length as f64 / duration.as_secs_f64()
            );

            if let Some(ref save_cb) = saver {
                let d = match build_episode_dict(py, &episode, iteration, worker_id) {
                    Ok(dict) => dict,
                    Err(e) => {
                        eprintln!(
                            "[Worker-{}] ⚠️ 构建 episode dict 失败: {}",
                            worker_id, e
                        );
                        continue;
                    }
                };
                if let Err(e) = save_cb.call1(py, (vec![d],)) {
                    eprintln!(
                        "[Worker-{}] ⚠️ Python save callback failed: {}",
                        worker_id, e
                    );
                }
            } else {
                eprintln!(
                    "[Worker-{}] ⚠️ 未配置Python save回调 (PY_SAVE_FUNC)，跳过保存",
                    worker_id
                );
            }

            game_count = game_count.saturating_add(1);
            if game_count >= games_per_iteration {
                iteration = iteration.saturating_add(1);
                println!(
                    "[Worker-{}] 📍 完成迭代 {} → 进入迭代 {}",
                    worker_id,
                    iteration - 1,
                    iteration
                );
                game_count = 0;
            }
        }

        println!("[Worker-{}] 数据收集器已停止", worker_id);
        Ok(())
    })
}
