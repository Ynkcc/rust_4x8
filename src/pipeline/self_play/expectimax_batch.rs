//! Expectimax + NNUE 自对弈运行器。
//!
//! 并发模型与 MCTS 的 EvalQueue batching 流水线不同：Expectimax 是 DFS 串行搜索、
//! NNUE 叶评估为 CPU 向量化算子，天然按局并行（局间多 worker，每局独占引擎搜索
//! 上下文；局内可选 Lazy SMP 多线程）。每局完成即流式写出 JSONL，
//! 样本格式与 `python/banqi/nnue/train.py::NnueSampleDataset` 契约一致。

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc;
use std::sync::Arc;

use crate::core::env::DarkChessEnv;
use crate::core::expectimax::ExpectimaxEngine;

use super::match_core::play_one_game_expectimax;
use super::NnueEpisode;

/// Expectimax 自对弈引擎配置。
#[derive(Clone, Debug)]
pub struct ExpectimaxSelfPlayConfig {
    pub node_budget: u64,
    pub max_depth: i32,
    /// 局内 Lazy SMP 搜索线程数（自对弈建议 1，把核心留给局间 worker）
    pub threads_per_search: usize,
}

impl Default for ExpectimaxSelfPlayConfig {
    fn default() -> Self {
        Self {
            node_budget: 500_000,
            max_depth: 8,
            threads_per_search: 1,
        }
    }
}

/// 自对弈统计（从选手 A 视角计胜负；A/B 轮换先后手）。
#[derive(Debug, Clone, Default)]
pub struct ExpectimaxSelfPlayStats {
    pub games: usize,
    pub a_wins: usize,
    pub b_wins: usize,
    pub draws: usize,
    pub steps: usize,
    pub nodes: u64,
}

/// 运行 Expectimax + NNUE 自对弈。
///
/// - `variant_id`：变体字符串（"4x8"/"dark"、"4x2"/"mini"、"4x4"），
///   决定环境构造器与 NNUE 特征维度（`.nnue` 文件须与变体匹配）。
/// - `num_workers`：局间并发 worker 数（1 = 单线程）。
/// - `seed`：Some 时第 i 局使用 `s + i` 确定性种子（A/B 轮换先后手）。
/// - `out_jsonl`：Some 时每局完成即追加一行 NNUE episode JSON。
pub fn run_expectimax_self_play(
    nnue_path: &str,
    config: &ExpectimaxSelfPlayConfig,
    n_games: usize,
    num_workers: usize,
    seed: Option<u64>,
    out_jsonl: Option<&Path>,
    variant_id: &str,
) -> Result<ExpectimaxSelfPlayStats, String> {
    if n_games == 0 {
        return Ok(ExpectimaxSelfPlayStats::default());
    }
    let make_env: fn() -> DarkChessEnv = match variant_id.to_ascii_lowercase().as_str() {
        "4x8" | "dark" | "" => DarkChessEnv::new,
        "4x2" | "mini" => DarkChessEnv::new_mini,
        "4x4" | "game4x4" => DarkChessEnv::new_4x4,
        other => return Err(format!("未知变体: {other:?}（应为 4x8 | 4x2 | 4x4）")),
    };
    let engine = {
        let mut e = ExpectimaxEngine::from_nnue_file(nnue_path)?;
        e.config.node_budget = config.node_budget;
        e.config.max_depth = config.max_depth;
        e.config.threads = config.threads_per_search;
        Arc::new(e)
    };

    let next_game = AtomicUsize::new(0);
    let (tx, rx) = mpsc::channel::<(usize, i32, NnueEpisode)>();

    let writer = out_jsonl
        .map(|p| File::create(p))
        .transpose()
        .map_err(|e| format!("创建 JSONL 输出文件失败: {e}"))?;
    let mut writer = writer.map(BufWriter::new);

    let mut stats = ExpectimaxSelfPlayStats::default();

    std::thread::scope(|scope| -> Result<(), String> {
        for _ in 0..num_workers.max(1) {
            let engine = Arc::clone(&engine);
            let next_game = &next_game;
            let tx = tx.clone();
            scope.spawn(move || loop {
                let i = next_game.fetch_add(1, Ordering::SeqCst);
                if i >= n_games {
                    break;
                }
                let player_a_is_red = (i % 2) == 0;
                let game_seed = seed.map(|s| s.wrapping_add(i as u64));
                let outcome = play_one_game_expectimax::<DarkChessEnv>(
                    &engine,
                    player_a_is_red,
                    game_seed,
                    make_env,
                );
                let ep = outcome
                    .nnue_episode
                    .expect("Expectimax 记录路径必产出 NnueEpisode");
                if tx.send((i, outcome.result, ep)).is_err() {
                    break;
                }
            });
        }
        drop(tx);

        // 主线程：收结果 + 流式写 JSONL + 聚合统计
        let mut last_error = None;
        let mut received = 0;
        while received < n_games {
            match rx.recv() {
                Ok((_, result, ep)) => {
                    received += 1;
                    stats.games += 1;
                    stats.steps += ep.game_length;
                    match result {
                        1 => stats.a_wins += 1,
                        -1 => stats.b_wins += 1,
                        _ => stats.draws += 1,
                    }
                    if let Some(w) = &mut writer {
                        let line = super::nnue_episode_to_jsonl(&ep);
                        if let Err(e) = writeln!(w, "{line}") {
                            last_error = Some(format!("写入 JSONL 失败: {e}"));
                            break;
                        }
                    }
                }
                Err(e) => {
                    last_error = Some(format!("worker 通道异常关闭: {e}"));
                    break;
                }
            }
        }
        match last_error {
            Some(e) => Err(e),
            None => Ok(()),
        }
    })?;

    if let Some(w) = &mut writer {
        w.flush().map_err(|e| format!("刷新 JSONL 输出失败: {e}"))?;
    }
    Ok(stats)
}
