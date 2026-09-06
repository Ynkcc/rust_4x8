// 临时测量：4x4 全明子环境下，以指定 node_budget/max_depth 跑 NNUE Expectimax
// 整局自对弈，统计“能否走到有胜负的终局”、步数、累计搜索节点、是否顶满预算。
//
// 用法: cargo run --release --bin tmp_reach -- [nnue] [depth] [budget] [ngames] [seed]
use std::time::Instant;

use banqi_4x8::core::env::{DarkChessEnv, GameEnv};
use banqi_4x8::core::expectimax::ExpectimaxEngine;

fn play_one(
    engine: &ExpectimaxEngine,
    seed: u64,
) -> (Option<i32>, usize, u64, usize) {
    let mut env = DarkChessEnv::new_4x4();
    env.seed = Some(seed);
    env.reset();

    let mut winner = None;
    let mut moves = 0usize;
    let mut total_nodes = 0u64;
    let mut budget_capped_moves = 0usize;
    let max_steps = <DarkChessEnv as GameEnv>::max_steps();

    loop {
        if env.check_game_over_conditions().0 {
            break;
        }
        let Some(res) = engine.search_par(&env) else { break };
        total_nodes += res.nodes;
        if res.nodes >= engine.config.node_budget {
            budget_capped_moves += 1;
        }
        moves += 1;
        let Ok((_, terminated, truncated, w)) = GameEnv::step(&mut env, res.action) else { break };
        if w.is_some() {
            winner = w;
        }
        if terminated || truncated {
            break;
        }
        if moves >= max_steps {
            break;
        }
    }
    (winner, moves, total_nodes, budget_capped_moves)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = args
        .get(1)
        .cloned()
        .unwrap_or_else(|| "python/outputs/4x4/nnue_loop/best.nnue".to_string());
    let depth = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(8);
    let budget = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(200_000u64);
    let n_games = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(4usize);
    let base_seed = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(1000u64);

    let dim_check = DarkChessEnv::new_4x4().config.nnue_feature_dim();
    let mut engine = match ExpectimaxEngine::from_nnue_file(&path) {
        Ok(e) => e,
        Err(err) => {
            eprintln!("load fail {path}: {err}");
            std::process::exit(2);
        }
    };
    engine.set_max_depth(depth);
    engine.set_node_budget(budget);
    engine.config.threads = 1;

    println!(
        "4x4-fullinfo  model_feature_dim={} env_feature_dim={} budget={budget} depth={depth} n={n_games}",
        engine.config
            .nnue_evaluator
            .as_ref()
            .map(|e| e.feature_dim)
            .unwrap_or(0),
        dim_check
    );
    if engine
        .config
        .nnue_evaluator
        .as_ref()
        .map(|e| e.feature_dim != dim_check)
        .unwrap_or(true)
    {
        eprintln!("维度不匹配，中止（请换 4x4 的 .nnue）");
        std::process::exit(2);
    }

    let mut decisive = 0;
    let mut draws = 0;
    let mut total_moves = 0usize;
    let mut total_nodes = 0u64;
    let mut any_capped = 0usize;
    let max_steps = <DarkChessEnv as GameEnv>::max_steps();

    let t0 = Instant::now();
    for i in 0..n_games {
        let seed = base_seed.wrapping_add(i as u64);
        let (w, m, nodes, capped) = play_one(&engine, seed);
        if w.is_some() {
            decisive += 1;
        } else {
            draws += 1;
        }
        total_moves += m;
        total_nodes += nodes;
        if capped > 0 {
            any_capped += 1;
        }
        let label = match w {
            Some(1) => "RED_WIN ",
            Some(-1) => "BLACK_WIN",
            _ => "DRAW/TRUNC",
        };
        println!(
            "  game{i:02} seed={seed:<6} {label} moves={m:>3}/<={max_steps} total_nodes={nodes:>12} capped_moves={capped}"
        );
    }
    let dt = t0.elapsed();
    println!(
        "SUMMARY decisive={decisive}/{n_games} draws={draws} avg_moves={:.1} avg_total_nodes={} any_capped_games={any_capped}/{n_games} elapsed={dt:?}",
        total_moves as f32 / n_games as f32,
        total_nodes / n_games.max(1) as u64,
    );
}
