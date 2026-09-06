use std::sync::Arc;
use std::time::Instant;
use banqi_4x8::core::env::DarkChessEnv;
use banqi_4x8::core::expectimax::{SearchConfig, search, search_par};
use banqi_4x8::inference::nnue::NnueEvaluator;

fn bench(path: &str, depth: i32, budget: u64, threads: usize, quiesce: bool, chance_red: i32) {
    let evaluator = match NnueEvaluator::load_from_file(path) {
        Ok(e) => e,
        Err(e) => {
            println!("load fail {path}: {e}");
            return;
        }
    };
    let env = DarkChessEnv::new();
    println!(
        "[dim] model={} env={} budget={budget} depth={depth} threads={threads}",
        evaluator.feature_dim,
        env.config.nnue_feature_dim()
    );
    if evaluator.feature_dim != env.config.nnue_feature_dim() {
        println!("dim mismatch, skip");
        return;
    }
    let cfg = SearchConfig {
        node_budget: budget,
        max_depth: depth,
        nnue_evaluator: Some(Arc::new(evaluator)),
        threads,
        quiesce,
        chance_reduction: chance_red,
        ..Default::default()
    };
    let t = Instant::now();
    let r = if threads <= 1 { search(&env, &cfg) } else { search_par(&env, &cfg) };
    let el = t.elapsed();
    match r {
        Some(res) => println!(
            "RESULT action={} value={:.4} depth={} nodes={} elapsed={:?}",
            res.action, res.value, res.depth, res.nodes, el
        ),
        None => println!("RESULT none (no legal move) elapsed={:?}", el),
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(1).cloned().unwrap_or_else(|| "python/checkpoints/nnue/base_v0.nnue".to_string());
    let depth = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(8);
    let budget = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(200_000);
    let threads = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(1);
    let quiesce = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(true);
    let chance_red = args.get(6).and_then(|s| s.parse().ok()).unwrap_or(1i32);
    bench(&path, depth, budget, threads, quiesce, chance_red);
}
