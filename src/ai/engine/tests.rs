//! 纯计算引擎单元测试。

use super::*;

#[test]
fn engine_returns_legal_action() {
    let mut env = DarkChessEnv::new();
    env.seed = Some(5);
    env.reset();
    let mut cfg = EngineConfig::default();
    cfg.node_budget = 50_000;
    cfg.max_depth = 6;
    let res = best_move(&env, &cfg).expect("应返回动作");
    // 动作必须合法
    let mut masks = vec![0i32; env.config.action_space_size];
    env.action_masks_into(&mut masks);
    assert_eq!(masks[res.action], 1, "引擎返回非法动作 {}", res.action);
    assert!(res.depth >= 1);
    assert!(res.nodes <= cfg.node_budget + 1000);
}

#[test]
fn engine_survives_random_games() {
    for seed in 1..=4u64 {
        let mut env = DarkChessEnv::new();
        env.seed = Some(seed);
        env.reset();
        let _ = seed;
        let mut cfg = EngineConfig::default();
        cfg.node_budget = 20_000;
        cfg.max_depth = 4;
        let mut steps = 0;
        loop {
            let Some(res) = best_move(&env, &cfg) else { break };
            let mut next = env;
            if next.step(res.action, None).is_err() {
                panic!("引擎返回非法动作 {}", res.action);
            }
            env = next;
            let (term, _, _) = env.check_game_over_conditions();
            steps += 1;
            if term || steps > 60 {
                break;
            }
        }
    }
}

/// 性能基准（`cargo test --release -- --ignored engine_bench`）：
/// 用于校准默认节点预算对应的延迟。
#[test]
#[ignore]
fn engine_bench() {
    let mut env = DarkChessEnv::new();
    env.seed = Some(99);
    env.reset();
    for &budget in &[100_000u64, 300_000, 1_000_000] {
        let cfg = EngineConfig {
            node_budget: budget,
            max_depth: 24,
            ..Default::default()
        };
        let t0 = std::time::Instant::now();
        let res = best_move(&env, &cfg);
        let dt = t0.elapsed();
        match res {
            Some(r) => println!(
                "budget={budget} dt={dt:?} nodes={} depth={} value={:.3}",
                r.nodes, r.depth, r.value
            ),
            None => println!("budget={budget} 无合法动作"),
        }
    }
}
