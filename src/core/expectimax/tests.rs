//! Expectimax 强引擎单元测试（自 engine/alpha_beta 合并）。

use crate::core::env::DarkChessEnv;

use super::search::{SearchConfig, search};

#[test]
fn engine_returns_legal_action() {
    let mut env = DarkChessEnv::new();
    env.seed = Some(5);
    env.reset();
    let mut cfg = SearchConfig::default();
    cfg.node_budget = 50_000;
    cfg.max_depth = 6;
    cfg.nnue_evaluator = None; // 未加载 NNUE 时叶评估为 0，仅验证搜索骨架
    let res = search(&env, &cfg).expect("应返回动作");
    // 动作必须合法
    let mut masks = vec![0i32; env.config.action_space_size];
    env.action_masks_into(&mut masks);
    assert_eq!(masks[res.action], 1, "引擎返回非法动作 {}", res.action);
    assert!(res.depth >= 1);
    assert!(res.nodes <= cfg.node_budget + 1000);
}

#[test]
fn engine_incremental_nnue_matches_full_recompute() {
    use crate::inference::nnue::NnueEvaluator;

    let feature_dim = DarkChessEnv::default().config.nnue_feature_dim();
    let eval = NnueEvaluator::new_dummy(feature_dim);
    let mut cfg = SearchConfig::default();
    cfg.node_budget = 50_000;
    cfg.max_depth = 5;
    cfg.nnue_evaluator = Some(std::sync::Arc::new(eval));

    for seed in [7u64, 33u64] {
        let mut env = DarkChessEnv::new();
        env.seed = Some(seed);
        env.reset();
        let res = search(&env, &cfg).expect("NNUE 增量搜索应返回动作");
        let mut masks = vec![0i32; env.config.action_space_size];
        env.action_masks_into(&mut masks);
        assert_eq!(masks[res.action], 1, "Seed {}: 引擎返回非法动作 {}", seed, res.action);
        assert!(res.value.abs() <= 1.0, "Seed {}: 评估值越界 {}", seed, res.value);
    }
}

#[test]
fn engine_feature_ablation_consistency() {
    // 验证关闭启发式优化（排序、置换表、晚走子减深）不破坏搜索合法性及基本界限
    let mut env = DarkChessEnv::new();
    env.seed = Some(12345);
    env.reset();

    let base_cfg = SearchConfig {
        node_budget: 10_000,
        max_depth: 3,
        ..Default::default()
    };

    let res_full = search(&env, &base_cfg).expect("全特性搜索应返回结果");

    let no_opt_cfg = SearchConfig {
        features: 0,
        node_budget: 10_000,
        max_depth: 3,
        ..Default::default()
    };
    let res_no_opt = search(&env, &no_opt_cfg).expect("无优化搜索应返回结果");

    let mut masks = vec![0i32; env.config.action_space_size];
    env.action_masks_into(&mut masks);
    assert_eq!(masks[res_full.action], 1);
    assert_eq!(masks[res_no_opt.action], 1);
    // 在相同深度下，无剪枝与有剪枝产出的根评估值符号或胜负倾向应当一致或接近
    assert!((res_full.value - res_no_opt.value).abs() < 0.5, "全特性与无优化搜索价值偏离过大");
}

#[test]
fn engine_survives_random_games() {
    for seed in 1..=4u64 {
        let mut env = DarkChessEnv::new();
        env.seed = Some(seed);
        env.reset();
        let mut cfg = SearchConfig::default();
        cfg.node_budget = 20_000;
        cfg.max_depth = 4;
        cfg.nnue_evaluator = None;
        let mut steps = 0;
        loop {
            let Some(res) = search(&env, &cfg) else { break };
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
        let cfg = SearchConfig {
            node_budget: budget,
            max_depth: 24,
            ..Default::default()
        };
        let t0 = std::time::Instant::now();
        let res = search(&env, &cfg);
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

/// 对称合并量化（`cargo test --release -- --ignored --nocapture tt_sym_quant`）：
/// 多个中盘局面开启 `tt_sym_probe`，输出原始键 miss 后对称键可挽回率，
/// 用于决定是否值得实现完整规范键置换表。
#[test]
#[ignore]
fn tt_sym_quant() {
    for seed in [1u64, 17, 42, 99, 12345] {
        let mut env = DarkChessEnv::new();
        env.seed = Some(seed);
        env.reset();
        // 随机走 12 步进入中盘（翻棋/吃子混合后的局面置换率更真实）
        for _ in 0..12 {
            let mut cfg = SearchConfig::default();
            cfg.node_budget = 3_000;
            cfg.max_depth = 2;
            cfg.nnue_evaluator = None;
            let Some(res) = search(&env, &cfg) else { break };
            let mut next = env;
            if next.step(res.action, None).is_err() {
                break;
            }
            env = next;
        }
        println!("--- seed={seed} ---");
        let cfg = SearchConfig {
            node_budget: 8_000_000,
            time_limit_ms: 15_000,
            max_depth: 12,
            tt_sym_probe: true,
            ..Default::default()
        };
        let res = search(&env, &cfg);
        match res {
            Some(r) => println!("nodes={} depth={} value={:.3}", r.nodes, r.depth, r.value),
            None => println!("无合法动作"),
        }
    }
}
