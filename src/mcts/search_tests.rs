// src/mcts/search_tests.rs
// 泛型 Gumbel MCTS 在井字棋（TicTacToeEnv，无机会节点）上的收敛测试。
//
// 验证目标：
// 1. 泛型化后 MCTS 能在「全为常规节点、无机会节点」的井字棋上正确运行；
// 2. 在 minimax 完美评估器的引导下，MCTS 收敛到最优策略：
//    - 先手 vs 随机：高胜率
//    - 后手 vs 随机：不败
//    - 先手/后手 vs minimax：全平（井字棋双方最优 = 平局）

use crate::game_env::{GameEnv, Player, TicTacToeEnv};
use crate::mcts::config::GumbelConfig;
use crate::mcts::evaluator::Evaluator;
use crate::mcts::search::GumbelMCTS;
use rand::prelude::*;
use std::cell::RefCell;
use std::collections::HashMap;

/// 从 `env.get_current_player()` 视角计算 minimax 价值：1=当前玩家胜，-1=负，0=平。
/// 带记忆化缓存（井字棋状态空间有限，缓存后每个局面只计算一次）。
fn minimax_cached(env: &TicTacToeEnv, cache: &mut HashMap<[i8; 9], i32>) -> i32 {
    let key = env.cells();
    if let Some(&v) = cache.get(&key) {
        return v;
    }
    let (terminated, _, winner) = env.check_game_over_conditions();
    let v = if terminated {
        match winner {
            Some(w) if w == env.get_current_player().val() => 1,
            Some(w) if w == env.get_current_player().opposite().val() => -1,
            _ => 0,
        }
    } else {
        let mut masks = [0i32; 9];
        env.action_masks_into(&mut masks);
        let mut best = -2;
        for a in 0..9 {
            if masks[a] == 1 {
                let mut next = *env;
                next.step(a).unwrap();
                // next 的返回值是「对手视角」，取反得到当前玩家视角
                best = best.max(-minimax_cached(&next, cache));
            }
        }
        best
    };
    cache.insert(key, v);
    v
}

/// 选择使 minimax 价值最大的动作（确定性，多解取第一个）。
fn minimax_action(env: &TicTacToeEnv) -> usize {
    let mut cache = HashMap::new();
    let mut masks = [0i32; 9];
    env.action_masks_into(&mut masks);
    let mut best_actions = Vec::new();
    let mut best_v = -2;
    for a in 0..9 {
        if masks[a] == 1 {
            let mut next = *env;
            next.step(a).unwrap();
            let v = -minimax_cached(&next, &mut cache);
            if v > best_v {
                best_v = v;
                best_actions.clear();
                best_actions.push(a);
            } else if v == best_v {
                best_actions.push(a);
            }
        }
    }
    best_actions[0]
}

fn random_action(env: &TicTacToeEnv) -> usize {
    let mut masks = [0i32; 9];
    env.action_masks_into(&mut masks);
    let legal: Vec<usize> = (0..9).filter(|&i| masks[i] == 1).collect();
    legal[thread_rng().gen_range(0..legal.len())]
}

/// 基于 minimax 的完美评估器：values 为当前玩家视角的胜负价值，policy 均匀。
/// 内部缓存 minimax 结果（MCTS 每步会反复评估大量叶子）。
struct TttMinimaxEvaluator {
    cache: RefCell<HashMap<[i8; 9], i32>>,
}

impl TttMinimaxEvaluator {
    fn new() -> Self {
        Self {
            cache: RefCell::new(HashMap::new()),
        }
    }
}

impl Evaluator<TicTacToeEnv> for TttMinimaxEvaluator {
    fn evaluate(&self, envs: &[TicTacToeEnv]) -> (Vec<Vec<f32>>, Vec<f32>) {
        let logits = vec![vec![0.0f32; 9]; envs.len()];
        let mut cache = self.cache.borrow_mut();
        let values: Vec<f32> = envs
            .iter()
            .map(|e| minimax_cached(e, &mut cache) as f32)
            .collect();
        (logits, values)
    }
}

enum Opponent {
    Random,
    Minimax,
}

/// 手动对局：MCTS 用 minimax 评估器；对手为 Random 或 Minimax。
/// 返回全局胜者（1=先手 Red 胜，-1=后手 Black 胜，0=平）。
///
/// 整局复用 MCTS 树（走 `step_next` 推进），验证树复用路径正确性。
fn play_ttt(mcts_is_red: bool, opponent: Opponent, sims: usize) -> i32 {
    let evaluator = TttMinimaxEvaluator::new();
    let config = GumbelConfig {
        num_simulations: sims,
        max_considered_actions: 9,
        c_scale: 1.0,
        gumbel_scale: 1.0,
    };
    let mut env = TicTacToeEnv::new();
    let mut mcts = GumbelMCTS::new(&env, &evaluator, config);

    loop {
        let is_red = env.get_current_player() == Player::Red;
        let mcts_turn = is_red == mcts_is_red;

        if mcts_turn {
            let result = match mcts.run() {
                Some(r) => r,
                None => break, // 无合法动作（不应发生）
            };
            let action = result.action;
            let (_, _, term, _, winner) = env.step(action).unwrap();
            mcts.step_next(&env, action);
            if term {
                return winner.unwrap_or(0);
            }
        } else {
            let action = match opponent {
                Opponent::Random => random_action(&env),
                Opponent::Minimax => minimax_action(&env),
            };
            let (_, _, term, _, winner) = env.step(action).unwrap();
            mcts.step_next(&env, action);
            if term {
                return winner.unwrap_or(0);
            }
        }
    }
    0
}

#[test]
fn ttt_mcts_first_player_beats_random() {
    let sims = 160;
    let n = 20;
    let mut wins = 0;
    let mut draws = 0;
    for _ in 0..n {
        match play_ttt(true, Opponent::Random, sims) {
            1 => wins += 1,
            0 => draws += 1,
            -1 => {}
            _ => unreachable!(),
        }
    }
    assert!(
        wins + draws >= (n as f32 * 0.95) as usize,
        "MCTS 先手 vs 随机: 胜={} 平={} 负={}，先手应几乎全胜",
        wins,
        draws,
        n - wins - draws
    );
    assert!(wins >= (n as f32 * 0.85) as usize);
}

#[test]
fn ttt_mcts_second_player_never_loses_to_random() {
    let sims = 160;
    let n = 20;
    let mut losses = 0;
    for _ in 0..n {
        // 返回 1 = 先手(Red)赢 = MCTS 后手输
        if play_ttt(false, Opponent::Random, sims) == 1 {
            losses += 1;
        }
    }
    assert!(
        losses == 0,
        "MCTS 后手 vs 随机: 输 {} 局，完美后手不应输",
        losses
    );
}

#[test]
fn ttt_mcts_draws_against_minimax() {
    let sims = 320;
    let n = 8;
    for _ in 0..n {
        assert_eq!(
            play_ttt(true, Opponent::Minimax, sims),
            0,
            "MCTS 先手 vs minimax 应全平"
        );
    }
    for _ in 0..n {
        assert_eq!(
            play_ttt(false, Opponent::Minimax, sims),
            0,
            "MCTS 后手 vs minimax 应全平"
        );
    }
}

/// 诊断：随机采样未结束局面，比较 MCTS 单步选择与 minimax 最优是否一致。
#[test]
fn ttt_mcts_single_step_matches_minimax() {
    use std::collections::HashSet;

    let mut seen: HashSet<[i8; 9]> = HashSet::new();
    let mut agree = 0;
    let mut total = 0;

    // 从空棋盘随机模拟出若干局面
    for _ in 0..200 {
        let mut env = TicTacToeEnv::new();
        loop {
            let (terminated, _, _) = env.check_game_over_conditions();
            if terminated {
                break;
            }
            let cells = env.cells();
            if cells.iter().filter(|&&c| c != 0).count() >= 3 && !seen.contains(&cells) {
                seen.insert(cells);
                total += 1;

                let eval = TttMinimaxEvaluator::new();
                let config = GumbelConfig {
                    num_simulations: 200,
                    max_considered_actions: 9,
                    c_scale: 1.0,
                    gumbel_scale: 1.0,
                };
                let mut mcts = GumbelMCTS::new(&env, &eval, config);
                if let Some(r) = mcts.run() {
                    let mcts_optimal_set: HashSet<usize> =
                        optimal_set(&env).into_iter().collect();
                    if mcts_optimal_set.contains(&r.action) {
                        agree += 1;
                    } else {
                        eprintln!(
                            "✗ 分歧: cells={:?} player={} optimal={:?} mcts={}",
                            cells,
                            env.get_current_player().val(),
                            optimal_set(&env),
                            r.action
                        );
                        if total >= 12 {
                            break;
                        }
                    }
                }
            }
            if env.cells().iter().filter(|&&c| c != 0).count() >= 9 {
                break;
            }
            let action = random_action(&env);
            let (_, _, term, _, _) = env.step(action).unwrap();
            if term {
                break;
            }
        }
        if total >= 12 {
            break;
        }
    }
    eprintln!("诊断: 一致 {}/{}", agree, total);
    assert!(agree * 10 >= total * 9, "MCTS 单步选择与 minimax 最优不一致过多");
}

/// 返回当前玩家视角下所有 minimax 最优动作。
fn optimal_set(env: &TicTacToeEnv) -> Vec<usize> {
    let mut cache = HashMap::new();
    let mut masks = [0i32; 9];
    env.action_masks_into(&mut masks);
    let mut best_v = -2;
    let mut best_actions = Vec::new();
    for a in 0..9 {
        if masks[a] == 1 {
            let mut next = *env;
            next.step(a).unwrap();
            let v = -minimax_cached(&next, &mut cache);
            if v > best_v {
                best_v = v;
                best_actions.clear();
                best_actions.push(a);
            } else if v == best_v {
                best_actions.push(a);
            }
        }
    }
    best_actions
}

/// 验证 MCTS 在空棋盘（开局）的第一次搜索可产出合法动作与有效策略。
#[test]
fn ttt_mcts_initial_search_produces_valid_action() {
    let evaluator = TttMinimaxEvaluator::new();
    let config = GumbelConfig {
        num_simulations: 32,
        max_considered_actions: 9,
        c_scale: 1.0,
        gumbel_scale: 1.0,
    };
    let env = TicTacToeEnv::new();
    let mut mcts = GumbelMCTS::new(&env, &evaluator, config);
    let result = mcts.run().expect("空棋盘应有合法动作");

    assert!(result.action < 9);
    let mut masks = [0i32; 9];
    env.action_masks_into(&mut masks);
    assert_eq!(masks[result.action], 1, "选中动作必须合法");

    assert_eq!(result.improved_policy.len(), 9);
    let sum: f32 = result.improved_policy.iter().sum();
    assert!((sum - 1.0).abs() < 1e-4, "improved_policy 应归一化, sum={}", sum);
    let legal_mass: f32 = result
        .improved_policy
        .iter()
        .enumerate()
        .filter(|&(i, _)| masks[i] == 1)
        .map(|(_, &p)| p)
        .sum();
    assert!(legal_mass > 0.99);
}
