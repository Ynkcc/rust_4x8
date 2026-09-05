// src/self_play/batched.rs - 批量（batched）流水线自对弈
//
// 通过 `BatchedTree` 同时驱动 `concurrency` 局游戏，把多棵树的 MCTS 叶子评估
// 合并成一个大 batch 送给 evaluator，大幅提升 GPU/推理吞吐。
//
// 语义与 `run_batch_self_play` 保持一致：返回 `num_games` 个非空 GameEpisode。
//
// 吞吐优化——流水线（pipeline）：
//   评估器跑在一个后台线程上，主线程负责 MCTS 选择/回填。
//   - 主线程把一批叶子 envs 交给后台线程后，**不必等结果**，立即去给其他
//     未被阻塞的树继续做选择，从而把 CPU MCTS 遍历与（模拟）推理重叠起来，
//     让单个 CPU 核心保持忙碌（同时多批在途，GPU 也连续工作）。
//   - 只有当所有活跃树都在等待在途批结果、且没有可推进的树时，主线程才会
//     阻塞等待后台线程返回。
//
// 正确性：与单线程版本一致——任何树的叶子在被评估/回填前，该树不会继续前进；
// 不同树之间互不依赖，可以安全地在评估期间推进其他树。

use crate::core::env::{GameEnv, ResNetObservation, Player};
use crate::core::mcts::batched::BatchedTree;
use crate::core::mcts::{Evaluator, GumbelConfig, PendingEval, health_logits_expectation};
use crate::pipeline::self_play::finalize_episode;
use crate::pipeline::self_play::GameEpisode;
use std::collections::{HashMap, VecDeque};
use std::sync::mpsc::{channel, Sender};
use std::sync::{Arc, Condvar, Mutex};
use std::thread;

/// 评估请求：一批待评估环境。
struct EvalRequest<G: GameEnv> {
    id: u64,
    envs: Vec<G>,
}

/// 评估响应：按请求顺序返回 logits 与 values（含可选的血量分桶 logits）。
struct EvalResponse {
    id: u64,
    logits: Vec<Vec<f32>>,
    values: Vec<f32>,
    health: Option<Vec<Vec<f32>>>,
}

/// 共享请求队列（多消费者）：多个评估线程从这里取批。
struct EvalQueue<G: GameEnv> {
    reqs: Mutex<VecDeque<EvalRequest<G>>>,
    cvar: Condvar,
}

impl<G: GameEnv> Default for EvalQueue<G> {
    fn default() -> Self {
        Self {
            reqs: Mutex::new(VecDeque::new()),
            cvar: Condvar::new(),
        }
    }
}

impl<G: GameEnv> EvalQueue<G> {
    fn push(&self, req: EvalRequest<G>) {
        let mut q = self.reqs.lock().unwrap();
        q.push_back(req);
        self.cvar.notify_one();
    }

    /// 关闭：唤醒所有等待线程，使其退出。
    fn shutdown(&self) {
        self.cvar.notify_all();
    }
}

/// 后台评估线程：循环从共享队列取批、评估、回传结果。
///
/// 队列关闭（shutdown）后 `pop` 仍返回 Some 的话会空转，因此用 `stopped` 标志。
fn eval_worker<G: GameEnv, E: Evaluator<G> + Sync>(
    evaluator: &E,
    queue: &EvalQueue<G>,
    tx: Sender<EvalResponse>,
    stopped: &Arc<Mutex<bool>>,
) {
    loop {
        let req = {
            let mut q = queue.reqs.lock().unwrap();
            loop {
                if *stopped.lock().unwrap() {
                    return;
                }
                if let Some(req) = q.pop_front() {
                    break req;
                }
                q = queue.cvar.wait(q).unwrap();
            }
        };
        let out = evaluator.evaluate(&req.envs);
        if tx
            .send(EvalResponse {
                id: req.id,
                logits: out.logits,
                values: out.values,
                health: out.health,
            })
            .is_err()
        {
            break;
        }
    }
}

/// 运行批量自对弈（流水线版）。
///
/// - `evaluator`：评估器（要求 `Sync`，供后台线程共享）
/// - `config`：自对弈配置（mcts_sims / max_considered_actions / gumbel_scale）
/// - `num_games`：目标对局总数
/// - `concurrency`：同时推进的并发对局数（越大单批 batch 越大，流水线也越深）
/// - `make_env`：环境工厂
///
/// 返回的 episode 顺序即为完成顺序，与对局内部步序无关。
///
/// 注意：调用方若在持有 Python GIL 时调用，应先 `py.allow_threads(...)` 释放 GIL，
/// 否则后台线程的 `Python::with_gil` 会与主线程互等死锁。
pub fn run_batched_self_play<G: GameEnv + Sync, E: Evaluator<G> + Sync>(
    evaluator: &E,
    config: &crate::pipeline::self_play::SelfPlayConfig,
    num_games: usize,
    concurrency: usize,
    make_env: fn() -> G,
) -> Vec<GameEpisode> {
    let concurrency = concurrency.max(1);
    let gumbel_cfg = GumbelConfig {
        num_simulations: config.mcts_sims,
        max_considered_actions: config.max_considered_actions,
        c_scale: config.c_scale,
        gumbel_scale: config.gumbel_scale,
        health_enabled: config.health_enabled,
        health_weight: config.health_weight,
        health_confidence_exp: config.health_confidence_exp,
    };

    // 共享请求队列 + 响应通道
    let queue = Arc::new(EvalQueue::<G>::default());
    let stopped = Arc::new(Mutex::new(false));
    let (resp_tx, resp_rx) = channel::<EvalResponse>();

    // 评估 worker 数量：与并发对局数挂钩，但限制上限，避免过多 Python/GIL 竞争。
    let num_workers = concurrency.min(8).max(1);

    // 由于需要借用 `&evaluator`（非 'static），使用 scoped 线程。
    // 主循环结束后必须调用 `queue.shutdown()` 唤醒所有 worker 退出，
    // 否则 `thread::scope` 在 join 时死锁。
    thread::scope(move |scope| {
        for _ in 0..num_workers {
            let q = Arc::clone(&queue);
            let st = Arc::clone(&stopped);
            let tx = resp_tx.clone();
            scope.spawn(move || eval_worker(evaluator, q.as_ref(), tx, &st));
        }

        let mut episodes: Vec<GameEpisode> = Vec::with_capacity(num_games);
        let mut done = 0;

        // 分批（wave）推进：每波启动 concurrency 局新游戏，全部结束后进入下一波。
        while done < num_games {
            let wave = concurrency.min(num_games - done);

            // 初始化本波的游戏树 + 每局的样本收集
            let mut trees: Vec<BatchedTree<'_, G, E>> = Vec::with_capacity(wave);
            let mut episode_data: Vec<
                Vec<(ResNetObservation, Vec<f32>, f32, f32, u32, Player, Vec<i32>, usize, bool)>,
            > = Vec::with_capacity(wave);
            for _ in 0..wave {
                let env = make_env();
                trees.push(BatchedTree::new(&env, evaluator, &gumbel_cfg));
                episode_data.push(Vec::new());
            }
            let mut active: Vec<bool> = vec![true; wave];
            // 每棵树是否正等待一个在途批（Some(batch_id)），等待期间不得继续选择
            let mut blocked: Vec<Option<u64>> = vec![None; wave];
            // 在途批：batch_id -> (每个 eval 属于哪棵树, 待评估项)
            let mut in_flight: HashMap<u64, (Vec<usize>, Vec<PendingEval<G>>)> = HashMap::new();
            let mut next_batch_id: u64 = 0;

            // 交替推进各树，直到本波全部结束
            while active.iter().any(|&a| a) {
                // 1) 完成已就绪（Ready）且未被阻塞的决策
                for i in 0..wave {
                    if !active[i] || blocked[i].is_some() {
                        continue;
                    }
                    if trees[i].finalize_step() {
                        if let Some(r) = &trees[i].result {
                            episode_data[i].push((
                                r.state.clone(),
                                r.improved_policy.clone(),
                                r.mcts_value,
                                r.completed_q,
                                r.root_visit_count,
                                r.player,
                                r.action_mask.clone(),
                                r.action,
                                true, // 批量流水线未做算力随机化，全部视为 Full Search
                            ));
                        }
                        if trees[i].game_over {
                            active[i] = false;
                        } else {
                            trees[i].start_next_step();
                        }
                    }
                }

                // 2) 从所有未被阻塞的活跃树上收集待评估项，合并成一批
                let mut pool: Vec<PendingEval<G>> = Vec::new();
                let mut pool_targets: Vec<usize> = Vec::new();
                let mut touched: Vec<usize> = Vec::new();
                for i in 0..wave {
                    if !active[i] || blocked[i].is_some() {
                        continue;
                    }
                    let mut local: Vec<PendingEval<G>> = Vec::new();
                    if trees[i].collect(&mut local) {
                        for p in local {
                            pool.push(p);
                            pool_targets.push(i);
                        }
                        touched.push(i);
                    }
                }

                // 3) 有收集到东西：提交给后台线程评估（非阻塞），并阻塞涉及到的树
                if !pool.is_empty() {
                    let batch_id = next_batch_id;
                    next_batch_id += 1;
                    let envs: Vec<G> = pool.iter().map(|p| p.env).collect();
                    queue.push(EvalRequest { id: batch_id, envs });
                    for &t in &touched {
                        blocked[t] = Some(batch_id);
                    }
                    in_flight.insert(batch_id, (pool_targets, pool));
                }

                // 4) 尽力 drain 后台线程已返回的批并回填（非阻塞轮询）
                while let Ok(resp) = resp_rx.try_recv() {
                    if let Some((targets, evals)) = in_flight.remove(&resp.id) {
                        if resp.logits.len() != evals.len() {
                            eprintln!(
                                "⚠️ batched_self_play: 批 {} 结果数量 {} != 待评估 {}，丢弃",
                                resp.id,
                                resp.logits.len(),
                                evals.len()
                            );
                            for &t in &targets {
                                blocked[t] = None;
                            }
                            continue;
                        }
                        // 按树分组回填
                        let mut by_tree: HashMap<usize, Vec<usize>> = HashMap::new();
                        for (k, &t) in targets.iter().enumerate() {
                            by_tree.entry(t).or_default().push(k);
                        }
                        for (t, idxs) in by_tree {
                            if !active[t] {
                                continue;
                            }
                            let mut applied: Vec<(&PendingEval<G>, &[f32], f32, f32)> =
                                Vec::with_capacity(idxs.len());
                            for &k in &idxs {
                                let health =
                                    health_logits_expectation(resp.health.as_deref(), k).unwrap_or(0.0);
                                applied.push((&evals[k], &resp.logits[k], resp.values[k], health));
                            }
                            trees[t].apply(&applied);
                            blocked[t] = None;
                        }
                    }
                }

                // 5) 没有任何树可推进（要么全结束，要么全部在等待在途批）→
                //    若还有在途批，阻塞等待一个结果后再继续，避免空转；
                //    若无在途批也无活跃树则结束本波。
                let has_active = active.iter().any(|&a| a);
                let any_unblocked = (0..wave).any(|i| active[i] && blocked[i].is_none());
                if has_active && !any_unblocked && !in_flight.is_empty() {
                    // 主线程在此阻塞，等待后台线程返回任意一个结果，然后继续
                    match resp_rx.recv() {
                        Ok(resp) => {
                            if let Some((targets, evals)) = in_flight.remove(&resp.id) {
                                if resp.logits.len() == evals.len() {
                                    let mut by_tree: HashMap<usize, Vec<usize>> = HashMap::new();
                                    for (k, &t) in targets.iter().enumerate() {
                                        by_tree.entry(t).or_default().push(k);
                                    }
                                    for (t, idxs) in by_tree {
                                        if active[t] {
                                            let mut applied: Vec<(&PendingEval<G>, &[f32], f32, f32)> =
                                                Vec::with_capacity(idxs.len());
                                            for &k in &idxs {
                                                let health = health_logits_expectation(
                                                    resp.health.as_deref(),
                                                    k,
                                                )
                                                .unwrap_or(0.0);
                                                applied.push((
                                                    &evals[k],
                                                    &resp.logits[k],
                                                    resp.values[k],
                                                    health,
                                                ));
                                            }
                                            trees[t].apply(&applied);
                                            blocked[t] = None;
                                        }
                                    }
                                } else {
                                    for &t in &targets {
                                        blocked[t] = None;
                                    }
                                }
                            }
                        }
                        Err(_) => break,
                    }
                } else if !has_active {
                    break;
                }
                // 其余情况（仍有未阻塞树，或刚从阻塞中被唤醒）→ 回到循环顶继续
            }

            // 收尾：把本波完成的局 finalize 成 GameEpisode
            for i in 0..wave {
                let game_length = episode_data[i].len();
                if game_length == 0 {
                    continue;
                }
                let winner = trees[i].step_outcome.2;
                let health_diff_red = trees[i]
                    .tree
                    .root_env()
                    .and_then(|e| e.terminal_health_diff_red());
                episodes.push(finalize_episode(
                    std::mem::take(&mut episode_data[i]),
                    winner,
                    health_diff_red,
                    None,
                ));
                done += 1;
                if done >= num_games {
                    break;
                }
            }
        }

        // 全部对局完成：置停止标志并唤醒所有 worker 退出（否则 scope 会死锁）
        *stopped.lock().unwrap() = true;
        queue.shutdown();

        episodes
    })
}
