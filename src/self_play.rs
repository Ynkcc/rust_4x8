// src/self_play.rs - 自对弈与数据生成模块 (同步版, 泛型化 G = 游戏环境)
//
// 本模块实现了自对弈（Self-Play）逻辑，用于生成强化学习所需的训练数据。
// 重构说明：
// - 移除异步依赖，改为同步执行
// - 直接持有模型引用，无需 Channel 通信
// - 使用 Gumbel AlphaZero MCTS
// - 泛型化：`G: GameEnv` 可为暗棋（DarkChessEnv）或井字棋（TicTacToeEnv），
//   环境由调用方以 `fn() -> G` 工厂注入。

use crate::game_env::{GameEnv, Observation, Player};
use crate::mcts::batched::BatchedTree;
use crate::mcts::{Evaluator, GumbelConfig, GumbelMCTS};
use crate::mcts::PendingEval;
use std::time::Instant;

// ================ 数据结构定义 ================

/// 游戏简要统计信息
#[derive(Debug, Clone)]
pub struct GameStats {
    /// 游戏总步数
    pub steps: usize,
    /// 获胜方: Some(1)=红胜, Some(-1)=黑胜, None/Some(0)=平局
    pub winner: Option<i32>,
}

/// 单局游戏的完整数据记录
///
/// 包含该局游戏中每一步的观测状态、MCTS 搜索产生的策略概率、
/// MCTS 估算的根节点价值、实际选择的动作以及最终的游戏结果。
///
/// 样本中的观测统一使用 `Observation`（各游戏按自身通道/尺寸编码），
/// 因此 `GameEpisode` 本身不携带游戏泛型参数。
#[derive(Debug, Clone)]
pub struct GameEpisode {
    /// 训练样本列表: (观测状态, 策略概率分布, MCTS根节点价值, completed_Q, 根节点访问次数, 最终回报, 动作掩码, 实际动作)
    pub samples: Vec<(Observation, Vec<f32>, f32, f32, u32, f32, Vec<i32>, usize)>,
    /// 游戏总步数
    pub game_length: usize,
    /// 获胜方
    pub winner: Option<i32>,
}

// ================ 场景定义 ================

/// 训练场景类型枚举（暗棋专用）
///
/// 预留扩展点：未来可实现特定残局/开局场景 (如 TwoAdvisors, HiddenThreats)。
/// 目前所有场景均退化为标准开局。
#[derive(Debug, Clone, Copy)]
pub enum ScenarioType {
    /// 场景1: 双士残局 (R_A vs B_A) — 未实现，回退为 Standard
    TwoAdvisors,
    /// 场景2: 隐藏威胁 (Hidden Threat) — 未实现，回退为 Standard
    HiddenThreats,
    /// 标准开局 - 正常的完整游戏
    Standard,
}

impl ScenarioType {
    /// 根据枚举值创建对应的游戏环境（当前所有场景均创建标准环境）
    pub fn create_env(&self) -> crate::game_env::DarkChessEnv {
        crate::game_env::DarkChessEnv::new()
    }

    /// 获取场景的描述名称
    pub fn name(&self) -> &'static str {
        match self {
            ScenarioType::TwoAdvisors => "TwoAdvisors (R_A vs B_A) [unimplemented=Standard]",
            ScenarioType::HiddenThreats => "HiddenThreats [unimplemented=Standard]",
            ScenarioType::Standard => "Standard",
        }
    }

    /// 获取该场景下的期望最优动作索引 (预留验证接口，未实现场景默认返回 0)
    pub fn expected_action(&self) -> usize {
        match self {
            ScenarioType::TwoAdvisors | ScenarioType::HiddenThreats | ScenarioType::Standard => 0,
        }
    }
}

// ================ 自对弈配置 ================

/// 自对弈配置
#[derive(Clone)]
pub struct SelfPlayConfig {
    /// 每次决策执行的 MCTS 模拟次数
    pub mcts_sims: usize,
    /// Gumbel Top-K 候选动作数
    pub max_considered_actions: usize,
    // 注意：根节点 Dirichlet 噪声注入已移除。Gumbel AlphaZero 的探索由
    // Gumbel 噪声（Top-K 采样）与 Sequential Halving 提供，根节点子节点
    // prior 不参与任何搜索决策（Top-K 用 logit、根选择不经 PUCT），
    // 注入 Dirichlet 无效，请勿重新添加 dirichlet_alpha / dirichlet_epsilon 字段。
    /// 温度采样的步数阈值
    pub temperature_steps: usize,
    /// 训练场景
    pub scenario: ScenarioType,
    /// PUCT 探索系数（c_puct）与训练目标 σ 的缩放因子。默认 1.0。
    pub c_scale: f32,
    /// Gumbel 噪声尺度（根节点 Top-K 采样探索强度）。默认 1.0（标准 Gumbel）。
    pub gumbel_scale: f32,
}

impl Default for SelfPlayConfig {
    fn default() -> Self {
        Self {
            mcts_sims: 64,
            max_considered_actions: 16,
            temperature_steps: 10,
            scenario: ScenarioType::Standard,
            c_scale: 1.0,
            gumbel_scale: 1.0,
        }
    }
}

// ================ 自对弈运行器 (同步) ================

/// 自对弈运行器
///
/// 直接持有评估器引用，同步执行。
/// `G` 为游戏环境类型，环境由 `make_env` 工厂创建。
pub struct SelfPlayRunner<'a, G: GameEnv, E: Evaluator<G>> {
    evaluator: &'a E,
    config: SelfPlayConfig,
    make_env: fn() -> G,
}

impl<'a, G: GameEnv, E: Evaluator<G>> SelfPlayRunner<'a, G, E> {
    /// 创建新的自对弈运行器
    pub fn new(evaluator: &'a E, config: SelfPlayConfig, make_env: fn() -> G) -> Self {
        Self {
            evaluator,
            config,
            make_env,
        }
    }

    /// 使用默认配置创建
    pub fn with_defaults(evaluator: &'a E, mcts_sims: usize, make_env: fn() -> G) -> Self {
        let config = SelfPlayConfig {
            mcts_sims,
            ..Default::default()
        };
        Self {
            evaluator,
            config,
            make_env,
        }
    }

    /// 执行一局完整的自对弈 (同步)
    pub fn play_episode(&self, _episode_num: usize) -> GameEpisode {
        let _start_time = Instant::now();

        // 1. 初始化环境
        let mut env = (self.make_env)();

        // 2. 配置 MCTS
        let mcts_config = GumbelConfig {
            num_simulations: self.config.mcts_sims,
            max_considered_actions: self.config.max_considered_actions,
            c_scale: self.config.c_scale,
            gumbel_scale: self.config.gumbel_scale,
        };
        let mut mcts = GumbelMCTS::new(&env, self.evaluator, mcts_config.clone());

        let mut episode_data = Vec::new();
        let mut step = 0;

        // 3. 游戏主循环
        loop {
            // 注意：这里不再注入根节点 Dirichlet 噪声 —— Gumbel AlphaZero 的
            // 探索由 Gumbel 噪声 + Sequential Halving 提供，根节点 prior 不参与
            // 搜索决策，注入无效（详见 src/mcts/search.rs 中的说明）。请勿加回。

            // --- MCTS 搜索 (同步) ---
            let search_result = match mcts.run() {
                Some(result) => result,
                None => {
                    // mcts.run() 返回 None = 当前玩家无合法走法 → 该玩家判负。
                    // 调用环境终止条件获取真实 winner，回填正确的 ±1 胜负。
                    let (_, _, winner) = env.check_game_over_conditions();
                    return finalize_episode(episode_data, winner);
                }
            };

            // --- 温度采样：前 temperature_steps 用 τ=1（探索），之后用 argmax（利用）---
            let temperature: f32 = if step < self.config.temperature_steps { 1.0 } else { 1e-3 };
            let sampled_action = {
                // Gumbel AlphaZero 标准动作选择：基于 completed Q 的温度 softmax（π ∝ exp(Q/τ)）
                let q_policy = mcts.get_root_completed_q_policy(temperature);
                GumbelMCTS::<G, E>::sample_action_from_policy(&q_policy, &search_result.action_mask)
            };
            let action = sampled_action;
            let completed_q = mcts.get_root_completed_q(action);

            // --- 收集样本数据 ---
            // 注意: improved_policy 仍使用 Gumbel AlphaZero 的 σ(Q) + logit 公式作为训练目标
            // 实际动作 action 一并记录，用于对局回放 / 文字棋谱还原与交叉校验
            episode_data.push((
                search_result.state,
                search_result.improved_policy,
                search_result.mcts_value,
                completed_q,
                search_result.root_visit_count,
                search_result.player,
                search_result.action_mask,
                action,
            ));

            // --- 执行动作 ---
            match env.step(action) {
                Ok((_, _, terminated, truncated, winner)) => {
                    // 推进 MCTS 树
                    mcts.step_next(&env, action);

                    if terminated || truncated {
                        // --- 游戏结束处理：统一回填 ---
                        return finalize_episode(episode_data, winner);
                    }
                }
                Err(e) => {
                    eprintln!("  ⚠️ 游戏错误 (step={}, action={}): {}", step, action, e);
                    return GameEpisode {
                        samples: Vec::new(),
                        game_length: step,
                        winner: None,
                    };
                }
            }

            // --- 步数限制检查：使用环境给定的步数上限 ---
            step += 1;
            if step >= G::max_steps() {
                // 步数上限截断：环境视其为 truncated 平局 (winner=Some(0))，
                // 与终局分支语义对齐，game_result 回填 0.0。
                return finalize_episode(episode_data, Some(0));
            }
        }
    }
}

// ================ 高级 API ================

/// 运行单局自对弈
///
/// `make_env` 为环境工厂（如 `DarkChessEnv::new` 或 `TicTacToeEnv::new`）。
pub fn run_self_play<G: GameEnv, E: Evaluator<G>>(
    evaluator: &E,
    config: &SelfPlayConfig,
    make_env: fn() -> G,
) -> GameEpisode {
    let runner = SelfPlayRunner::new(evaluator, config.clone(), make_env);
    runner.play_episode(0)
}

/// 批量运行多局自对弈
pub fn run_batch_self_play<G: GameEnv, E: Evaluator<G>>(
    evaluator: &E,
    config: &SelfPlayConfig,
    num_games: usize,
    make_env: fn() -> G,
) -> Vec<GameEpisode> {
    (0..num_games)
        .map(|i| {
            let runner = SelfPlayRunner::new(evaluator, config.clone(), make_env);
            runner.play_episode(i)
        })
        .collect()
}

// ============================================================================
// 批量（batched）自对弈
// ============================================================================
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

use std::collections::{HashMap, VecDeque};
use std::sync::mpsc::{channel, Sender};
use std::sync::{Arc, Condvar, Mutex};
use std::thread;

/// 评估请求：一批待评估环境。
struct EvalRequest<G: GameEnv> {
    id: u64,
    envs: Vec<G>,
}

/// 评估响应：按请求顺序返回 logits 与 values。
struct EvalResponse {
    id: u64,
    logits: Vec<Vec<f32>>,
    values: Vec<f32>,
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
        let (logits, values) = evaluator.evaluate(&req.envs);
        if tx
            .send(EvalResponse {
                id: req.id,
                logits,
                values,
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
/// - `config`：自对弈配置（mcts_sims / max_considered_actions / temperature_steps）
/// - `num_games`：目标对局总数
/// - `concurrency`：同时推进的并发对局数（越大单批 batch 越大，流水线也越深）
/// - `make_env`：环境工厂
///
/// 返回的 episode 顺序即为完成顺序（先完成的先返回），与对局内部步序无关。
///
/// 注意：调用方若在持有 Python GIL 时调用，应先 `py.allow_threads(...)` 释放 GIL，
/// 否则后台线程的 `Python::with_gil` 会与主线程互等死锁。
pub fn run_batched_self_play<G: GameEnv + Sync, E: Evaluator<G> + Sync>(
    evaluator: &E,
    config: &SelfPlayConfig,
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
                Vec<(Observation, Vec<f32>, f32, f32, u32, Player, Vec<i32>, usize)>,
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

            // 辅助：把一棵树的叶子收集进当前批
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
                            let mut applied: Vec<(&PendingEval<G>, &[f32], f32)> =
                                Vec::with_capacity(idxs.len());
                            for &k in &idxs {
                                applied.push((&evals[k], &resp.logits[k], resp.values[k]));
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
                                            let mut applied: Vec<(&PendingEval<G>, &[f32], f32)> =
                                                Vec::with_capacity(idxs.len());
                                            for &k in &idxs {
                                                applied.push(
                                                    (&evals[k], &resp.logits[k], resp.values[k]),
                                                );
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
                episodes.push(finalize_episode(std::mem::take(&mut episode_data[i]), winner));
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

// ================ 辅助函数 ================

/// 按 winner 统一回填 episode_data 并构造 GameEpisode。
///
/// - reward_red：winner=Some(1) → 1.0，Some(-1) → -1.0，None/Some(0) → 0.0；
/// - 每个样本按该样本玩家的视角换算 game_result（红方视角为正）；
/// - game_length 统一为「已完成步数」= 样本数，消除各终止路径语义差 1 的不一致。
///
/// 该函数同时被三条终止路径调用：MCTS None 分支（无合法走法判负）、
/// 终局分支（terminated/truncated）、步数上限分支。
pub(crate) fn finalize_episode(
    episode_data: Vec<(Observation, Vec<f32>, f32, f32, u32, Player, Vec<i32>, usize)>,
    winner: Option<i32>,
) -> GameEpisode {
    let game_length = episode_data.len();
    let reward_red: f32 = match winner {
        Some(1) => 1.0,
        Some(-1) => -1.0,
        _ => 0.0,
    };
    let samples = episode_data
        .into_iter()
        .map(|(obs, p, mcts_val, completed_q, root_visit_count, player, mask, action)| {
            let game_result_val: f32 = if player.val() == 1 {
                reward_red
            } else {
                -reward_red
            };
            (
                obs,
                p,
                mcts_val,
                completed_q,
                root_visit_count,
                game_result_val,
                mask,
                action,
            )
        })
        .collect();
    GameEpisode {
        samples,
        game_length,
        winner,
    }
}

/// 选择 completed_Q 最大的动作（确定性）
pub fn select_completed_q_action<G: GameEnv, E: Evaluator<G>>(
    mcts: &GumbelMCTS<G, E>,
    masks: &[i32],
) -> (usize, f32) {
    let mut best_action: Option<usize> = None;
    let mut best_completed_q = f32::NEG_INFINITY;

    for (action, &mask) in masks.iter().enumerate() {
        if mask != 1 {
            continue;
        }
        let completed_q = mcts.get_root_completed_q(action);
        if completed_q > best_completed_q {
            best_completed_q = completed_q;
            best_action = Some(action);
        }
    }

    let action = best_action.expect("无有效动作");
    (action, best_completed_q)
}

/// 获取 Top-K 动作 (用于调试)
pub fn get_top_k_actions(probs: &[f32], k: usize) -> Vec<(usize, f32)> {
    let mut indexed: Vec<(usize, f32)> = probs.iter().enumerate().map(|(i, &p)| (i, p)).collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    indexed.into_iter().take(k).collect()
}
