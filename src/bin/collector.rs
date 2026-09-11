// src/bin/collector.rs — 统一 Collector 进程（统一训练架构 L3，单机/分布式共用）
//
// 主循环（§3.1 同构）：
//   task = registry.get_task()
//   model = registry.ensure_model()
//   episodes = self_play::run(...)   # 复用 pipeline/self_play 主干（Rust ONNX 推理）
//   episode_store.put / registry.report_episode
//
// backend=local     : LocalRegistry（notify watch 热重载）+ LocalEpisodeStore 落盘
// backend=scheduler : SchedulerRegistry（gRPC GetTask + R2 预签名直传；selfplay/rating 双任务）

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use clap::Parser;
use flate2::write::GzEncoder;
use flate2::Compression;
use rayon::ThreadPoolBuilder;

use banqi_4x8::core::env::DarkChessEnv;
use banqi_4x8::core::env::variants::{Game4x4Env, MiniDarkChessEnv};
use banqi_4x8::core::env::traits::GameEnv;
use banqi_4x8::inference::onnx::{OnnxModel, OnnxEvaluator};
use banqi_4x8::pipeline::self_play::{
    PlayerSpec, SelfPlayConfig, ScenarioType, MatchParams, run_match_core,
};
use banqi_4x8::registry::{LocalEpisodeStore, LocalRegistry, SchedulerConfig, SchedulerRegistry};
use banqi_4x8::registry::scheduler_registry::pb::TaskKind;

#[derive(Parser, Debug)]
#[command(name = "banqi-collector", about = "统一自对弈采集进程（Collector）")]
struct Args {
    /// 变体 id：4x8 / 4x4 / mini（backend=local 使用；backend=scheduler 由服务端下发）
    #[arg(long, default_value = "4x8")]
    variant: String,
    /// 协作后端：local（单机） / scheduler（分布式）
    #[arg(long, default_value = "local")]
    backend: String,
    /// backend=local：onnx 模型路径（Registry watch 指针）
    #[arg(long, default_value = "outputs/model_latest.onnx")]
    model_path: String,
    /// backend=local：episode 落盘根目录
    #[arg(long, default_value = "outputs/episodes")]
    out_dir: String,
    /// backend=scheduler：调度器 gRPC 地址（http://host:port）
    #[arg(long, default_value = "http://127.0.0.1:50051")]
    scheduler_endpoint: String,
    /// backend=scheduler：worker 标识
    #[arg(long, default_value = "")]
    worker_id: String,
    /// backend=scheduler：网络/模型本地缓存目录
    #[arg(long, default_value = "outputs/distributed_cache")]
    cache_dir: String,
    /// 推理设备：cpu / auto
    #[arg(long, default_value = "auto")]
    device: String,
    #[arg(long, default_value_t = 64)]
    mcts_sims: usize,
    #[arg(long, default_value_t = 16)]
    max_considered_actions: usize,
    /// 每批自对弈局数（scheduler 模式下为单批上限）
    #[arg(long, default_value_t = 64)]
    games_per_iter: usize,
    /// 自对弈线程数（默认 = CPU 核数）
    #[arg(long, default_value_t = 0)]
    threads: usize,
    /// 总批数上限（0 = 无限循环）
    #[arg(long, default_value_t = 0)]
    iterations: usize,
    #[arg(long, default_value_t = 0)]
    worker_num: usize,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let config = SelfPlayConfig {
        mcts_sims: args.mcts_sims,
        max_considered_actions: args.max_considered_actions,
        scenario: ScenarioType::Standard,
        c_scale: 1.0,
        gumbel_scale: 1.0,
        playout_cap_random_enabled: false,
        fast_mcts_sims: args.mcts_sims / 4,
        full_search_prob: 0.25,
        ..Default::default()
    };

    match args.backend.as_str() {
        "local" => run_local(args, config),
        "scheduler" => run_scheduler(args, config),
        other => anyhow::bail!("未知 backend: {other}（可选 local / scheduler）"),
    }
}

// ============================================================================
// backend = local（单机形态）
// ============================================================================

fn run_local(args: Args, config: SelfPlayConfig) -> Result<()> {
    let stop = Arc::new(AtomicBool::new(false));
    {
        let stop = Arc::clone(&stop);
        ctrlc::set_handler(move || stop.store(true, Ordering::SeqCst))
            .context("注册 Ctrl-C handler 失败")?;
    }

    let mut registry = LocalRegistry::new(&args.model_path, &args.device)?;
    let store = LocalEpisodeStore::new(&args.out_dir, args.worker_num);
    let pool = build_pool(args.threads)?;

    println!("=== banqi-collector 启动（local） ===");
    println!("variant={} model={} out={}", args.variant, args.model_path, args.out_dir);
    println!(
        "mcts_sims={} games_per_iter={} threads={}",
        args.mcts_sims, args.games_per_iter, pool.current_num_threads()
    );

    let mut iteration: usize = 0;
    let mut total_games: usize = 0;
    while !stop.load(Ordering::SeqCst) && (args.iterations == 0 || iteration < args.iterations) {
        let model = registry.ensure_model()?;

        let started = Instant::now();
        let result = run_variant_dispatch(
            &args.variant,
            Arc::clone(&model),
            Arc::clone(&model),
            &config,
            args.games_per_iter,
            true,
            &pool,
        )?;
        let duration = started.elapsed();

        let stored = store.put(&args.variant, iteration, &result.episodes)?;
        total_games += stored;
        println!(
            "[iter {iteration}] 🎮 {stored} 局（步均 {:.1}）→ {out} 落盘，耗时 {:.1}s",
            avg_steps(&result.episodes),
            duration.as_secs_f64(),
            out = args.out_dir,
        );
        iteration += 1;
    }

    println!("=== banqi-collector 退出：累计 {total_games} 局 / {iteration} 批 ===");
    Ok(())
}

// ============================================================================
// backend = scheduler（分布式形态）
// ============================================================================

fn run_scheduler(args: Args, mut config: SelfPlayConfig) -> Result<()> {
    let pool = build_pool(args.threads)?;
    let mut registry = SchedulerRegistry::new(SchedulerConfig {
        endpoint: args.scheduler_endpoint.clone(),
        worker_id: if args.worker_id.is_empty() {
            format!("worker-{}", std::process::id())
        } else {
            args.worker_id.clone()
        },
        client_version: env!("CARGO_PKG_VERSION").to_string(),
        cache_dir: args.cache_dir.clone().into(),
        device: args.device.clone(),
    })?;

    println!("=== banqi-collector 启动（scheduler） ===");
    println!(
        "endpoint={} cache={} threads={}（variant 由服务端 GetTask 下发）",
        args.scheduler_endpoint, args.cache_dir, pool.current_num_threads()
    );

    const BACKOFF: Duration = Duration::from_secs(30);
    let mut iteration: usize = 0;
    while args.iterations == 0 || iteration < args.iterations {
        let task = match registry.get_task()? {
            Some(t) => {
                registry.set_running_task(&t.task_id);
                t
            }
            None => {
                std::thread::sleep(BACKOFF);
                continue;
            }
        };

        // 服务端下发的 mcts_sims 覆盖本地值（0 = 不覆盖）
        if task.mcts_sims > 0 {
            config.mcts_sims = task.mcts_sims;
            config.fast_mcts_sims = task.mcts_sims / 4;
        }

        let started = Instant::now();
        match task.kind {
            TaskKind::TaskSelfplay => {
                let model = registry.model(&task.network_sha)?;
                let result = run_variant_dispatch(
                    &task.variant,
                    Arc::clone(&model),
                    Arc::clone(&model),
                    &config,
                    task.games,
                    true,
                    &pool,
                )?;
                let gz = episodes_gz(&result.episodes)?;
                let total_steps: usize =
                    result.episodes.iter().map(|e| e.game_length).sum();
                // 批内聚合胜方（调度端仅作日志/统计用）
                let winner = if result.wins > result.losses {
                    1
                } else if result.wins < result.losses {
                    -1
                } else {
                    0
                };
                registry.report_episode(
                    &task.task_id,
                    &task.network_sha,
                    result.episodes.len(),
                    total_steps,
                    winner,
                    gz,
                )?;
                println!(
                    "[iter {iteration}] selfplay task={} 🎮 {} 局（步均 {:.1}）耗时 {:.1}s",
                    task.task_id,
                    result.episodes.len(),
                    avg_steps(&result.episodes),
                    started.elapsed().as_secs_f64()
                );
                registry.add_completed_games(result.episodes.len());
            }
            TaskKind::TaskRating => {
                let candidate = registry.model(&task.network_sha)?;
                let opponent = registry.model(&task.opponent_sha)?;
                // 换色配对要求偶数局
                let n = (task.games - task.games % 2).max(2);
                let result = run_variant_dispatch(
                    &task.variant,
                    candidate,
                    opponent,
                    &config,
                    n,
                    false,
                    &pool,
                )?;
                let pairs = pairs_from_outcomes(&result.game_outcomes);
                println!(
                    "[iter {iteration}] rating task={} games={} w/l/d={}/{}/{} pairs={:?} 耗时 {:.1}s",
                    task.task_id,
                    n,
                    result.wins,
                    result.losses,
                    result.draws,
                    pairs,
                    started.elapsed().as_secs_f64()
                );
                registry.add_completed_games(n);
                registry.report_match_result(
                    &task.task_id,
                    &task.network_sha,
                    &task.opponent_sha,
                    n,
                    result.wins,
                    result.losses,
                    result.draws,
                    pairs,
                )?;
            }
            TaskKind::TaskNone => unreachable!("get_task 已过滤 TASK_NONE"),
        }
        iteration += 1;
    }
    Ok(())
}
fn build_pool(threads: usize) -> Result<rayon::ThreadPool> {
    ThreadPoolBuilder::new()
        .num_threads(if threads > 0 { threads } else { num_cpus::get() })
        .build()
        .context("构建 rayon 线程池失败")
}
fn avg_steps(episodes: &[banqi_4x8::pipeline::self_play::GameEpisode]) -> f64 {
    if episodes.is_empty() {
        0.0
    } else {
        episodes.iter().map(|e| e.game_length as f64).sum::<f64>() / episodes.len() as f64
    }
}

/// 按变体分发到泛型主干（run_match_core 需静态类型 G）。
#[allow(clippy::too_many_arguments)]
fn run_variant_dispatch(
    variant: &str,
    model_a: Arc<OnnxModel>,
    model_b: Arc<OnnxModel>,
    config: &SelfPlayConfig,
    n_games: usize,
    record_episodes: bool,
    pool: &rayon::ThreadPool,
) -> Result<banqi_4x8::pipeline::self_play::MatchResult> {
    match variant {
        "4x8" => run_games::<DarkChessEnv>(model_a, model_b, config, n_games, record_episodes, pool),
        "4x4" => run_games::<Game4x4Env>(model_a, model_b, config, n_games, record_episodes, pool),
        "mini" => run_games::<MiniDarkChessEnv>(model_a, model_b, config, n_games, record_episodes, pool),
        other => anyhow::bail!("未知变体: {other}（可选 4x8 / 4x4 / mini）"),
    }
}

fn run_games<G>(
    model_a: Arc<OnnxModel>,
    model_b: Arc<OnnxModel>,
    config: &SelfPlayConfig,
    n_games: usize,
    record_episodes: bool,
    pool: &rayon::ThreadPool,
) -> Result<banqi_4x8::pipeline::self_play::MatchResult>
where
    G: GameEnv
        + banqi_4x8::pipeline::self_play::AsDarkChessRef
        + banqi_4x8::pipeline::self_play::SeedableEnv
        + Send
        + Sync
        + Default
        + 'static,
{
    let spec_a = PlayerSpec::ModelEval(Arc::new(OnnxEvaluator::<G>::new(model_a)));
    let spec_b = PlayerSpec::ModelEval(Arc::new(OnnxEvaluator::<G>::new(model_b)));
    let result = run_match_core(MatchParams {
        player_a: &spec_a,
        player_b: &spec_b,
        n_games,
        config,
        seed: None,
        record_episodes,
        model_sims: config.mcts_sims,
        thread_pool: Some(pool),
        make_env: G::default,
    });
    if result.episodes.is_empty() && result.nnue_episodes.is_empty() && record_episodes {
        anyhow::bail!("自对弈 0 局产出（检查模型与配置）");
    }
    Ok(result)
}

/// episodes 序列化为 jsonl.gz 内存块（与 LocalEpisodeStore 行格式一致）。
fn episodes_gz(episodes: &[banqi_4x8::pipeline::self_play::GameEpisode]) -> Result<Vec<u8>> {
    use banqi_4x8::pipeline::self_play::serialize::episode_to_dict_json;
    use std::io::Write;
    let mut gz = GzEncoder::new(Vec::new(), Compression::default());
    for ep in episodes {
        writeln!(gz, "{}", episode_to_dict_json(ep)).context("序列化 episode 失败")?;
    }
    gz.finish().context("gzip 收尾失败")
}

/// 从逐局结果推导五项成对计数（换色配对：i 与 i+1 一组，均为 candidate 视角）。
/// 得分映射 负=0 / 和=1 / 胜=2，两局得分之和 0..4 对应 LL/LD/DD/DW/WW。
fn pairs_from_outcomes(outcomes: &[i32]) -> [usize; 5] {
    let score = |r: i32| match r {
        1 => 2usize,
        0 => 1,
        _ => 0,
    };
    let mut pairs = [0usize; 5];
    for chunk in outcomes.chunks(2) {
        if chunk.len() == 2 {
            pairs[score(chunk[0]) + score(chunk[1])] += 1;
        }
    }
    pairs
}
