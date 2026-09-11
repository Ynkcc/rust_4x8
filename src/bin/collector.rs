// src/bin/collector.rs — 统一 Collector 进程（统一训练架构 L3，单机/分布式共用）
//
// 主循环（§3.1 同构）：
//   task = registry.get_task()      # local: 启动参数 + watch 模型路径
//   model = registry.ensure_model() # local: notify watch 热重载
//   episodes = self_play::run(...)  # 复用 pipeline/self_play 主干（Rust ONNX 推理，无 Python 回调）
//   episode_store.put(episodes)     # local: LocalEpisodeStore jsonl.gz 目录
//
// 分布式形态（backend=scheduler：GetTask/ReportEpisode/R2 直传）后续接入。

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result};
use clap::Parser;
use rayon::ThreadPoolBuilder;

use banqi_4x8::core::env::DarkChessEnv;
use banqi_4x8::core::env::variants::{Game4x4Env, MiniDarkChessEnv};
use banqi_4x8::core::env::traits::GameEnv;
use banqi_4x8::inference::onnx::OnnxEvaluator;
use banqi_4x8::pipeline::self_play::{PlayerSpec, SelfPlayConfig, ScenarioType, MatchParams, run_match_core};
use banqi_4x8::registry::{LocalEpisodeStore, LocalRegistry};

#[derive(Parser, Debug)]
#[command(name = "banqi-collector", about = "统一自对弈采集进程（Collector）")]
struct Args {
    /// 变体 id：4x8 / 4x4 / mini
    #[arg(long, default_value = "4x8")]
    variant: String,
    /// 协作后端：local（单机） / scheduler（分布式，暂未实现）
    #[arg(long, default_value = "local")]
    backend: String,
    /// onnx 模型路径（Registry watch 指针）
    #[arg(long, default_value = "outputs/model_latest.onnx")]
    model_path: String,
    /// episode 落盘根目录
    #[arg(long, default_value = "outputs/episodes")]
    out_dir: String,
    /// 推理设备：cpu / auto
    #[arg(long, default_value = "auto")]
    device: String,
    #[arg(long, default_value_t = 64)]
    mcts_sims: usize,
    #[arg(long, default_value_t = 16)]
    max_considered_actions: usize,
    /// 每批自对弈局数
    #[arg(long, default_value_t = 64)]
    games_per_iter: usize,
    /// 自对弈线程数（默认 = CPU 核数）
    #[arg(long, default_value_t = 0)]
    threads: usize,
    /// 总批数上限（0 = 无限循环）
    #[arg(long, default_value_t = 0)]
    iterations: usize,
    #[arg(long, default_value_t = 0)]
    worker_id: usize,
}

fn main() -> Result<()> {
    let args = Args::parse();
    if args.backend != "local" {
        anyhow::bail!("backend={:?} 暂未实现（scheduler 形态待接入），仅支持 local", args.backend);
    }

    let stop = Arc::new(AtomicBool::new(false));
    {
        let stop = Arc::clone(&stop);
        ctrlc::set_handler(move || stop.store(true, Ordering::SeqCst))
            .context("注册 Ctrl-C handler 失败")?;
    }

    let mut registry = LocalRegistry::new(&args.model_path, &args.device)?;
    let store = LocalEpisodeStore::new(&args.out_dir, args.worker_id);
    let pool = ThreadPoolBuilder::new()
        .num_threads(if args.threads > 0 { args.threads } else { num_cpus::get() })
        .build()
        .context("构建 rayon 线程池失败")?;

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

    println!("=== banqi-collector 启动 ===");
    println!("variant={} backend={} model={} out={}", args.variant, args.backend, args.model_path, args.out_dir);
    println!("mcts_sims={} games_per_iter={} threads={}", args.mcts_sims, args.games_per_iter, pool.current_num_threads());

    let mut iteration: usize = 0;
    let mut total_games: usize = 0;
    while !stop.load(Ordering::SeqCst) && (args.iterations == 0 || iteration < args.iterations) {
        let task = registry.get_task(&args.variant, args.mcts_sims, args.games_per_iter);
        let model = registry.ensure_model()?;

        let started = Instant::now();
        let n_games = run_variant_dispatch(
            &args.variant,
            model,
            &config,
            task.games_per_iter,
            &pool,
        )?;
        let duration = started.elapsed();

        let stored = store.put(&args.variant, iteration, &n_games.episodes)?;
        total_games += stored;
        println!(
            "[iter {iteration}] 🎮 {stored} 局（步均 {:.1}）→ {out} 落盘，耗时 {:.1}s",
            if n_games.episodes.is_empty() { 0.0 } else {
                n_games.episodes.iter().map(|e| e.game_length as f64).sum::<f64>() / stored.max(1) as f64
            },
            duration.as_secs_f64(),
            out = args.out_dir,
        );
        iteration += 1;
    }

    println!("=== banqi-collector 退出：累计 {total_games} 局 / {iteration} 批 ===");
    Ok(())
}

/// 按变体分发到泛型主干（run_match_core 需静态类型 G）。
fn run_variant_dispatch(
    variant: &str,
    model: Arc<banqi_4x8::inference::onnx::OnnxModel>,
    config: &SelfPlayConfig,
    n_games: usize,
    pool: &rayon::ThreadPool,
) -> Result<banqi_4x8::pipeline::self_play::MatchResult> {
    match variant {
        "4x8" => run_collector::<DarkChessEnv>(model, config, n_games, pool),
        "4x4" => run_collector::<Game4x4Env>(model, config, n_games, pool),
        "mini" => run_collector::<MiniDarkChessEnv>(model, config, n_games, pool),
        other => anyhow::bail!("未知变体: {other}（可选 4x8 / 4x4 / mini）"),
    }
}

fn run_collector<G>(
    model: Arc<banqi_4x8::inference::onnx::OnnxModel>,
    config: &SelfPlayConfig,
    n_games: usize,
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
    let evaluator: Arc<OnnxEvaluator<G>> = Arc::new(OnnxEvaluator::new(model));
    let spec = PlayerSpec::ModelEval(evaluator);
    let result = run_match_core(MatchParams {
        player_a: &spec,
        player_b: &spec,
        n_games,
        config,
        seed: None,
        record_episodes: true,
        model_sims: config.mcts_sims,
        thread_pool: Some(pool),
        make_env: G::default,
    });
    if result.episodes.is_empty() && result.nnue_episodes.is_empty() {
        anyhow::bail!("自对弈 0 局产出（检查模型与配置）");
    }
    Ok(result)
}
