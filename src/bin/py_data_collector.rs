// src/py_data_collector.rs
//
// 独立二进制 (banqi-py-collector)：使用外部 Python 预测器 (远程 Torch 模型)
// 生成自对弈数据，并以 JSONL 格式保存到本地磁盘。
//
// 数据流：
//   1. 在独立线程运行 Python 预测器 (--predictor 指定模块:函数)
//   2. Rust 侧用 Gumbel MCTS 驱动自对弈，需要时通过 PyO3 调用 Python 评估
//   3. 每完成一局就序列化为 dict 并写入 JSONL (--output)
//
// 子模块：
//   - memory_estimator: 内存占用估算（独立文件）

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result, anyhow};
use pyo3::prelude::*;

use banqi_4x8::core::env::types::Player;
use banqi_4x8::core::env::DarkChessEnv;
use banqi_4x8::engine::movegen::generate_moves;
use banqi_4x8::bridge::python::episode_to_dict_darkchess;
use banqi_4x8::bridge::python::py_evaluator::PyEvaluator;
use banqi_4x8::pipeline::self_play::{SelfPlayConfig, run_self_play};
use banqi_4x8::utils::memory_estimator;

/// 加载 Python 预测器函数 (PyEvaluator)。
fn load_python_predictor(py: Python, spec: &str) -> Result<Py<PyAny>> {
    let parts: Vec<&str> = spec.splitn(2, ':').collect();
    if parts.len() != 2 {
        return Err(anyhow!(
            "预测器规格应为 'module:function'，收到: {}",
            spec
        ));
    }
    let (module_name, func_name) = (parts[0], parts[1]);
    let func = py.import(module_name)?.getattr(func_name)?;
    Ok(func.into())
}

/// 加载 Python 数据保存器函数 (可选)。
fn load_python_saver(py: Python, spec: &str) -> Option<Py<PyAny>> {
    if spec.is_empty() {
        return None;
    }
    match py.import("pathlib") {
        Ok(_) => py
            .import(spec.split(':').next().unwrap_or(""))
            .ok()
            .and_then(|m| m.getattr(spec.split(':').nth(1).unwrap_or("")).ok())
            .map(|f| f.into()),
        Err(_) => None,
    }
}

/// 将自对弈 episode 构建为可被 JSON 序列化的 dict（兼容 Python 端 load_data）。
fn build_episode_dict(py: Python, ep: &banqi_4x8::pipeline::self_play::GameEpisode) -> PyResult<Py<pyo3::types::PyDict>> {
    episode_to_dict_darkchess(py, ep).map(|b| b.unbind())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let mut predictor_spec = "predictor:predict".to_string();
    let mut output_path = "self_play_data.jsonl".to_string();
    let mut games = 1usize;
    let mut mcts_sims = 64usize;
    let mut max_considered_actions = 16usize;
    let mut temperature_steps = 12usize;
    let mut concurrency = 1usize;
    let mut verbose = false;
    let mut estimate_memory = false;
    let mut memory_report = false;
    let mut show_help = false;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--predictor" => {
                i += 1;
                predictor_spec = args
                    .get(i)
                    .context("缺少 --predictor 参数值")?
                    .clone();
            }
            "--output" => {
                i += 1;
                output_path = args.get(i).context("缺少 --output 参数值")?.clone();
            }
            "--games" => {
                i += 1;
                games = args
                    .get(i)
                    .context("缺少 --games 参数值")?
                    .parse()
                    .context("无效的 games 数值")?;
            }
            "--mcts-sims" => {
                i += 1;
                mcts_sims = args
                    .get(i)
                    .context("缺少 --mcts-sims 参数值")?
                    .parse()
                    .context("无效的 mcts-sims 数值")?;
            }
            "--max-considered-actions" => {
                i += 1;
                max_considered_actions = args
                    .get(i)
                    .context("缺少 --max-considered-actions 参数值")?
                    .parse()
                    .context("无效的 max-considered-actions 数值")?;
            }
            "--temperature-steps" => {
                i += 1;
                temperature_steps = args
                    .get(i)
                    .context("缺少 --temperature-steps 参数值")?
                    .parse()
                    .context("无效的 temperature-steps 数值")?;
            }
            "--concurrency" => {
                i += 1;
                concurrency = args
                    .get(i)
                    .context("缺少 --concurrency 参数值")?
                    .parse()
                    .context("无效的 concurrency 数值")?;
            }
            "--verbose" => verbose = true,
            "--estimate-memory" => estimate_memory = true,
            "--memory-report" => memory_report = true,
            "--help" | "-h" => show_help = true,
            other => eprintln!("⚠️  未知参数: {}", other),
        }
        i += 1;
    }

    if show_help {
        println!(
            "用法: banqi-py-collector [选项]\n\
             \n选项:\n\
             \t--predictor <module:func>   远程 Python 预测器 (默认 predictor:predict)\n\
             \t--output <path>             JSONL 输出路径 (默认 self_play_data.jsonl)\n\
             \t--games <n>                 生成游戏局数\n\
             \t--mcts-sims <n>             MCTS 模拟次数\n\
             \t--max-considered-actions <n> Gumbel 最大考虑动作数\n\
             \t--temperature-steps <n>     温度下降步数\n\
             \t--concurrency <n>           并行游戏数\n\
             \t--verbose                   输出单步棋谱\n\
             \t--estimate-memory           估算单局挂起内存\n\
             \t--memory-report             打印完整内存报告\n\
             \t--help, -h                  显示帮助"
        );
        return Ok(());
    }

    if memory_report {
        memory_estimator::print_full_memory_report(mcts_sims, concurrency.max(1));
        return Ok(());
    }

    println!("🚀 启动暗棋自对弈数据收集器");
    println!("   预测器: {}", predictor_spec);
    println!("   输出:   {}", output_path);
    println!("   游戏数: {}", games);
    println!("   MCTS:   sims={}, max_considered={}", mcts_sims, max_considered_actions);

    // 初始化 Python 解释器 (在独立线程中运行预测器)
    let predict_result: Result<(Py<PyAny>, Option<Py<PyAny>>)> = Python::attach(|py| {
        let predict_fn = load_python_predictor(py, &predictor_spec)
            .map_err(|e| anyhow!("加载预测器失败: {}", e))?;
        let saver_fn = load_python_saver(py, "");
        Ok((predict_fn, saver_fn))
    });

    let (predict_fn, _saver_fn) = predict_result?;

    let evaluator = PyEvaluator::new(predict_fn);
    let cfg = SelfPlayConfig {
        mcts_sims,
        max_considered_actions,
        temperature_steps,
        scenario: banqi_4x8::pipeline::self_play::ScenarioType::Standard,
        c_scale: 1.0,
        gumbel_scale: 1.0,
    };

    let running = Arc::new(AtomicBool::new(true));
    let r = running.clone();
    ctrlc::set_handler(move || {
        println!("\n⚠️  收到中断信号，正在优雅退出...");
        r.store(false, Ordering::SeqCst);
    })
    .ok();

    let start_time = Instant::now();
    let mut completed = 0usize;
    let mut file = std::fs::File::create(&output_path)
        .with_context(|| format!("无法创建输出文件: {}", output_path))?;

    while completed < games {
        if !running.load(Ordering::SeqCst) {
            eprintln!("🛑 中断：已完成 {} 局", completed);
            break;
        }

        let episode_start = Instant::now();
        let episode = run_self_play(&evaluator, &cfg, DarkChessEnv::new);
        let episode_duration = episode_start.elapsed();

        if episode.samples.is_empty() {
            eprintln!("⚠️  生成了空游戏数据，跳过");
            continue;
        }

        // 序列化为 dict 并写入 JSONL
        let dict_json = Python::attach(|py| -> Result<String> {
            let dict = build_episode_dict(py, &episode)?;
            let json_str = py
                .import("json")?
                .getattr("dumps")?
                .call1((dict,))?;
            let s: String = json_str.extract()?;
            Ok(s)
        })?;

        use std::io::Write;
        writeln!(file, "{}", dict_json)?;

        completed += 1;
        let winner = match episode.winner {
            Some(1) => "红胜",
            Some(-1) => "黑胜",
            _ => "平局",
        };
        println!(
            "✅ Game #{} 完成: 步数={}, 结果={}, 耗时={:.1}s (累计 {:.1}s)",
            completed,
            episode.game_length,
            winner,
            episode_duration.as_secs_f64(),
            start_time.elapsed().as_secs_f64()
        );

        if verbose {
            println!("   棋谱 (示例):");
            let mut env = DarkChessEnv::new();
            env.reset();
            for (idx, (_, policy, _, _, _, _, mask, action, _)) in episode.samples.iter().enumerate() {
                let moves = generate_moves(&env, env.get_current_player());
                if let Some(mv) = moves.iter().find(|m| m.action == *action) {
                    println!(
                        "     第{}手 [{}]: {:?} (policy_top={:.3})",
                        idx + 1,
                        if env.get_current_player() == Player::Red { "红" } else { "黑" },
                        mv,
                        policy.iter().cloned().fold(0.0_f32, f32::max)
                    );
                }
                let _ = mask;
                if env.step(*action, None).is_ok() {
                    let (term, _, _) = env.check_game_over_conditions();
                    if term {
                        break;
                    }
                }
            }
        }

        if estimate_memory {
            let single = memory_estimator::estimate_single_game_suspended(
                mcts_sims,
                episode.game_length / 2,
                episode.game_length,
            );
            println!(
                "   📊 本局训练数据内存 ≈ {:.1} MB ({} 步样本)",
                single.total_mb, episode.game_length
            );
        }
    }

    println!(
        "\n🏁 完成！共生成 {} 局，保存到: {}\n   总耗时: {:.1}s",
        completed,
        output_path,
        start_time.elapsed().as_secs_f64()
    );

    Ok(())
}
