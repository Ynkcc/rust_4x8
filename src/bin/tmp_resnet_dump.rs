// tmp_resnet_dump.rs - ResNet 神经网络输入特征人工验证工具
//
// 在 4x4 变体环境中随机行动，每手将神经网络输入特征（棋盘张量 + 标量向量）
// 解码为人类可读表述写入文件，供人工核对特征编码与实际局面是否一致。
//
// 用法: cargo run --bin tmp_resnet_dump [输出文件路径] [局数]
// 默认输出: outputs/resnet_decode_4x4.txt

use banqi_4x8::core::env::config::{GameConfig, game_4x4_config};
use banqi_4x8::core::env::types::{Piece, PieceType, Player};
use banqi_4x8::core::env::variants::game4x4::Game4x4Env;
use banqi_4x8::pipeline::replay::{
    decode_board_with_config, decode_scalar_state, format_board, format_scalar_state, piece_name,
};
use rand::seq::SliceRandom;
use rand::thread_rng;

fn main() {
    let out_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "outputs/resnet_decode_4x4.txt".to_string());
    let games: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);

    let cfg = game_4x4_config();
    let mut rng = thread_rng();
    let mut report = String::new();

    report.push_str(&format!(
        "4x4 变体 ResNet 输入特征解码验证（{} 局随机对局）\n\
         棋盘通道布局: [0..{}] 己方明子 | [{}..{}] 对方明子 | [{}] 暗子 | [{}] 空位\n\
         标量布局: [0]步数/判和上限 [1]己HP [2]敌HP | 存活向量×2 | 暗子向量×2（按当前行棋方视角）\n\n",
        games, cfg.num_active, cfg.num_active, 2 * cfg.num_active,
        2 * cfg.num_active, 2 * cfg.num_active + 1,
    ));

    for game in 0..games {
        let mut env = Game4x4Env::new();
        let mut masks = vec![0i32; cfg.action_space_size];
        let mut step = 0usize;

        report.push_str(&format!("# ===== 第 {} 局 =====\n", game + 1));

        loop {
            let obs = env.get_resnet_state();
            let board_flat: Vec<f32> = obs.board.iter().copied().collect();
            let scalars: Vec<f32> = obs.scalars.iter().copied().collect();
            let player = env.get_current_player();

            // 维度一致性自检：实际编码长度 vs config 声明
            if scalars.len() != cfg.resnet_scalar_feature_count {
                report.push_str(&format!(
                    "[警告] 标量长度不一致: 实际编码 {} vs config.resnet_scalar_feature_count {}\n",
                    scalars.len(),
                    cfg.resnet_scalar_feature_count
                ));
            }

            let slots = decode_board_with_config(&board_flat, player, &cfg);
            let scalar_result = decode_scalar_state(&scalars, &cfg);

            report.push_str(&format!("\n--- 第 {} 局 第 {} 手（{} 行棋）---\n", game + 1, step + 1, player));
            report.push_str("解码棋盘（来自 NN 输入棋盘张量）:\n");
            report.push_str(&format_board(&slots, &cfg));
            report.push_str(&format!("解码标量（来自 NN 输入标量向量）: {}\n", format_scalar_state(&scalars, &cfg, player)));
            report.push_str(&format!(
                "暗子向量明细 | 己方暗子: {} | 对方暗子: {}\n",
                fmt_counts(&scalar_result.my_hidden, player, &cfg),
                fmt_counts(&scalar_result.opp_hidden, player.opposite(), &cfg),
            ));

            env.action_masks_into(&mut masks);
            let valid: Vec<usize> = masks
                .iter()
                .enumerate()
                .filter_map(|(i, &v)| (v == 1).then_some(i))
                .collect();
            if valid.is_empty() {
                report.push_str("无合法动作，结束。\n");
                break;
            }
            let action = *valid.choose(&mut rng).unwrap();
            report.push_str(&format!("随机动作: {}\n", action));

            match env.step(action) {
                Ok((_reward, terminated, truncated, winner)) => {
                    step += 1;
                    if terminated || truncated {
                        report.push_str(&format!(
                            "对局结束（共 {} 手），winner = {:?}\n",
                            step, winner
                        ));
                        break;
                    }
                }
                Err(e) => panic!("step 失败: {}", e),
            }
            if step >= cfg.max_steps_per_episode {
                break;
            }
        }
        report.push('\n');
    }

    if let Some(parent) = std::path::Path::new(&out_path).parent() {
        std::fs::create_dir_all(parent).expect("创建输出目录失败");
    }
    std::fs::write(&out_path, &report).expect("写出报告失败");
    println!("已写入 {}", out_path);
}

/// 按棋子类型将计数向量格式化为 "红兵x2 红炮x1" 形式。
fn fmt_counts(counts: &[u8], player: Player, cfg: &GameConfig) -> String {
    let mut parts = Vec::new();
    for (ci, &pt) in cfg.active_types.iter().enumerate().take(cfg.num_active) {
        if counts[ci] > 0 {
            parts.push(format!(
                "{}x{}",
                piece_name(Piece::new(PieceType::from_index(pt), player)),
                counts[ci]
            ));
        }
    }
    if parts.is_empty() {
        "无".to_string()
    } else {
        parts.join(" ")
    }
}
