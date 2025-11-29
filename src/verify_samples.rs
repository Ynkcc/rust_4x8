// verify_samples.rs - 验证训练样本数据质量
//
// 从 training_samples.db 中读取样本并重建局面，用于人工判断训练数据是否正确

use anyhow::Result;
use rusqlite::Connection;
use ndarray::Array;
use banqi_3x4::game_env::Observation;

fn main() -> Result<()> {
    let db_path = "training_samples.db";
    let conn = Connection::open(db_path)?;
    
    println!("从 {} 读取训练样本...\n", db_path);
    
    // 查询距离终局正好4步的样本（game_length - step_in_game = 4）
    let mut stmt = conn.prepare(
        "SELECT board_state, scalar_state, policy_probs, value_target, action_mask, 
                game_length, step_in_game, iteration, episode_type
         FROM training_samples 
         WHERE (game_length - step_in_game) = 2 AND value_target=1
         ORDER BY RANDOM()
         LIMIT 10"
    )?;
    
    let samples = stmt.query_map([], |row| {
        let board_bytes: Vec<u8> = row.get(0)?;
        let scalar_bytes: Vec<u8> = row.get(1)?;
        let probs_bytes: Vec<u8> = row.get(2)?;
        let value: f32 = row.get(3)?;
        let mask_bytes: Vec<u8> = row.get(4)?;
        let game_length: i64 = row.get(5)?;
        let step_in_game: i64 = row.get(6)?;
        let iteration: i64 = row.get(7)?;
        let episode_type: String = row.get(8)?;
        
        let board_data: Vec<f32> = board_bytes.chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();
        
        let scalar_data: Vec<f32> = scalar_bytes.chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();
        
        let probs: Vec<f32> = probs_bytes.chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();
        
        let mask: Vec<i32> = mask_bytes.chunks_exact(4)
            .map(|chunk| i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();
        
        // 调试：打印实际数据长度
        eprintln!("🐛 DEBUG: board_data.len()={}, scalar_data.len()={}, probs.len()={}, mask.len()={}", 
            board_data.len(), scalar_data.len(), probs.len(), mask.len());
        
        // 根据实际长度推断形状
        // 期望: board = 2*8*3*4 = 192 或 1*8*3*4 = 96 (如果禁用了状态堆叠)
        let board = if board_data.len() == 192 {
            Array::from_shape_vec((2, 8, 3, 4), board_data)
                .expect("Failed to reshape board data (2,8,3,4)")
        } else if board_data.len() == 96 {
            // 如果是单帧，添加一个维度以保持一致
            let mut padded = vec![0.0f32; 192];
            padded[..96].copy_from_slice(&board_data);
            Array::from_shape_vec((2, 8, 3, 4), padded)
                .expect("Failed to reshape board data (1,8,3,4)")
        } else {
            panic!("Unexpected board_data length: {}", board_data.len());
        };
        
        let scalars = Array::from_vec(scalar_data);
        
        let obs = Observation { board, scalars };
        
        Ok((obs, probs, value, mask, game_length, step_in_game, iteration, episode_type))
    })?;
    
    let mut count = 0;
    for sample in samples {
        let (obs, probs, value, mask, game_length, step_in_game, iteration, episode_type) = sample?;
        count += 1;
        
        println!("========== 样本 #{} ==========", count);
        println!("来源: Iteration {}, 类型: {}", iteration, episode_type);
        println!("游戏长度: {} 步, 当前步: {}, 距离终局: {} 步", 
            game_length, step_in_game, game_length - step_in_game);
        println!("价值标签: {:.3}", value);
        println!();
        
        // 尝试重建局面 - 从 observation 中提取信息
        print_observation(&obs);
        
        // 打印策略分布（top 10）
        let mut indexed_probs: Vec<(usize, f32)> = probs.iter()
            .enumerate()
            .map(|(i, &p)| (i, p))
            .collect();
        indexed_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        println!("\n策略分布 (Top 10):");
        for (i, (action, prob)) in indexed_probs.iter().take(10).enumerate() {
            let action_desc = get_action_description(*action);
            println!("  #{}: action {:2} {} prob={:.4}", 
                i + 1, action, action_desc, prob);
        }
        
        // 打印有效动作
        let valid_actions: Vec<usize> = mask.iter()
            .enumerate()
            .filter_map(|(i, &m)| if m == 1 { Some(i) } else { None })
            .collect();
        println!("\n有效动作数: {} / 46", valid_actions.len());
        
        // 检查策略是否集中
        let max_prob = probs.iter().cloned().fold(0.0f32, f32::max);
        let entropy: f32 = probs.iter()
            .filter(|&&p| p > 1e-8)
            .map(|&p| -p * p.ln())
            .sum();
        println!("策略质量: max_prob={:.3}, entropy={:.3}", max_prob, entropy);
        
        println!("\n");
    }
    
    println!("共重建 {} 个样本", count);
    
    Ok(())
}

/// 打印 Observation 的关键信息
fn print_observation(obs: &Observation) {
    // Board 形状: (2, 8, 3, 4) 但实际存储时被展平
    // 编码格式（来自 game_env.rs get_board_state_tensor）:
    // - 外层循环：棋子类型 (0=Soldier, 1=Advisor, 2=General)
    // - 内层循环：位置 (0..11)
    // 数据布局: [my_soldier[0..11], my_advisor[0..11], my_general[0..11], 
    //            opp_soldier[0..11], opp_advisor[0..11], opp_general[0..11],
    //            hidden[0..11], empty[0..11]]
    
    println!("棋盘状态 (当前帧):");
    
    // 直接访问底层数据
    let board_data = obs.board.as_slice().unwrap();
    
    println!("      0         1         2         3");
    println!("   +---------+---------+---------+---------+");
    
    for row in 0..3 {
        print!(" {} |", (b'A' + row as u8) as char);
        
        for col in 0..4 {
            let pos = row * 4 + col; // 线性位置 (0..11)
            let mut piece_char = "   .    ";
            
            // 检查当前帧 (frame=0) 的各个通道
            // frame0 数据从 0..96
            let frame0_start = 0;
            let positions_per_channel = 12;
            
            // 8个通道: 3(my) + 3(opp) + 1(hidden) + 1(empty)
            for channel in 0..8 {
                let channel_start = frame0_start + channel * positions_per_channel;
                let idx = channel_start + pos;
                
                if idx < board_data.len() && board_data[idx] > 0.5 {
                    piece_char = match channel {
                        // 我方棋子 (从当前玩家视角)
                        0 => " My_Sol ",
                        1 => " My_Adv ",
                        2 => " My_Gen ",
                        // 对手棋子
                        3 => " Op_Sol ",
                        4 => " Op_Adv ",
                        5 => " Op_Gen ",
                        // 特殊位置
                        6 => "   ?    ", // Hidden
                        7 => "   .    ", // Empty (实际上空位应该不显示)
                        _ => "   ??   ",
                    };
                    break;
                }
            }
            
            print!(" {}", piece_char);
        }
        
        println!(" |");
        println!("   +---------+---------+---------+---------+");
    }
    
    // 从 scalars 中提取信息
    println!("\n标量特征摘要:");
    if obs.scalars.len() >= 56 {
        println!("  特征向量长度: {}", obs.scalars.len());
        let scalars_data = obs.scalars.as_slice().unwrap();
        println!("  前10个特征: {:?}", &scalars_data[..10.min(scalars_data.len())]);
    }
}

/// 获取动作描述
fn get_action_description(action: usize) -> String {
    const REVEAL_ACTIONS_COUNT: usize = 12;
    
    if action < REVEAL_ACTIONS_COUNT {
        // 翻棋动作
        let pos = action;
        let row = pos / 4;
        let col = pos % 4;
        format!("(reveal@{:2} [{},{}])", pos, (b'A' + row as u8) as char, col)
    } else {
        // 移动动作：需要重建 from->to 的映射
        // 按照 game_env.rs 的 initialize_lookup_tables 逻辑重建
        let mut idx = REVEAL_ACTIONS_COUNT;
        let moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]; // 上、下、左、右
        let dir_names = ["上", "下", "左", "右"];
        
        for r1 in 0..3 {
            for c1 in 0..4 {
                let _from_sq = r1 * 4 + c1;
                for (dir_idx, (dr, dc)) in moves.iter().enumerate() {
                    let r2 = r1 as i32 + dr;
                    let c2 = c1 as i32 + dc;
                    
                    if r2 >= 0 && r2 < 3 && c2 >= 0 && c2 < 4 {
                        if idx == action {
                            let _to_sq = (r2 as usize) * 4 + (c2 as usize);
                            return format!("(move [{},{}]->{} to [{},{}])", 
                                (b'A' + r1 as u8) as char, c1, dir_names[dir_idx],
                                (b'A' + r2 as u8) as char, c2);
                        }
                        idx += 1;
                    }
                }
            }
        }
        
        format!("(unknown action {})", action)
    }
}
