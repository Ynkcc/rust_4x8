// test_observation.rs - 测试 Observation 的编码和解析

use banqi_3x4::game_env::{DarkChessEnv, Player};

fn main() {
    println!("=== 测试 Observation 编码和解析 ===\n");
    
    // 创建一个已知的测试场景
    let mut env = DarkChessEnv::new();
    env.setup_two_advisors(Player::Black);
    
    println!("设置场景: setup_two_advisors (黑方视角)");
    println!("当前玩家: {:?}", env.get_current_player());
    
    // 获取 observation
    let obs = env.get_state();
    
    // 打印原始数据
    let board_data = obs.board.as_slice().unwrap();
    let scalars_data = obs.scalars.as_slice().unwrap();
    
    println!("\n--- 原始数据 ---");
    println!("board shape: {:?}", obs.board.shape());
    println!("board.len(): {}", board_data.len());
    println!("scalars.len(): {}", scalars_data.len());
    
    // 解析并显示棋盘
    println!("\n--- 解析后的棋盘 (frame 0) ---");
    println!("编码格式: [my_soldier[0..11], my_advisor[0..11], my_general[0..11],");
    println!("           opp_soldier[0..11], opp_advisor[0..11], opp_general[0..11],");
    println!("           hidden[0..11], empty[0..11]]");
    println!();
    
    // 打印每个通道的数据
    let positions_per_channel = 12;
    let channel_names = [
        "My Soldier  ",
        "My Advisor  ",
        "My General  ",
        "Opp Soldier ",
        "Opp Advisor ",
        "Opp General ",
        "Hidden      ",
        "Empty       ",
    ];
    
    for (channel, name) in channel_names.iter().enumerate() {
        let start = channel * positions_per_channel;
        let end = start + positions_per_channel;
        print!("{}: ", name);
        for i in start..end {
            print!("{:.0} ", board_data[i]);
        }
        println!();
    }
    
    // 重建棋盘显示
    println!("\n--- 棋盘可视化 ---");
    println!("      0         1         2         3");
    println!("   +---------+---------+---------+---------+");
    
    for row in 0..3 {
        print!(" {} |", (b'A' + row as u8) as char);
        
        for col in 0..4 {
            let pos = row * 4 + col;
            let mut piece_char = "   .    ";
            
            // 检查8个通道
            for channel in 0..8 {
                let idx = channel * positions_per_channel + pos;
                if board_data[idx] > 0.5 {
                    piece_char = match channel {
                        0 => " My_Sol ",
                        1 => " My_Adv ",
                        2 => " My_Gen ",
                        3 => " Op_Sol ",
                        4 => " Op_Adv ",
                        5 => " Op_Gen ",
                        6 => "   ?    ",
                        7 => "        ", // Empty 不显示
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
    
    // 验证预期
    println!("\n--- 验证预期 ---");
    println!("期望看到:");
    println!("  - C1 位置: 红方士 (从黑方视角看是对手士) -> Op_Adv");
    println!("  - A3 位置: 黑方士 (从黑方视角看是我方士) -> My_Adv");
    
    // 检查具体位置
    let c1_pos = 2 * 4 + 1; // row=2, col=1 -> pos=9
    let a3_pos = 0 * 4 + 3; // row=0, col=3 -> pos=3
    
    println!("\n实际数据检查:");
    println!("C1 (pos={}): ", c1_pos);
    for channel in 0..8 {
        let idx = channel * positions_per_channel + c1_pos;
        if board_data[idx] > 0.5 {
            println!("  -> 通道 {} ({}): {}", channel, channel_names[channel], board_data[idx]);
        }
    }
    
    println!("A3 (pos={}): ", a3_pos);
    for channel in 0..8 {
        let idx = channel * positions_per_channel + a3_pos;
        if board_data[idx] > 0.5 {
            println!("  -> 通道 {} ({}): {}", channel, channel_names[channel], board_data[idx]);
        }
    }
}
