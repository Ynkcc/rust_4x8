use banqi_3x4::*;
use rand::seq::SliceRandom;
use rand::thread_rng;

fn main() {
    let mut env = DarkChessEnv::new();
    let mut rng = thread_rng();

    println!("=== Rust 迷你暗棋 (3x4) - 随机对局演示 ===\n");
    println!("开始执行随机策略测试...\n");

    let obs = env.reset();
    println!("游戏重置完成");
    println!("初始棋盘状态张量形状: {:?}", obs.board.shape());
    println!("初始标量状态形状: {:?}\n", obs.scalars.shape());

    let mut step_count = 0;

    loop {
        env.print_board();

        let masks = env.action_masks();
        let valid_actions: Vec<usize> = masks
            .iter()
            .enumerate()
            .filter_map(|(idx, &val)| if val == 1 { Some(idx) } else { None })
            .collect();

        if valid_actions.is_empty() {
            println!("当前玩家无棋可走，游戏结束。");
            break;
        }

        let action = *valid_actions.choose(&mut rng).unwrap();
        
        let action_desc = if action < REVEAL_ACTIONS_COUNT {
            let sq = action;
            format!("翻棋 (位置 {})", sq)
        } else {
            format!("移动/吃子 (动作 {})", action)
        };

        println!("Step {}: 执行动作 {} -> {}\n", step_count, action, action_desc);

        match env.step(action, None) {
            Ok((_obs, _reward, terminated, truncated, winner)) => {
                step_count += 1;

                if terminated || truncated {
                    env.print_board();
                    println!("\n=== 游戏结束 ===");
                    println!("总步数: {}", step_count);
                    
                    if let Some(w) = winner {
                        match w {
                            0 => println!("结果: 和棋"),
                            1 => println!("结果: 红方获胜"),
                            -1 => println!("结果: 黑方获胜"),
                            _ => println!("结果: 未知代码 {}", w),
                        }
                    }
                    break;
                }
            },
            Err(e) => {
                panic!("Step 执行逻辑错误: {}", e);
            }
        }

        // 限制最大步数以防无限循环
        if step_count >= 200 {
            println!("\n达到最大步数限制，游戏结束。");
            break;
        }
    }
    
    println!("\n验证结束。");
}
