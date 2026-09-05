// src/game_env/board/tests.rs
// DarkChessEnv 单元测试。

use super::*;

/// 随机走子对局，持续检查每个观测的 bitboard 一致性。
#[test]
fn random_game_keeps_board_consistent() {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let max_games = 200;
    for game in 0..max_games {
        let mut env = DarkChessEnv::new();
        let mut steps = 0;
        let mut action_history: Vec<usize> = Vec::new();
        let mut consistent = true;
        let mut fail_step = 0;
        let mut fail_active = 0usize;
        let mut fail_sq = 0usize;
        let mut last_from_slot: Option<Slot> = None;
        let mut last_to_slot: Option<Slot> = None;
        while steps < 200 {
            let masks = env.action_masks();
            let legal: Vec<usize> = masks
                .iter()
                .enumerate()
                .filter(|&(_, &m)| m == 1)
                .map(|(i, _)| i)
                .collect();
            if legal.is_empty() {
                break;
            }
            let action = legal[rng.gen_range(0..legal.len())];
            let coords = action_lookup_tables(&env.config).action_to_coords[action].clone();
            last_to_slot = if coords.len() == 2 {
                Some(env.board[coords[1]])
            } else {
                None
            };
            last_from_slot = if coords.len() == 2 {
                Some(env.board[coords[0]])
            } else {
                None
            };
            match env.step(action, None) {
                Ok((_, terminated, truncated, _)) => {
                    action_history.push(action);
                    let b = env.get_resnet_state().board.as_slice().unwrap().to_vec();
                    for sq in 0..env.config.total_positions {
                        let active = (0..env.config.resnet_board_channels)
                            .filter(|&pt| b[pt * env.config.total_positions + sq] > 0.5)
                            .count();
                        if active != 1 {
                            consistent = false;
                            fail_step = steps + 1;
                            fail_active = active;
                            fail_sq = sq;
                            break;
                        }
                    }
                    if !consistent {
                        break;
                    }
                    if terminated || truncated {
                        break;
                    }
                    steps += 1;
                }
                Err(e) => {
                    panic!("第{steps}步: env.step 返回 Err: {e}");
                }
            }
        }
        if !consistent {
            let coords: Vec<String> = action_history
                .iter()
                .map(|&a| {
                    let t = action_lookup_tables(&env.config);
                    match t.action_to_coords[a].len() {
                        1 => format!("翻({})", t.action_to_coords[a][0]),
                        _ => format!(
                            "({}->{})",
                            t.action_to_coords[a][0], t.action_to_coords[a][1]
                        ),
                    }
                })
                .collect();
            panic!(
                "第{game}局 第{fail_step}步: 第{fail_sq}格 归属通道数={fail_active} (应为1)\n\
                 最后动作 from_slot={:?} to_slot={:?}\n动作序列: {coords:?}",
                last_from_slot, last_to_slot
            );
        }
    }
}
