// src/replay/describe.rs - 中文棋谱描述生成

use crate::game_env::actions::action_lookup_tables;
use crate::game_env::board::DarkChessEnv;
use crate::game_env::config::{MAX_POSITIONS, darkchess_config, GameConfig};
use crate::game_env::types::{Player, Slot};

use super::decode::decode_board_with_config;
use super::scalar::{survival_from_scalars, survival_to_dead};
use super::util::{coord_str, dead_str, format_board, piece_name, player_at};

/// 描述单个动作（中文，带坐标）。
///
/// - 翻棋：翻开(0,a)
/// - 移动/炮击（吃明子）：红马(0,a)->黑兵(1,b)
/// - 移动/炮击（走到空位或暗子）：红马(0,a)->(1,b)
fn describe_action(slots: &[Slot], action: usize, cfg: &GameConfig) -> String {
    assert!(
        action < cfg.action_space_size,
        "动作索引越界: {} (动作空间大小={})",
        action,
        cfg.action_space_size
    );
    let tables = action_lookup_tables(cfg);
    let coords = &tables.action_to_coords[action];
    if coords.len() == 1 {
        format!("翻开{}", coord_str(coords[0], cfg.cols))
    } else {
        let (from_sq, to_sq) = (coords[0], coords[1]);
        let from_piece = match slots[from_sq] {
            Slot::Revealed(p) => p,
            _ => panic!("第{}格应为已翻开的棋子（动作 {}）", from_sq, action),
        };
        let from_name = piece_name(from_piece);
        match slots[to_sq] {
            Slot::Revealed(tp) => format!(
                "{}{}->{}{}",
                from_name,
                coord_str(from_sq, cfg.cols),
                piece_name(tp),
                coord_str(to_sq, cfg.cols)
            ),
            _ => format!(
                "{}{}->{}",
                from_name,
                coord_str(from_sq, cfg.cols),
                coord_str(to_sq, cfg.cols)
            ),
        }
    }
}

/// 从对局记录解析完整的中文棋谱描述（config 驱动，支持 4x8/4x2/4x4）。
///
/// 参数对应 `GameEpisode` 序列化后的数组（见 `py::episode_to_dict`）：
/// - `boards`: 每手的棋盘观测张量（扁平）
/// - `scalars`: 每手的标量特征（长度 ≥ 3：步数/己方血量/对方血量，均为归一化）
/// - `action_masks`: 每手的合法动作掩码
/// - `actions`: 每手实际选择的动作索引
///
/// 函数内部会：
/// 1. 用 boards/scalars 逐手还原棋盘，并重建 DarkChessEnv；
/// 2. 重新生成 action_masks 与传入的逐元素比较，**断言一致**；
/// 3. 断言 actions[i] 一定在对应合法掩码内；
/// 4. 输出含坐标的中文棋谱（阵营由手数奇偶 i%2 决定）。
pub fn describe_record_with_config(
    boards: &[Vec<f32>],
    scalars: &[Vec<f32>],
    action_masks: &[Vec<i32>],
    actions: &[usize],
    cfg: &GameConfig,
) -> String {
    assert_eq!(
        boards.len(),
        actions.len(),
        "boards 与 actions 长度不一致: {} vs {}",
        boards.len(),
        actions.len()
    );
    assert_eq!(
        action_masks.len(),
        actions.len(),
        "action_masks 与 actions 长度不一致: {} vs {}",
        action_masks.len(),
        actions.len()
    );
    assert_eq!(
        scalars.len(),
        actions.len(),
        "scalars 与 actions 长度不一致: {} vs {}",
        scalars.len(),
        actions.len()
    );
    assert!(!boards.is_empty(), "记录为空，无可描述的手数");

    let mut out = String::new();
    for i in 0..boards.len() {
        let player = player_at(i);
        let slots = decode_board_with_config(&boards[i], player, cfg);

        // 每手直接用存活数推导双方已阵亡棋子（阵亡 = 总数 - 存活）
        let (red_survival, black_survival) = survival_from_scalars(&scalars[i], player, cfg);
        let red_dead_vec = survival_to_dead(&red_survival, cfg);
        let black_dead_vec = survival_to_dead(&black_survival, cfg);

        // --- 重建环境并断言 action_masks 一致 ---
        let mut board_array = [Slot::Empty; MAX_POSITIONS];
        board_array[..cfg.total_positions].copy_from_slice(&slots);
        let env = DarkChessEnv::from_board_with_config(board_array, player, *cfg);
        let rebuilt_mask = env.action_masks();
        assert_eq!(
            rebuilt_mask.len(),
            action_masks[i].len(),
            "第{}手: 重建掩码长度 {} != 记录掩码长度 {}",
            i + 1,
            rebuilt_mask.len(),
            action_masks[i].len()
        );
        let mismatch = rebuilt_mask
            .iter()
            .zip(action_masks[i].iter())
            .enumerate()
            .find(|(_, (a, b))| a != b)
            .map(|(idx, (a, b))| (idx, *a, *b));
        if let Some((idx, a, b)) = mismatch {
            panic!(
                "第{}手: 重建 action_masks 与记录不一致，首个不同动作索引 = {}（重建={}，记录={}）",
                i + 1,
                idx,
                a,
                b
            );
        }

        // --- 断言实际行动在合法掩码内 ---
        let action = actions[i];
        assert!(
            action < cfg.action_space_size,
            "第{}手: action 越界: {} (动作空间大小={})",
            i + 1,
            action,
            cfg.action_space_size
        );
        assert!(
            action_masks[i][action] == 1,
            "第{}手: 实际行动 actions[{}] = {} 不在合法掩码内（掩码值={}）",
            i + 1,
            i,
            action,
            action_masks[i][action]
        );

        // --- 血量（scalars[1]=己方HP/上限，scalars[2]=对方HP/上限，按当前行棋方换算） ---
        assert!(
            scalars[i].len() >= 3,
            "第{}手: scalars 长度不足 3: {}",
            i + 1,
            scalars[i].len()
        );
        let hp_max = cfg.initial_health as f32;
        let my_hp = (scalars[i][1] * hp_max).round() as i32;
        let opp_hp = (scalars[i][2] * hp_max).round() as i32;
        let (red_hp, black_hp) = match player {
            Player::Red => (my_hp, opp_hp),
            Player::Black => (opp_hp, my_hp),
        };

        // --- 合法行动列表（用重建掩码，已断言与记录一致） ---
        let legal: Vec<String> = rebuilt_mask
            .iter()
            .enumerate()
            .filter(|(_, m)| **m == 1)
            .map(|(a, _)| describe_action(&slots, a, cfg))
            .collect();
        let actual = describe_action(&slots, action, cfg);

        let turn_label = match player {
            Player::Red => "红方回合",
            Player::Black => "黑方回合",
        };
        let red_dead = dead_str(Player::Red, &red_dead_vec);
        let black_dead = dead_str(Player::Black, &black_dead_vec);

        out.push_str(&"=".repeat(60));
        out.push_str("\n\n");
        out.push_str(&format!("第{}手\n{}\n\n", i + 1, turn_label));
        out.push_str("棋盘（数字=行，字母=列）:\n");
        out.push_str(&format_board(&slots, cfg));
        out.push('\n');
        out.push_str(&format!("红方血量: {}\n", red_hp));
        out.push_str(&format!("红方已阵亡棋子: {}\n", red_dead));
        out.push('\n');
        out.push_str(&format!("黑方血量: {}\n", black_hp));
        out.push_str(&format!("黑方已阵亡棋子: {}\n", black_dead));
        out.push('\n');
        out.push_str("合法行动:\n");
        out.push_str(&legal.join("、"));
        out.push_str("\n\n");
        out.push_str("实际行动:\n");
        out.push_str(&actual);
        out.push('\n');
    }
    out
}

/// 4x8 默认变体入口（向后兼容）。
pub fn describe_record(
    boards: &[Vec<f32>],
    scalars: &[Vec<f32>],
    action_masks: &[Vec<i32>],
    actions: &[usize],
) -> String {
    describe_record_with_config(boards, scalars, action_masks, actions, &darkchess_config())
}
