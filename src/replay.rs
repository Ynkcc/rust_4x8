// src/replay.rs - 从对局记录还原人类可读的棋谱文字描述
//
// 设计说明：
// - 阵营由手数奇偶决定：第 i 手（0 基）i%2==0 → 红方、i%2==1 → 黑方，
//   因此无需外部手动传入己方颜色；
// - 用 boards/scalars 逐手还原棋盘 → 重建 DarkChessEnv → 重新生成 action_masks，
//   与记录中的 action_masks 逐元素断言一致；
// - 断言记录的 actions[i] 一定在合法掩码内；
// - 输出中文棋谱描述，动作带坐标，如：红马(0,a)->黑兵(1,b)。

use crate::game_env::actions::action_lookup_tables;
use crate::game_env::board::DarkChessEnv;
use crate::game_env::constants::*;
use crate::game_env::types::*;

/// 根据手数（0 基）确定当前行棋方：偶数为红方，奇数为黑方。
fn player_at(step: usize) -> Player {
    if step % 2 == 0 {
        Player::Red
    } else {
        Player::Black
    }
}

fn piece_type_from_idx(idx: usize) -> PieceType {
    match idx {
        0 => PieceType::Soldier,
        1 => PieceType::Cannon,
        2 => PieceType::Horse,
        3 => PieceType::Chariot,
        4 => PieceType::Elephant,
        5 => PieceType::Advisor,
        6 => PieceType::General,
        _ => panic!("非法棋子类型索引: {}", idx),
    }
}

fn color_prefix(player: Player) -> &'static str {
    match player {
        Player::Red => "红",
        Player::Black => "黑",
    }
}

/// 棋子的中文名称（含颜色前缀），如 红马 / 黑兵 / 红帅 / 黑将。
fn piece_name(piece: Piece) -> String {
    let name = match piece.piece_type {
        PieceType::General => match piece.player {
            Player::Red => "帅",
            Player::Black => "将",
        },
        PieceType::Cannon => "炮",
        PieceType::Horse => "马",
        PieceType::Chariot => "车",
        PieceType::Elephant => "象",
        PieceType::Advisor => "士",
        PieceType::Soldier => "兵",
    };
    format!("{}{}", color_prefix(piece.player), name)
}

/// 坐标文本：行用数字、列用字母，如 (0,a)。
fn coord_str(sq: usize) -> String {
    let r = sq / BOARD_COLS;
    let c = sq % BOARD_COLS;
    format!("({},{})", r, (b'a' + c as u8) as char)
}

/// 将观测张量解码为绝对棋盘的槽位数组。
///
/// 通道布局见 features.rs：前 7 通道为己方棋子、7..14 为敌方棋子、
/// 14 为暗子、15 为空位。"己方/敌方"以当前行棋方为视角。
pub fn decode_board(board: &[f32], current_player: Player) -> Vec<Slot> {
    assert_eq!(
        board.len(),
        BOARD_CHANNELS * TOTAL_POSITIONS,
        "board 张量长度错误: 期望 {}，实际 {}",
        BOARD_CHANNELS * TOTAL_POSITIONS,
        board.len()
    );
    let opp = current_player.opposite();
    let mut slots = Vec::with_capacity(TOTAL_POSITIONS);
    for sq in 0..TOTAL_POSITIONS {
        let hidden = board[(BOARD_CHANNELS - 2) * TOTAL_POSITIONS + sq];
        let empty = board[(BOARD_CHANNELS - 1) * TOTAL_POSITIONS + sq];
        let slot = if hidden > 0.5 {
            Slot::Hidden
        } else if empty > 0.5 {
            Slot::Empty
        } else {
            let mut piece: Option<Piece> = None;
            for pt in 0..NUM_PIECE_TYPES {
                if board[pt * TOTAL_POSITIONS + sq] > 0.5 {
                    piece = Some(Piece::new(piece_type_from_idx(pt), current_player));
                    break;
                }
            }
            if piece.is_none() {
                for pt in 0..NUM_PIECE_TYPES {
                    if board[(NUM_PIECE_TYPES + pt) * TOTAL_POSITIONS + sq] > 0.5 {
                        piece = Some(Piece::new(piece_type_from_idx(pt), opp));
                        break;
                    }
                }
            }
            match piece {
                Some(p) => Slot::Revealed(p),
                None => panic!("第 {} 格无法解码为任何棋格状态", sq),
            }
        };
        slots.push(slot);
    }
    slots
}

/// 描述单个动作（中文，带坐标）。
///
/// - 翻棋：翻开(0,a)
/// - 移动/炮击（吃明子）：红马(0,a)->黑兵(1,b)
/// - 移动/炮击（走到空位或暗子）：红马(0,a)->(1,b)
fn describe_action(slots: &[Slot], action: usize) -> String {
    assert!(
        action < ACTION_SPACE_SIZE,
        "动作索引越界: {} (动作空间大小={})",
        action,
        ACTION_SPACE_SIZE
    );
    let tables = action_lookup_tables();
    let coords = &tables.action_to_coords[action];
    if coords.len() == 1 {
        format!("翻开{}", coord_str(coords[0]))
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
                coord_str(from_sq),
                piece_name(tp),
                coord_str(to_sq)
            ),
            _ => format!("{}{}->{}", from_name, coord_str(from_sq), coord_str(to_sq)),
        }
    }
}

/// 从 scalars 的存活向量中解析出某一方的存活棋子数量。
///
/// 存活编码（见 features.rs）：对每种棋子按 `PIECE_MAX_COUNTS[pt]` 分块，
/// 每块用 `count` 个 1 + `(max - count)` 个 0 表示该种棋子存活 count 个。
/// `start` 为存活向量在 scalars 中的起始偏移。
fn parse_survival(scalars: &[f32], start: usize) -> [u8; NUM_PIECE_TYPES] {
    let mut counts = [0u8; NUM_PIECE_TYPES];
    let mut offset = start;
    for pt in 0..NUM_PIECE_TYPES {
        let max = PIECE_MAX_COUNTS[pt];
        let mut c = 0u8;
        for k in 0..max {
            if scalars[offset + k] > 0.5 {
                c += 1;
            }
        }
        counts[pt] = c;
        offset += max;
    }
    counts
}

/// 从一手的 scalars 解析出双方存活棋子数，返回 (红方存活, 黑方存活)。
///
/// scalars 布局：`[0]` 步数、`[1]` 当前行棋方 HP、`[2]` 对方 HP、
/// `[3..3+16]` 当前行棋方存活、`[3+16..]` 对方存活。
fn survival_from_scalars(scalars: &[f32], cur_player: Player) -> ([u8; NUM_PIECE_TYPES], [u8; NUM_PIECE_TYPES]) {
    let my_counts = parse_survival(scalars, 3);
    let opp_counts = parse_survival(scalars, 3 + SURVIVAL_VECTOR_SIZE);
    match cur_player {
        Player::Red => (my_counts, opp_counts),
        Player::Black => (opp_counts, my_counts),
    }
}

/// 由某一方的存活棋子数推导该方已阵亡棋子列表。
///
/// 阵亡数 = 该类棋子总数 - 存活数。存活数直接从 scalars 得出，天然包含吃暗子
/// 与炮吃己方暗子的情况（只要被吃，存活数就会减少），无需差分、无需特殊处理。
fn survival_to_dead(survival: &[u8; NUM_PIECE_TYPES]) -> Vec<PieceType> {
    let mut dead = Vec::with_capacity(TOTAL_PIECES_PER_PLAYER);
    for pt in 0..NUM_PIECE_TYPES {
        let total = PIECE_MAX_COUNTS[pt];
        let alive = survival[pt] as usize;
        for _ in 0..total.saturating_sub(alive) {
            dead.push(piece_type_from_idx(pt));
        }
    }
    dead
}

fn pad_cell(s: &str) -> String {
    // 中文字符按 2 列显示宽度计算，保证棋盘对齐
    let display_w = if s.is_ascii() { s.len() } else { s.chars().count() * 2 };
    let pad = 4usize.saturating_sub(display_w);
    format!("{}{}", s, " ".repeat(pad))
}

fn format_board(slots: &[Slot]) -> String {
    let mut s = String::new();
    s.push_str("    ");
    for c in 0..BOARD_COLS {
        s.push_str(&pad_cell(&((b'a' + c as u8) as char).to_string()));
        s.push_str("  ");
    }
    s.push('\n');
    for r in 0..BOARD_ROWS {
        s.push_str(&format!("{}   ", r));
        for c in 0..BOARD_COLS {
            let cell = match slots[r * BOARD_COLS + c] {
                Slot::Hidden => "?".to_string(),
                Slot::Empty => ".".to_string(),
                Slot::Revealed(p) => piece_name(p),
            };
            s.push_str(&pad_cell(&cell));
            s.push_str("  ");
        }
        s.push('\n');
    }
    s
}

fn dead_str(player: Player, dead_list: &[PieceType]) -> String {
    if dead_list.is_empty() {
        "（无）".to_string()
    } else {
        dead_list
            .iter()
            .map(|&pt| piece_name(Piece::new(pt, player)))
            .collect::<Vec<_>>()
            .join("、")
    }
}

/// 从对局记录解析完整的中文棋谱描述。
///
/// 参数对应 `GameEpisode` 序列化后的数组（见 `py::episode_to_dict`）：
/// - `boards`: 每手的棋盘观测张量（16 通道 × 32 格，扁平）
/// - `scalars`: 每手的标量特征（长度 ≥ 3：步数/己方血量/对方血量，均为归一化）
/// - `action_masks`: 每手的合法动作掩码
/// - `actions`: 每手实际选择的动作索引
///
/// 函数内部会：
/// 1. 用 boards/scalars 逐手还原棋盘，并重建 DarkChessEnv；
/// 2. 重新生成 action_masks 与传入的逐元素比较，**断言一致**；
/// 3. 断言 actions[i] 一定在对应合法掩码内；
/// 4. 输出含坐标的中文棋谱（阵营由手数奇偶 i%2 决定）。
pub fn describe_record(
    boards: &[Vec<f32>],
    scalars: &[Vec<f32>],
    action_masks: &[Vec<i32>],
    actions: &[usize],
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
        let slots = decode_board(&boards[i], player);

        // 每手直接用存活数推导双方已阵亡棋子（阵亡 = 总数 - 存活）
        let (red_survival, black_survival) = survival_from_scalars(&scalars[i], player);
        let red_dead_vec = survival_to_dead(&red_survival);
        let black_dead_vec = survival_to_dead(&black_survival);

        // --- 重建环境并断言 action_masks 一致 ---
        let mut board_array = [Slot::Empty; TOTAL_POSITIONS];
        board_array.copy_from_slice(&slots);
        let env = DarkChessEnv::from_board(board_array, player);
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
            action < ACTION_SPACE_SIZE,
            "第{}手: action 越界: {} (动作空间大小={})",
            i + 1,
            action,
            ACTION_SPACE_SIZE
        );
        assert!(
            action_masks[i][action] == 1,
            "第{}手: 实际行动 actions[{}] = {} 不在合法掩码内（掩码值={}）",
            i + 1,
            i,
            action,
            action_masks[i][action]
        );

        // --- 血量（scalars[1]=己方HP/60，scalars[2]=对方HP/60，按当前行棋方换算） ---
        assert!(
            scalars[i].len() >= 3,
            "第{}手: scalars 长度不足 3: {}",
            i + 1,
            scalars[i].len()
        );
        let my_hp = (scalars[i][1] * INITIAL_HEALTH_POINTS as f32).round() as i32;
        let opp_hp = (scalars[i][2] * INITIAL_HEALTH_POINTS as f32).round() as i32;
        let (red_hp, black_hp) = match player {
            Player::Red => (my_hp, opp_hp),
            Player::Black => (opp_hp, my_hp),
        };

        // --- 合法行动列表（用重建掩码，已断言与记录一致） ---
        let legal: Vec<String> = rebuilt_mask
            .iter()
            .enumerate()
            .filter(|(_, m)| **m == 1)
            .map(|(a, _)| describe_action(&slots, a))
            .collect();
        let actual = describe_action(&slots, action);

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
        out.push_str(&format_board(&slots));
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
