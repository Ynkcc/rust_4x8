// src/ai/movegen.rs
// 快速走子生成器：直接从 DarkChessEnv 生成合法动作列表。
//
// 语义与 `rules.rs::get_action_masks_for_player_into` 严格一致：
//   - 翻棋：所有 Hidden 格 → 动作索引 = 格号；
//   - 常规移动：非炮棋子正交一格 → 空位 / 可吃的对方明子；
//   - 炮击：沿直线找到“炮架”（第一个棋子），越过炮架后的第一个棋子为目标
//     （目标可为暗子[机会]或对方明子[吃子]，不能是己方明子）。
// 等级规则：兵0<炮1<马2<车3<象4<士5<将6；兵克将、将怕兵；炮翻山吃任意子。
//
// 与 `action_masks` 的一致性由 `generate_moves_matches_action_masks` 单测保证。

use crate::core::env::actions::{action_lookup_tables, pack_coords};
use crate::core::env::types::{PieceType, Player, Slot};
use crate::core::env::DarkChessEnv;

/// 一条生成的走子/翻棋动作。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Move {
    /// 动作空间索引
    pub action: usize,
    /// 源格（翻棋时 from == to）
    pub from: usize,
    /// 目标格
    pub to: usize,
    /// 是否为机会动作（目标是暗子：翻棋或吃暗子）
    pub is_chance: bool,
    /// 是否为吃明子（目标是对方已翻开的明子）
    pub is_capture: bool,
    /// 纯翻棋
    pub is_flip: bool,
}

/// 判断攻击方能否吃掉防守方（暗棋等级规则）。
///
/// 等级索引（PieceType 枚举序）：兵0 炮1 马2 车3 象4 士5 将6，
/// 索引越大等级越高；特例：兵克将、将怕兵。
pub fn can_capture(attacker: PieceType, defender: PieceType) -> bool {
    if attacker == PieceType::Soldier && defender == PieceType::General {
        return true; // 兵克将
    }
    if attacker == PieceType::General && defender == PieceType::Soldier {
        return false; // 将怕兵
    }
    (attacker as usize) >= (defender as usize)
}

/// 生成 `player` 在当前局面的全部合法动作。
pub fn generate_moves(env: &DarkChessEnv, player: Player) -> Vec<Move> {
    let cfg = &env.config;
    let cols = cfg.cols as i32;
    let rows = cfg.rows as i32;
    let total = cfg.total_positions;
    let slots = env.get_board_slots();
    let lookup = action_lookup_tables(cfg);
    let mut moves = Vec::new();

    let in_bounds = |r: i32, c: i32| r >= 0 && r < rows && c >= 0 && c < cols;
    let to_sq = |r: i32, c: i32| (r * cols + c) as usize;

    // 1. 翻棋
    for sq in 0..total {
        if matches!(slots[sq], Slot::Hidden) {
            if let Some(&action) = lookup.coords_to_action.get(&pack_coords(&[sq])) {
                moves.push(Move {
                    action,
                    from: sq,
                    to: sq,
                    is_chance: true,
                    is_capture: false,
                    is_flip: true,
                });
            }
        }
    }

    // 2. 移动 / 炮击
    const ORTHO: [(i32, i32); 4] = [(1, 0), (-1, 0), (0, 1), (0, -1)];
    for from in 0..total {
        let piece = match slots[from] {
            Slot::Revealed(p) if p.player == player => p,
            _ => continue,
        };
        let r1 = (from / cfg.cols) as i32;
        let c1 = (from % cfg.cols) as i32;

        if piece.piece_type == PieceType::Cannon {
            // 炮：沿直线找“炮架”（第一个非空格），越过炮架后的第一个棋子为目标。
            for (dr, dc) in ORTHO {
                let mut sr = r1 + dr;
                let mut sc = c1 + dc;
                while in_bounds(sr, sc) && matches!(slots[to_sq(sr, sc)], Slot::Empty) {
                    sr += dr;
                    sc += dc;
                }
                if !in_bounds(sr, sc) {
                    continue; // 该方向无炮架
                }
                // 越过炮架找目标
                let mut tr = sr + dr;
                let mut tc = sc + dc;
                while in_bounds(tr, tc) && matches!(slots[to_sq(tr, tc)], Slot::Empty) {
                    tr += dr;
                    tc += dc;
                }
                if !in_bounds(tr, tc) {
                    continue; // 炮架后无目标
                }
                let to = to_sq(tr, tc);
                // 目标不能是己方明子（可为暗子[机会]或对方明子[吃子]）
                if matches!(slots[to], Slot::Revealed(p) if p.player == player) {
                    continue;
                }
                if let Some(&action) = lookup.coords_to_action.get(&pack_coords(&[from, to])) {
                    let is_chance = matches!(slots[to], Slot::Hidden);
                    moves.push(Move {
                        action,
                        from,
                        to,
                        is_chance,
                        is_capture: !is_chance,
                        is_flip: false,
                    });
                }
            }
        } else {
            // 常规移动：正交一格
            for (dr, dc) in ORTHO {
                let r2 = r1 + dr;
                let c2 = c1 + dc;
                if !in_bounds(r2, c2) {
                    continue;
                }
                let to = to_sq(r2, c2);
                match slots[to] {
                    Slot::Empty => {
                        if let Some(&action) =
                            lookup.coords_to_action.get(&pack_coords(&[from, to]))
                        {
                            moves.push(Move {
                                action,
                                from,
                                to,
                                is_chance: false,
                                is_capture: false,
                                is_flip: false,
                            });
                        }
                    }
                    Slot::Revealed(def)
                        if def.player != player
                            && can_capture(piece.piece_type, def.piece_type) =>
                    {
                        if let Some(&action) =
                            lookup.coords_to_action.get(&pack_coords(&[from, to]))
                        {
                            moves.push(Move {
                                action,
                                from,
                                to,
                                is_chance: false,
                                is_capture: true,
                                is_flip: false,
                            });
                        }
                    }
                    _ => {}
                }
            }
        }
    }
    moves
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;
    use rand::SeedableRng;

    /// 与 `action_masks_into` 的动作集合逐位一致。
    fn assert_moves_match(env: &DarkChessEnv) {
        let mut masks = vec![0i32; env.config.action_space_size];
        env.action_masks_into(&mut masks);
        let mut gen_actions = vec![0i32; env.config.action_space_size];
        for m in generate_moves(env, env.get_current_player()) {
            gen_actions[m.action] = 1;
        }
        assert_eq!(
            masks, gen_actions,
            "movegen 与 action_masks 不一致 (player={})",
            env.get_current_player()
        );
    }

    /// 走一步并验证生成动作与掩码始终一致（多种随机种子、随机对局）。
    #[test]
    fn generate_moves_matches_action_masks() {
        for seed in 1..=12u64 {
            let mut env = DarkChessEnv::new();
            env.seed = Some(seed);
            env.reset();
            let mut rng = rand::rngs::StdRng::seed_from_u64(seed.wrapping_mul(0x9E37_79B9));
            for step in 0..80 {
                assert_moves_match(&env);
                let mut masks = vec![0i32; env.config.action_space_size];
                env.action_masks_into(&mut masks);
                let legal: Vec<usize> = (0..env.config.action_space_size)
                    .filter(|&i| masks[i] == 1)
                    .collect();
                if legal.is_empty() {
                    break;
                }
                let action = legal[rng.gen_range(0..legal.len())];
                match env.step(action, None) {
                    Ok((_, _, terminated, _, _)) => {
                        if terminated {
                            break;
                        }
                    }
                    Err(e) => panic!("seed={seed} step={step}: {e}"),
                }
            }
        }
    }

    /// 元属性抽查：翻棋动作目标为 Hidden；吃子动作为对方明子；合法动作可执行。
    #[test]
    fn move_metadata_is_consistent() {
        let mut env = DarkChessEnv::new();
        env.seed = Some(7);
        env.reset();
        for m in generate_moves(&env, env.get_current_player()) {
            assert_eq!(env.is_chance_action(m.action), m.is_chance, "action={}", m.action);
            if m.is_flip {
                assert!(m.is_chance && m.from == m.to);
            }
            if m.is_capture {
                assert!(!m.is_chance);
            }
            let mut next = env;
            assert!(next.step(m.action, None).is_ok(), "非法动作 {}", m.action);
        }
    }
}
