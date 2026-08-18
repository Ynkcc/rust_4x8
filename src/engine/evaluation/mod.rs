// src/ai/eval.rs
// 多特征启发式评估（纯计算）——参照 misty-banqi 引擎的评估体系，适配本项目 DarkChessEnv。
//
// 评估维度：
//   - 物质价值（明子，校正价值表） + 覆盖物质（暗子按公共袋价值计） → 全存活物质：
//     翻棋不改变物质总量，只有吃子才改变，消除“翻出己方好子即虚增”的幻象；
//   - 将帅情境价值（king_ctx）：敌兵存活越少，将越难被杀，将价值动态上升；
//   - 自适应支配价值（domination）：明子的支配者（能吃它的存活敌子）越少越接近“不朽”；
//   - 价值加权机动性（vmob）：每个合法走子按其棋子的价值加权；
//   - 将帅危险度（general danger）：邻接敌兵 + 空逃路线 + 角落围困（近程+逃跑感知）。
//
// 价值表（校正表，仅用于搜索评估，不影响吃子扣血规则）：
//   参照 misty-banqi `VALUE = [将30, 士10, 象10, 车14, 马8, 炮12, 兵4]`，
//   按本项目 PieceType 索引（兵0 炮1 马2 车3 象4 士5 将6）映射。

use crate::core::env::types::{PieceType, Player, Slot};
use crate::core::env::DarkChessEnv;

use super::movegen::{can_capture, generate_moves};

/// 校正价值表（索引 = PieceType：兵0 炮1 马2 车3 象4 士5 将6）。
pub const CORRECTED_VALUES: [f32; 7] = [4.0, 12.0, 8.0, 14.0, 10.0, 10.0, 30.0];

/// 评估归一化尺度（与初始血量 60 同量纲）。
pub const EVAL_SCALE: f32 = 60.0;

/// 评估参数。
#[derive(Clone, Copy, Debug)]
pub struct EvalParams {
    /// 搜索评估专用价值表（索引 = PieceType）
    pub values: [f32; 7],
    /// 机动性权重
    pub w_mob: f32,
    /// 将帅安全权重
    pub w_king: f32,
    /// 支配价值强度（自适应吃子权）
    pub dom_k: f32,
}

impl Default for EvalParams {
    fn default() -> Self {
        Self {
            values: CORRECTED_VALUES,
            w_mob: 0.8,
            w_king: 0.6,
            dom_k: 0.5,
        }
    }
}

/// 从棋盘点位统计双方已翻开的明子数 [player_idx][type_idx]。
fn revealed_counts(env: &DarkChessEnv) -> [[u32; 7]; 2] {
    let mut revealed = [[0u32; 7]; 2];
    for slot in env.get_board_slots() {
        if let Slot::Revealed(p) = slot {
            revealed[p.player.idx()][p.piece_type as usize] += 1;
        }
    }
    revealed
}

/// 双方剩余暗子数 [player_idx][type_idx]（公共信息：总数 - 已明 - 已死，双方都可推导）。
fn hidden_counts(env: &DarkChessEnv) -> [[u32; 7]; 2] {
    let cfg = &env.config;
    let revealed = revealed_counts(env);
    let mut dead = [[0u32; 7]; 2];
    for &p in &[Player::Red, Player::Black] {
        for &t in env.get_dead_pieces(p) {
            dead[p.idx()][t as usize] += 1;
        }
    }
    let mut hidden = [[0u32; 7]; 2];
    for pl in 0..2 {
        for t in 0..7 {
            hidden[pl][t] = (cfg.piece_counts[t] as u32)
                .saturating_sub(revealed[pl][t])
                .saturating_sub(dead[pl][t]);
        }
    }
    hidden
}

/// 双方存活子数 [player_idx][type_idx] = 明子 + 暗子（= 总数 - 已死）。
fn alive_counts(env: &DarkChessEnv) -> [[u32; 7]; 2] {
    let mut alive = revealed_counts(env);
    let hidden = hidden_counts(env);
    for pl in 0..2 {
        for t in 0..7 {
            alive[pl][t] += hidden[pl][t];
        }
    }
    alive
}

/// 明子物质差（persp 视角，+为 persp 领先）。
fn material(env: &DarkChessEnv, persp: Player, values: &[f32; 7]) -> f32 {
    let mut m = 0.0;
    for slot in env.get_board_slots() {
        if let Slot::Revealed(p) = slot {
            let v = values[p.piece_type as usize];
            m += if p.player == persp { v } else { -v };
        }
    }
    m
}

/// 覆盖物质（暗子按公共袋价值计），使翻棋不改变总物质。
fn covered_material(hidden: &[[u32; 7]; 2], persp: usize, values: &[f32; 7]) -> f32 {
    let opp = 1 - persp;
    let mut m = 0.0;
    for t in 0..7 {
        m += hidden[persp][t] as f32 * values[t];
        m -= hidden[opp][t] as f32 * values[t];
    }
    m
}

/// 将帅情境价值：将帅的额外价值随“对方存活兵数”减少而上升
/// （只有兵——或炮架——能杀将，敌兵耗尽时将几乎不可杀）。
fn king_ctx_term(alive: &[[u32; 7]; 2], persp: usize) -> f32 {
    // 敌兵存活数索引：{0..=5} → 加分（将价值 30 的 0.47..0.0 比例）。
    const KING_CTX: [f32; 6] = [14.1, 4.1, 0.8, 0.2, 0.0, 0.0];
    let me = persp;
    let opp = 1 - persp;
    let sol = [
        alive[me][PieceType::Soldier as usize],
        alive[opp][PieceType::Soldier as usize],
    ];
    let gen_alive = [
        alive[me][PieceType::General as usize] > 0,
        alive[opp][PieceType::General as usize] > 0,
    ];
    let mut t = 0.0;
    if gen_alive[0] {
        t += KING_CTX[(sol[1] as usize).min(5)]; // 我方将 vs 敌兵
    }
    if gen_alive[1] {
        t -= KING_CTX[(sol[0] as usize).min(5)]; // 敌将 vs 我兵
    }
    t
}

/// 自适应支配价值：明子按“存活敌人中能吃掉它的数量”折价，
/// 支配者越少越接近“不朽”；炮总能翻山吃任意子，恒为支配者。
fn domination_value(
    env: &DarkChessEnv,
    alive: &[[u32; 7]; 2],
    persp: usize,
    values: &[f32; 7],
    dom_k: f32,
) -> f32 {
    let mut total = 0.0;
    for slot in env.get_board_slots() {
        if let Slot::Revealed(p) = slot {
            let r = p.piece_type as usize;
            let mine = p.player.idx() == persp;
            let enemy = if mine { 1 - persp } else { persp };
            let mut dom = 0u32;
            for d in 0..7 {
                if d == PieceType::Cannon as usize
                    || can_capture(PieceType::from_index(d), p.piece_type)
                {
                    dom += alive[enemy][d];
                }
            }
            let bonus = values[r] * dom_k / (1.0 + dom as f32);
            total += if mine { bonus } else { -bonus };
        }
    }
    total
}

/// 价值加权机动性：每个合法走子按其棋子的价值加权，再按平均价值归一化。
fn mobility_valued(env: &DarkChessEnv, player: Player, values: &[f32; 7]) -> f32 {
    const AVG_VAL: f32 = 12.0;
    let mut counts = [0u32; 32];
    for m in generate_moves(env, player) {
        counts[m.from] += 1;
    }
    let mut s = 0.0;
    let slots = env.get_board_slots();
    for from in 0..env.config.total_positions {
        if counts[from] > 0 {
            if let Slot::Revealed(p) = &slots[from] {
                s += values[p.piece_type as usize] * counts[from] as f32;
            }
        }
    }
    s / AVG_VAL
}

/// 将帅危险度（近程 + 逃跑感知 + 围困）：
///   - 威胁 = Σ 敌兵 1/2^(距吃掉还需步数 - 1)；只计已翻开的敌兵（只应对具体威胁）；
///   - 逃跑 = 将周围空格中不被敌兵相邻的安全空格数；danger = 威胁 / (1 + 逃跑)；
///   - 围困 = 将所在角落/边界的潜伏惩罚，按存活敌兵数缩放。
fn general_danger(env: &DarkChessEnv, color: usize) -> f32 {
    const ORTHO: [(i32, i32); 4] = [(1, 0), (-1, 0), (0, 1), (0, -1)];
    const CONF_K: f32 = 0.15;
    let cfg = &env.config;
    let cols = cfg.cols as i32;
    let rows = cfg.rows as i32;
    let total = cfg.total_positions;
    let slots = env.get_board_slots();
    let opp = 1 - color;
    let in_bounds = |r: i32, c: i32| r >= 0 && r < rows && c >= 0 && c < cols;
    let sq_of = |r: i32, c: i32| (r * cols + c) as usize;
    let is_enemy_soldier = |sq: usize| {
        matches!(slots[sq], Slot::Revealed(p)
            if p.player.idx() == opp && p.piece_type == PieceType::Soldier)
    };

    // 找己方已翻开的将（暗子将暂不可见，无具体威胁）。
    let gpos = (0..total).find(|&sq| {
        matches!(slots[sq], Slot::Revealed(p)
            if p.player.idx() == color && p.piece_type == PieceType::General)
    });
    let Some(gpos) = gpos else { return 0.0 };
    let gr = (gpos / cfg.cols) as i32;
    let gc = (gpos % cfg.cols) as i32;

    let mut empty_staging: Vec<usize> = Vec::with_capacity(4);
    let mut threat = 0.0f32;
    let mut escape = 0u32;
    for (dr, dc) in ORTHO {
        let (r, c) = (gr + dr, gc + dc);
        if !in_bounds(r, c) {
            continue;
        }
        let s = sq_of(r, c);
        if matches!(slots[s], Slot::Empty) {
            empty_staging.push(s);
            // 该空格只有不被敌兵相邻才算可逃（走到那里下回合即被抓）。
            let mut safe = true;
            for (er, ec) in ORTHO {
                let (nr, nc) = (r + er, c + ec);
                if in_bounds(nr, nc) && is_enemy_soldier(sq_of(nr, nc)) {
                    safe = false;
                    break;
                }
            }
            if safe {
                escape += 1;
            }
        } else if is_enemy_soldier(s) {
            threat += 1.0; // 敌兵已邻接 → 下一步即吃（p=1）
        }
    }

    // 多源 BFS（从将的空邻格出发，只穿过空格）：找到的敌兵距目标 k 步 → 还需 k+1 步。
    if !empty_staging.is_empty() {
        let mut dist = [u8::MAX; 32];
        let mut queue = [0usize; 32];
        let mut qlen = 0usize;
        for &s in &empty_staging {
            dist[s] = 0;
            queue[qlen] = s;
            qlen += 1;
        }
        let mut head = 0usize;
        let mut found = [false; 32];
        while head < qlen {
            let cur = queue[head];
            head += 1;
            let d = dist[cur];
            if d >= 5 {
                continue; // 超过 5 步威胁权重 1/2^5 可忽略
            }
            let cr = (cur / cfg.cols) as i32;
            let cc = (cur % cfg.cols) as i32;
            for (dr, dc) in ORTHO {
                let (nr, nc) = (cr + dr, cc + dc);
                if !in_bounds(nr, nc) {
                    continue;
                }
                let n = sq_of(nr, nc);
                if is_enemy_soldier(n) && !found[n] {
                    found[n] = true;
                    let p = (d as u32) + 2; // (d+1) 步走到邻格 + 1 步吃
                    threat += 0.5_f32.powi((p - 1) as i32);
                } else if matches!(slots[n], Slot::Empty) && dist[n] == u8::MAX {
                    dist[n] = d + 1;
                    queue[qlen] = n;
                    qlen += 1;
                }
            }
        }
    }

    // 围困（潜伏）：角 2 / 边 1 / 中心 0，按存活敌兵数缩放（没有敌兵则角落无妨）。
    let onboard = ORTHO
        .iter()
        .filter(|(dr, dc)| in_bounds(gr + dr, gc + dc))
        .count();
    let confine = (4 - onboard) as f32;
    let alive = alive_counts(env);
    let enemy_sol = alive[opp][PieceType::Soldier as usize];
    let latent = (enemy_sol.min(4) as f32) / 4.0;
    threat / (1.0 + escape as f32) + CONF_K * confine * latent
}

/// 从 `persp` 视角评估当前局面，输出范围约 [-1, 1]（tanh 归一化）。
pub fn evaluate_for(env: &DarkChessEnv, persp: Player, params: &EvalParams) -> f32 {
    let me = persp.idx();
    let opp = 1 - me;
    let hidden = hidden_counts(env);
    let alive = alive_counts(env);
    let values = &params.values;

    let mut mat = material(env, persp, values);
    mat += covered_material(&hidden, me, values);
    mat += king_ctx_term(&alive, me);
    mat += domination_value(env, &alive, me, values, params.dom_k);

    let mob = mobility_valued(env, persp, values) - mobility_valued(env, persp.opposite(), values);
    let king = params.w_king * (general_danger(env, me) - general_danger(env, opp));

    ((mat + params.w_mob * mob - king) / EVAL_SCALE).tanh()
}

/// 从“当前玩家”视角评估（搜索叶子 / MCTS 评估器通用入口）。
pub fn evaluate(env: &DarkChessEnv, params: &EvalParams) -> f32 {
    evaluate_for(env, env.get_current_player(), params)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eval_finite_and_bounded() {
        let mut env = DarkChessEnv::new();
        env.seed = Some(11);
        env.reset();
        let params = EvalParams::default();
        let v = evaluate(&env, &params);
        assert!(v.is_finite());
        assert!((-1.0..=1.0).contains(&v));
        // 视角对称：双方视角评估互为相反数。
        let me = env.get_current_player();
        let v_me = evaluate_for(&env, me, &params);
        let v_opp = evaluate_for(&env, me.opposite(), &params);
        assert!((v_me + v_opp).abs() < 1e-4, "v_me={v_me} v_opp={v_opp}");
    }

    #[test]
    fn flip_does_not_change_full_material() {
        // 全存活物质（明子+暗子）在翻棋前后应保持不变（从固定视角）。
        let mut env = DarkChessEnv::new();
        env.seed = Some(3);
        env.reset();
        let params = EvalParams::default();
        let persp = env.get_current_player();
        let me = persp.idx();
        let before = material(&env, persp, &params.values)
            + covered_material(&hidden_counts(&env), me, &params.values);
        // 找一个翻棋动作并执行
        let flip = generate_moves(&env, env.get_current_player())
            .iter()
            .find(|m| m.is_flip)
            .map(|m| m.action);
        if let Some(a) = flip {
            let mut next = env;
            let _ = next.step(a, None);
            let after = material(&next, persp, &params.values)
                + covered_material(&hidden_counts(&next), me, &params.values);
            assert!((before - after).abs() < 1e-3, "before={before} after={after}");
        }
    }
}
