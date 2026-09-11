// 环境种子设置与底层棋盘访问辅助 trait（自 pipeline/self_play/match_core.rs 迁入，
// 因需访问 DarkChessEnv 的 pub(crate) 内部，必须与环境同 crate）。

use crate::core::env::{DarkChessEnv, Game4x4Env, MiniDarkChessEnv};

pub trait AsDarkChessRef {
    fn as_darkchess_ref(&self) -> &DarkChessEnv;
}

impl AsDarkChessRef for DarkChessEnv {
    fn as_darkchess_ref(&self) -> &DarkChessEnv {
        self
    }
}

impl AsDarkChessRef for MiniDarkChessEnv {
    fn as_darkchess_ref(&self) -> &DarkChessEnv {
        &self.inner
    }
}

impl AsDarkChessRef for Game4x4Env {
    fn as_darkchess_ref(&self) -> &DarkChessEnv {
        &self.inner
    }
}

pub trait SeedableEnv {
    fn set_seed(&mut self, seed: u64);
}

impl SeedableEnv for DarkChessEnv {
    fn set_seed(&mut self, seed: u64) {
        self.seed = Some(seed);
        self.reset_internal_state();
        self.initialize_board();
    }
}

impl SeedableEnv for MiniDarkChessEnv {
    fn set_seed(&mut self, seed: u64) {
        self.inner.seed = Some(seed);
        self.inner.reset_internal_state();
        self.inner.initialize_board();
    }
}

impl SeedableEnv for Game4x4Env {
    fn set_seed(&mut self, seed: u64) {
        self.inner.seed = Some(seed);
        self.inner.reset_internal_state();
        self.inner.initialize_board();
    }
}
