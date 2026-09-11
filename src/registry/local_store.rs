// src/registry/local_store.rs — LocalEpisodeStore（单机 EpisodeStore 实现）
//
// 数据出口统一：episode 以 jsonl.gz 落盘目录（每批一个文件，文件名含迭代号/时间戳），
// 字段契约与 PyO3 `episode_to_dict` / gRPC 路径完全一致（复用 serialize::episode_to_dict_json），
// Python 侧由 banqi.infra.episode_store.LocalEpisodeStore 无差别扫描消费。

use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use flate2::write::GzEncoder;
use flate2::Compression;

use crate::pipeline::self_play::serialize::episode_to_dict_json;
use crate::pipeline::self_play::GameEpisode;

pub struct LocalEpisodeStore {
    dir: PathBuf,
    worker_id: usize,
}

impl LocalEpisodeStore {
    pub fn new(dir: impl AsRef<Path>, worker_id: usize) -> Self {
        Self {
            dir: dir.as_ref().to_path_buf(),
            worker_id,
        }
    }

    pub fn root(&self) -> &Path {
        &self.dir
    }

    /// 将一批 episode 写入 `<dir>/<variant>/iter{iter:06}_{ts}_w{worker}.jsonl.gz`，
    /// 返回写入的局数。空批次直接返回 0，不产生文件。
    pub fn put(&self, variant: &str, iteration: usize, episodes: &[GameEpisode]) -> Result<usize> {
        if episodes.is_empty() {
            return Ok(0);
        }
        let out_dir = self.dir.join(variant);
        std::fs::create_dir_all(&out_dir)
            .with_context(|| format!("创建 episode 目录失败: {}", out_dir.display()))?;

        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis();
        let path = out_dir.join(format!("iter{iteration:06}_{ts}_w{}.jsonl.gz", self.worker_id));

        let f = File::create(&path)
            .with_context(|| format!("创建 episode 文件失败: {}", path.display()))?;
        let mut gz = GzEncoder::new(f, Compression::default());
        for ep in episodes {
            writeln!(gz, "{}", episode_to_dict_json(ep))
                .with_context(|| format!("写入 episode 失败: {}", path.display()))?;
        }
        gz.finish()
            .with_context(|| format!("收尾 episode 文件失败: {}", path.display()))?;
        Ok(episodes.len())
    }
}
