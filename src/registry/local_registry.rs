// src/registry/local_registry.rs — LocalRegistry（单机 Registry 实现）
//
// 语义：Registry 指针变更（导出路径模型文件被替换）→ Collector 感知换网。
// 通过 notify watch 模型文件所在目录，触发 dirty 标记；ensure_model 惰性重载，
// 重载失败保留旧模型并打印告警（与 data_collector 的 mtime 轮询语义一致，但
// 统一为文件 watch 以与分布式 SchedulerRegistry 的 GetTask 触发方式同构）。

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::Arc;

use anyhow::{Context, Result};
use notify::{RecursiveMode, Watcher};

use crate::inference::onnx::OnnxModel;

/// Collector 一次迭代要执行的任务（单机形态下参数来自启动配置，静态不变）。
#[derive(Debug, Clone)]
pub struct CollectorTask {
    pub variant: String,
    pub model_path: PathBuf,
    pub mcts_sims: usize,
    pub games_per_iter: usize,
}

pub struct LocalRegistry {
    model_path: PathBuf,
    device: String,
    _watcher: notify::RecommendedWatcher,
    dirty: Arc<AtomicBool>,
    current: Option<Arc<OnnxModel>>,
}

impl LocalRegistry {
    /// watch 模型文件所在目录（非递归）；任何事件都置 dirty，
    /// 由 ensure_model 时比对路径/加载结果决定是否真正重载。
    pub fn new(model_path: impl AsRef<Path>, device: &str) -> Result<Self> {
        let model_path = model_path.as_ref().to_path_buf();
        let dirty = Arc::new(AtomicBool::new(true));
        let dirty_flag = Arc::clone(&dirty);

        let (_tx, rx): (
            mpsc::Sender<Result<notify::Event, notify::Error>>,
            mpsc::Receiver<Result<notify::Event, notify::Error>>,
        ) = mpsc::channel();
        let mut watcher = notify::recommended_watcher(move |res: Result<notify::Event, notify::Error>| {
            if res.is_ok() {
                dirty_flag.store(true, Ordering::Release);
            }
        })
        .context("创建 notify watcher 失败")?;
        if let Some(parent) = model_path.parent() {
            watcher
                .watch(parent, RecursiveMode::NonRecursive)
                .with_context(|| format!("watch 目录失败: {}", parent.display()))?;
        }
        // 清空挂起事件，避免把 watch 之前的旧事件当作变更
        while rx.try_recv().is_ok() {}

        Ok(Self {
            model_path,
            device: device.to_string(),
            _watcher: watcher,
            dirty,
            current: None,
        })
    }

    pub fn get_task(&self, variant: &str, mcts_sims: usize, games_per_iter: usize) -> CollectorTask {
        CollectorTask {
            variant: variant.to_string(),
            model_path: self.model_path.clone(),
            mcts_sims,
            games_per_iter,
        }
    }

    /// 返回当前应使用的模型；模型文件有变更时重载。
    /// 重载失败时：已有旧模型则告警保留，否则返回错误（首次加载失败不可继续）。
    pub fn ensure_model(&mut self) -> Result<Arc<OnnxModel>> {
        if !self.dirty.load(Ordering::Acquire) {
            if let Some(m) = &self.current {
                return Ok(Arc::clone(m));
            }
            self.dirty.store(true, Ordering::Release); // 无当前模型时强制走加载
        }
        self.dirty.store(false, Ordering::Release);

        let path = self.model_path.display().to_string();
        match OnnxModel::new(&path, &self.device) {
            Ok(model) => {
                let arc = Arc::new(model);
                self.current = Some(Arc::clone(&arc));
                println!("[registry] ✅ 模型加载/重载成功: {path}");
                Ok(arc)
            }
            Err(e) => {
                if let Some(old) = &self.current {
                    eprintln!("[registry] ⚠️ 模型重载失败（保留旧模型）: {e}");
                    Ok(Arc::clone(old))
                } else {
                    Err(anyhow::anyhow!("模型首次加载失败: {path}: {e}"))
                }
            }
        }
    }
}
