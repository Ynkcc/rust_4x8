// mongodb_storage.rs - MongoDB存储模块（同步版本）
//
// 提供按整局游戏存储训练数据的功能

use crate::self_play::GameEpisode;
use anyhow::Result;
use bson::{doc, Document};
use mongodb::{Client, Collection};
use serde::{Deserialize, Serialize};
use tokio::runtime::Runtime;
use tokio::sync::mpsc;

// ================ 数据结构 ================

struct SaveJob {
    iteration: usize,
    episodes: Vec<GameEpisode>,
}

/// 单个样本的数据结构（用于MongoDB存储）
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SampleDocument {
    pub board_state: Vec<f32>,
    pub scalar_state: Vec<f32>,
    pub policy_probs: Vec<f32>,
    pub mcts_value: f32,
    pub completed_q: f32,
    pub root_visit_count: u32,
    pub game_result_value: f32,
    pub action_mask: Vec<i32>,
    pub step_in_game: usize,
}

/// 整局游戏的数据结构
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GameDocument {
    pub iteration: usize,
    pub game_length: usize,
    pub winner: Option<i32>,
    pub samples: Vec<SampleDocument>,
    pub timestamp: bson::DateTime,
}

// ================ MongoDB客户端 ================

pub struct MongoStorage {
    client: Client,
    db_name: String,
    collection_name: String,
    runtime: Runtime,
    sender: mpsc::UnboundedSender<SaveJob>,
}

impl MongoStorage {
    /// 创建新的MongoDB存储客户端（同步版本）
    pub fn new(uri: &str, db_name: &str, collection_name: &str) -> Result<Self> {
        let runtime = Runtime::new()?;
        
        let client = runtime.block_on(async {
            Client::with_uri_str(uri).await
        })?;

        // 测试连接
        runtime.block_on(async {
            client
                .database("admin")
                .run_command(doc! { "ping": 1 })
                .await
        })?;

        println!(
            "MongoDB连接成功: 数据库={}, 集合={}",
            db_name, collection_name
        );

        // 创建异步通道和后台处理任务
        let (tx, mut rx) = mpsc::unbounded_channel::<SaveJob>();
        let client_clone = client.clone();
        let db_name_clone = db_name.to_string();
        let collection_name_clone = collection_name.to_string();

        runtime.spawn(async move {
            let collection = client_clone
                .database(&db_name_clone)
                .collection::<Document>(&collection_name_clone);

            while let Some(job) = rx.recv().await {
                if let Err(e) = Self::process_save_job(&collection, job).await {
                    eprintln!("❌ [MongoDB Background] 保存失败: {}", e);
                }
            }
        });

        Ok(Self {
            client,
            db_name: db_name.to_string(),
            collection_name: collection_name.to_string(),
            runtime,
            sender: tx,
        })
    }

    /// 获取集合引用
    fn get_collection(&self) -> Collection<Document> {
        self.client
            .database(&self.db_name)
            .collection(&self.collection_name)
    }

    /// 保存一批游戏数据（非阻塞）
    pub fn save_games(&self, iteration: usize, episodes: Vec<GameEpisode>) -> Result<()> {
        if episodes.is_empty() {
            return Ok(());
        }

        self.sender.send(SaveJob {
            iteration,
            episodes,
        })?;

        Ok(())
    }

    /// 内部异步处理函数
    async fn process_save_job(collection: &Collection<Document>, job: SaveJob) -> Result<()> {
        let mut documents = Vec::new();

        for episode in job.episodes {
            if episode.samples.is_empty() {
                // 仅在调试极其详细时打印，避免刷屏
                continue;
            }
            
            let mut sample_docs = Vec::new();

            for (step_idx, (obs, probs, mcts_val, completed_q, root_visit_count, game_result_val, mask)) in episode.samples.iter().enumerate() {
                let board_state: Vec<f32> = obs.board.as_slice().unwrap().to_vec();
                let scalar_state: Vec<f32> = obs.scalars.as_slice().unwrap().to_vec();

                sample_docs.push(SampleDocument {
                    board_state,
                    scalar_state,
                    policy_probs: probs.clone(),
                    mcts_value: *mcts_val,
                    completed_q: *completed_q,
                    root_visit_count: *root_visit_count,
                    game_result_value: *game_result_val,
                    action_mask: mask.clone(),
                    step_in_game: step_idx,
                });
            }

            let game_doc = GameDocument {
                iteration: job.iteration,
                game_length: episode.game_length,
                winner: episode.winner,
                samples: sample_docs,
                timestamp: bson::DateTime::now(),
            };

            let doc = bson::to_document(&game_doc)?;
            documents.push(doc);
        }

        if !documents.is_empty() {
            let count = documents.len();
            collection.insert_many(documents).await?;
            println!(
                "  [MongoDB Background] 已保存 {} 局游戏 (iteration={})",
                count,
                job.iteration
            );
        }

        Ok(())
    }

    /// 获取数据库中的总游戏数
    pub fn count_games(&self) -> Result<u64> {
        let collection = self.get_collection();
        let count = self.runtime.block_on(async {
            collection.count_documents(doc! {}).await
        })?;
        Ok(count)
    }

    /// 获取指定迭代范围内的游戏统计
    pub fn get_iteration_stats(&self, iteration: usize) -> Result<IterationStats> {
        let collection = self.get_collection();

        let filter = doc! { "iteration": iteration as i64 };
        
        let total_games = self.runtime.block_on(async {
            collection.count_documents(filter.clone()).await
        })?;

        let (red_wins, black_wins, draws, total_samples) = self.runtime.block_on(async {
            let mut cursor = collection.find(filter).await?;
            let mut red_wins = 0u64;
            let mut black_wins = 0u64;
            let mut draws = 0u64;
            let mut total_samples = 0u64;

            while cursor.advance().await? {
                let raw_doc = cursor.current();
                if let Ok(game) = bson::from_slice::<GameDocument>(raw_doc.as_bytes()) {
                    match game.winner {
                        Some(1) => red_wins += 1,
                        Some(-1) => black_wins += 1,
                        _ => draws += 1,
                    }
                    total_samples += game.samples.len() as u64;
                }
            }

            Ok::<_, mongodb::error::Error>((red_wins, black_wins, draws, total_samples))
        })?;

        Ok(IterationStats {
            iteration,
            total_games,
            red_wins,
            black_wins,
            draws,
            total_samples,
        })
    }

    /// 删除旧数据，只保留最近N个iteration的数据
    pub fn cleanup_old_iterations(&self, keep_recent: usize) -> Result<()> {
        let collection = self.get_collection();

        let iterations = self.runtime.block_on(async {
            let pipeline = vec![
                doc! { "$group": { "_id": "$iteration" } },
                doc! { "$sort": { "_id": -1 } },
            ];

            let mut cursor = collection.aggregate(pipeline).await?;
            let mut iterations = Vec::new();

            while cursor.advance().await? {
                let raw_doc = cursor.current();
                if let Ok(iter_value) = raw_doc.get_i32("_id") {
                    iterations.push(iter_value as usize);
                } else if let Ok(iter_value) = raw_doc.get_i64("_id") {
                    iterations.push(iter_value as usize);
                }
            }

            Ok::<_, mongodb::error::Error>(iterations)
        })?;

        if iterations.len() > keep_recent {
            let iterations_to_delete: Vec<i64> = iterations
                .iter()
                .skip(keep_recent)
                .map(|&i| i as i64)
                .collect();

            if !iterations_to_delete.is_empty() {
                let deleted_count = self.runtime.block_on(async {
                    let filter = doc! { "iteration": { "$in": iterations_to_delete.clone() } };
                    let result = collection.delete_many(filter).await?;
                    Ok::<_, mongodb::error::Error>(result.deleted_count)
                })?;

                println!(
                    "  [MongoDB] 清理旧数据: 删除了 {} 局游戏",
                    deleted_count
                );
            }
        }

        Ok(())
    }
}

// ================ 统计信息 ================

#[derive(Debug, Clone)]
pub struct IterationStats {
    pub iteration: usize,
    pub total_games: u64,
    pub red_wins: u64,
    pub black_wins: u64,
    pub draws: u64,
    pub total_samples: u64,
}

impl IterationStats {
    pub fn print(&self) {
        if self.total_games == 0 {
            println!("  [MongoDB] Iteration {} 统计: 无数据", self.iteration);
            return;
        }
        println!("  [MongoDB] Iteration {} 统计:", self.iteration);
        println!("    总游戏数: {}", self.total_games);
        println!("    红方胜: {} ({:.1}%)", self.red_wins, self.red_wins as f32 / self.total_games as f32 * 100.0);
        println!("    黑方胜: {} ({:.1}%)", self.black_wins, self.black_wins as f32 / self.total_games as f32 * 100.0);
        println!("    平局: {} ({:.1}%)", self.draws, self.draws as f32 / self.total_games as f32 * 100.0);
        println!("    总样本数: {}", self.total_samples);
    }
}
