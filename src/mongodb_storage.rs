// mongodb_storage.rs - MongoDB存储模块
//
// 提供按整局游戏存储训练数据的功能

use crate::self_play::GameEpisode;
use anyhow::Result;
use bson::{doc, Bson, Document};
use mongodb::{options::ClientOptions, Client, Collection};
use serde::{Deserialize, Serialize};

// ================ 数据结构 ================

/// 单个样本的数据结构（用于MongoDB存储）
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SampleDocument {
    pub board_state: Vec<f32>,
    pub scalar_state: Vec<f32>,
    pub policy_probs: Vec<f32>,
    pub mcts_value: f32,           // MCTS根节点的价值（用于训练）
    pub game_result_value: f32,    // 游戏结果价值（用于分析）
    pub action_mask: Vec<i32>,
    pub step_in_game: usize,
}

/// 整局游戏的数据结构
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GameDocument {
    pub iteration: usize,
    pub game_length: usize,
    pub winner: Option<i32>, // Some(1)=红胜, Some(-1)=黑胜, None/Some(0)=平局
    pub samples: Vec<SampleDocument>,
    pub timestamp: bson::DateTime,
}

// ================ MongoDB客户端 ================

pub struct MongoStorage {
    client: Client,
    db_name: String,
    collection_name: String,
}

impl MongoStorage {
    /// 创建新的MongoDB存储客户端
    pub async fn new(uri: &str, db_name: &str, collection_name: &str) -> Result<Self> {
        let client_options = ClientOptions::parse(uri).await?;
        let client = Client::with_options(client_options)?;

        // 测试连接
        client
            .database("admin")
            .run_command(doc! { "ping": 1 })
            .await?;

        println!(
            "MongoDB连接成功: 数据库={}, 集合={}",
            db_name, collection_name
        );

        Ok(Self {
            client,
            db_name: db_name.to_string(),
            collection_name: collection_name.to_string(),
        })
    }

    /// 获取集合引用
    fn get_collection(&self) -> Collection<Document> {
        self.client
            .database(&self.db_name)
            .collection(&self.collection_name)
    }

    /// 保存一批游戏数据
    pub async fn save_games(&self, iteration: usize, episodes: &[GameEpisode]) -> Result<()> {
        if episodes.is_empty() {
            return Ok(());
        }

        let collection = self.get_collection();
        let mut documents = Vec::new();

        for episode in episodes {
            let mut sample_docs = Vec::new();

            for (step_idx, (obs, probs, mcts_val, game_result_val, mask)) in episode.samples.iter().enumerate() {
                let board_state: Vec<f32> = obs.board.as_slice().unwrap().to_vec();
                let scalar_state: Vec<f32> = obs.scalars.as_slice().unwrap().to_vec();

                sample_docs.push(SampleDocument {
                    board_state,
                    scalar_state,
                    policy_probs: probs.clone(),
                    mcts_value: *mcts_val,
                    game_result_value: *game_result_val,
                    action_mask: mask.clone(),
                    step_in_game: step_idx,
                });
            }

            let game_doc = GameDocument {
                iteration,
                game_length: episode.game_length,
                winner: episode.winner,
                samples: sample_docs,
                timestamp: bson::DateTime::now(),
            };

            // 序列化为BSON文档
            let doc = bson::to_document(&game_doc)?;
            documents.push(doc);
        }

        // 批量插入
        collection.insert_many(documents).await?;

        println!(
            "  [MongoDB] 已保存 {} 局游戏到数据库 (iteration={})",
            episodes.len(),
            iteration
        );

        Ok(())
    }

    /// 获取数据库中的总游戏数
    pub async fn count_games(&self) -> Result<u64> {
        let collection = self.get_collection();
        let count = collection.count_documents(doc! {}).await?;
        Ok(count)
    }

    /// 获取指定迭代范围内的游戏统计
    pub async fn get_iteration_stats(&self, iteration: usize) -> Result<IterationStats> {
        let collection = self.get_collection();

        let filter = doc! { "iteration": iteration as i64 };
        let total_games = collection.count_documents(filter.clone()).await?;

        let mut cursor = collection.find(filter).await?;
        let mut red_wins = 0u64;
        let mut black_wins = 0u64;
        let mut draws = 0u64;
        let mut total_samples = 0u64;

        use futures::stream::TryStreamExt;
        while let Some(doc) = cursor.try_next().await? {
            if let Ok(game) = bson::from_document::<GameDocument>(doc) {
                match game.winner {
                    Some(1) => red_wins += 1,
                    Some(-1) => black_wins += 1,
                    _ => draws += 1,
                }
                total_samples += game.samples.len() as u64;
            }
        }

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
    pub async fn cleanup_old_iterations(&self, keep_recent: usize) -> Result<()> {
        let collection = self.get_collection();

        // 查找所有不同的iteration值
        let pipeline = vec![
            doc! { "$group": { "_id": "$iteration" } },
            doc! { "$sort": { "_id": -1 } },
        ];

        let mut cursor = collection.aggregate(pipeline).await?;
        let mut iterations = Vec::new();

        use futures::stream::TryStreamExt;
        while let Some(doc) = cursor.try_next().await? {
            if let Some(Bson::Int32(iter)) = doc.get("_id") {
                iterations.push(*iter as usize);
            } else if let Some(Bson::Int64(iter)) = doc.get("_id") {
                iterations.push(*iter as usize);
            }
        }

        if iterations.len() > keep_recent {
            let iterations_to_delete: Vec<i64> = iterations
                .iter()
                .skip(keep_recent)
                .map(|&i| i as i64)
                .collect();

            if !iterations_to_delete.is_empty() {
                let filter = doc! { "iteration": { "$in": iterations_to_delete.clone() } };
                let result = collection.delete_many(filter).await?;

                println!(
                    "  [MongoDB] 清理旧数据: 删除了 {} 局游戏 (iterations: {:?})",
                    result.deleted_count, iterations_to_delete
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
        println!("  [MongoDB] Iteration {} 统计:", self.iteration);
        println!("    总游戏数: {}", self.total_games);
        println!("    红方胜: {} ({:.1}%)", self.red_wins, self.red_wins as f32 / self.total_games as f32 * 100.0);
        println!("    黑方胜: {} ({:.1}%)", self.black_wins, self.black_wins as f32 / self.total_games as f32 * 100.0);
        println!("    平局: {} ({:.1}%)", self.draws, self.draws as f32 / self.total_games as f32 * 100.0);
        println!("    总样本数: {}", self.total_samples);
    }
}
