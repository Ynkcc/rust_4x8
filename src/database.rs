// database.rs - 数据库操作模块
//
// 提供SQLite数据库相关操作，用于存储和读取训练样本

use crate::game_env::Observation;
use anyhow::Result;
use rusqlite::{Connection, params};

// ================ 数据库操作 ================

pub fn init_database(db_path: &str) -> Result<Connection> {
    let conn = Connection::open(db_path)?;
    
    conn.execute(
        "CREATE TABLE IF NOT EXISTS training_samples (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            iteration INTEGER NOT NULL,
            episode_type TEXT NOT NULL,
            board_state BLOB NOT NULL,
            scalar_state BLOB NOT NULL,
            policy_probs BLOB NOT NULL,
            value_target REAL NOT NULL,
            action_mask BLOB NOT NULL,
            game_length INTEGER NOT NULL,
            step_in_game INTEGER NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )",
        [],
    )?;
    
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_iteration ON training_samples(iteration)",
        [],
    )?;
    
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_episode_type ON training_samples(episode_type)",
        [],
    )?;
    
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_game_length ON training_samples(game_length)",
        [],
    )?;
    
    println!("数据库初始化完成: {}", db_path);
    Ok(conn)
}

pub fn save_samples_to_db(
    conn: &mut Connection,
    iteration: usize,
    episode_type: &str,
    samples: &[(Observation, Vec<f32>, f32, Vec<i32>)],
    game_length: usize,
) -> Result<()> {
    let tx = conn.transaction()?;
    {
        let mut stmt = tx.prepare(
            "INSERT INTO training_samples 
             (iteration, episode_type, board_state, scalar_state, policy_probs, value_target, action_mask, game_length, step_in_game) 
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)"
        )?;
        
        for (step_idx, (obs, probs, value, mask)) in samples.iter().enumerate() {
            let board_bytes: Vec<u8> = obs.board.as_slice().unwrap()
                .iter()
                .flat_map(|&x| x.to_le_bytes())
                .collect();
            
            let scalar_bytes: Vec<u8> = obs.scalars.as_slice().unwrap()
                .iter()
                .flat_map(|&x| x.to_le_bytes())
                .collect();
            
            let probs_bytes: Vec<u8> = probs.iter()
                .flat_map(|&x| x.to_le_bytes())
                .collect();
            
            let mask_bytes: Vec<u8> = mask.iter()
                .flat_map(|&x| x.to_le_bytes())
                .collect();
            
            stmt.execute(params![
                iteration as i64,
                episode_type,
                board_bytes,
                scalar_bytes,
                probs_bytes,
                value,
                mask_bytes,
                game_length as i64,
                step_idx as i64,
            ])?;
        }
    }
    tx.commit()?;
    Ok(())
}

pub fn load_samples_from_db(conn: &Connection) -> Result<Vec<(Observation, Vec<f32>, f32, Vec<i32>)>> {
    let mut stmt = conn.prepare(
        "SELECT board_state, scalar_state, policy_probs, value_target, action_mask 
         FROM training_samples"
    )?;
    
    let samples = stmt.query_map([], |row| {
        let board_bytes: Vec<u8> = row.get(0)?;
        let scalar_bytes: Vec<u8> = row.get(1)?;
        let probs_bytes: Vec<u8> = row.get(2)?;
        let value: f32 = row.get(3)?;
        let mask_bytes: Vec<u8> = row.get(4)?;
        
        let board_data: Vec<f32> = board_bytes.chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();
        
        let scalar_data: Vec<f32> = scalar_bytes.chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();
        
        let probs: Vec<f32> = probs_bytes.chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();
        
        let mask: Vec<i32> = mask_bytes.chunks_exact(4)
            .map(|chunk| i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();
        
        use ndarray::Array;
        let board = Array::from_shape_vec((2, 8, 3, 4), board_data)
            .expect("Failed to reshape board data");
        let scalars = Array::from_vec(scalar_data);
        
        let obs = Observation { board, scalars };
        
        Ok((obs, probs, value, mask))
    })?;
    
    let mut result = Vec::new();
    for sample in samples {
        result.push(sample?);
    }
    
    Ok(result)
}

/// 加载距离游戏结束指定步数以内的样本
pub fn load_endgame_samples_from_db(conn: &Connection, max_steps_from_end: usize) -> Result<Vec<(Observation, Vec<f32>, f32, Vec<i32>)>> {
    let mut stmt = conn.prepare(
        "SELECT board_state, scalar_state, policy_probs, value_target, action_mask, game_length, step_in_game
         FROM training_samples
         WHERE (game_length - step_in_game) <= ?1"
    )?;
    
    let samples = stmt.query_map([max_steps_from_end as i64], |row| {
        let board_bytes: Vec<u8> = row.get(0)?;
        let scalar_bytes: Vec<u8> = row.get(1)?;
        let probs_bytes: Vec<u8> = row.get(2)?;
        let value: f32 = row.get(3)?;
        let mask_bytes: Vec<u8> = row.get(4)?;
        
        let board_data: Vec<f32> = board_bytes.chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();
        
        let scalar_data: Vec<f32> = scalar_bytes.chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();
        
        let probs: Vec<f32> = probs_bytes.chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();
        
        let mask: Vec<i32> = mask_bytes.chunks_exact(4)
            .map(|chunk| i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();
        
        use ndarray::Array;
        // STATE_STACK_SIZE=1, 所以shape是 (1, 8, 3, 4) 而不是 (2, 8, 3, 4)
        let board = Array::from_shape_vec((1, 8, 3, 4), board_data)
            .expect("Failed to reshape board data");
        let scalars = Array::from_vec(scalar_data);
        
        let obs = Observation { board, scalars };
        
        Ok((obs, probs, value, mask))
    })?;
    
    let mut result = Vec::new();
    for sample in samples {
        result.push(sample?);
    }
    
    println!("从数据库加载了 {} 个距离游戏结束 {} 步以内的样本", result.len(), max_steps_from_end);
    
    Ok(result)
}