//! 暗棋 NNUE 评估网络引擎

use std::fs::File;
use std::io::{Read, Result as IoResult};
use std::path::Path;

use crate::core::env::DarkChessEnv;
use super::feature::{Accumulator, TRANSFORMER_OUT_DIM};

const FC1_OUT_DIM: usize = 32;

/// NNUE 评估网络引擎
///
/// `feature_dim` 由加载的权重文件尺寸反推，或构造时指定；
/// 与 `env.config.nnue_feature_dim()` 不一致时评估会告警并跳过越界特征。
#[derive(Clone, Debug)]
pub struct NnueEvaluator {
    pub feature_dim: usize,
    pub feature_weights: Vec<f32>,
    pub feature_bias: Vec<f32>,
    pub fc1_weights: Vec<f32>,
    pub fc1_bias: Vec<f32>,
    pub fc2_weights: Vec<f32>,
    pub fc2_bias: f32,
}

/// 由权重文件总浮点数反推特征维度。
/// 布局：OUT*DIM + OUT + FC1*OUT + FC1 + FC1 + 1（OUT=TRANSFORMER_OUT_DIM）。
fn infer_feature_dim(total_floats: usize) -> usize {
    let fixed = 2 * TRANSFORMER_OUT_DIM + 2 * FC1_OUT_DIM + 1;
    (total_floats.saturating_sub(fixed)) / TRANSFORMER_OUT_DIM
}

impl Default for NnueEvaluator {
    fn default() -> Self {
        Self::new_dummy(crate::core::env::darkchess_config().nnue_feature_dim())
    }
}

impl NnueEvaluator {
    pub fn new_dummy(feature_dim: usize) -> Self {
        let feature_weights = vec![0.01; TRANSFORMER_OUT_DIM * feature_dim];
        let feature_bias = vec![0.0; TRANSFORMER_OUT_DIM];
        let fc1_weights = vec![0.02; FC1_OUT_DIM * TRANSFORMER_OUT_DIM];
        let fc1_bias = vec![0.0; FC1_OUT_DIM];
        let fc2_weights = vec![0.05; FC1_OUT_DIM];
        let fc2_bias = 0.0;

        Self {
            feature_dim,
            feature_weights,
            feature_bias,
            fc1_weights,
            fc1_bias,
            fc2_weights,
            fc2_bias,
        }
    }

    pub fn load_from_file<P: AsRef<Path>>(path: P) -> IoResult<Self> {
        let mut file = File::open(path)?;
        let mut buf = Vec::new();
        file.read_to_end(&mut buf)?;

        let total_floats = buf.len() / 4;
        let feature_dim = infer_feature_dim(total_floats);

        let mut offset = 0;
        let read_f32_slice = |buf: &[u8], offset: &mut usize, count: usize| -> Vec<f32> {
            let mut res = Vec::with_capacity(count);
            for _ in 0..count {
                if *offset + 4 > buf.len() {
                    res.push(0.0);
                } else {
                    let bytes = [buf[*offset], buf[*offset + 1], buf[*offset + 2], buf[*offset + 3]];
                    res.push(f32::from_le_bytes(bytes));
                    *offset += 4;
                }
            }
            res
        };

        let feature_weights = read_f32_slice(&buf, &mut offset, TRANSFORMER_OUT_DIM * feature_dim);
        let feature_bias = read_f32_slice(&buf, &mut offset, TRANSFORMER_OUT_DIM);
        let fc1_weights = read_f32_slice(&buf, &mut offset, FC1_OUT_DIM * TRANSFORMER_OUT_DIM);
        let fc1_bias = read_f32_slice(&buf, &mut offset, FC1_OUT_DIM);
        let fc2_weights = read_f32_slice(&buf, &mut offset, FC1_OUT_DIM);
        let fc2_bias = read_f32_slice(&buf, &mut offset, 1).first().copied().unwrap_or(0.0);

        Ok(Self {
            feature_dim,
            feature_weights,
            feature_bias,
            fc1_weights,
            fc1_bias,
            fc2_weights,
            fc2_bias,
        })
    }

    pub fn compute_accumulator(&self, active_features: &[usize]) -> Accumulator {
        let mut acc = Accumulator::default();
        acc.vals.copy_from_slice(&self.feature_bias);

        for &feat_idx in active_features {
            if feat_idx < self.feature_dim {
                let w_offset = feat_idx * TRANSFORMER_OUT_DIM;
                for j in 0..TRANSFORMER_OUT_DIM {
                    acc.vals[j] += self.feature_weights[w_offset + j];
                }
            }
        }
        acc
    }

    pub fn forward_accumulator(&self, acc: &Accumulator) -> f32 {
        let mut h0 = [0.0f32; TRANSFORMER_OUT_DIM];
        for i in 0..TRANSFORMER_OUT_DIM {
            h0[i] = acc.vals[i].clamp(0.0, 1.0);
        }

        let mut h1 = [0.0f32; FC1_OUT_DIM];
        for i in 0..FC1_OUT_DIM {
            let mut sum = self.fc1_bias[i];
            let row_offset = i * TRANSFORMER_OUT_DIM;
            for j in 0..TRANSFORMER_OUT_DIM {
                sum += self.fc1_weights[row_offset + j] * h0[j];
            }
            h1[i] = sum.clamp(0.0, 1.0);
        }

        let mut out = self.fc2_bias;
        for i in 0..FC1_OUT_DIM {
            out += self.fc2_weights[i] * h1[i];
        }

        out.tanh()
    }

    pub fn evaluate(&self, env: &DarkChessEnv) -> f32 {
        let expect_dim = env.config.nnue_feature_dim();
        if expect_dim != self.feature_dim {
            eprintln!(
                "[nnue] 警告: 权重特征维度({})与环境推导维度({})不一致，越界特征将被跳过",
                self.feature_dim, expect_dim
            );
        }
        let active = env.nnue_active_features();
        let acc = self.compute_accumulator(&active);
        self.forward_accumulator(&acc)
    }
}
