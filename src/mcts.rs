// code_files/src/mcts.rs
///大语言模型注意:
///不要修改在机会节点的全量展开，以及后续在机会节点的所有子节点模拟
///不要移除显式判断父子节点玩家是否一致，以确定价值是否取反
///不要移除此块注释，如果你移除，我将要求重新生成，直到你学会保留本注释为止
/// ==========================================================================
use crate::{DarkChessEnv, Piece, PieceType, Player, Slot, ACTION_SPACE_SIZE};
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use tokio::sync::{RwLock, Semaphore};
use futures::future::join_all;
use futures::future::BoxFuture;

// ============================================================================
// 1. 节点定义 (Node Definition)
// ============================================================================

/// MCTS 树节点
///
/// 包含搜索树节点的统计信息、子节点指针以及对应的游戏状态。
/// 支持两种节点类型：
/// - **决策节点 (State Node)**: 玩家采取行动的节点。
/// - **机会节点 (Chance Node)**: 处理随机事件（如翻棋）的节点，其分支代表不同的随机结果。
#[derive(Debug)]
pub struct MctsNode {
    /// 访问次数 (N)
    pub visit_count: u32,
    /// 虚拟损失 (Virtual Loss) - 用于并发搜索时防止过度探索同一路径
    pub virtual_loss: u32,
    /// 价值总和 (W) - 从该节点对应玩家的视角累积
    pub value_sum: f32,
    /// 先验概率 (P) - 由神经网络策略头输出
    pub prior: f32,
    /// 子节点映射 (针对 State Node)
    /// Key: 动作索引 (Action Index), Value: 对应的子节点 (线程安全)
    pub children: HashMap<usize, Arc<RwLock<MctsNode>>>,
    /// 标记该节点是否已经扩展过（即是否已经计算过子节点的先验概率）
    pub is_expanded: bool,

    // --- Chance Node 相关属性 ---
    /// 是否为机会节点 (Chance Node)
    /// 当上一步动作包含不确定性（如翻开暗子）时，当前节点为机会节点。
    pub is_chance_node: bool,
    pub is_root_node: bool,
    /// 可能的后续状态映射 (针对 Chance Node)
    /// Key: 结果 ID (Outcome ID, 代表具体的棋子类型), Value: (该结果的概率, 对应的子节点)
    pub possible_states: HashMap<usize, (f32, Arc<RwLock<MctsNode>>)>,

    // --- 游戏环境 ---
    /// 存储该节点对应的游戏环境状态
    /// State Node 通常持有环境快照。
    pub env: Option<Box<DarkChessEnv>>,
}

impl MctsNode {
    /// 创建新节点
    pub fn new(prior: f32, is_chance_node: bool, env: Option<DarkChessEnv>, is_root_node: bool) -> Self {
        Self {
            visit_count: 0,
            virtual_loss: 0,
            value_sum: 0.0,
            prior,
            children: HashMap::new(),
            is_expanded: false,
            is_chance_node,
            is_root_node,
            possible_states: HashMap::new(),
            env: env.map(Box::new),
        }
    }

    /// 获取当前节点对应的行动玩家
    pub fn player(&self) -> Player {
        self.env
            .as_ref()
            .expect("Node must have environment")
            .as_ref()
            .get_current_player()
    }

    /// 获取节点的平均价值 Q(s, a)
    /// 计算公式: (W - virtual_loss) / (N + virtual_loss)
    /// 
    /// Virtual Loss 机制：假设每个虚拟访问都会导致失败（价值 -1.0），
    /// 从而降低该节点的 Q 值，防止多个线程同时探索同一路径。
    pub fn q_value(&self) -> f32 {
        let total_visits = self.visit_count + self.virtual_loss;
        if total_visits == 0 {
            0.0
        } else {
            // 修正：分子也要减去 virtual_loss，假设虚拟访问都会导致失败
            // 对于价值范围 [-1, 1]，每个虚拟访问假设价值为 -1.0
            (self.value_sum - self.virtual_loss as f32) / total_visits as f32
        }
    }
}

/// 辅助函数：为翻开的棋子生成唯一 ID
/// 用于在 Chance Node 中区分不同的翻棋结果 (Outcome)。
/// 映射规则:
/// 0-6: 红方 [兵, 炮, 马, 车, 象, 士, 将]
/// 7-13: 黑方 [兵, 炮, 马, 车, 象, 士, 将]
fn get_outcome_id(piece: &Piece) -> usize {
    let type_idx = match piece.piece_type {
        PieceType::Soldier => 0,
        PieceType::Cannon => 1,
        PieceType::Horse => 2,
        PieceType::Chariot => 3,
        PieceType::Elephant => 4,
        PieceType::Advisor => 5,
        PieceType::General => 6,
    };
    let player_offset = match piece.player {
        Player::Red => 0,
        Player::Black => 7,
    };
    type_idx + player_offset
}

// ============================================================================
// 2. 评估接口 (Evaluation Interface) - 异步化
// ============================================================================

/// 状态评估器 trait (Async)
/// 用于抽象神经网络或其他估值函数的接口。
#[async_trait::async_trait]
pub trait Evaluator: Send + Sync {
    /// 评估给定状态，返回 (策略概率分布, 状态价值)
    async fn evaluate(&self, env: &DarkChessEnv) -> (Vec<f32>, f32);
}

// ============================================================================
// MCTS 主逻辑 (异步并发版)
// ============================================================================

/// MCTS 配置参数
#[derive(Clone)]
pub struct MCTSConfig {
    /// PUCT 探索常数 (C_puct)
    pub cpuct: f32,
    /// 模拟/推理总次数限制
    pub num_simulations: usize,
    /// 虚拟损失值 (Virtual Loss)
    pub virtual_loss: f32,
    /// 并行 Worker 数量 (并发度)
    pub num_mcts_workers: usize,
    /// 限制同时等待推理的任务数量 (Semaphore Permits)
    pub max_pending_inference: usize,
    /// Dirichlet 噪声参数 alpha
    pub dirichlet_alpha: f32,
    /// Dirichlet 噪声权重 epsilon
    pub dirichlet_epsilon: f32,
    /// 是否为训练模式
    pub train: bool,
}

impl Default for MCTSConfig {
    fn default() -> Self {
        Self {
            cpuct: 1.0,
            num_simulations: 400,
            virtual_loss: 1.0, // 通常取 1.0 或 3.0
            num_mcts_workers: 16,
            max_pending_inference: 8, // 限制等待 GPU 的并发数
            dirichlet_alpha: 0.3,
            dirichlet_epsilon: 0.25,
            train: false,
        }
    }
}

/// 蒙特卡洛树搜索 (MCTS) 主结构
pub struct MCTS<E: Evaluator> {
    /// 根节点使用 Arc<RwLock> 以支持并发访问
    pub root: Arc<RwLock<MctsNode>>,
    evaluator: Arc<E>,
    config: MCTSConfig,
    /// 推理信号量，用于限制并发推理请求数量
    inference_sem: Arc<Semaphore>,
}

impl<E: Evaluator + 'static> MCTS<E> {
    /// 创建新的 MCTS 实例
    pub fn new(env: &DarkChessEnv, evaluator: Arc<E>, config: MCTSConfig) -> Self {
        let root = MctsNode::new(1.0, false, Some(env.clone()), true);
        let max_inference = config.max_pending_inference.max(1);
        Self {
            root: Arc::new(RwLock::new(root)),
            evaluator,
            config,
            inference_sem: Arc::new(Semaphore::new(max_inference)),
        }
    }

    /// 应用 Dirichlet 噪声到策略概率分布
    /// 
    /// # 参数
    /// - `policy_probs`: 策略概率分布（会被原地修改）
    /// - `masks`: 动作掩码（1 表示合法动作，0 表示非法动作）
    /// - `config`: MCTS 配置参数
    fn apply_dirichlet_noise(policy_probs: &mut [f32], masks: &[i32], config: &MCTSConfig) {
        use rand::distributions::Distribution;
        use rand_distr::Dirichlet as DirichletDist;
        
        let valid_actions: Vec<usize> = masks
            .iter()
            .enumerate()
            .filter_map(|(i, &m)| if m == 1 { Some(i) } else { None })
            .collect();
        
        if valid_actions.len() > 1 {
            let alpha = vec![config.dirichlet_alpha; valid_actions.len()];
            let dirichlet = DirichletDist::new(&alpha).unwrap();
            let noise = dirichlet.sample(&mut rand::thread_rng());
            
            for (i, &idx) in valid_actions.iter().enumerate() {
                policy_probs[idx] = (1.0 - config.dirichlet_epsilon) * policy_probs[idx]
                    + config.dirichlet_epsilon * noise[i] as f32;
            }
        }
    }

    /// 推进搜索树 (Tree Reuse)
    pub async fn step_next(&mut self, env: &DarkChessEnv, action: usize) {
        // 获取 root 的写锁，因为我们要替换 root
        let mut root_guard = self.root.write().await;
        
        // 尝试从 children 中移除对应的子节点
        if let Some(child_arc) = root_guard.children.remove(&action) {
            // 这里我们需要获取子节点的锁来检查它的属性
            // 注意：因为我们要将 child_arc 提升为新的 root，
            // 我们不能持有 child 的锁太久，这里主要是为了逻辑判断
            let is_chance_node;
            {
                let child = child_arc.read().await;
                is_chance_node = child.is_chance_node;
            }

            if is_chance_node {
                // 如果子节点是 Chance Node
                let slot = env.get_target_slot(action);
                match slot {
                    Slot::Revealed(piece) => {
                        let outcome_id = get_outcome_id(&piece);
                        let next_node_opt = {
                            let mut child_guard = child_arc.write().await;
                             child_guard.possible_states.remove(&outcome_id)
                        };

                        if let Some((_, next_node)) = next_node_opt {
                            // 成功找到后续节点，替换 root
                            // 将子节点提升为根节点，需要设置 is_root_node = true
                            {
                                let mut next_node_guard = next_node.write().await;
                                next_node_guard.is_root_node = true;
                            }
                            drop(root_guard);
                            self.root = next_node;
                            
                            // 如果是训练模式，且新根节点已经扩展过，对其子节点应用 Dirichlet 噪声
                            if self.config.train {
                                self.apply_root_dirichlet_noise().await;
                            }
                            return;
                        }
                    }
                    _ => panic!("Expected revealed piece at action position in Chance Node"),
                }
                // 没找到分支，重置
                drop(root_guard);
                self.root = Arc::new(RwLock::new(MctsNode::new(1.0, false, Some(env.clone()), true)));
            } else {
                // 确定性节点，直接复用，将子节点提升为根节点
                {
                    let mut child_guard = child_arc.write().await;
                    child_guard.is_root_node = true;
                }
                drop(root_guard);
                self.root = child_arc;
                
                // 如果是训练模式，且新根节点已经扩展过，对其子节点应用 Dirichlet 噪声
                if self.config.train {
                    self.apply_root_dirichlet_noise().await;
                }
            }
        } else {
            // 无法复用
            drop(root_guard);
            self.root = Arc::new(RwLock::new(MctsNode::new(1.0, false, Some(env.clone()), true)));
        }
    }

    /// 对根节点应用 Dirichlet 噪声（仅在训练模式且根节点已扩展时有效）
    async fn apply_root_dirichlet_noise(&self) {
        let root = self.root.read().await;
        
        // 只有在根节点已扩展且有子节点时才应用噪声
        if !root.is_expanded || root.children.is_empty() {
            return;
        }
        
        let env = root.env.as_ref().expect("Root must have env").as_ref();
        let mut masks = vec![0; ACTION_SPACE_SIZE];
        env.action_masks_into(&mut masks);
        
        // 收集所有子节点及其 action
        let children_actions: Vec<(usize, Arc<RwLock<MctsNode>>)> = root
            .children
            .iter()
            .map(|(&action, child)| (action, child.clone()))
            .collect();
        
        drop(root); // 释放读锁
        
        // 收集当前的 prior 值
        let mut policy_probs = vec![0.0; ACTION_SPACE_SIZE];
        for (action, child_arc) in &children_actions {
            let child = child_arc.read().await;
            policy_probs[*action] = child.prior;
        }
        
        // 应用 Dirichlet 噪声
        Self::apply_dirichlet_noise(&mut policy_probs, &masks, &self.config);
        
        // 更新子节点的 prior 值
        for (action, child_arc) in children_actions {
            let mut child = child_arc.write().await;
            child.prior = policy_probs[action];
        }
    }

    /// 执行 MCTS 搜索 (异步并行)
    /// 
    /// 启动多个 Worker 任务，直到完成指定数量的推理/评估。
    pub async fn run(&self) -> Option<usize> {
        let counter = Arc::new(AtomicUsize::new(0));
        let mut tasks = Vec::new();

        for _ in 0..self.config.num_mcts_workers {
            let root = self.root.clone();
            let evaluator = self.evaluator.clone();
            let config = self.config.clone();
            let sem = self.inference_sem.clone();
            let counter = counter.clone();

            let handle = tokio::spawn(async move {
                loop {
                    // 检查是否达到模拟次数限制
                    let current_count = counter.load(Ordering::Relaxed);
                    if current_count >= config.num_simulations {
                        break;
                    }

                    // 执行一次模拟
                    let (cost, _) = Self::simulate(
                        root.clone(),
                        None,
                        &evaluator,
                        &config,
                        &sem,
                    ).await;

                    // 增加计数 (使用本次模拟产生的 cost，通常为 1，机会节点展开时可能 > 1)
                    counter.fetch_add(cost, Ordering::Relaxed);
                }
            });
            tasks.push(handle);
        }

        // 等待所有任务完成
        join_all(tasks).await;

        // 搜索结束，从根节点选择访问次数最多的动作
        let root = self.root.read().await;
        root.children
            .iter()
            .max_by_key(|(_, child_arc)| {
                // 这里需要短暂获取子节点锁来读取 visit_count
                // 由于此时没有写操作了，使用 blocking_read 或者 try_read 都可以，这里用 try_read 避免死锁风险
                // 但在 async fn 中最好还是 await read
                // 为了简单，我们假设 run 结束时没有 contention
                 if let Ok(guard) = child_arc.try_read() {
                     guard.visit_count
                 } else {
                     0
                 }
            })
            .map(|(action, _)| *action)
    }

    /// 递归模拟函数 (Async)
    ///
    /// # 参数
    /// - `node_arc`: 当前节点的 Arc<RwLock>
    /// - `incoming_action`: 进入该节点的前置动作
    /// - `sem`: 推理信号量
    ///
    /// # 返回
    /// (cost, value)
    fn simulate<'a>(
        node_arc: Arc<RwLock<MctsNode>>,
        incoming_action: Option<usize>,
        evaluator: &'a Arc<E>,
        config: &'a MCTSConfig,
        sem: &'a Arc<Semaphore>,
    ) -> BoxFuture<'a, (usize, f32)> {
        Box::pin(async move {
            // 1. 获取基本信息 (Read Lock)
            // 尽可能短地持有锁
            let (is_chance_node, is_expanded, is_root_node, env_clone, player) = {
                let node = node_arc.read().await;
                let env = node.env.as_ref().expect("Node must have env").as_ref().clone();
                (node.is_chance_node, node.is_expanded, node.is_root_node, env, node.player())
            };

            // 检查游戏结束
            let mut masks = vec![0; ACTION_SPACE_SIZE];
            env_clone.action_masks_into(&mut masks);
            if masks.iter().all(|&x| x == 0) {
                // 游戏结束，更新统计 (Write Lock)
                let mut node = node_arc.write().await;
                node.visit_count += 1;
                node.value_sum += -1.0;
                return (1, -1.0);
            }

            // ========================================================================
            // Case A: Chance Node (机会节点)
            // ========================================================================
            if is_chance_node {
                let reveal_pos = incoming_action.expect("Chance node must have incoming action");

                // --- 1. 未扩展：全量展开 ---
                if !is_expanded {
                    // 统计剩余隐藏棋子 (逻辑未变)
                    let mut counts = [0; 14];
                    for p in &env_clone.hidden_pieces {
                        counts[get_outcome_id(p)] += 1;
                    }
                    let total_hidden = env_clone.hidden_pieces.len() as f32;
                    
                    // 准备所有可能的子环境 (Outcome)
                    let mut tasks = Vec::new();
                    let mut outcome_ids = Vec::new();
                    let mut probs = Vec::new();

                    for outcome_id in 0..14 {
                        if counts[outcome_id] > 0 {
                            let prob = counts[outcome_id] as f32 / total_hidden;
                            
                            let mut next_env = env_clone.clone();
                            let specific_piece = next_env
                                .hidden_pieces
                                .iter()
                                .find(|p| get_outcome_id(p) == outcome_id)
                                .expect("Piece not found")
                                .clone();
                            let _ = next_env.step(reveal_pos, Some(specific_piece));
                            
                            // 创建子节点（非根节点）
                            let child_node = Arc::new(RwLock::new(
                                MctsNode::new(1.0, false, Some(next_env), false)
                            ));
                            
                            tasks.push(child_node.clone());
                            outcome_ids.push(outcome_id);
                            probs.push(prob);
                        }
                    }

                    // 并发递归模拟所有子节点 (Fully Expand & Simulate Children)
                    // 这里我们使用了 futures::future::join_all 实现并发
                    let futures = tasks.iter().map(|child| {
                        Self::simulate(child.clone(), None, evaluator, config, sem)
                    });
                    let results = join_all(futures).await;

                    let mut total_eval_cost = 0;
                    let mut total_weighted_value = 0.0;
                    let mut outcomes_map = HashMap::new();

                    for (i, (cost, val)) in results.into_iter().enumerate() {
                        total_eval_cost += cost;
                        let child_node: &Arc<RwLock<MctsNode>> = &tasks[i];
                        let prob = probs[i];
                        
                        // 获取子节点 Player 用于价值对齐
                        let child_player = {
                            child_node.read().await.player()
                        };

                        let aligned_value = Self::value_from_child_perspective(
                            player,
                            child_player,
                            val,
                        );
                        total_weighted_value += prob * aligned_value;
                        
                        outcomes_map.insert(outcome_ids[i], (prob, child_node.clone()));
                    }

                    // 获取 Write Lock 更新当前节点
                    let mut node = node_arc.write().await;
                    node.is_expanded = true;
                    node.possible_states = outcomes_map;
                    node.visit_count += 1;
                    node.value_sum += total_weighted_value;

                    return (total_eval_cost, total_weighted_value);
                } 
                
                // --- 2. 已扩展：并发遍历所有分支 ---
                // 获取所有子分支 (Read Lock)
                let outcomes: Vec<(f32, Arc<RwLock<MctsNode>>)> = {
                    let node = node_arc.read().await;
                    node.possible_states.values().cloned().collect()
                };

                // 并发模拟
                let futures = outcomes.iter().map(|(_, child)| {
                    Self::simulate(child.clone(), None, evaluator, config, sem)
                });
                let results = join_all(futures).await;

                let mut total_cost = 0;
                let mut total_weighted_value = 0.0;

                for (i, (cost, val)) in results.into_iter().enumerate() {
                    total_cost += cost;
                    let (prob, child_node): &(f32, Arc<RwLock<MctsNode>>) = &outcomes[i];
                    let child_player = child_node.read().await.player();
                    
                    let aligned_value = Self::value_from_child_perspective(
                        player,
                        child_player,
                        val,
                    );
                    total_weighted_value += prob * aligned_value;
                }

                // 更新统计 (Write Lock)
                let mut node = node_arc.write().await;
                node.visit_count += 1;
                node.value_sum += total_weighted_value;

                return (total_cost, total_weighted_value);
            }

            // ========================================================================
            // Case B: State Node (决策节点)
            // ========================================================================
            
            // --- 1. 扩展 (Expansion) ---
            if !is_expanded {
                // 使用 Semaphore 限制并发推理
                let _permit = sem.acquire().await.expect("Semaphore closed");
                let (mut policy_probs, value) = evaluator.evaluate(&env_clone).await;
                drop(_permit); // 显式释放许可 (虽然离开作用域也会释放)

                // Dirichlet 噪声 (仅在根节点且训练模式)
                // 使用 is_root_node 属性来判断是否为根节点
                if config.train && is_root_node {
                    Self::apply_dirichlet_noise(&mut policy_probs, &masks, config);
                }

                // 创建子节点
                let mut new_children = HashMap::new();
                for (action_idx, &mask) in masks.iter().enumerate() {
                    if mask == 1 {
                        let prior = policy_probs[action_idx];
                        let target_is_hidden = matches!(env_clone.get_target_slot(action_idx), Slot::Hidden);
                        
                        let child_env = if target_is_hidden {
                            Some(env_clone.clone())
                        } else {
                            let mut t = env_clone.clone();
                            let _ = t.step(action_idx, None);
                            Some(t)
                        };

                        // 创建子节点（非根节点）
                        let child_node = Arc::new(RwLock::new(
                            MctsNode::new(prior, target_is_hidden, child_env, false)
                        ));
                        new_children.insert(action_idx, child_node);
                    }
                }

                // 更新节点 (Write Lock)
                let mut node = node_arc.write().await;
                node.is_expanded = true;
                node.children = new_children;
                node.visit_count += 1;
                node.value_sum += value;

                return (1, value);
            }

            // --- 2. 选择 (Selection with Virtual Loss) ---
            let (action, best_child_arc) = {
                // 获取写锁：因为我们要应用 Virtual Loss，必须修改 node 状态
                let mut node = node_arc.write().await;
                let sqrt_total_visits = ((node.visit_count + node.virtual_loss) as f32).sqrt();
                
                let mut best_action = None;
                let mut best_score = f32::NEG_INFINITY;
                let mut best_child_arc = None;

                for (&act, child_arc) in &node.children {
                    // 读取子节点信息 (Read Lock)
                    let (child_q, child_visits, child_prior, child_player) = {
                        let c = child_arc.read().await;
                        // 子节点 Q 值计算也应包含其自身的 Virtual Loss (通常)
                        // 这里我们使用 q_value() 封装
                        (c.q_value(), c.visit_count + c.virtual_loss, c.prior, c.player())
                    };

                    let adjusted_q = Self::value_from_child_perspective(player, child_player, child_q);
                    
                    // PUCT: Q + c * P * sqrt(N_parent) / (1 + N_child)
                    let u_score = config.cpuct * child_prior * sqrt_total_visits / (1.0 + child_visits as f32);
                    // 减去 Virtual Loss * 系数? 
                    // 传统的 Virtual Loss 做法是：在访问计数上加，这会降低 U term。
                    // 同时通常会暂时降低 value_sum (视为输)，降低 Q term。
                    // 简化版：仅增加 visit_count (parent & child) 就足以产生惩罚效果。
                    
                    let score = adjusted_q + u_score;

                    if score > best_score {
                        best_score = score;
                        best_action = Some(act);
                        best_child_arc = Some(child_arc.clone());
                    }
                }

                // 应用 Virtual Loss
                node.virtual_loss += config.virtual_loss as u32;
                if let Some(child) = &best_child_arc {
                    let mut c = child.write().await;
                    c.virtual_loss += config.virtual_loss as u32;
                }

                (best_action.expect("No valid child"), best_child_arc.expect("No valid child"))
            };

            // --- 3. 递归 (Recursion) ---
            // 此时已释放父节点锁，允许其他线程访问父节点
            let (cost, child_v) = Self::simulate(best_child_arc.clone(), Some(action), evaluator, config, sem).await;

            // --- 4. 反向传播 (Backpropagation & Remove Virtual Loss) ---
            let child_player = best_child_arc.read().await.player();
            let my_value = Self::value_from_child_perspective(player, child_player, child_v);

            // 更新当前节点 (Write Lock)
            let mut node = node_arc.write().await;
            
            // 移除 Virtual Loss
            if node.virtual_loss >= config.virtual_loss as u32 {
                node.virtual_loss -= config.virtual_loss as u32;
            } else {
                node.virtual_loss = 0; // 防御性处理
            }
            
            // 也要移除子节点的 Virtual Loss
            {
                let mut c = best_child_arc.write().await;
                if c.virtual_loss >= config.virtual_loss as u32 {
                    c.virtual_loss -= config.virtual_loss as u32;
                } else {
                    c.virtual_loss = 0;
                }
            }

            node.visit_count += 1;
            node.value_sum += my_value;

            (cost, my_value)
        })
    }

    /// 获取根节点的访问概率分布（即策略 π）
    pub async fn get_root_probabilities(&self) -> Vec<f32> {
        let mut probs = vec![0.0; ACTION_SPACE_SIZE];
        let root = self.root.read().await;
        let total = root.visit_count as f32;
        if total == 0.0 {
            return probs;
        }

        for (&action, child_arc) in &root.children {
            if action < probs.len() {
                let child = child_arc.read().await;
                probs[action] = child.visit_count as f32 / total;
            }
        }
        probs
    }

    /// 价值视角转换辅助函数
    fn value_from_child_perspective(
        parent_player: Player,
        child_player: Player,
        child_value: f32,
    ) -> f32 {
        if parent_player == child_player {
            child_value
        } else {
            -child_value
        }
    }
}