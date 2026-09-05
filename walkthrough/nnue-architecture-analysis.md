# Pikafish NNUE 架构分析：MoE 核查与 Bucketed Layer Stacks 收益评估

> 分析对象：`src/nnue`
> 核心问题：① 是否使用了 MoE 架构？② 若无，实际采用的「多模型硬路由」设计价值如何？
> 结论：**未使用 MoE**；实际为「共享特征变换器 + 按子力配置硬分桶的 N 个独立头部」，收益极高而成本近乎为零。

---

## 0. 结论摘要

| 问题 | 结论 |
|---|---|
| 是否使用 MoE | **否**。无门控网络、无可学习路由、无 Top-k 加权组合 |
| 实际架构 | `HalfKAv2_hm + FullThreats` 稀疏特征 → **共享** FeatureTransformer(1024维) → 按**子力配置**硬选 16 个独立稠密 MLP 之一 → 标量输出 |
| 易混淆点 | 存在**两套语义无关的分桶机制**（24 路特征桶 vs 16 路 layer stack），极易误判为 MoE |
| 分桶的内存成本 | 16 个 head 共 ≈ **0.535 MiB**，仅占全网 **0.83%** |
| 分桶的推理成本 | **零**。每次 eval 只跑 1 个 head，路由为查表 |
| 收益本质 | 用 1/16 的推理算力，买到 16 倍的头部容量 |
| 成立前提 | **头部必须远小于特征变换器**。若 `L2` 从 32 放大到 512，设计收益崩塌 |

---

## 1. 架构全景

### 1.1 目录结构

```
src/nnue/
├── features/
│   ├── half_ka_v2_hm.{h,cpp}     # PSQ 特征集（子力位置 + 将的位置）
│   └── full_threats.{h,cpp}      # 威胁特征集（攻击关系）
├── layers/
│   ├── affine_transform.h                  # 稠密全连接
│   ├── affine_transform_sparse_input.h     # 稀疏输入全连接（fc_0）
│   ├── clipped_relu.h                      # 无参数
│   └── sqr_clipped_relu.h                  # 无参数
├── network.{h,cpp}               # 网络加载 / 评估入口
├── nnue_accumulator.{h,cpp}      # 增量累加器
├── nnue_architecture.h           # ★ 网络结构定义
├── nnue_feature_transformer.h    # ★ 特征变换器
├── nnue_common.h                 # 类型与常量
├── nnz_helper.h                  # 稀疏前向优化
└── simd.h
```

`layers/` 下**只有 4 个层实现，没有任何门控/路由层** —— 这是排除 MoE 的第一条硬证据。

### 1.2 数据流

```
Position
   │
   ├─ make_feature_bucket()      → 将位置(6) × 攻击(4) = 24 → 决定累加器权重行
   │                                 ↓
   ├──────────────────────► Accumulator（1024维，增量更新，★所有 layer stack 共享）
   │                                 ↓
   ├─ make_layer_stack_bucket()  → 车/马炮数量 → 决定 ①PSQT 切片  ②选用哪个 head
   │                                 ↓
   │                          transformedFeatures (1024维, u8)
   │                                 ↓
   │                          network[bucket].propagate(...)   ← 只跑 1 个
   │                                 ↓
   │                          fc_0(1024→32) → [sqr|clip] → fc_1(64→32) → [sqr|clip] → fc_2(128→1)
   │                                 ↓
   └──────────────────────► psqt(子力项) + positional(位置项) → 最终评估值
```

### 1.3 维度常量

```43:48:src/nnue/nnue_architecture.h
constexpr IndexType L1 = 1024;
constexpr int       L2 = 32;
constexpr int       L3 = 32;

constexpr IndexType PSQTBuckets = 16;
constexpr IndexType LayerStacks = 16;
```

---

## 2. MoE 核查：逐项核验

MoE 的三个必要条件，逐条比对：

| MoE 必要条件 | 本实现 | 证据 |
|---|---|---|
| **门控/路由网络** | ✗ 无 | `layers/` 仅 4 个层；全目录 grep `moe\|expert\|gate\|router\|topk\|softmax` 命中 0（其余 `switch` 命中是 C++ 关键字） |
| **可学习路由权重** | ✗ 无 | 权重文件仅读写 `fc_0/ac_0/fc_1/ac_1/fc_2`，无路由参数；路由键为编译期常量表 |
| **Top-k 加权组合多专家** | ✗ 无 | 严格 Top-1，其余 15 个 head 完全不参与 |

网络主干为**单条稠密路径**：

```57:68:src/nnue/nnue_architecture.h
struct NetworkArchitecture {
    static constexpr IndexType TransformedFeatureDimensions = L1;
    static constexpr int       FC_0_OUTPUTS                 = L2;
    static constexpr int       FC_1_OUTPUTS                 = L3;

    Layers::AffineTransformSparseInput<TransformedFeatureDimensions, FC_0_OUTPUTS> fc_0;
    Layers::SqrClippedReLU<FC_0_OUTPUTS, WeightScaleBits + 1>                      ac_sqr_0;
    Layers::ClippedReLU<FC_0_OUTPUTS, WeightScaleBits + 1>                         ac_0;
    Layers::AffineTransform<FC_0_OUTPUTS * 2, FC_1_OUTPUTS>                        fc_1;
    Layers::SqrClippedReLU<FC_1_OUTPUTS, WeightScaleBits>                          ac_sqr_1;
    Layers::ClippedReLU<FC_1_OUTPUTS, WeightScaleBits>                             ac_1;
    Layers::AffineTransform<FC_0_OUTPUTS * 2 + FC_1_OUTPUTS * 2, 1>                fc_2;
```

即 `1024 → 32 → 32 → 1`。激活层确认无参数：

```57:59:src/nnue/layers/clipped_relu.h
    // Read network parameters
    bool read_parameters(std::istream&) { return true; }
```

---

## 3. ★ 两套分桶机制辨析（最易误判处）

代码中存在**两套语义完全无关的分桶**，桶数也不同。混淆二者是误判为 MoE 的主因。

> **重要澄清**：`PSQTBuckets = 16` **只属于 layer stack 一侧**（用于 `psqtAccumulation` 索引），与特征桶数**无关**。特征桶数为 24（6 将桶 × 4 攻击桶），二者数值不等。

| | **Feature Bucket** | **Layer Stack Bucket** |
|---|---|---|
| 桶数 | **24**（6 将桶 × 4 攻击桶） | **16**（`LayerStacks` / `PSQTBuckets`） |
| 路由键 | 将的位置 + 攻击关系 | **子力配置**（车/马炮数量） |
| 计算函数 | `make_feature_bucket()` | `make_layer_stack_bucket()` |
| 作用位置 | 特征索引 → 决定累加器用哪组权重行 | ① PSQT 切片 ② 选用哪个 head |
| 是否可学习 | 否 | 否 |
| 与 MoE 关系 | 无（输入特征工程） | 无（硬路由条件计算） |

### 3.1 Feature Bucket：输入特征分桶

```64:72:src/nnue/features/half_ka_v2_hm.cpp
std::tuple<int, bool, int> HalfKAv2_hm::make_feature_bucket(Color           perspective,
                                                            const Position& pos) {
    const Square ksq           = pos.king_square(perspective);
    const Square oksq          = pos.king_square(~perspective);
    auto [king_bucket, mirror] = KingBuckets[ksq][oksq][requires_mid_mirror(pos, perspective)];
    auto attack_bucket         = make_attack_bucket(pos, perspective);
    auto bucket                = king_bucket * 4 + attack_bucket;

    return {bucket, mirror, attack_bucket};
}
```

`king_bucket` 的取值范围由 `KingBuckets` 表决定，其取值集合为 `{0,1,2,3,4,5}`，即 **6 个将桶**：

```83:88:src/nnue/features/half_ka_v2_hm.h
        constexpr u8 KingBuckets[SQUARE_NB] = {
          // clang-format off
          0,  0,  0,  0,  1, M(0),  0,  0,  0,
          0,  0,  0,  2,  3, M(2),  0,  0,  0,
          0,  0,  0,  4,  5, M(4),  0,  0,  0,
          0,  0,  0,  0,  0,   0 ,  0,  0,  0,
```

这与 `Dimensions` 的定义一致：

```54:54:src/nnue/features/half_ka_v2_hm.h
    static constexpr IndexType Dimensions = 6 * AttackBucketNB * PS_NB;   // 6*4*689 = 16,536
```

即 **6 将桶 × 4 攻击桶 = 24** 种 feature bucket，决定稀疏特征的索引偏移与镜像。属于**特征工程**，与专家路由无关。

### 3.2 Layer Stack Bucket：按子力配置硬路由

```76:88:src/nnue/features/half_ka_v2_hm.cpp
IndexType HalfKAv2_hm::make_layer_stack_bucket(const Position& pos) {
    static constexpr auto LayerStackBuckets = [] {
        MultiArray<u8, 3, 3, 5, 5> v{};
        for (u8 us_rook = 0; us_rook <= 2; ++us_rook)
            for (u8 opp_rook = 0; opp_rook <= 2; ++opp_rook)
                for (u8 us_knight_cannon = 0; us_knight_cannon <= 4; ++us_knight_cannon)
                    for (u8 opp_knight_cannon = 0; opp_knight_cannon <= 4; ++opp_knight_cannon)
                        v[us_rook][opp_rook][us_knight_cannon][opp_knight_cannon] = [&] {
                            if (us_rook == opp_rook)
                                return us_rook * 4
                                     + int(us_knight_cannon + opp_knight_cannon >= 4) * 2
                                     + int(us_knight_cannon == opp_knight_cannon);
                            else if (us_rook == 2 && opp_rook == 1)
```

**路由键是 4 个棋子计数**（我方车数、对方车数、我方马炮数、对方马炮数），**不含任何位置信息**，且为编译期常量表。

### 3.3 关键：分桶只切在极小的头部

`bucket` 参数进入 `transform()` 后，**仅用于 PSQT 切片**：

```204:222:src/nnue/nnue_feature_transformer.h
    i32 transform(const Position&                             pos,
                  AccumulatorStack&                           accumulatorStack,
                  AccumulatorCaches&                          cache,
                  OutputType*                                 output,
                  int                                         bucket,
                  [[maybe_unused]] NNZInfo<OutputDimensions>& nnzInfo) const {
        accumulatorStack.evaluate(pos, *this, cache);
        const auto& accumulatorState = accumulatorStack.latest();

        const Color perspectives[2]  = {pos.side_to_move(), ~pos.side_to_move()};
        const auto& psqtAccumulation = accumulatorState.psqtAccumulation;
        const auto  psqt =
          (psqtAccumulation[perspectives[0]][bucket] - psqtAccumulation[perspectives[1]][bucket])
          / 2;

        const auto& accumulation = accumulatorState.accumulation;

        for (IndexType p = 0; p < 2; ++p)
            transform_perspective(accumulation[perspectives[p]], output, p, nnzInfo);
```

昂贵的 1024 维 `accumulation` **完全不依赖 bucket**，由全部 16 个 layer stack 共享，且在搜索中增量更新。

累加器结构印证 PSQT 按 layer stack 分桶：

```45:49:src/nnue/nnue_accumulator.h
struct alignas(CacheLineSize) Accumulator {
    std::array<std::array<i16, L1>, COLOR_NB>          accumulation;
    std::array<std::array<i32, PSQTBuckets>, COLOR_NB> psqtAccumulation;
    std::array<bool, COLOR_NB>                         computed = {};
};
```

### 3.4 推理时只跑 1 个 head

```112:126:src/nnue/network.cpp
NetworkOutput Network::evaluate(const Position&    pos,
                                AccumulatorStack&  accumulatorStack,
                                AccumulatorCaches& cache) const {

    constexpr u64 alignment = CacheLineSize;

    alignas(alignment) TransformedFeatureType transformedFeatures[FeatureTransformer::BufferSize];

    ASSERT_ALIGNED(transformedFeatures, alignment);

    NNZInfo<L1> nnzInfo;

    const int  bucket     = PSQFeatureSet::make_layer_stack_bucket(pos);
    const auto psqt       = featureTransformer.transform(pos, accumulatorStack, cache,
                                                         transformedFeatures, bucket, nnzInfo);
```

> **注意**：`trace_evaluate()` 会循环全部 16 个 bucket，但那是调试输出专用路径，不在搜索热路径上。

```186:197:src/nnue/network.cpp
    NnueEvalTrace t{};
    t.correctBucket = PSQFeatureSet::make_layer_stack_bucket(pos);
    for (IndexType bucket = 0; bucket < LayerStacks; ++bucket)
    {
        NNZInfo<L1> nnzInfo;
        const auto materialist = featureTransformer.transform(pos, accumulatorStack, cache,
                                                              transformedFeatures, bucket, nnzInfo);
        const auto positional  = network[bucket].propagate(transformedFeatures, nnzInfo);

        t.psqt[bucket]       = static_cast<Value>(materialist / OutputScale);
        t.positional[bucket] = static_cast<Value>(positional / OutputScale);
    }
```

---

## 4. 成本量化

### 4.1 数据类型与输入维度

```58:62:src/nnue/nnue_common.h
using BiasType         = i16;
using ThreatWeightType = i8;
using WeightType       = i8;
using PSQTWeightType   = i32;
using IndexType        = u32;
```

```54:54:src/nnue/features/half_ka_v2_hm.h
    static constexpr IndexType Dimensions = 6 * AttackBucketNB * PS_NB;   // 6*4*689 = 16,536
```
```37:37:src/nnue/features/full_threats.h
    static constexpr IndexType Dimensions = 45547;
```

```89:103:src/nnue/nnue_feature_transformer.h
    static constexpr IndexType ThreatInputDimensions = ThreatFeatureSet::Dimensions;
    static constexpr IndexType PsqDimensions         = PSQFeatureSet::Dimensions;
    static constexpr IndexType InputDimensions       = PsqDimensions + ThreatInputDimensions;
    static constexpr IndexType OutputDimensions      = HalfDimensions;
    static constexpr IndexType ThreatWeightSize      = ThreatInputDimensions * HalfDimensions;
    static constexpr IndexType ThreatPsqtWeightSize  = ThreatInputDimensions * PSQTBuckets;

    using BiasesArray       = std::array<BiasType, HalfDimensions>;
    using WeightArray       = std::array<WeightType, HalfDimensions * PsqDimensions>;
    using ThreatWeightArray = std::array<ThreatWeightType, ThreatWeightSize>;
    using PsqtWeightArray   = std::array<PSQTWeightType, PSQTBuckets * PsqDimensions>;
    using ThreatPsqtArray   = std::array<PSQTWeightType, ThreatPsqtWeightSize>;

    // Size of forward propagation buffer
    static constexpr usize BufferSize = OutputDimensions * sizeof(OutputType);
```

### 4.2 内存开销测算

**特征变换器（共享，仅 1 份）**

| 数组 | 元素数 × 类型 | 字节 | MiB |
|---|---|---|---|
| `weights` | 1024 × 16536 × i8 | 16,932,864 | 16.15 |
| `threatWeights` | 45547 × 1024 × i8 | 46,640,128 | 44.48 |
| `psqtWeights` | 16 × 16536 × i32 | 1,058,304 | 1.01 |
| `threatPsqtWeights` | 45547 × 16 × i32 | 2,915,008 | 2.78 |
| `biases` | 1024 × i16 | 2,048 | ~0 |
| **合计** | | **67,548,352** | **≈ 64.42** |

**单个 head（NetworkArchitecture）**

| 层 | 权重 | 偏置 | 小计 |
|---|---|---|---|
| `fc_0` (1024→32) | 32 × 1024 × i8 = 32,768 | 32 × i16 = 64 | 32,832 |
| `fc_1` (64→32) | 32 × 64 × i8 = 2,048 | 32 × i16 = 64 | 2,112 |
| `fc_2` (128→1) | 1 × 128 × i8 = 128 | 1 × i16 = 2 | 130 |
| `ac_*` ×4 | 0 | 0 | 0 |
| **单 head 合计** | | | **35,074 B ≈ 34.25 KiB** |

（`fc_0` 的 `PaddedInputDimensions = ceil_to_multiple(1024, 32) = 1024`）

**16 个 head = 561,184 B ≈ 0.535 MiB**

> ### 关键比例
> **16 个 head 仅占全网 `561,184 / 67,548,352` ≈ 0.83%**

### 4.3 三项成本汇总

| 成本维度 | 开销 | 说明 |
|---|---|---|
| **推理** | **零** | 每次 eval 只跑 1 个 head；路由为 `MultiArray<u8,3,3,5,5>` 查表，输入是 4 个棋子计数（可增量维护） |
| **内存** | **+0.83%** | 且任一时刻活跃仅 34.25 KiB，稳驻 L1/L2 |
| **训练** | **几乎不变** | 占算力绝大多数的共享特征变换器仍在 100% 数据上训练；head 仅 3.5 万参数，1/16 数据量绰绰有余 |

---

## 5. 收益机制

### 5.1 核心：用 1/16 算力买到 16 倍头部容量

head 极小（`L2 = L3 = 32`）。单个 32 维瓶颈层要同时容纳「双车残局」「车马炮对攻」「马炮残局」等**性质完全不同**的估值逻辑，容量严重不足。

| | **16 × 32 分桶** | **单层 512 稠密** |
|---|---|---|
| 总 head 容量 | 512 隐单元 | 512 隐单元 |
| 每次推理 MACs | 1024×32 + 64×32 + 128 ≈ **3.5 万** | 1024×512 + 1024×32 + 1088 ≈ **55.8 万** |
| head 内存 | 0.535 MiB | ≈ 0.5 MiB |
| 单 head 专业化 | 完全专精一种子力配置 | 必须兼顾全部 |

**同样内存、同样总容量，推理算力只要约 1/16。**

这正是 MoE 想要的条件计算收益，但路由是免费的手写函数 —— 无需门控网络、无需负载均衡损失、无额外推理开销。

### 5.2 路由键是极强的领域先验

「子力配置决定局面性质」是象棋的核心先验：双方车/马炮数量基本决定了局面的性质与胜负判据。用领域知识直接给定分区，比让门控网络去学，既省成本又更可控、可解释。

---

## 6. 与替代方案对比

### 6.1 硬路由 vs 可学习 MoE

| | MoE（学习路由） | 本设计（硬路由） |
|---|---|---|
| 路由参数 | 需训练，占权重文件 | 编译期常量，0 字节 |
| 路由推理开销 | 一次前向 + softmax/topk | 一次查表 |
| 训练复杂度 | 需负载均衡损失、易不稳定 | 无 |
| 可解释性 | 路由语义难解释 | 语义明确（子力配置） |
| 组合方式 | Top-k 加权 | Top-1 |

在象棋场景下，学习路由大概率会重新发现「子力配置」这个分区，却要付出门控参数、训练不稳定、推理开销的代价。**用不可学习的领域知识换掉可学习路由，是本设计的精髓而非缺陷。**

### 6.2 硬路由 vs 单层大 head

分桶方案在「内存相同、总容量相同」下，**推理算力仅需 1/16**，且每个 head 完全专精。只要子力分区是好的归纳偏置，分桶方案严格更优。

---

## 7. 风险与适用边界

### 7.1 分桶边界的估值跳变 —— 已被巧妙规避

分桶键全为**子力计数**（`us_rook` / `opp_rook` / `us_knight_cannon` / `opp_knight_cannon`）。象棋中子力只在吃子时改变，而吃子节点的估值本来就剧烈跳变。**把不连续点藏在「本来就不连续」的地方，额外代价被掩盖。**

> 这是本设计**最关键的选择**：若按位置特征分桶，连续行棋就可能跨桶，估值抖动会直接破坏搜索的平滑性假设。

### 7.2 数据碎片化

稀有子力配置的桶样本量小，可能训练不足。缓解手段：`LayerStackBuckets` 采用粗量化（车数上限 2、马炮数上限 4），保证每个桶足够宽。

### 7.3 桶函数必须与训练器严格一致

引擎与训练器的 `make_layer_stack_bucket` 任何偏差都会**静默产生错配的网络**。目前依赖 `get_hash_value()` 与权重文件结构做校验，属于工程脆弱点，无编译期保证。

### 7.4 桶数并非越多越好

收益源于「子力配置决定局面性质」这一先验，边际收益随桶数快速衰减；而数据碎片化与边界跳变风险线性上升。`LayerStacks = 16` 是经验甜点，继续翻倍大概率得不偿失。

### 7.5 ★ 成立前提（最重要）

**该设计的价值高度依赖「头部必须远小于特征变换器」。**

- 当前：head 占 0.83%，复制 16 份 ≈ 免费
- 若把 `L2` 从 32 放大到 512：head 与特征变换器同量级，16 倍复制立刻变成 16 倍内存开销，收益瞬间崩塌

它成立的前提，正是 NNUE 中「**昂贵的共享表示 + 极廉价的专用读出**」这一结构特点。

---

## 8. 总体判断

**收益/成本比极高，属于「几乎无成本的纯收益」。** 按重要性排序：

1. **分桶位置选得极准** —— 切在占总模型 0.83% 的头部，共享了占 99.17% 的昂贵特征变换器
2. **推理零开销** —— 这是硬路由相对稠密大 head 的决定性优势
3. **路由键是象棋核心先验**，且跳变被刻意藏在吃子节点

> **一句话总结**：这不是通用架构创新，而是把 NNUE 的结构特点（头小身大）与象棋的领域先验（子力决定局面）**同时利用到极致**的实现技巧。收益真实且巨大，但**不可迁移到「头大」的模型上**。

---

## 9. 关键代码位置索引

| 关注点 | 位置 |
|---|---|
| 网络结构定义 | `src/nnue/nnue_architecture.h:57-161` |
| 维度常量 | `src/nnue/nnue_architecture.h:43-48` |
| 前向传播 | `src/nnue/nnue_architecture.h:101-147` |
| 评估入口（单 head） | `src/nnue/network.cpp:112-126` |
| 调试评估（全 16 head） | `src/nnue/network.cpp:176-200` |
| 权重读写 | `src/nnue/network.cpp:284-306` |
| 特征变换器 | `src/nnue/nnue_feature_transformer.h:80-222` |
| `transform()`（分桶切点） | `src/nnue/nnue_feature_transformer.h:204-222` |
| 累加器结构 | `src/nnue/nnue_accumulator.h:45-49` |
| Feature Bucket | `src/nnue/features/half_ka_v2_hm.cpp:64-73` |
| Layer Stack Bucket | `src/nnue/features/half_ka_v2_hm.cpp:76-88` |
| PSQ 特征维度 | `src/nnue/features/half_ka_v2_hm.h:48-54` |
| 将桶表（6 桶） | `src/nnue/features/half_ka_v2_hm.h:80-114` |
| 威胁特征维度 | `src/nnue/features/full_threats.h:37` |
| 数据类型定义 | `src/nnue/nnue_common.h:58-62` |
| 激活层无参数 | `src/nnue/layers/clipped_relu.h:57-59`、`src/nnue/layers/sqr_clipped_relu.h:58-60` |
| 全连接层定义 | `src/nnue/layers/affine_transform.h:126-141` |
| 稀疏全连接层定义 | `src/nnue/layers/affine_transform_sparse_input.h:44-68` |
| 网络哈希校验 | `src/nnue/nnue_architecture.h:71-85` |
| 网络大小打印 | `src/nnue/network.cpp:163-172` |

---

## 10. 附：复现核查方法

```bash
# 1. 排除 MoE：全目录搜索门控/路由关键词
grep -rniE "moe|mixture.of.experts|expert|router|topk|top_k|softmax" src/nnue/

# 2. 确认 layers/ 下无门控层
ls src/nnue/layers/

# 3. 确认推理只跑 1 个 head
sed -n '112,126p' src/nnue/network.cpp

# 4. 确认分桶只影响 PSQT，不影响 1024 维共享特征
sed -n '204,222p' src/nnue/nnue_feature_transformer.h

# 5. 确认激活层无参数
grep -n "read_parameters" src/nnue/layers/*.h
```

实际运行时可观察 UCI 启动日志打印的网络尺寸，与本文 §4.2 的 64.42 MiB 估算交叉验证：

```163:172:src/nnue/network.cpp
    if (f)
    {
        usize size = sizeof(featureTransformer) + sizeof(NetworkArchitecture) * LayerStacks;
        f("NNUE evaluation using " + evalfilePath.string() + " ("
          + std::to_string(size / (1024 * 1024)) + "MiB, ("
          + std::to_string(featureTransformer.InputDimensions) + ", "
          + std::to_string(network[0].TransformedFeatureDimensions) + ", "
          + std::to_string(network[0].FC_0_OUTPUTS) + ", " + std::to_string(network[0].FC_1_OUTPUTS)
          + ", 1))");
    }
```
