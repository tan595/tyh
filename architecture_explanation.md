# Reg2ST 改进方案详解：多分辨率空间建模 + HECLIP 对比学习

## 目录

1. [原始 Reg2ST 架构](#1-原始-reg2st-架构)
2. [改进后架构](#2-改进后架构)
3. [改动对比总结](#3-改动对比总结)
4. [模块详解：多分辨率空间建模 (Multi-Resolution)](#4-模块详解多分辨率空间建模)
5. [模块详解：HECLIP Mix 对比学习](#5-模块详解heclip-mix-对比学习)
6. [数据格式说明](#6-数据格式说明)
7. [消融实验结果](#7-消融实验结果)
8. [背景知识补充](#8-背景知识补充)

---

## 1. 原始 Reg2ST 架构

### 架构图（ASCII）

```
输入数据
├── image features: [N, 1024]     (Phikon-v2 预提取)
├── gene expression: [N, D_out]   (log-normalized)
├── spatial position: [N, 2]      (离散坐标 0~63)
├── ori counts: [N, D_out]        (原始计数，用于 ZINB)
└── scale factors: [N]            (库大小归一化因子)

                    ┌─────────────────────────┐
  image [N,1024] ──>│  Image Encoder           │──> i_f [N,1024]
                    │  Linear→GELU→Drop→Linear │
                    └────────────┬─────────────┘
                                 │
                    ┌────────────▼─────────────┐
                    │  Gene Projection          │──> proj_i_f [N,1024]
                    │  Linear→GELU→Linear       │
                    └────────────┬─────────────┘
                                 │
  pos [N,2] ──> embed_x + embed_y ──> i_ct = i_f + x_emb + y_emb
                                 │
                                 │        ┌──────────────────────────┐
                                 │        │ Gene Encoder             │
  gene [N,D_out] ────────────────┼───────>│ Linear→GELU→Drop→Linear │──> g_f [N,1024]
                                 │        └────────────┬─────────────┘
                                 │                     │
                                 │              L2 Normalize
                                 │                     │
                    ┌────────────▼─────────────┐       │
                    │  Decoder (Cross-Attention)│       │
                    │  Q = i_ct                 │       │
                    │  K,V = proj_i_f           │       │
                    │  6 layers × 8 heads       │       │
                    │  训练时 75% mask           │       │
                    └────────────┬─────────────┘       │
                                 │                     │
                    ┌────────────▼─────────────┐       │
                    │  WiKG (动态图卷积)         │       │
                    │  head/tail projection     │       │
                    │  top-k 邻居选择           │       │
                    │  gated attention          │       │
                    │  bi-interaction 聚合      │       │
                    └────────────┬─────────────┘       │
                                 │                     │
                    ┌────────────▼─────────────┐       │
                    │  ZINB Heads               │       │
                    │  mean / disp / pi         │       │
                    └────────────┬─────────────┘       │
                                 │                     │
                    ┌────────────▼─────────────┐       │
                    │  Gene Head                │       │
                    │  Linear→GELU→Drop→Linear  │──> i_g [N,D_out]  (预测值)
                    └──────────────────────────┘       │
                                                       │
                    ┌──────────────────────────────────┘
                    │
            ========== 损失函数 ==========
            │
            ├── MSE Loss:    ||gene_truth - i_g||²
            ├── InfoNCE Loss:  0.5 × BCE(i_f @ g_f^T × scale, I)
            ├── Proj Loss:   ||g_f - proj_i_f||²
            └── ZINB Loss:   0.25 × ZINB(ori_counts, mean, disp, pi)
```

### 原始代码存在的 Bug

1. **Decoder 双重调用**（`model.py:162-167`）：训练时先用 mask 调用 decoder，但第 167 行又无条件调用了一次 decoder（无 mask），覆盖了 mask 版本的结果。这意味着 mask 机制在原始代码中实际上**没有生效**。
2. **学习率调度器硬编码** `iter_per_epoch=31`（`model.py:252`）：应该根据实际训练集大小设置。
3. **`mask.cuda()` 硬编码**（`model.py:164`）：在多 GPU 场景下可能导致设备不匹配。

---

## 2. 改进后架构

### 架构图（ASCII）

```
输入数据（同上）

                    ┌─────────────────────────┐
  image [N,1024] ──>│  Image Encoder           │──> i_f [N,1024]
                    │  Linear→GELU→Drop→Linear │
                    └────────────┬─────────────┘
                                 │
                    ┌────────────▼─────────────┐
                    │  Gene Projection          │──> proj_i_f [N,1024]
                    │  Linear→GELU→Linear       │
                    └────────────┬─────────────┘
                                 │
  pos [N,2] ──> embed_x + embed_y ──> i_ct = i_f + x_emb + y_emb
                                 │
                                 │
     ┌───────────────────────────┼───────────────────────────┐
     │                           │                           │
     │  ★ 新增模块 1a            │                           │  ★ 新增模块 1b
     │  ┌────────────────────┐   │                           │  ┌────────────────────┐
     │  │ NeighborEncoder    │   │ target_f                  │  │ GlobalEncoder       │
     │  │ (局部空间交叉注意力)│   │  = i_ct                   │  │ (全局自注意力)       │
     │  │                    │   │                           │  │                      │
     │  │ 1.KNN找k=6个空间  │   │                           │  │ 2层Transformer      │
     │  │   最近邻           │   │                           │  │ Encoder             │
     │  │ 2.cross-attn:     │   │                           │  │ 输入: 所有spot的i_f  │
     │  │   Q=当前spot       │   │                           │  │                      │
     │  │   K,V=邻居特征    │   │                           │  └──────────┬───────────┘
     │  └──────────┬─────────┘   │                           │             │
     │             │             │                           │             │
     │             ▼             ▼                           │             ▼
     │  ┌────────────────────────────────────────────────────────────────────┐
     │  │  ★ 新增模块 1c: MultiResFusion (门控融合)                          │
     │  │                                                                    │
     │  │  concat = [target_f, neighbor_f, global_f]  → [N, 3072]           │
     │  │  gate = sigmoid(Linear(concat))                                    │
     │  │  fused = LayerNorm(Linear(concat * gate))   → [N, 1024]           │
     │  │                                                                    │
     │  │  per_res_heads: 各分辨率独立预测 → distillation loss               │
     │  └────────────────────────────┬───────────────────────────────────────┘
     │                               │
     │                               ▼  fused (替代原始 i_ct 作为 Decoder 的 Q)
     │
     │              ┌────────────────▼─────────────┐
     │              │  Decoder (Cross-Attention)    │
     │              │  Q = fused (★ 此处改变)        │
     │              │  K,V = proj_i_f               │
     │              │  6 layers × 8 heads           │
     │              │  训练时 75% mask（已修复生效）  │
     │              └────────────────┬──────────────┘
     │                               │
     │              ┌────────────────▼──────────────┐
     │              │  WiKG (动态图卷积)             │
     │              └────────────────┬──────────────┘
     │                               │
     │              ┌────────────────▼──────────────┐
     │              │  ZINB Heads (mean/disp/pi)    │
     │              └────────────────┬──────────────┘
     │                               │
     │              ┌────────────────▼──────────────┐
     │              │  Gene Head → i_g [N,D_out]    │
     │              └──────────────────────────────┘

            ========== 损失函数 ==========
            │
            ├── MSE Loss:     ||gene_truth - i_g||²
            ├── Proj Loss:    ||g_f - proj_i_f||²
            ├── ZINB Loss:    w_zinb × ZINB(ori, mean, disp, pi)
            │
            ├── ★ 对比损失 (改进):
            │   ├── hard_con = InfoNCE(logits_img, logits_gene)
            │   ├── ★ heclip_con = ImageCentricSoftTarget(logits_img, i_f)
            │   └── con_loss = (1-λ) × hard_con + λ × heclip_con
            │
            └── ★ Distillation Loss: w_fusion × MSE(per_res_pred, fused_pred)
```

---

## 3. 改动对比总结

| 组件 | 原始 Reg2ST | 改进后 |
|------|------------|--------|
| **Decoder 的 Query** | `i_ct = i_f + pos_emb` | 多分辨率融合后的 `fused`（融合了局部邻域 + 全局上下文） |
| **空间建模** | 仅离散位置嵌入 (embed_x + embed_y) | 位置嵌入 + NeighborEncoder(k-NN 交叉注意力) + GlobalEncoder(全局自注意力) |
| **对比损失** | 纯 hard InfoNCE | HECLIP mix: (1-λ)×hard + λ×image-centric soft target |
| **训练 mask** | Bug：mask 未生效 | 已修复：mask 正确应用 |
| **LR 调度器** | 硬编码 iter_per_epoch=31 | 根据实际训练集大小动态设置 |
| **额外损失** | 无 | fusion distillation loss（各分辨率预测对齐融合预测） |

### 改动不涉及的部分（保持不变）

- Image Encoder 结构
- Gene Projection / Gene Head 结构
- Gene Linear (gene encoder)
- WiKG 动态图卷积
- ZINB 分布建模头 (mean/disp/pi)
- Decoder 内部结构 (6层 cross-attention)
- 数据预处理流程和数据加载

---

## 4. 模块详解：多分辨率空间建模

### 4.1 动机

原始 Reg2ST 的空间信息仅通过离散位置嵌入 (`embed_x`, `embed_y`) 引入，将 (x, y) 坐标映射为可学习向量加到图像特征上。这种方式：
- **无法建模邻域关系**：每个 spot 独立编码，不知道相邻 spot 的表达模式
- **无法捕获全局空间结构**：无法感知组织中远距离的空间模式（如区域边界、梯度变化）

### 4.2 NeighborEncoder（局部空间邻域建模）

```python
class NeighborEncoder(nn.Module):
    # 输入: query [N, dim], neighbor_feats [N, k, dim]
    # 输出: [N, dim]
```

**工作流程：**
1. 根据空间坐标用 KD-Tree 找到每个 spot 的 k=6 个最近邻
2. 提取这些邻居的图像特征 `i_f[neighbor_idx]`，得到 `[N, k, dim]`
3. 用交叉注意力聚合：Query = 当前 spot，Key/Value = 邻居特征
4. 残差连接 + FFN

**直觉**：基因表达具有空间自相关性（Tobler 地理学第一定律），近处的 spot 表达模式相似。通过交叉注意力让模型学会从邻居中提取有用的空间上下文。

### 4.3 GlobalEncoder（全局空间建模）

```python
class GlobalEncoder(nn.Module):
    # 输入: all_feats [N, dim]
    # 输出: [N, dim]
```

**工作流程：**
1. 将所有 spot 的特征 `i_f` 作为序列输入 2 层 Transformer Encoder
2. 通过自注意力机制，每个 spot 可以关注到所有其他 spot
3. LayerNorm 输出

**直觉**：捕获跨区域的长程依赖关系，例如肿瘤边界与基质区域之间的基因表达梯度。

### 4.4 MultiResFusion（门控融合）

```python
class MultiResFusion(nn.Module):
    # 输入: target_f [N,dim], neighbor_f [N,dim], global_f [N,dim]
    # 输出: fused [N,dim], per_res {3个独立预测头}
```

**工作流程：**
1. 拼接三个分辨率的特征：`[target_f, neighbor_f, global_f]` → `[N, 3×dim]`
2. 门控选择：`gate = sigmoid(Linear(concat))`，让模型学习每个分辨率的重要性
3. 投影回原维度：`fused = LayerNorm(Linear(concat × gate))` → `[N, dim]`
4. 三个独立预测头分别产出各分辨率的预测，用于 distillation loss

**Distillation Loss**：让各分辨率的独立预测对齐融合后的预测，确保每个分支都学到有意义的表示，防止某个分支退化。

---

## 5. 模块详解：HECLIP Mix 对比学习

### 5.1 原始 InfoNCE 的问题

原始 Reg2ST 使用 hard InfoNCE 对比损失：

```
labels = I (单位矩阵)
loss = BCE(logit_scale × img_feat @ gene_feat^T, I)
```

这假设：同一 spot 的 (image, gene) 是正对，所有其他 spot 都是负对。

**问题**：在空间转录组数据中，相邻 spot 的表达高度相似。将它们作为负样本会给模型施加错误的监督信号 — 让模型把实际上相似的样本推远。

### 5.2 HECLIP Image-Centric Soft Target

**核心思想**：用图像特征之间的相似度作为"软标签"，替代硬的 0/1 标签。

```python
def image_centric_loss(self, logits_per_image, i_f):
    # 1. 计算图像特征间的余弦相似度
    img_sim = normalize(i_f) @ normalize(i_f)^T

    # 2. 用 logit_scale / temperature 缩放后取 softmax
    targets = softmax(img_sim × logit_scale / temp)

    # 3. 计算 KL 散度（cross-entropy 形式）
    return -mean(sum(targets × log_softmax(logits_per_image)))
```

**关键超参数**：
- `temp = 0.5`：温度参数，控制软标签的"尖锐"程度
- `logit_scale ≈ 14.3`（从 `log(1/0.07)` 初始化），所以 `target_scale = 14.3 / 0.5 ≈ 28.6`
- `λ = 0.2`：mix 权重

**最终对比损失**：
```
con_loss = (1 - 0.2) × hard_InfoNCE + 0.2 × image_centric_soft
```

**直觉**：如果两个 spot 的组织形态学图像很相似（如都在肿瘤区域），那么它们的基因表达也应该相似。软标签把这种先验知识注入对比学习，避免"假负样本"问题。

---

## 6. 数据格式说明

### 6.1 数据来源

| 数据集 | 切片数 | Folds | 基因数 (D_out) | 平台 |
|--------|--------|-------|----------------|------|
| HER2ST | 36 张 HE 切片 | 32-fold (leave-one-out) | 785 高变异基因 | ST (原始空间转录组) |
| Skin (CSCC) | 12 张切片 (P2/P5/P9/P10 × 3 重复) | 12-fold | 171 高变异基因 | ST |

### 6.2 每个样本的数据结构

DataLoader 每次返回一个切片 (batch_size=1)，squeeze 后每个 batch 的结构：

| 变量 | 形状 | 含义 |
|------|------|------|
| `g` (gene) | `[N, D_out]` | log-normalized 基因表达矩阵，N 为 spot 数量 |
| `i` (image) | `[N, 1024]` | Phikon-v2 提取的图像特征，每个 spot 对应一个 224×224 的 patch |
| `pos` | `[N, 2]` | 离散化的空间坐标，范围 0~63，用于位置嵌入查表 |
| `centers` | `[N, 2]` | 原始像素坐标，用于可视化 |
| `oris` | `[N, D_out]` | 原始未归一化的计数数据，用于 ZINB 损失 |
| `sfs` | `[N]` | Scale factors（库大小归一化因子），用于 ZINB 损失中缩放均值 |

### 6.3 图像特征提取

- 使用 **Phikon-v2**（基于 ViT 的病理学基础模型）预提取每个 spot 的 1024 维特征
- 每个 spot 对应组织切片上一个 224×224 像素的 patch
- 特征提取是离线完成的，训练时直接加载，不需要反向传播到 ViT

### 6.4 基因表达预处理

1. 选择高变异基因 (HVG)：HER2ST 取 785 个，Skin 取 171 个
2. Log-normalization: `log1p(counts / total × scale_factor)`
3. 原始计数 (`oris`) 保留用于 ZINB 损失建模

---

## 7. 消融实验结果

### Skin 数据集 12-fold 全折交叉验证

| 配置 | Gene PCC | Spot PCC | Spearman | RMSE |
|------|----------|----------|----------|------|
| Baseline (Reg2ST) | 0.1805 | — | — | — |
| + HECLIP | 0.1877 (+4.0%) | — | — | — |
| + Multi-Resolution | 0.2060 (+14.1%) | — | — | — |
| + MR + HECLIP | 0.2208 (+22.3%) | — | — | — |

**关键观察**：
- 两个模块独立有效，且效果**超线性叠加**（4% + 14% → 22%）
- 多分辨率模块提升更大，说明空间建模是原始模型的主要瓶颈
- HECLIP 在与多分辨率结合时效果更好（单独 4%，结合后贡献约 7%），说明更好的空间表示让软对比学习有更多"真实相似对"可以利用

---

## 8. 背景知识补充

### 8.1 空间转录组 (Spatial Transcriptomics) 简介

空间转录组技术可以在保留空间位置信息的前提下测量基因表达。传统单细胞 RNA-seq 需要将组织解离，丢失了细胞在组织中的位置。ST 技术通过在载玻片上固定带有位置条码的寡核苷酸探针，在原位捕获 mRNA，从而同时获得：
- **基因表达谱**：每个空间位置（spot）的基因表达计数
- **空间坐标**：每个 spot 在组织切片上的 (x, y) 位置
- **组织学图像**：H&E 染色的全切片病理图像

**本任务的目标**：从组织学图像（H&E 染色）预测基因表达，即 Image → Gene Expression。这有重要的临床价值：H&E 染色成本低、标准化程度高，而 ST 测序昂贵且通量有限。

### 8.2 CLIP 式对比学习

CLIP (Contrastive Language-Image Pre-training) 的核心思想是通过对比学习对齐两个模态的表示空间。在 Reg2ST 中：

- **图像模态**：每个 spot 的病理图像特征
- **基因模态**：每个 spot 的基因表达向量

对比学习的目标是让同一 spot 的 (image, gene) 对在特征空间中靠近，不同 spot 的对远离。具体实现为：

```
similarity_matrix = logit_scale × img_features @ gene_features^T
```

这是一个 N×N 的矩阵，对角线元素是正对的相似度，非对角线是负对的相似度。InfoNCE loss 让对角线值最大化。

**`logit_scale`** 是一个可学习参数，初始化为 `exp(log(1/0.07)) ≈ 14.3`，用于缩放相似度矩阵到合适的梯度范围。

### 8.3 Cross-Attention (交叉注意力) 与 Decoder

Transformer 的注意力机制：
```
Attention(Q, K, V) = softmax(Q × K^T / √d) × V
```

- **Self-Attention**：Q, K, V 来自同一输入（如 GlobalEncoder）
- **Cross-Attention**：Q 来自一个输入，K/V 来自另一个输入

在 Reg2ST 的 Decoder 中：
- Q = `i_ct`（加了位置嵌入的图像特征），代表"在这个空间位置，我想查询什么信息"
- K, V = `proj_i_f`（投影后的图像特征），代表"全局可供查询的信息库"
- 通过交叉注意力，每个 spot 可以从所有 spot 的特征中检索相关信息

**训练时的 75% mask**：随机遮蔽 75% 的注意力连接，防止模型依赖简单的复制策略（直接复制自己的特征），迫使模型学习从其他 spot 的特征中推断。

### 8.4 WiKG (Weighted Knowledge Graph)

WiKG 是一种动态图神经网络模块：

1. **Head/Tail 投影**：将输入特征分别投影为"头实体"和"尾实体"
2. **Top-K 邻居选择**：通过注意力得分选择每个节点最相关的 k 个邻居（不是基于空间距离，而是基于特征相似度）
3. **Gated Knowledge Attention**：门控机制决定从每个邻居获取多少信息
4. **Bi-Interaction 聚合**：同时考虑"加法"和"乘法"交互
   - `sum_embedding = LeakyReLU(Linear(x + neighbor_agg))`
   - `bi_embedding = LeakyReLU(Linear(x × neighbor_agg))`
   - `output = sum_embedding + bi_embedding`

WiKG 在 Decoder 之后运行，进一步整合图结构信息。

### 8.5 ZINB (Zero-Inflated Negative Binomial) 损失

基因表达计数数据有两个特点：
1. **过度离散 (overdispersion)**：方差远大于均值，普通 Poisson 分布无法拟合
2. **零膨胀 (zero inflation)**：大量基因在大量 spot 中表达量为 0

ZINB 分布建模这两个特点：
```
P(X = 0) = π + (1 - π) × NB(0; μ, θ)    # 零来自两个来源
P(X = k) = (1 - π) × NB(k; μ, θ)         # 非零值来自 NB 分布
```

其中：
- `π` (pi)：零膨胀概率，由 Sigmoid 输出
- `μ` (mean)：NB 分布的均值，由 exp 激活保证为正
- `θ` (disp)：NB 分布的离散参数，由 softplus 激活

模型预测三个参数 (mean, disp, pi)，ZINB loss 最大化原始计数数据在预测分布下的对数似然。

### 8.6 Pearson 相关系数 (PCC) 作为评估指标

```
PCC = Σ((x - x̄)(y - ȳ)) / (√Σ(x - x̄)² × √Σ(y - ȳ)²)
```

- **Gene-wise PCC** (dim=1)：对每个基因，计算所有 spot 上预测值和真实值的相关系数。衡量模型能否捕捉每个基因的空间表达模式。
- **Spot-wise PCC** (dim=0)：对每个 spot，计算所有基因上预测值和真实值的相关系数。衡量模型能否恢复每个空间位置的基因表达谱。
- 最终报告所有基因/spot 的平均 PCC。

### 8.7 学习率调度策略

采用 Warmup + Cosine Decay：
1. **Warmup** (前 10 个 epoch)：学习率从 1e-5 线性增长到 1e-4
2. **Cosine Decay** (之后的 epoch)：学习率按余弦曲线从 1e-4 衰减到 1e-6

这是深度学习中的常用策略：warmup 防止训练初期梯度不稳定，cosine decay 在后期精细调整参数。

### 8.8 PyTorch Lightning 训练框架

代码使用 PyTorch Lightning 封装训练循环：
- `training_step()`：定义前向传播和损失计算
- `validation_step()`：定义验证逻辑
- `test_step()`：定义测试逻辑，保存预测结果到 AnnData
- `configure_optimizers()`：定义优化器和学习率调度
- `ModelCheckpoint`：自动保存验证 PCC 最高的模型
- `CSVLogger`：记录训练指标到 CSV 文件

---

## 附：运行命令参考

```bash
# Baseline
python run_all.py --dataset skin --model reg2st --run_tag baseline

# + HECLIP only
python run_all.py --dataset skin --model final --use_heclip \
    --heclip_target_temp 0.5 --heclip_mix_lambda 0.2 --run_tag heclip

# + Multi-Resolution only
python run_all.py --dataset skin --model final --use_multires --run_tag mr

# + MR + HECLIP (完整模型)
python run_all.py --dataset skin --model final --use_multires --use_heclip \
    --heclip_target_temp 0.5 --heclip_mix_lambda 0.2 --run_tag mr_heclip
```
