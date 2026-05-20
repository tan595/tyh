# Reg2ST

- 对应论文：`/root/workspace/1/论文/reg2st.pdf`
- 对应代码：`/root/workspace/1/Reg2ST-main`
- 一句话定位：先用对比学习对齐图像与基因特征，再预测 gene feature，并通过动态 GNN 建模 spot 关系来做空间表达预测。

## 1. 摘要
- 背景：ST 同时保留基因表达和空间位置，但采样成本高；从 HE 图像恢复 ST 是高性价比方向。
- 挑战：已有方法一是没有充分利用基因表达中潜在模式，二是图像特征提取不够深，三是用 KNN 构图过于依赖几何距离。
- 方法：Reg2ST 把图像与空间转录组看作同一 spot 的两种模态表达，先用对比学习缩小模态差距，再用图像特征预测 gene feature，并通过 cross-attention 与动态 GNN 融合图像、位置和基因关系。
- 贡献：在 HER2+ 和 cSCC 数据上，相比 Hist2ST、THItoGene、HGGEP 等方法获得更高 PCC，同时兼顾统计显著性和计算效率。

## 2. 引言
- 问题背景：病理图像中有丰富的结构和细胞形态信息，而 ST 给出了这些结构背后的分子表达。
- 历史发展：ST-Net、HisToGene、Hist2ST、THItoGene、HGGEP、OmiCLIP/Loki 等方法分别从 CNN、Transformer、GNN、对比学习等角度推进该任务。
- 作者动机：作者认为现有方法仍有三个缺口：没有充分“借用”基因表达潜在模式；图像特征提取不够灵活；KNN 图只能表达距离近，表达不了更深层的功能或层级关系。
- 方法与贡献落点：Reg2ST 想做的是“对齐 + 中间层预测 gene feature + 动态图关系学习”的一体化设计。

## 3. 相关工作
- 问题发展历史：图像到 ST 预测方法逐渐从直接回归，发展到加入 Transformer/GNN，再到对比学习和 foundation model。
- 算法发展历史：CNN 更擅长局部纹理，Transformer 更擅长全局依赖，GNN 更擅长 spot 关系建模；近年的方法开始把这些路线组合起来。
- 本文相对差异：Reg2ST 不是纯检索式方法，也不是纯回归式方法，而是在对比对齐之后继续显式预测 gene feature，并用动态图来传播 spot 关系。

## 4. 方法
- 动机：如果先把图像和基因特征拉近，再让图像端去预测基因端的潜在表示，模型就更容易利用表达中的潜在模式。
- 核心流程：先用图像编码器与基因编码器提取特征；用 contrastive learning 最小化二者距离；再由图像特征预测 gene feature；把 spot 坐标嵌入图像特征，与预测 gene feature 通过 cross-attention 融合；最后用 dynamic GNN 建图并输出表达。
- 算法理解：Reg2ST 的关键不是“直接从 image 到 gene”，而是先让图像学会长得像 gene feature，再把这种中间表示送进更强的图结构模块。
- 与 KNN 的差异：传统 KNN 图只按物理距离连边，Reg2ST 的图是动态变化的，更依赖预测 gene feature 和图像特征本身的相似性。

## 5. 实验
- 实验内容：在 HER2+ 和 cSCC 数据集上做空间表达预测，主要指标为 mean/median PCC，并给出统计检验和效率比较。
- 实验现象：Reg2ST 在两个数据集上都超过 README 中列出的主要基线，说明“对比对齐 + 动态图”这条路线有效。
- 原因分析：对比学习缩小模态间差距，预测 gene feature 强化了图像到表达的桥接，动态图又进一步利用了 spot 间的潜在关系。

## 6. 结论
- 总结全文：Reg2ST 的贡献在于提出了一条介于 BLEEP 式检索和 Hist2ST 式直接预测之间的中间路线：先对齐，再预测 gene feature，再图传播。
- 展望：后续可以把图像编码器升级为更强病理基础模型，并把动态图构建做得更可解释。

## 附：代码实现对应
- `code/model.py`：主模型 `Reg2ST`，包含对比学习、gene feature 预测、动态 GNN 和损失。
- `code/attention.py`：cross-attention 模块。
- `code/wikg.py`：动态图/图传播模块。
- `code/train.py` / `code/predict.py`：训练与推理入口。
- `code/herst.py`：HER2+ 与 cSCC 数据处理。
- `code/pikon.py`：Phikon 特征相关脚本。
