# HESCAPE

- 对应论文：`/root/workspace/1/论文/hescape.pdf`
- 对应代码：`/root/workspace/1/hescape-main`
- 一句话定位：一个面向病理图像-基因表达跨模态学习的大规模 benchmark，而不是单一预测模型。

## 1. 摘要
- 背景：空间转录组让图像和基因表达天然成对出现，但该领域缺少统一的多模态 benchmark。
- 挑战：不同基因 panel、不同器官、不同 donor 带来强批次效应，使得跨模态预训练结果难以公平比较。
- 方法：HESCAPE 汇总了 6 个 Xenium panel、54 个 donor、约 62 万个 image-gene pair，并系统比较多个图像编码器、基因编码器和预训练策略。
- 贡献：发现基因编码器比图像编码器更决定跨模态对齐效果；空间预训练的基因模型更强；但对比学习虽提升突变分类，却会降低直接的基因表达预测，批次效应是关键原因。

## 2. 引言
- 问题背景：病理 foundation model 已经很强，但当图像与高维基因表达结合时，是否真的能靠对比学习学到有用联合表示，还缺少系统证据。
- 历史发展：图像端已有大量 pathology encoders，基因端也有 scFoundation、Geneformer、Nicheformer 等 foundation models，但两边结合后的表现尚不清楚。
- 作者动机：作者希望把“哪种图像编码器更好”“哪种基因编码器更好”“对比预训练到底对下游有没有帮助”这些问题一次性讲清楚。
- 方法与贡献落点：HESCAPE 通过标准化数据、统一训练和统一评测，回答的是“方法论上的比较问题”。

## 3. 相关工作
- 问题发展历史：单模态 benchmark 已较成熟，而多模态空间组学 benchmark 仍稀缺。
- 算法发展历史：图像编码器从 CTransPath、UNI 到 Gigapath；基因编码器从 scVI/DRVI 到 scFoundation/Nicheformer；多模态方法从简单对齐走向 foundation model。
- 本文相对差异：HESCAPE 不提出单个新模型，而是提出评测体系，并重点分析批次效应和真实下游任务差距。

## 4. 方法
- 动机：没有统一 benchmark 时，方法间比较常常不公平，很难知道真正有效的因素是什么。
- 核心流程：构建 one-to-one 的 image-gene paired 数据；选择 5 个图像编码器和 4 个基因编码器；在 contrastive pretraining 下训练；再评估 retrieval、mutation classification、gene expression prediction 等任务。
- 算法理解：HESCAPE 的价值不在结构复杂，而在把“模型、数据、指标、训练协议”都标准化，方便定位性能来源。
- 关键发现：跨模态检索提升，并不必然等价于基因表达预测提升。

## 5. 实验
- 实验内容：比较不同编码器、不同预训练方式在检索和下游任务上的表现，并专门分析 batch effect。
- 实验现象：空间预训练基因编码器显著提升对齐；但 contrastive pretraining 对 mutation classification 有利，对 expression prediction 却常常变差。
- 原因分析：对比学习更擅长学习“跨模态匹配性”，但表达预测要求对细粒度数值结构更敏感，而批次效应会破坏这种数值一致性。

## 6. 结论
- 总结全文：HESCAPE 的核心结论是，跨模态预训练不是天然对所有下游任务都有利，尤其在空间转录组里，批次效应是决定成败的核心变量。
- 展望：后续需要开发更强的 batch-robust multimodal learning，而不仅仅是把 CLIP 直接套到病理和基因上。

## 附：代码实现对应
- `src/hescape/models/clip.py`：跨模态 CLIP 模型，整合图像编码器与基因编码器。
- `src/hescape/modules/pretrain_module.py`：预训练模块。
- `src/hescape/data_modules/image_gexp_dataset.py`：图像-表达配对数据集。
- `experiments/hescape_pretrain/train.py`：benchmark 训练入口。
- `experiments/drvi_pretrain/*`：基因编码器预训练与分析。
