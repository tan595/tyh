# STPath

- 对应论文：`/root/workspace/1/论文/STPath.pdf`
- 对应代码：`/root/workspace/1/STPath-main`
- 一句话定位：一个面向多器官、多技术、全基因覆盖的生成式 ST foundation model。

## 1. 摘要
- 背景：已有从 WSI 预测 ST 的方法常常依赖固定器官、固定技术或小范围基因面板，泛化有限。
- 挑战：要同时跨 17 个器官、4 种技术并覆盖近 39000 个基因，单纯做小数据集 fine-tuning 明显不够。
- 方法：STPath 在大规模 WSI-ST 配对数据上进行预训练，把图像、基因、器官类型和测序技术一并送入 geometry-aware Transformer，并通过 masked gene prediction 学习。
- 贡献：无需针对下游任务再做专门微调，即可在 6 类任务、23 个数据集、14 个生物标志物上表现强劲；在 HEST-Bench 上取得最佳 Pearson。

## 2. 引言
- 问题背景：ST 与病理结合能更细地刻画肿瘤微环境，但目前测序通量和成本仍限制应用。
- 历史发展：早期方法大多基于预训练病理编码器，在小数据集上训练表达头；结果往往局限于特定器官、特定平台和固定基因集。
- 作者动机：作者希望做一个真正意义上的基础模型，让同一个模型跨器官、跨技术、跨任务地复用，而不是每换一个数据集就重新训练一遍。
- 方法与贡献落点：STPath 把 ST 预测看成生成式预训练问题，而不是单一监督回归问题。

## 3. 相关工作
- 问题发展历史：病理 foundation model 已在图像端逐渐成熟，但 ST 预测端还停留在数据集级别的专门模型。
- 算法发展历史：从 patch 特征回归到图结构建模，再到 foundation model，核心趋势是“更大规模预训练 + 更强泛化能力”。
- 本文相对差异：STPath 同时把 organ type 和 tech type 显式作为条件输入，这使它天然适合做跨域泛化。

## 4. 方法
- 动机：如果模型只在一个器官或一个技术上训练，就很难学到跨场景稳定的形态-分子映射。
- 核心流程：先用大规模数据集构建预训练语料；把每个 spot 的图像特征、基因表达、器官标签和技术标签编码成 token；再用空间感知 Transformer 做 masked gene prediction。
- 算法理解：它本质上类似“空间版多模态 BERT/Transformer”，通过遮蔽表达并要求模型恢复，逼迫模型学习基因间依赖和空间依赖。
- 推理理解：推理时可以直接预测表达，也可以在少量提示 spot 的条件下做更强的恢复。

## 5. 实验
- 实验内容：在 gene expression prediction、spot imputation、spatial clustering、biomarker prediction、mutation prediction、survival prediction 六类任务上评估。
- 实验现象：STPath 在 HEST-Bench 上取得最高 PCC，并在提示少量 spot 时进一步增强；同时还能提升下游生存预测和突变预测。
- 原因分析：大规模预训练让模型学到了更稳定的跨器官表示，而 geometry-aware Transformer 使其更好利用 spot 之间的空间结构。

## 6. 结论
- 总结全文：STPath 把 ST 预测从“小模型做单任务”推进到“基础模型做多任务”，是这篇工作的最大意义。
- 展望：如果未来继续扩展到单细胞、蛋白和多切片三维重建，STPath 这类模型会更接近真正的组织多组学基础模型。

## 附：代码实现对应
- `stpath/model/model.py`：核心 `STFM` 模型，含输入编码、backbone 与表达头。
- `stpath/data/*`：训练/推理数据集与采样逻辑。
- `stpath/tokenization/*`：图像、基因、器官、技术等 token 化。
- `stpath/hest_utils/*`：病理基础编码器封装与 HEST/ST 数据工具。
- `tutorial/HEST-Bench.ipynb`：下游任务示例。
