# STModule

- 对应论文：`/root/workspace/1/论文/STModule.pdf`
- 对应代码：`/root/workspace/1/STModule-main`
- 一句话定位：把组织表达图谱分解成“组织模块 + 空间分布图”，强调模块级而非细胞类型级的空间组分发现。

## 1. 摘要
- 背景：SRT 数据中既有肿瘤、免疫、浸润、结构区室等多类信号，但现有方法往往只盯着 SVG、空间 domain 或细胞类型比例。
- 挑战：组织模块可能重叠、尺度不同、跨细胞类型共享，单一聚类或单基因分析很难拆开。
- 方法：STModule 用 Bayesian spatial factorization 同时估计模块的空间图和基因载荷，把表达矩阵分解成多个 tissue modules。
- 贡献：相比已有方法，STModule 能捕获更广的生物信号，发现更稳健、可转移的模块基因集合，并支持把学到的模块迁移到新切片上估计空间图。

## 2. 引言
- 问题背景：组织中的功能单元往往不是单一细胞类型，而是由多个细胞共同构成的 recurrent communities。
- 历史发展：SRT 分析先后发展出 SVG 检测、spatial domain clustering、cell type deconvolution 等路线。
- 作者动机：这些路线虽然各有价值，但都不是直接面向“组织模块发现”设计的，因此容易遗漏跨细胞类型或多尺度的组织信号。
- 方法与贡献落点：作者希望以组织模块为最小分析单位，直接同时学习“模块在哪里”和“模块由哪些基因定义”。

## 3. 相关工作
- 问题发展历史：从单基因的空间变异检测，到整片组织的 domain 分区，再到细胞类型解卷积，组织级分析对象逐渐复杂。
- 算法发展历史：SVG 方法偏特征选择，domain clustering 偏区域划分，deconvolution 偏细胞组成；三者都没有直接建模模块重叠和多尺度。
- 本文相对差异：STModule 用概率因子分解直接把表达矩阵写成“空间地图 × 模块基因负荷”，是更贴近模块概念的建模方式。

## 4. 方法
- 动机：组织模块既要有空间连续性，也要有一组共同表达的基因，最好还能允许不同模块部分重叠。
- 核心流程：把表达矩阵 `Y` 分解为 spot-module 空间活动 `P` 和 module-gene 载荷 `G`；在贝叶斯框架中加入噪声建模和空间先验；训练后得到模块空间图和模块基因。
- 算法理解：这相当于把传统 NMF/因子分解加入了“空间平滑”和“贝叶斯不确定性”，更适合 noisy 的 SRT 数据。
- 迁移能力：作者还设计了 spatial map estimation，可把一个切片上学到的模块投射到新切片上。

## 5. 实验
- 实验内容：在模拟数据和多种真实 SRT 数据上，与 SVG、domain clustering、deconvolution 等方法比较识别能力和生物学解释能力。
- 实验现象：STModule 在模块识别、空间组件可解释性和跨切片可转移性方面表现更强，能识别肿瘤、免疫浸润、上皮内瘤变等多种信号。
- 原因分析：因为模型不是先做区域划分再找基因，而是从一开始就把“空间图”和“基因集合”联合学习，所以更容易得到稳定模块。

## 6. 结论
- 总结全文：STModule 的价值在于把 SRT 分析对象从“单基因”或“单区域”提升到了“组织模块”。
- 展望：未来可以把它拓展到更高分辨率的细胞级 SRT，或和病理图像共同建模，做图像引导的 tissue module discovery。

## 附：代码实现对应
- `R/run_STModule.r`：主入口，运行模块发现和新切片空间图估计。
- `R/spatial_factorization.r` / `R/spatial_factorization_gpu.r`：CPU/GPU 版贝叶斯空间因子分解。
- `R/spatial_map_estimation.r` / `R/spatial_map_estimation_gpu.r`：新切片模块投影。
- `R/result_analysis.r`：空间图和关联基因可视化。
- `R/data_preprocessing.r`、`R/utils.r`：预处理和工具函数。
