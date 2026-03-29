# 代码补充说明

这份文档补充三类情况：
- 本地有代码，但没有对应 PDF：`istar-master`
- 本地 PDF 与同名仓库不匹配：`DANet-main`、`miso_code-main`
- 本地仓库与论文同名但版本目标不同：`STimage-master`

## 1. DANet-main
- 代码定位：这是一个“从 H&E 预测空间基因表达”的项目，不是 `DANet.pdf` 那篇白细胞域泛化论文。
- 方法主线：整体仍是 CLIP/BLEEP 风格的图像-表达对齐，但 spot encoder 被替换成了带 `Mamba` 的序列建模器，强调 dynamic alignment。
- 关键文件：
- `main.py`：训练入口，负责构建数据集、训练循环和测试。
- `models_ours.py`：主模型，包含 `ImageEncoder`、`SpotEncoder`、`ProjectionHead` 和 `CLIPModel`。
- `dataset.py`：数据加载。
- `models_BLEEP.py`、`models_Mamba.py`、`models_NO.py`、`models_RKAN.py`：不同对照或消融版本。
- 代码理解：它本质上是在 BLEEP 框架里替换和加强表达侧编码器，让 spot 表征不再是简单线性向量，而是经过序列模型建模。

## 2. miso_code-main
- 代码定位：这是 MISO 模型实现，不是 `miso_code.pdf` 那篇 benchmark。
- 方法主线：MISO 强调 multiscale integration of spatial omics with tumor morphology，更像多尺度 bag/MIL 风格的图像-组学整合框架。
- 关键文件：
- `miso/train.py`：训练入口。
- `miso/models/simple_mil.py`：MIL 聚合模块。
- `miso/engine/training.py`、`miso/engine/losses.py`：训练和损失。
- `miso/engine/super_resolved_inference.py`：超分辨率推理。
- `miso/scripts/process_data.py`、`process_her2st.py`：数据预处理。
- 代码理解：仓库更偏“模型工程实现”，核心在多尺度 patch 聚合和肿瘤形态与空间 omics 的联合建模。

## 3. STimage-master
- 代码定位：本地仓库更像 STimage 的早期 Python/TensorFlow 工具包，重点在从图像预测表达、估计不确定性、解释模型。
- 方法主线：包含负二项/线性等多种预测头，强调 robustness、uncertainty 和 interpretability。
- 关键文件：
- `stimage/01_Preprocessing.py` 到 `stimage/06_Predict_external.py`：完整分析流水线。
- `stimage/_model.py`：`CNN_NB_model`、`CNN_linear_model`、`CNN_NB_multiple_genes` 等核心模型。
- `stimage/_data_generator.py`、`_img_normaliser.py`、`_utils.py`：数据和图像工具。
- `Tutorials/*`：训练、预测、解释和外部推理示例。
- 代码理解：仓库更偏“图像到基因表达预测平台”，而不是 2025 年论文里那个“54 种整合策略选择框架”。

## 4. istar-master
- 代码定位：这是 iStar 的代码，实现“从 spot-level 到 near-single-cell 的超分辨率组织结构重建”。
- 方法主线：融合 ST 与 histology，把 spot 级表达提升到接近单细胞分辨率，更偏 super-resolution / imputation，而不是纯图像到表达预测。
- 关键文件：
- `train.py`：模型训练。
- `extract_features.py`、`hipt_4k.py`、`vision_transformer*.py`：图像特征抽取与视觉 backbone。
- `impute.py`、`impute_by_basic.py`：表达补全与超分辨率推理。
- `plot_imputed.py`、`plot_spots.py`、`visual.py`：可视化。
- 代码理解：iStar 的核心思想是“借 histology 把低分辨率 ST 细化”，更接近 spatial upsampling，而不是直接用图像生成表达。
