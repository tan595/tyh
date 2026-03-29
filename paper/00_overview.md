# `/root/workspace/1` 论文与代码总览

## 1. 本次整理范围
- 已整理本地 `论文` 目录中的 19 篇 PDF，每篇都单独写成一个 `md`。
- 对于能和代码仓库直接对应的项目，我在对应论文文档末尾加入了“代码实现对应”。
- 对于“论文与同名仓库不匹配”或“只有代码没有本地 PDF”的情况，我单独写在 [code_supplement.md](./code_supplement.md)。

## 2. 论文与仓库对应关系
| 论文 | 代码仓库 | 备注 |
| --- | --- | --- |
| BLEEP-.pdf | BLEEP-main | 匹配 |
| CMRCNet.pdf | CMRCNet-main | 匹配 |
| DANet.pdf | DANet-main | **不匹配**：PDF 是白细胞分类论文，仓库是 ST 预测代码 |
| DIALOGUE.pdf | DIALOGUE-master | 匹配 |
| GHIST.pdf | GHIST-main | 匹配 |
| HECLIP.pdf | HECLIP-main | 匹配 |
| HGGEP.pdf | HGGEP-main | 匹配 |
| HiST.pdf | HiST-main | 匹配 |
| Loki.pdf | Loki-main | 匹配 |
| ST-mLiver.pdf | ST-mLiver-master | 匹配 |
| STModule.pdf | STModule-main | 匹配 |
| STPath.pdf | STPath-main | 匹配 |
| THItoGene.pdf | THItoGene-main | 匹配 |
| cNMF.pdf | cNMF-main | 匹配 |
| hescape.pdf | hescape-main | 匹配 |
| mclSTExp.pdf | mclSTExp-main | 匹配 |
| miso_code.pdf | miso_code-main | **不匹配**：PDF 是 benchmark，仓库是 MISO 模型实现 |
| reg2st.pdf | Reg2ST-main | 匹配 |
| stImage.pdf | STimage-master | **版本不完全一致**：论文是 2025 年整合框架，仓库更像早期预测工具实现 |

## 3. 额外代码仓库
- `istar-master`：只有代码和 README，没有本地 PDF。
- 这两部分都放在 [code_supplement.md](./code_supplement.md)。

## 4. 阅读顺序建议
1. 先看 BLEEP、CMRCNet、HECLIP、mclSTExp、THItoGene、HGGEP，理解“从 H&E 预测 ST”的主线。
2. 再看 GHIST、HiST、Loki、STPath，理解“更强监督/基础模型/单细胞分辨率/临床任务扩展”。
3. 最后看 DIALOGUE、cNMF、STModule、ST-mLiver、hescape、miso_code、stImage，理解“程序发现、组织模块、benchmark 与分析框架”。
4. `Reg2ST` 可以放在 BLEEP、CMRCNet 之后看，适合用来理解“对比学习 + 直接预测 + 图结构增强”的混合路线。
