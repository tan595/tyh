"""
10x Visium数据集类
支持加载10x Genomics官方的Visium空间转录组数据
"""
import numpy as np
import torch
import pandas as pd
import scanpy as sc
import scprep as scp
from pathlib import Path
from torch.utils.data import Dataset
from collections import defaultdict
import os


class ViT_10xVisium(Dataset):
    """
    10x Visium空间转录组数据集

    支持加载10x官方格式的Visium数据：
    - filtered_feature_bc_matrix.h5
    - spatial/tissue_positions_list.csv
    - spatial/scalefactors_json.json
    - spatial/tissue_hires_image.png
    """

    def __init__(self, root, sample_names, train=True, fold=0,
                 n_top_genes=1000, flatten=True, ori=False, adj=False):
        """
        Args:
            root: Visium数据根目录
            sample_names: 样本名称列表（如['sample1', 'sample2']）
            train: True为训练集，False为测试集
            fold: 交叉验证的折数
            n_top_genes: 高变基因数量
            flatten: 是否展平特征
            ori: 是否保留原始基因表达（用于ZINB损失）
            adj: 是否计算邻接矩阵（暂不使用）
        """
        super(ViT_10xVisium, self).__init__()

        self.root = Path(root)
        self.train = train
        self.ori = ori
        self.adj = adj
        self.flatten = flatten
        self.n_top_genes = n_top_genes

        # 交叉验证划分
        all_samples = sorted(sample_names)
        te_names = [all_samples[fold]]
        tr_names = [s for s in all_samples if s != te_names[0]]

        if self.train:
            self.names = tr_names
        else:
            self.names = te_names

        print(f"{'Train' if train else 'Test'} samples: {self.names}")

        # 加载所有样本
        print('Loading Visium data...')
        self.adata_dict = {}
        for name in self.names:
            sample_path = self.root / name
            if not sample_path.exists():
                raise FileNotFoundError(f"Sample not found: {sample_path}")

            # 使用scanpy加载Visium数据
            adata = sc.read_visium(path=str(sample_path))
            self.adata_dict[name] = adata

        # 高变基因选择（在所有样本上统一）
        print('Selecting highly variable genes...')
        self.gene_list = self._select_hvg(self.adata_dict, n_top_genes)
        print(f'Selected {len(self.gene_list)} genes')

        # 提取基因表达数据
        self.exp_dict = {}
        for name, adata in self.adata_dict.items():
            # 筛选高变基因
            adata_hvg = adata[:, self.gene_list].copy()

            # 归一化 + log变换
            exp = scp.transform.log(
                scp.normalize.library_size_normalize(adata_hvg.X.toarray())
            )
            self.exp_dict[name] = exp

        # 保留原始表达（用于ZINB损失）
        if self.ori:
            self.ori_dict = {}
            self.counts_dict = {}
            for name, adata in self.adata_dict.items():
                adata_hvg = adata[:, self.gene_list].copy()
                ori_exp = adata_hvg.X.toarray()
                self.ori_dict[name] = ori_exp

                # 计算缩放因子
                n_counts = ori_exp.sum(1)
                sf = n_counts / np.median(n_counts)
                self.counts_dict[name] = sf

        # 提取空间坐标
        self.center_dict = {}
        self.loc_dict = {}
        for name, adata in self.adata_dict.items():
            # 像素坐标（用于可视化）
            spatial_coords = adata.obsm['spatial']
            self.center_dict[name] = spatial_coords.astype(int)

            # 网格坐标（用于位置编码）
            # Visium的array坐标通常在obs中
            if 'array_row' in adata.obs.columns and 'array_col' in adata.obs.columns:
                loc = adata.obs[['array_row', 'array_col']].values
            else:
                # 如果没有array坐标，使用归一化的像素坐标
                loc = spatial_coords / 100  # 简单归一化
            self.loc_dict[name] = loc

        # 加载或生成图像特征
        print('Loading image features...')
        self.img_dict = {}
        for name in self.names:
            # 尝试加载预提取的Phikon-v2特征
            feature_path = self.root / f"{name}_phikonv2.npy"
            if feature_path.exists():
                img_f = np.load(feature_path)
                print(f"  Loaded pre-extracted features: {name}")
            else:
                # 如果没有预提取特征，使用占位符
                # 实际使用时需要先提取特征
                n_spots = len(self.adata_dict[name])
                img_f = np.random.randn(n_spots, 1024).astype(np.float32)
                print(f"  Warning: Using random features for {name} (need to extract Phikon-v2 features)")

            self.img_dict[name] = img_f

        # 数据集索引
        self.lengths = [len(self.exp_dict[name]) for name in self.names]
        self.cumlen = np.cumsum(self.lengths)
        self.id2name = dict(enumerate(self.names))

    def _select_hvg(self, adata_dict, n_top_genes):
        """
        在所有样本上选择高变基因

        策略：合并所有样本，计算高变基因，取交集
        """
        # 合并所有样本
        adata_list = [adata.copy() for adata in adata_dict.values()]

        # 对每个样本计算高变基因
        hvg_sets = []
        for adata in adata_list:
            # 归一化
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)

            # 计算高变基因
            sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes)
            hvg = adata.var_names[adata.var.highly_variable].tolist()
            hvg_sets.append(set(hvg))

        # 取交集（确保所有样本都有这些基因）
        common_hvg = set.intersection(*hvg_sets)

        if len(common_hvg) < n_top_genes * 0.5:
            print(f"Warning: Only {len(common_hvg)} common HVGs found")
            # 如果交集太小，改用并集并取前n_top_genes
            all_hvg = set.union(*hvg_sets)
            return sorted(list(all_hvg))[:n_top_genes]

        return sorted(list(common_hvg))[:n_top_genes]

    def __getitem__(self, index):
        """
        获取指定索引的样本数据

        Returns:
            data: [基因表达, 图像特征, 空间位置, 像素坐标, (原始表达, 缩放因子)]
        """
        ID = self.id2name[index]

        exps = self.exp_dict[ID]
        img_f = self.img_dict[ID]
        centers = self.center_dict[ID]
        loc = self.loc_dict[ID]

        positions = torch.LongTensor(loc)

        data = [torch.Tensor(exps), img_f, positions, torch.Tensor(centers)]

        if self.ori:
            oris = self.ori_dict[ID]
            sfs = self.counts_dict[ID]
            data += [torch.Tensor(oris), torch.Tensor(sfs)]

        return data

    def __len__(self):
        """返回数据集中样本（组织切片）的数量"""
        return len(self.exp_dict)


def test_visium_dataset():
    """测试Visium数据集加载"""

    print("="*60)
    print("测试Visium数据集加载")
    print("="*60)

    # 测试配置
    root = "data/visium"
    sample_names = ["CytAssist_11mm_FFPE_Human_Colorectal_Cancer"]

    try:
        # 创建数据集
        dataset = ViT_10xVisium(
            root=root,
            sample_names=sample_names,
            train=False,
            fold=0,
            n_top_genes=1000,
            ori=True
        )

        print(f"\n✓ 数据集创建成功")
        print(f"  样本数: {len(dataset)}")
        print(f"  基因数: {len(dataset.gene_list)}")

        # 测试数据加载
        data = dataset[0]
        print(f"\n✓ 数据加载成功")
        print(f"  基因表达: {data[0].shape}")
        print(f"  图像特征: {data[1].shape}")
        print(f"  空间位置: {data[2].shape}")
        print(f"  像素坐标: {data[3].shape}")
        if len(data) > 4:
            print(f"  原始表达: {data[4].shape}")
            print(f"  缩放因子: {data[5].shape}")

        print("\n✓ 测试通过！")

    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_visium_dataset()
