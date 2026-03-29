"""
Visium数据集完整实验脚本
支持Multi-Tissue Tumor Atlas的LOOCV实验
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger

from visium_dataset import ViT_10xVisium
from model import Reg2ST
from model_mrcmr import MRCMR
from performance import get_metrics
from utils import parser_option
import json


def run_visium_experiments(args):
    """运行Visium数据集实验"""

    torch.set_float32_matmul_precision('high')

    # 读取Multi-Tissue配置
    config_path = "tools/multi_tissue_atlas_plan.json"
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        # 获取所有肿瘤样本
        all_samples = []
        tumor_tissues = ['breast_cancer', 'colorectal_cancer', 'prostate_cancer',
                        'lung_cancer', 'ovarian_cancer', 'brain']

        for tissue in tumor_tissues:
            if tissue in config['all_datasets']:
                for sample in config['all_datasets'][tissue]:
                    all_samples.append(sample['name'])

        print(f"Multi-Tissue Atlas: {len(all_samples)} samples")
        for i, name in enumerate(all_samples, 1):
            print(f"  {i}. {name}")
    else:
        # 如果没有配置文件，使用命令行指定的样本
        all_samples = args.visium_samples.split(',')
        print(f"Using samples from command line: {len(all_samples)} samples")

    n_folds = len(all_samples)
    model_name = getattr(args, 'model', 'reg2st')

    # 动态确定基因数（从第一个样本推断）
    print("\n检测基因数...")
    test_dataset = ViT_10xVisium(
        root=args.visium_root,
        sample_names=[all_samples[0]],
        train=False,
        fold=0,
        n_top_genes=args.n_top_genes,
        flatten=False,
        ori=True
    )
    args.dim_out = len(test_dataset.gene_list)
    print(f"基因数: {args.dim_out}")

    log_dir = f"visium_{model_name}_logs"

    all_pcc = []
    all_p = []
    all_metrics = []

    for fold in range(n_folds):
        print(f"\n{'='*60}")
        print(f"  {model_name.upper()} | Visium Multi-Tissue | Fold {fold}/{n_folds-1}")
        print(f"  Test sample: {all_samples[fold]}")
        print(f"{'='*60}")

        # 数据
        train_data = ViT_10xVisium(
            root=args.visium_root,
            sample_names=all_samples,
            train=True,
            fold=fold,
            n_top_genes=args.n_top_genes,
            flatten=False,
            ori=True,
            adj=False
        )

        val_data = ViT_10xVisium(
            root=args.visium_root,
            sample_names=all_samples,
            train=False,
            fold=fold,
            n_top_genes=args.n_top_genes,
            flatten=False,
            ori=True,
            adj=False
        )

        train_loader = DataLoader(train_data, batch_size=1, shuffle=False, pin_memory=True)
        val_loader = DataLoader(val_data, batch_size=1, shuffle=False, pin_memory=True)

        args.n_train_samples = len(train_data)

        # 模型
        if model_name == 'mrcmr':
            model = MRCMR(args=args)
        else:
            model = Reg2ST(args=args)

        # 训练
        csv_logger = CSVLogger(log_dir, name=f"fold{fold}")
        trainer = pl.Trainer(
            logger=csv_logger,
            precision=32,
            max_epochs=args.epochs,
            accelerator='gpu',
            devices=[args.device_id],
            log_every_n_steps=5,
            enable_progress_bar=True,
            enable_checkpointing=False
        )
        trainer.fit(model, train_loader, val_loader)

        # 测试
        trainer2 = pl.Trainer(
            precision=32,
            accelerator='gpu',
            devices=[args.device_id],
            enable_progress_bar=False
        )
        trainer2.test(model, val_loader)

        pcc = np.nanmean(model.p)
        all_pcc.append(pcc)
        all_p.append(model.p)

        # 多指标评估
        pred = model.data.X
        truth_g, truth_i, truth_pos, truth_centers, _, _ = next(iter(val_loader))
        truth = truth_g.squeeze(0).numpy()
        metrics = get_metrics(pred, truth)
        all_metrics.append(metrics)

        print(f"  Fold {fold} | gene_PCC={metrics['mean_gene_pcc']:.4f}  "
              f"spot_PCC={metrics['mean_spot_pcc']:.4f}  "
              f"Spearman={metrics['mean_gene_spearman']:.4f}  "
              f"RMSE={metrics['rmse']:.4f}")

    # 汇总
    all_pcc = np.array(all_pcc)
    mean_gene_pcc = np.mean([m['mean_gene_pcc'] for m in all_metrics])
    mean_spot_pcc = np.mean([m['mean_spot_pcc'] for m in all_metrics])
    mean_spearman = np.mean([m['mean_gene_spearman'] for m in all_metrics])
    mean_rmse = np.mean([m['rmse'] for m in all_metrics])

    print(f"\n{'='*60}")
    print(f"  FINAL RESULTS: {model_name.upper()} on Visium Multi-Tissue")
    print(f"{'='*60}")
    print(f"  Gene PCC:  {mean_gene_pcc:.4f} +/- {np.std([m['mean_gene_pcc'] for m in all_metrics]):.4f}")
    print(f"  Spot PCC:  {mean_spot_pcc:.4f} +/- {np.std([m['mean_spot_pcc'] for m in all_metrics]):.4f}")
    print(f"  Spearman:  {mean_spearman:.4f} +/- {np.std([m['mean_gene_spearman'] for m in all_metrics]):.4f}")
    print(f"  RMSE:      {mean_rmse:.4f} +/- {np.std([m['rmse'] for m in all_metrics]):.4f}")

    # 保存结果
    result_file = f"{model_name}_visium_results.npy"
    np.save(result_file, {
        'pcc_per_fold': all_pcc,
        'p_per_fold': all_p,
        'metrics_per_fold': all_metrics,
        'sample_names': all_samples,
        'summary': {
            'mean_gene_pcc': mean_gene_pcc,
            'mean_spot_pcc': mean_spot_pcc,
            'mean_spearman': mean_spearman,
            'mean_rmse': mean_rmse,
        }
    })
    print(f"  Results saved to {result_file}")


if __name__ == "__main__":
    args = parser_option()

    # Visium特定参数
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--visium_root', type=str, default='data/visium',
                        help='Visium数据根目录')
    parser.add_argument('--visium_samples', type=str, default='',
                        help='样本名称列表（逗号分隔），留空则从config读取')
    parser.add_argument('--n_top_genes', type=int, default=1000,
                        help='高变基因数量')
    args_visium, _ = parser.parse_known_args()

    # 合并参数
    args.visium_root = args_visium.visium_root
    args.visium_samples = args_visium.visium_samples
    args.n_top_genes = args_visium.n_top_genes

    run_visium_experiments(args)
