"""
快速验证脚本：测试Visium数据集是否能正常工作
使用1个样本跑1个epoch，验证数据加载和模型forward
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from visium_dataset import ViT_10xVisium
from model import Reg2ST
from model_mrcmr import MRCMR
from utils import parser_option
import argparse


def quick_validate():
    """快速验证Visium数据集"""

    print("="*80)
    print("Visium数据集快速验证")
    print("="*80)

    # 配置
    parser = argparse.ArgumentParser()
    parser.add_argument('--visium_root', type=str, default='data/visium',
                        help='Visium数据根目录')
    parser.add_argument('--sample', type=str,
                        default='CytAssist_11mm_FFPE_Human_Colorectal_Cancer',
                        help='测试样本名称')
    parser.add_argument('--model', type=str, default='reg2st',
                        choices=['reg2st', 'mrcmr'],
                        help='测试模型')
    parser.add_argument('--device_id', type=int, default=0)
    args_test = parser.parse_args()

    # 获取完整参数
    args = parser_option()
    args.visium_root = args_test.visium_root
    args.device_id = args_test.device_id

    # Visium数据集的基因数（需要根据实际情况调整）
    args.dim_out = 1000  # 高变基因数量
    args.epochs = 1  # 只跑1个epoch

    print(f"\n配置:")
    print(f"  数据目录: {args.visium_root}")
    print(f"  测试样本: {args_test.sample}")
    print(f"  模型: {args_test.model}")
    print(f"  基因数: {args.dim_out}")

    # 创建数据集
    print(f"\n{'='*60}")
    print("步骤1: 加载数据")
    print(f"{'='*60}")

    try:
        dataset = ViT_10xVisium(
            root=args.visium_root,
            sample_names=[args_test.sample],
            train=False,
            fold=0,
            n_top_genes=args.dim_out,
            flatten=False,
            ori=True,
            adj=False
        )

        print(f"✓ 数据集加载成功")
        print(f"  样本数: {len(dataset)}")
        print(f"  基因数: {len(dataset.gene_list)}")

        # 测试数据加载
        data = dataset[0]
        print(f"\n✓ 数据格式验证:")
        print(f"  基因表达: {data[0].shape}")
        print(f"  图像特征: {data[1].shape}")
        print(f"  空间位置: {data[2].shape}")

    except Exception as e:
        print(f"✗ 数据加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 创建DataLoader
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    # 创建模型
    print(f"\n{'='*60}")
    print("步骤2: 创建模型")
    print(f"{'='*60}")

    try:
        args.n_train_samples = 1  # 单样本

        if args_test.model == 'mrcmr':
            model = MRCMR(args=args)
        else:
            model = Reg2ST(args=args)

        print(f"✓ 模型创建成功: {args_test.model}")

    except Exception as e:
        print(f"✗ 模型创建失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 测试forward
    print(f"\n{'='*60}")
    print("步骤3: 测试模型forward")
    print(f"{'='*60}")

    try:
        model.eval()
        with torch.no_grad():
            batch = next(iter(loader))
            g, i, pos, centers, oris, sfs = batch

            print(f"  Batch shapes:")
            print(f"    gene: {g.shape}")
            print(f"    image: {i.shape}")
            print(f"    pos: {pos.shape}")

            # Forward pass
            g = g.squeeze(0)
            i = i.squeeze(0)
            pos = pos.squeeze(0)

            if args_test.model == 'mrcmr':
                i_g, *_ = model.forward(g, i, pos)
            else:
                i_g, *_ = model.forward(g, i, pos)

            print(f"\n✓ Forward成功")
            print(f"  输出shape: {i_g.shape}")
            print(f"  预期shape: {g.shape}")

            if i_g.shape == g.shape:
                print(f"\n✓ 输出shape正确！")
            else:
                print(f"\n✗ 输出shape不匹配")
                return False

    except Exception as e:
        print(f"✗ Forward失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 测试训练1个epoch
    print(f"\n{'='*60}")
    print("步骤4: 测试训练1个epoch")
    print(f"{'='*60}")

    try:
        trainer = pl.Trainer(
            max_epochs=1,
            accelerator='gpu' if torch.cuda.is_available() else 'cpu',
            devices=[args.device_id] if torch.cuda.is_available() else 1,
            enable_progress_bar=True,
            enable_checkpointing=False,
            logger=False
        )

        trainer.fit(model, loader, loader)

        print(f"\n✓ 训练测试成功")

    except Exception as e:
        print(f"✗ 训练测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 总结
    print(f"\n{'='*80}")
    print("✓ 所有测试通过！")
    print("="*80)
    print("\n可以开始完整实验:")
    print(f"  python run_visium.py --visium_root {args.visium_root}")
    print("="*80)

    return True


if __name__ == "__main__":
    success = quick_validate()
    sys.exit(0 if success else 1)
