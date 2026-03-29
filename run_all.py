"""
Batch training + evaluation for all folds.
Usage: python run_all.py --model pdg_st --dataset her2st
       python run_all.py --model reg2st --dataset her2st
       python run_all.py --model mrcmr --dataset her2st
"""
import os
import sys
import numpy as np
import torch
_original_torch_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs.setdefault('weights_only', False)
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_load

# Patch lightning_fabric as well (it caches its own reference)
import lightning_fabric.utilities.cloud_io as _cloud_io
_cloud_io.torch.load = _patched_load

# Add code dir to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from herst import ViT_HER2ST, ViT_SKIN
from model import Reg2ST
from model_pdg import PDG_ST
from model_mrcmr import MRCMR
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from performance import get_R, get_metrics
from utils import parser_option


def run_all(args):
    torch.set_float32_matmul_precision('high')
    args.divide_size = 1
    model_name = getattr(args, 'model', 'reg2st')

    if args.dataset == "her2st":
        n_folds = 32
        args.dim_out = 785
    else:
        n_folds = 12
        args.dim_out = 171

    log_dir = f"/root/autodl-tmp/{model_name}_{args.dataset}_logs"

    all_pcc = []
    all_p = []
    all_metrics = []

    for fold in range(n_folds):
        print(f"\n{'='*60}")
        print(f"  {model_name.upper()} | {args.dataset} | Fold {fold}/{n_folds-1}")
        print(f"{'='*60}")

        # Data
        if args.dataset == "her2st":
            train_data = ViT_HER2ST(train=True, flatten=False, ori=True, adj=False, fold=fold)
            val_data = ViT_HER2ST(train=False, flatten=False, ori=True, adj=False, fold=fold)
        else:
            train_data = ViT_SKIN(train=True, flatten=False, ori=True, adj=False, fold=fold)
            val_data = ViT_SKIN(train=False, flatten=False, ori=True, adj=False, fold=fold)

        train_loader = DataLoader(train_data, batch_size=1, shuffle=False, pin_memory=True)
        val_loader = DataLoader(val_data, batch_size=1, shuffle=False, pin_memory=True)

        args.n_train_samples = len(train_data)

        # Model
        if model_name == 'pdg_st':
            model = PDG_ST(args=args)
        elif model_name == 'mrcmr':
            model = MRCMR(args=args)
        else:
            model = Reg2ST(args=args)

        # Train (no checkpoint saving)
        csv_logger = CSVLogger(log_dir, name=f"fold{fold}")
        trainer = pl.Trainer(
            logger=csv_logger, precision=32,
            max_epochs=args.epochs,
            accelerator='gpu', devices=[args.device_id],
            log_every_n_steps=5,
            enable_progress_bar=True,
            enable_checkpointing=False
        )
        trainer.fit(model, train_loader, val_loader)

        # Test directly with in-memory model (no checkpoint needed)
        trainer2 = pl.Trainer(
            precision=32, accelerator='gpu', devices=[args.device_id],
            enable_progress_bar=False
        )
        trainer2.test(model, val_loader)

        pcc = np.nanmean(model.p)
        all_pcc.append(pcc)
        all_p.append(model.p)

        # Multi-metric evaluation
        pred = model.data.X
        truth_g, truth_i, truth_pos, truth_centers, _, _ = next(iter(val_loader))
        truth = truth_g.squeeze(0).numpy()
        metrics = get_metrics(pred, truth)
        all_metrics.append(metrics)
        print(f"  Fold {fold} | gene_PCC={metrics['mean_gene_pcc']:.4f}  "
              f"spot_PCC={metrics['mean_spot_pcc']:.4f}  "
              f"Spearman={metrics['mean_gene_spearman']:.4f}  "
              f"RMSE={metrics['rmse']:.4f}")

    # Summary
    all_pcc = np.array(all_pcc)
    mean_gene_pcc = np.mean([m['mean_gene_pcc'] for m in all_metrics])
    mean_spot_pcc = np.mean([m['mean_spot_pcc'] for m in all_metrics])
    mean_spearman = np.mean([m['mean_gene_spearman'] for m in all_metrics])
    mean_rmse = np.mean([m['rmse'] for m in all_metrics])

    print(f"\n{'='*60}")
    print(f"  FINAL RESULTS: {model_name.upper()} on {args.dataset}")
    print(f"{'='*60}")
    print(f"  Gene PCC:  {mean_gene_pcc:.4f} +/- {np.std([m['mean_gene_pcc'] for m in all_metrics]):.4f}")
    print(f"  Spot PCC:  {mean_spot_pcc:.4f} +/- {np.std([m['mean_spot_pcc'] for m in all_metrics]):.4f}")
    print(f"  Spearman:  {mean_spearman:.4f} +/- {np.std([m['mean_gene_spearman'] for m in all_metrics]):.4f}")
    print(f"  RMSE:      {mean_rmse:.4f} +/- {np.std([m['rmse'] for m in all_metrics]):.4f}")
    per_fold_pcc = [f"{m['mean_gene_pcc']:.4f}" for m in all_metrics]
    print(f"  Per-fold gene PCC: {per_fold_pcc}")

    # Save results
    result_file = f"{model_name}_{args.dataset}_results.npy"
    np.save(result_file, {
        'pcc_per_fold': all_pcc,
        'p_per_fold': all_p,
        'metrics_per_fold': all_metrics,
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
    run_all(args)
