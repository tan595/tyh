"""
Batch training + evaluation for all folds.
Supported models:
  - reg2st  (baseline)
  - final   (unified ablation: --use_multires / --use_heclip)
"""
import os
import sys
from datetime import datetime
import numpy as np
import torch

_original_torch_load = torch.load


def _patched_load(*args, **kwargs):
    kwargs.setdefault('weights_only', False)
    return _original_torch_load(*args, **kwargs)


torch.load = _patched_load

try:
    import lightning_fabric.utilities.cloud_io as _cloud_io
    _cloud_io.torch.load = _patched_load
except Exception:
    pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from herst import ViT_HER2ST, ViT_SKIN
from model import Reg2ST
from model_final import FinalModel
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from performance import get_metrics
from utils import parser_option, seed_torch


def run_all(args):
    torch.set_float32_matmul_precision('high')
    seed_torch(42)
    args.divide_size = 1

    model_name = getattr(args, 'model', 'reg2st')
    output_root = getattr(args, 'output_root', '/root/autodl-tmp/final_outputs')
    run_tag = getattr(args, 'run_tag', None)
    if run_tag is None or str(run_tag).strip() == '':
        run_tag = datetime.utcnow().strftime('%Y%m%d_%H%M%S') + f'_{os.getpid()}'
    tag_suffix = f'_{run_tag}'
    print(f'  Run tag: {run_tag}')

    if args.dataset == 'her2st':
        n_folds = 32
        args.dim_out = 785
    else:
        n_folds = 12
        args.dim_out = 171

    folds_str = getattr(args, 'folds', '')
    if folds_str and str(folds_str).strip():
        fold_list = [int(x.strip()) for x in str(folds_str).split(',')]
    else:
        fold_list = list(range(n_folds))

    log_dir = os.path.join(output_root, f"{model_name}_{args.dataset}_runall_logs{tag_suffix}")

    all_pcc = []
    all_p = []
    all_metrics = []

    for fold_idx, fold in enumerate(fold_list):
        print(f"\n{'=' * 60}")
        print(f"  {model_name.upper()} | {args.dataset} | Fold {fold} ({fold_idx + 1}/{len(fold_list)})")
        print(f"{'=' * 60}")

        if args.dataset == 'her2st':
            train_data = ViT_HER2ST(train=True, flatten=False, ori=True, adj=False, fold=fold)
            val_data = ViT_HER2ST(train=False, flatten=False, ori=True, adj=False, fold=fold)
        else:
            train_data = ViT_SKIN(train=True, flatten=False, ori=True, adj=False, fold=fold)
            val_data = ViT_SKIN(train=False, flatten=False, ori=True, adj=False, fold=fold)

        train_loader = DataLoader(train_data, batch_size=1, shuffle=False, pin_memory=True)
        val_loader = DataLoader(val_data, batch_size=1, shuffle=False, pin_memory=True)

        args.n_train_samples = len(train_data)

        if model_name == 'final':
            model = FinalModel(args=args)
        else:
            model = Reg2ST(args=args)

        save_dir = os.path.join(output_root, f"{args.dataset}_{model_name}_runall{tag_suffix}")
        os.makedirs(save_dir, exist_ok=True)

        ckpt_callback = ModelCheckpoint(
            dirpath=save_dir,
            filename=f"fold{fold}_best",
            save_top_k=1,
            monitor='pcc',
            mode='max',
            save_on_train_epoch_end=False,
        )

        csv_logger = CSVLogger(log_dir, name=f"fold{fold}")
        trainer = pl.Trainer(
            logger=csv_logger,
            precision=32,
            max_epochs=args.epochs,
            accelerator='gpu',
            devices=[args.device_id],
            callbacks=[ckpt_callback],
            log_every_n_steps=5,
            enable_progress_bar=True,
        )
        trainer.fit(model, train_loader, val_loader)

        best_path = ckpt_callback.best_model_path
        print(f"  Loading best checkpoint: {best_path}")
        trainer2 = pl.Trainer(
            precision=32,
            accelerator='gpu',
            devices=[args.device_id],
            enable_progress_bar=False,
        )
        trainer2.test(model, val_loader, ckpt_path=best_path)

        pcc = np.nanmean(model.p)
        all_pcc.append(pcc)
        all_p.append(model.p)

        pred = model.data.X
        truth_g, _, _, _, _, _ = next(iter(val_loader))
        truth = truth_g.squeeze(0).numpy()
        metrics = get_metrics(pred, truth)
        all_metrics.append(metrics)
        print(
            f"  Fold {fold} | gene_PCC={metrics['mean_gene_pcc']:.4f}  "
            f"spot_PCC={metrics['mean_spot_pcc']:.4f}  "
            f"Spearman={metrics['mean_gene_spearman']:.4f}  "
            f"RMSE={metrics['rmse']:.4f}"
        )

    all_pcc = np.array(all_pcc)
    mean_gene_pcc = np.mean([m['mean_gene_pcc'] for m in all_metrics])
    mean_spot_pcc = np.mean([m['mean_spot_pcc'] for m in all_metrics])
    mean_spearman = np.mean([m['mean_gene_spearman'] for m in all_metrics])
    mean_rmse = np.mean([m['rmse'] for m in all_metrics])

    print(f"\n{'=' * 60}")
    print(f"  FINAL RESULTS: {model_name.upper()} on {args.dataset}")
    print(f"{'=' * 60}")
    print(f"  Gene PCC:  {mean_gene_pcc:.4f} +/- {np.std([m['mean_gene_pcc'] for m in all_metrics]):.4f}")
    print(f"  Spot PCC:  {mean_spot_pcc:.4f} +/- {np.std([m['mean_spot_pcc'] for m in all_metrics]):.4f}")
    print(f"  Spearman:  {mean_spearman:.4f} +/- {np.std([m['mean_gene_spearman'] for m in all_metrics]):.4f}")
    print(f"  RMSE:      {mean_rmse:.4f} +/- {np.std([m['rmse'] for m in all_metrics]):.4f}")
    per_fold_pcc = [f"{m['mean_gene_pcc']:.4f}" for m in all_metrics]
    print(f"  Per-fold gene PCC: {per_fold_pcc}")

    os.makedirs(output_root, exist_ok=True)
    result_file = os.path.join(output_root, f"{model_name}_{args.dataset}_results{tag_suffix}.npy")
    np.save(
        result_file,
        {
            'pcc_per_fold': all_pcc,
            'p_per_fold': all_p,
            'metrics_per_fold': all_metrics,
            'summary': {
                'mean_gene_pcc': mean_gene_pcc,
                'mean_spot_pcc': mean_spot_pcc,
                'mean_spearman': mean_spearman,
                'mean_rmse': mean_rmse,
            },
        },
    )
    print(f"  Results saved to {result_file}")


if __name__ == '__main__':
    args = parser_option()
    run_all(args)
