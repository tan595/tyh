import os
import sys

import numpy as np
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader


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
from model import FinalModel
from performance import get_metrics
from utils import parser_option, seed_torch


def _default_ckpt_candidates(args):
    model_name = 'final'
    tag = str(getattr(args, 'run_tag', '') or '').strip()
    suffix = f'_{tag}' if tag else ''
    output_root = getattr(args, 'output_root', '')
    return [
        os.path.join(output_root, f'{args.dataset}_{model_name}_runall{suffix}', f'fold{args.fold}_model.ckpt'),
        os.path.join(output_root, f'{args.dataset}_{model_name}_model{suffix}', f'fold{args.fold}_model.ckpt'),
        os.path.join(output_root, f'{args.dataset}_{model_name}_runall{suffix}', f'fold{args.fold}.ckpt'),
        os.path.join(output_root, f'{args.dataset}_{model_name}_model{suffix}', f'fold{args.fold}.ckpt'),
    ]


def resolve_ckpt_path(args):
    if getattr(args, 'ckpt_path', ''):
        ckpt_path = os.path.expanduser(args.ckpt_path)
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f'checkpoint not found: {ckpt_path}')
        return ckpt_path

    for path in _default_ckpt_candidates(args):
        if path and os.path.exists(path):
            return path

    candidates = '\n  '.join(_default_ckpt_candidates(args))
    raise FileNotFoundError(
        'No checkpoint found. Pass --ckpt_path explicitly, or use --output_root '
        f'and --run_tag so predict.py can infer one.\nTried:\n  {candidates}'
    )


def build_loader(args):
    if args.dataset == 'her2st':
        args.dim_out = 785
        args.n_train_samples = 31
        val_data = ViT_HER2ST(train=False, flatten=False, ori=True, adj=False, fold=args.fold)
    else:
        args.dim_out = 171
        args.n_train_samples = 11
        val_data = ViT_SKIN(train=False, flatten=False, ori=True, adj=False, fold=args.fold)
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False, pin_memory=True)
    return val_loader


def predict(args):
    torch.set_float32_matmul_precision('high')
    seed_torch(getattr(args, 'seed', 42))

    ckpt_path = resolve_ckpt_path(args)
    val_loader = build_loader(args)
    model = FinalModel(args=args)

    accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'
    devices = [args.device_id] if torch.cuda.is_available() else 1
    trainer = pl.Trainer(
        precision=32,
        accelerator=accelerator,
        devices=devices,
        enable_progress_bar=False,
        logger=False,
    )
    print(f'Loading checkpoint: {ckpt_path}')
    trainer.test(model, val_loader, ckpt_path=ckpt_path)

    pred = model.data.X
    truth_g, _, _, centers, _, _ = next(iter(val_loader))
    truth = truth_g.squeeze(0).numpy()
    spatial = centers.squeeze(0).numpy()
    metrics = get_metrics(pred, truth)

    print(
        f"Fold {args.fold} | gene_PCC={metrics['mean_gene_pcc']:.4f}  "
        f"spot_PCC={metrics['mean_spot_pcc']:.4f}  "
        f"Spearman={metrics['mean_gene_spearman']:.4f}  "
        f"RMSE={metrics['rmse']:.4f}"
    )

    if getattr(args, 'save_pred', ''):
        save_path = os.path.expanduser(args.save_pred)
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        np.save(
            save_path,
            {
                'pred': pred,
                'truth': truth,
                'spatial': spatial,
                'gene_pcc': model.p,
                'gene_pvalue': model.r,
                'metrics': metrics,
                'checkpoint': ckpt_path,
                'args': vars(args),
            },
        )
        print(f'Saved prediction: {save_path}')


if __name__ == '__main__':
    predict(parser_option())
