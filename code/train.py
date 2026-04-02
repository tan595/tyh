from herst import ViT_HER2ST, ViT_SKIN
from model import Reg2ST
from model_final import FinalModel
from pytorch_lightning.loggers import CSVLogger
import torch
import os
from datetime import datetime

from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from utils import *


def train(args):
    torch.set_float32_matmul_precision('high')
    seed_torch(42)
    fold = args.fold
    args.divide_size = 1

    model_name = getattr(args, 'model', 'reg2st')
    output_root = getattr(args, 'output_root', '/root/autodl-tmp/final_outputs')
    run_tag = getattr(args, 'run_tag', None)
    if run_tag is None or str(run_tag).strip() == '':
        run_tag = datetime.utcnow().strftime('%Y%m%d_%H%M%S') + f'_{os.getpid()}'
    tag_suffix = f'_{run_tag}'
    print(f'Run tag: {run_tag}')

    save_dir = os.path.join(output_root, f"{args.dataset}_{model_name}_model{tag_suffix}")
    os.makedirs(save_dir, exist_ok=True)

    val_checkpoint_callback = ModelCheckpoint(
        dirpath=save_dir,
        filename=f"fold{fold}_model",
        save_last=True,
        save_top_k=1,
        monitor='pcc',
        mode='max',
        save_on_train_epoch_end=False,
    )

    logger_name = os.path.join(output_root, f"{args.dataset}_{model_name}_logs{tag_suffix}")
    csv_logger = CSVLogger(logger_name, name=f"fold{fold}")

    if args.dataset == 'her2st':
        train_data = ViT_HER2ST(train=True, flatten=False, ori=True, adj=False, fold=fold)
        val_data = ViT_HER2ST(train=False, flatten=False, ori=True, adj=False, fold=fold)
    else:
        args.dim_out = 171
        train_data = ViT_SKIN(train=True, flatten=False, ori=True, adj=False, fold=fold)
        val_data = ViT_SKIN(train=False, flatten=False, ori=True, adj=False, fold=fold)

    train_loader = DataLoader(train_data, batch_size=1, shuffle=False, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False, pin_memory=True)
    args.n_train_samples = len(train_data)

    if model_name == 'final':
        flags = []
        if getattr(args, 'use_multires', False):
            flags.append('MR')
        if getattr(args, 'use_heclip', False):
            flags.append(f'HECLIP(temp={args.heclip_target_temp},lam={args.heclip_mix_lambda})')
        print(f'Using FinalModel [{"+".join(flags) if flags else "baseline"}]')
        model = FinalModel(args=args)
    else:
        print('Using Reg2ST baseline model')
        model = Reg2ST(args=args)

    trainer = pl.Trainer(
        logger=csv_logger,
        precision=32,
        max_epochs=args.epochs,
        accelerator='gpu',
        devices=[args.device_id],
        callbacks=[val_checkpoint_callback],
        log_every_n_steps=5,
    )
    trainer.fit(model, train_loader, val_loader)


if __name__ == '__main__':
    args = parser_option()
    train(args)
