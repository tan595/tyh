from herst import ViT_HER2ST, ViT_SKIN
from model import Reg2ST
from model_pdg import PDG_ST
from model_mrcmr import MRCMR
from pytorch_lightning.loggers import CSVLogger
import torch
import numpy as np
import os


from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from performance import get_R
from utils import *

def predict(args):
    torch.set_float32_matmul_precision('high')
    model_name = getattr(args, 'model', 'reg2st')
    if args.dataset == "her2st":
        val_data = ViT_HER2ST(train=False, flatten=False,ori=True, adj=False, fold=args.fold)
        args.dim_out = 785
    else:
        val_data = ViT_SKIN(train=False, flatten=False, ori=True, adj=False, fold=args.fold)
        args.dim_out = 171
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False, pin_memory=True)
    ckpt_dir = getattr(args, 'ckpt', None)
    if ckpt_dir is None:
        # Try fold-specific first, then last.ckpt
        fold_ckpt = f"/root/autodl-tmp/reg2st_outputs/{args.dataset}_{model_name}_model/fold{args.fold}_model.ckpt"
        last_ckpt = f"/root/autodl-tmp/reg2st_outputs/{args.dataset}_{model_name}_model/last.ckpt"
        if os.path.exists(fold_ckpt):
            ckpt_dir = fold_ckpt
        elif os.path.exists(last_ckpt):
            ckpt_dir = last_ckpt
        else:
            raise FileNotFoundError(f"No checkpoint found in /root/autodl-tmp/reg2st_outputs/{args.dataset}_{model_name}_model/")
    print(f"Loading checkpoint from {ckpt_dir}")
    if model_name == 'pdg_st':
        model = PDG_ST(args)
    elif model_name == 'mrcmr':
        model = MRCMR(args)
    else:
        model = Reg2ST(args)
    trainer = pl.Trainer(precision=32, max_epochs=args.epochs,
                         accelerator='gpu', devices=[args.device_id])
    trainer.test(model, val_loader, ckpt_path=ckpt_dir)
        
if __name__ == "__main__":
    args = parser_option()
    predict(args)
    