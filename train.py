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

def train(args):
    torch.set_float32_matmul_precision('high')
    i = args.fold
    args.divide_size = 1
    model_name = getattr(args, 'model', 'reg2st')
    save_dir = f"/root/autodl-tmp/reg2st_outputs/{args.dataset}_{model_name}_model/"
    os.makedirs(save_dir, exist_ok=True)

    val_checkpoint_callback = ModelCheckpoint(
    dirpath=save_dir,
    filename=f"fold{i}_model",
    save_last=True,
    save_top_k=1,
    monitor='pcc',
    mode='max',
    save_on_train_epoch_end=False
    )
    logger_name = f'/root/autodl-tmp/reg2st_outputs/{args.dataset}_{model_name}_logs/'
    csv_logger = CSVLogger(logger_name, name=f"fold{i}")
    if args.dataset == "her2st":
        train_data = ViT_HER2ST(train=True, flatten=False,ori=True, adj=False, fold=i)
        val_data = ViT_HER2ST(train=False, flatten=False,ori=True, adj=False, fold=i)
    else:
        args.dim_out = 171
        train_data = ViT_SKIN(train=True, flatten=False, ori=True, adj=False, fold=i)
        val_data = ViT_SKIN(train=False, flatten=False, ori=True, adj=False, fold=i)

    train_loader = DataLoader(train_data, batch_size=1, shuffle=False, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False, pin_memory=True)

    if model_name == 'pdg_st':
        print(f"Using PDG-ST model")
        model = PDG_ST(args=args)
    elif model_name == 'mrcmr':
        print(f"Using MR-CMR model")
        model = MRCMR(args=args)
    else:
        print(f"Using Reg2ST model")
        model = Reg2ST(args=args)

    trainer = pl.Trainer(logger=csv_logger, precision=32, max_epochs=args.epochs,
                         accelerator='gpu', devices=[args.device_id],
                         callbacks=[val_checkpoint_callback],
                         log_every_n_steps=5)
    trainer.fit(model, train_loader, val_loader)
       
        
if __name__ == "__main__":
    args = parser_option()
    train(args)
    