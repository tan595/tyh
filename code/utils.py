import random
from typing import List
import pandas as pd
import scanpy as sc
import math
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
try:
    import torch.distributed.optim as optim
    from torch.distributed.optim import zero_redundancy_optimizer
    import torch.distributed as dist
except Exception:
    pass

import logging

from timm.scheduler.cosine_lr import CosineLRScheduler
from timm.scheduler.step_lr import StepLRScheduler
from timm.scheduler.scheduler import Scheduler
from collections import OrderedDict
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import h5py
from torch_geometric.data import Data
import yaml
import argparse
import os

from loss import DiceLoss, BinaryCrossEntropyLoss, CrossEntropyLoss
import types

class LR_Scheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_epochs, warmup_lr, num_epochs, base_lr, final_lr, iter_per_epoch, constant_predictor_lr=False):
        self.base_lr = base_lr
        self.constant_predictor_lr = constant_predictor_lr
        warmup_iter = iter_per_epoch * warmup_epochs
        warmup_lr_schedule = np.linspace(warmup_lr, base_lr, warmup_iter)
        decay_iter = iter_per_epoch * (num_epochs - warmup_epochs)
        cosine_lr_schedule = final_lr + 0.5 * (base_lr - final_lr) * (1 + np.cos(np.pi * np.arange(decay_iter) / decay_iter))
        self.lr_schedule = np.concatenate((warmup_lr_schedule, cosine_lr_schedule))
        self.optimizer = optimizer
        self.iter = 0
        self.current_lr = 0

    def step(self):
        for param_group in self.optimizer.param_groups:
            if self.constant_predictor_lr and param_group['name'] == 'predictor':
                param_group['lr'] = self.base_lr
            else:
                lr = param_group['lr'] = self.lr_schedule[self.iter]
        self.iter += 1
        self.current_lr = lr
        return lr

    def get_lr(self):
        return self.current_lr

class MeanAct(nn.Module):
    def __init__(self):
        super(MeanAct, self).__init__()

    def forward(self, x):
        return torch.clamp(torch.exp(x), min=1e-5, max=1e6)

class DispAct(nn.Module):
    def __init__(self):
        super(DispAct, self).__init__()

    def forward(self, x):
        return torch.clamp(F.softplus(x), min=1e-4, max=1e4)

def ZINB_loss(x, mean, disp, pi, scale_factor=1.0, ridge_lambda=0.0):
    eps = 1e-10
    if isinstance(scale_factor, float):
        scale_factor = np.full((len(mean),), scale_factor)
    scale_factor = scale_factor[:, None]
    mean = mean * scale_factor
    t1 = torch.lgamma(disp + eps) + torch.lgamma(x + 1.0) - torch.lgamma(x + disp + eps)
    t2 = (disp + x) * torch.log(1.0 + (mean / (disp + eps))) + (x * (torch.log(disp + eps) - torch.log(mean + eps)))
    nb_final = t1 + t2
    nb_case = nb_final - torch.log(1.0 - pi + eps)
    zero_nb = torch.pow(disp / (disp + mean + eps), disp)
    zero_case = -torch.log(pi + ((1.0 - pi) * zero_nb) + eps)
    result = torch.where(torch.le(x, 1e-8), zero_case, nb_case)
    if ridge_lambda > 0:
        ridge = ridge_lambda * torch.square(pi)
        result += ridge
    result = torch.mean(result)
    return result

def generate_masked_tensor(n, zero_prob=0.75):
    matrix = torch.ones(n, n)
    rand_tensor = torch.rand(n, n)
    mask = (rand_tensor < zero_prob) & (torch.eye(n) == 0)
    matrix[mask] = 0
    return matrix

def list_npy_files(directory):
    npy_files = []
    for file in os.listdir(directory):
        if file.endswith('.npy'):
            npy_files.append(os.path.join(directory, file))
    return npy_files

def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
def update_config_from_file(cfg_file):
    with open(cfg_file, 'r') as stream:
        params = yaml.safe_load(stream)
    return params

def dict_to_namespace(d):
    namespace = types.SimpleNamespace()
    for key, value in d.items():
        if isinstance(value, dict):
            setattr(namespace, key, dict_to_namespace(value))
        else:
            setattr(namespace, key, value)
    return namespace
def parser_option():
    parser = argparse.ArgumentParser('gene prediction', add_help=True)
    parser.add_argument('--name', type=str, default='test_model')
    parser.add_argument('--save_name', type=str, help='path saves xg and xi')
    parser.add_argument('--cfg_file', type=str, default='config.yaml')


    parser.add_argument('--dataset', type=str, default='her2st')


    parser.add_argument('--dim_in', type=int, default=1024)
    parser.add_argument('--dim_hidden', type=int, default=1024)
    parser.add_argument('--dim_out', type=int, default=785)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--wikg_top', type=int, default=6)
    parser.add_argument('--decoder_layer', type=int, default=6)
    parser.add_argument('--decoder_head', type=int, default=8)


    parser.add_argument('--mask_rate', type=float, default=0.75)
    parser.add_argument('--w_con', type=float, default=0.5)
    parser.add_argument('--w_zinb', type=float, default=0.25)


    parser.add_argument('--heclip_target_temp', type=float, default=0.5)
    parser.add_argument('--heclip_mix_lambda', type=float, default=0.2,
                        help='HECLIP weight in mix: con = (1-lam)*hard + lam*heclip')


    parser.add_argument('--multiscale_k_small', type=int, default=4,
                        help='K for small neighbourhood KV scale (local context)')
    parser.add_argument('--multiscale_k_large', type=int, default=7,
                        help='K for large neighbourhood KV scale (regional context)')
    parser.add_argument('--multiscale_k_query', type=int, default=14,
                        help='K for CCF query scale (k_query > k_large, fully independent of KV)')
    parser.add_argument('--multiscale_heads', type=int, default=4,
                        help='attention heads in MultiScaleNeighborFusion CCF')
    parser.add_argument('--w_branch', type=float, default=0.1,
                        help='weight for HiFusion branch supervision loss')


    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--fold', type=int, default=0, help='fold number')
    parser.add_argument('--device_id', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed for reproducible training')
    parser.add_argument('--output_root', type=str, default='/root/autodl-tmp/final_outputs',
                        help='root directory for outputs')
    parser.add_argument('--folds', type=str, default='',
                        help='comma-separated fold indices, empty = all folds')
    parser.add_argument('--run_tag', type=str, default='',
                        help='experiment tag for path disambiguation')
    parser.add_argument('--ckpt_path', type=str, default='',
                        help='checkpoint path used by predict.py')
    parser.add_argument('--save_pred', type=str, default='',
                        help='optional .npy path for predict.py outputs')

    args_cmd, unknown = parser.parse_known_args()
    if unknown:
        parser.error(
            f"unrecognised arguments: {' '.join(unknown)}\n"
            f"  Tip: code1 is the final full model only; use code2 for experimental switches."
        )



    cli_keys = set()
    for arg in os.sys.argv[1:]:
        token = arg.split('=')[0]
        if token.startswith('--'):
            cli_keys.add(token[2:])

    cfg_file = args_cmd.cfg_file
    if os.path.exists(cfg_file):
        yaml_params = update_config_from_file(cfg_file)
        for key, value in yaml_params.items():
            if key not in cli_keys and hasattr(args_cmd, key):
                setattr(args_cmd, key, value)
    return args_cmd

def find_folders(path: str) -> List:
    files = os.listdir(path)
    folders = [f for f in files if os.path.isdir(os.path.join(path, f))]
    return folders

def count_spot(path, name) -> List:

    folders = find_folders(path)

    count = []
    for folder in folders:
        file = os.path.join(path, folder, "spatial", name)

        df = pd.read_csv(file)

        spot_count = df["in_tissue"].value_counts()

        count.append({folder: spot_count[1]})
    return count

def check_spot_position(path, name):
    adata = sc.read_visium(path=path, count_file=name,load_images=True)
    print(adata.obsm["spatial"])

def test_ck_spot_position(path, name):
    check_spot_position(path, name)

def test_cnt_num(path, name):
    count_num = count_spot(path, name)
    print(count_num)

def large_mat_mul(input_a: np.ndarray, input_b: np.ndarray, batch: int =32) -> np.ndarray:
    m = input_a.shape[0]
    block_m = math.floor(m / batch)
    out = []
    for i in range(batch):
        start = i * block_m
        end = (i + 1) * block_m
        new_a = input_a[start:end]
        out_i = np.matmul(new_a, input_b)
        out.append(out_i)
    out = np.concatenate(out, axis=0)
    remain_a = input_a[batch * block_m:m]
    remain_o = np.matmul(remain_a, input_b)
    output = np.concatenate((out, remain_o), axis=0)
    return output

def mat_mul(input_a: np.ndarray, input_b: np.ndarray) -> np.ndarray:
    m = input_a.shape[0]
    if m > 100000:
        out = large_mat_mul(input_a, input_b)
    else:
        out = np.matmul(input_a, input_b)

    return out

def get_approximate_basis(matrix: np.ndarray,
                          q=6,
                          niter=2,
                          ) -> np.ndarray:
    niter = 2 if niter is None else niter
    _, n = matrix.shape[-2:]

    r = np.random.randn(n, q)

    matrix_t = matrix.T

    q, _ = np.linalg.qr(mat_mul(matrix, r))
    for _ in range(niter):
        q = np.linalg.qr(mat_mul(matrix_t, q))[0]
        q = np.linalg.qr(mat_mul(matrix, q))[0]
    return q

@torch.jit.script
def convert_to_single_emb(x: torch.Tensor, offset: int = 512) -> torch.Tensor:

    feature_num = x.size(1) if len(x.size()) > 1 else 1
    feature_offset = 1 + torch.arange(0, feature_num * offset, offset, dtype=torch.long)
    x = x + feature_offset
    return x

def check_keywords_in_name(name, keywords=()):
    isin = False
    for keyword in keywords:
        if keyword in name:
            isin = True
    return isin

def check_keywords_in_dict(name, keywords_dict):
    for k, v in keywords_dict.items():
        if k in name:
            return v
    return None


def set_weight_decay_and_lr(model, weight_decay, base_lr, skip_list=(), skip_keywords=(),
                            lr_layer_decay=None, lr_layer_decay_ratio=None, layerwise_lr=True):
    parameters = {}
    no_decay_name = []
    lr_ratio_log = {}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue


        if len(param.shape) == 1 or name.endswith(".bias") or (
                name in skip_list) or check_keywords_in_name(
                    name, skip_keywords):
            wd = 0.
            no_decay_name.append(name)
        else:
            wd = weight_decay


        if lr_layer_decay:

            assert hasattr(model, 'lr_decay_keywords')
            lr_ratio_keywords = model.lr_decay_keywords(lr_layer_decay_ratio)

            ratio = check_keywords_in_dict(name, lr_ratio_keywords)
            if ratio is not None:
                lr = ratio * base_lr
            else:
                lr = base_lr

            lr_ratio_log[name] = (base_lr, ratio, wd, param.requires_grad)
        else:
            lr = base_lr
        group_name = f"weight_decay_{str(wd)}_lr_{str(lr)}"
        if group_name not in parameters:
            parameters[group_name] = {'params': [param], 'weight_decay': wd, 'lr': lr}
        else:
            parameters[group_name]['params'].append(param)


    if layerwise_lr:

        for k, v in lr_ratio_log.items():
            print(k, v)
    parameters = list(parameters.values())
    return parameters

def check_keywords_in_name(name, keywords=()):
    isin = False
    for keyword in keywords:
        if keyword in name:
            isin = True
    return isin


def check_keywords_in_dict(name, keywords_dict):
    for k, v in keywords_dict.items():
        if k in name:
            return v
    return None


class LinearLRScheduler(Scheduler):

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        t_initial: int,
        lr_min_rate: float,
        warmup_t=0,
        warmup_lr_init=0.,
        t_in_epochs=True,
        noise_range_t=None,
        noise_pct=0.67,
        noise_std=1.0,
        noise_seed=42,
        initialize=True,
    ) -> None:
        super().__init__(optimizer,
                         param_group_field="lr",
                         noise_range_t=noise_range_t,
                         noise_pct=noise_pct,
                         noise_std=noise_std,
                         noise_seed=noise_seed,
                         initialize=initialize)

        self.t_initial = t_initial
        self.lr_min_rate = lr_min_rate
        self.warmup_t = warmup_t
        self.warmup_lr_init = warmup_lr_init
        self.t_in_epochs = t_in_epochs
        if self.warmup_t:
            self.warmup_steps = [(v - warmup_lr_init) / self.warmup_t
                                 for v in self.base_values]
            super().update_groups(self.warmup_lr_init)
        else:
            self.warmup_steps = [1 for _ in self.base_values]

    def _get_lr(self, t):
        if t < self.warmup_t:
            lrs = [self.warmup_lr_init + t * s for s in self.warmup_steps]
        else:
            t = t - self.warmup_t
            total_t = self.t_initial - self.warmup_t
            lrs = [
                v - ((v - v * self.lr_min_rate) * (t / total_t))
                for v in self.base_values
            ]
        return lrs

    def get_epoch_values(self, epoch: int):
        if self.t_in_epochs:
            return self._get_lr(epoch)
        else:
            return None

    def get_update_values(self, num_updates: int):
        if not self.t_in_epochs:
            return self._get_lr(num_updates)
        else:
            return None


def build_scheduler(config, optimizer, n_iter_per_epoch):
    num_steps = int(config.epochs * n_iter_per_epoch)

    warmup_steps = int(config.TRAIN.WARMUP_EPOCHS * n_iter_per_epoch)
    decay_steps = int(config.TRAIN.LR_SCHEDULER.DECAY_EPOCHS *
                      n_iter_per_epoch)

    print(f'warm_up_steps: {warmup_steps}')
    lr_scheduler = None

    if config.TRAIN.LR_SCHEDULER.NAME == 'cosine':
        lr_scheduler = CosineLRScheduler(
            optimizer,
            t_initial=num_steps,

            lr_min=float(config.TRAIN.MIN_LR),
            warmup_lr_init=float(config.TRAIN.WARMUP_LR),
            warmup_t=0,
            cycle_limit=1,
            t_in_epochs=False,
        )
        base_values = lr_scheduler.base_values
        l = []
        for v in base_values:
            l.append(float(v))
        lr_scheduler.base_values = l

    elif config.TRAIN.LR_SCHEDULER.NAME == 'linear':
        lr_scheduler = LinearLRScheduler(
            optimizer,
            t_initial=num_steps,
            lr_min_rate=0.01,
            warmup_lr_init=config.TRAIN.WARMUP_LR,
            warmup_t=warmup_steps,
            t_in_epochs=False,
        )
    elif config.TRAIN.LR_SCHEDULER.NAME == 'step':
        lr_scheduler = StepLRScheduler(
            optimizer,
            decay_t=decay_steps,
            decay_rate=config.TRAIN.LR_SCHEDULER.DECAY_RATE,
            warmup_lr_init=config.TRAIN.WARMUP_LR,
            warmup_t=warmup_steps,
            t_in_epochs=False,
        )

    return lr_scheduler

def build_criterion(config):
    if config.TRAIN.CRITERION == 'mse':
        criterion = torch.nn.MSELoss()
    elif config.TRAIN.CRITERION == 'bce':
        criterion = BinaryCrossEntropyLoss(config.TRAIN.CLASS_WEIGHTS, config.TRAIN.REDUCE_ZERO_LABEL)
    elif config.TRAIN.CRITERION == 'ce':
        criterion = CrossEntropyLoss(config.TRAIN.CLASS_WEIGHTS, config.TRAIN.REDUCE_ZERO_LABEL)
    elif config.TRAIN.CRITERION == 'dice':
        criterion = DiceLoss(config.TRAIN.REDUCE_ZERO_LABEL)
    else:
        raise ValueError(f'unknown {config.TRAIN.CRITERION}')
    return criterion


def auto_resume_helper(output_dir):
    checkpoints = os.listdir(output_dir)
    checkpoints = [ckpt for ckpt in checkpoints if ckpt.endswith('pth')]
    print(f'All checkpoints founded in {output_dir}: {checkpoints}')
    if len(checkpoints) > 0:
        latest_checkpoint = max(
            [os.path.join(output_dir, d) for d in checkpoints],
            key=os.path.getmtime)
        print(f'The latest checkpoint founded: {latest_checkpoint}')
        resume_file = latest_checkpoint
    else:
        resume_file = None
    return resume_file

def load_ema_checkpoint(config, path, model_ema, logger):
    logger.info(
        f'==============> Resuming form {config.MODEL.RESUME}....................'
    )
    checkpoint = torch.load(path, map_location='cpu')

    assert isinstance(checkpoint, dict)
    if 'model_ema' in checkpoint:
        new_state_dict = OrderedDict()
        for k, v in checkpoint['model_ema'].items():
            if model_ema.ema_has_module:
                name = 'module.' + k if not k.startswith('module') else k
            else:
                name = k
            new_state_dict[name] = v
        msg = model_ema.ema.load_state_dict(new_state_dict, strict=False)
        logger.info(msg)
        logger.info('Loaded state_dict_ema')
    else:
        logger.warning(
            'Failed to find state_dict_ema, starting from loaded model weights'
        )

    performance_ema = 0
    if 'performance_ema' in checkpoint:
        performance_ema = checkpoint['performance_ema']
    if 'ema_decay' in checkpoint:
        model_ema.decay = checkpoint['ema_decay']
    return performance_ema


def load_checkpoint(config, path, model, optimizer, lr_scheduler, scaler, logger):


    checkpoint = torch.load(path, map_location='cpu')

    print('resuming model')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    logger.info(msg)
    performance = 0.0
    if not config.EVAL_MODE and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
        if optimizer is not None:
            print('resuming optimizer')
            try:
                optimizer.load_state_dict(checkpoint['optimizer'])
            except:
                print('resume optimizer failed')
        if lr_scheduler is not None:
            print('resuming lr_scheduler')
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])



        if 'amp' in checkpoint and config.amp_opt_level != 'O0' and checkpoint[
                'config'].amp_opt_level != 'O0':
            scaler.load_state_dict(checkpoint['amp'])
        logger.info(
            f"=> loaded successfully {path} (epoch {checkpoint['epoch']})"
        )
        if 'performance' in checkpoint:
            performance = checkpoint['performance']




    return performance

def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt

def pad_1d_unsqueeze(x, padlen):
    x = x + 1
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen], dtype=x.dtype)
        new_x[:xlen] = x
        x = new_x
    return x.unsqueeze(0)


def pad_2d_unsqueeze(x, padlen):
    x = x + 1
    xlen, xdim = x.size()
    if xlen < padlen:
        new_x = x.new_zeros([padlen, xdim], dtype=x.dtype)
        new_x[:xlen, :] = x
        x = new_x
    return x.unsqueeze(0)


def pad_attn_bias_unsqueeze(x, padlen):
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen], dtype=x.dtype).fill_(float("-inf"))
        new_x[:xlen, :xlen] = x
        new_x[xlen:, :xlen] = 0
        x = new_x
    return x.unsqueeze(0)


def pad_edge_type_unsqueeze(x, padlen):
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen, x.size(-1)], dtype=x.dtype)
        new_x[:xlen, :xlen, :] = x
        x = new_x
    return x.unsqueeze(0)


def pad_spatial_pos_unsqueeze(x, padlen):
    x = x + 1
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen], dtype=x.dtype)
        new_x[:xlen, :xlen] = x
        x = new_x
    return x.unsqueeze(0)

def pad_1d_unsqueeze_nan(x, padlen):
    x = x + 1
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen], dtype=x.dtype).float()
        new_x[:] = float('nan')
        new_x[:xlen] = x
        x = new_x
    return x.unsqueeze(0)

def pad_3d_unsqueeze(x, padlen1, padlen2, padlen3):
    x = x + 1
    xlen1, xlen2, xlen3, xlen4 = x.size()
    if xlen1 < padlen1 or xlen2 < padlen2 or xlen3 < padlen3:
        new_x = x.new_zeros([padlen1, padlen2, padlen3, xlen4], dtype=x.dtype)
        new_x[:xlen1, :xlen2, :xlen3, :] = x
        x = new_x
    return x.unsqueeze(0)

def collator(items, max_node=1024, multi_hop_max_dist=20, spatial_pos_max=20):




    items = [item for item in items if item is not None and item['x'].size(0) <= max_node]
    items = [
        (
            item['edge_feature'],
            item['attn_edge_type'],
            item['spatial_pos'],


            item['edge_attr'],
            item['edge_index'],
            item['x'],
            item['edge_input'][:, :, :multi_hop_max_dist, :],

            item['y']
        )
        for item in items
    ]
    (
        attn_biases,
        attn_edge_types,
        spatial_poses,


        edge_attrs,
        edge_indexes,
        xs,
        edge_inputs,
        ys,
    ) = zip(*items)

    for idx, _ in enumerate(attn_biases):
        attn_biases[idx][1:, 1:][spatial_poses[idx] >= spatial_pos_max] = float("-inf")
    max_node_num = max(i.size(0) for i in xs)
    max_dist = max(i.size(-2) for i in edge_inputs)



    if ys[0].size(0) == 1:
        y = torch.cat(ys)
    else:
        try:
            max_edge_num = max([y.size(0) for y in ys])
            y = torch.cat([pad_1d_unsqueeze_nan(i, max_edge_num) for i in ys])
        except:
            y = torch.cat([pad_1d_unsqueeze_nan(i, max_node_num) for i in ys])

    x = torch.cat([pad_2d_unsqueeze(i, max_node_num) for i in xs])
    edge_input = torch.cat(
        [pad_3d_unsqueeze(i, max_node_num, max_node_num, max_dist) for i in edge_inputs]
    )
    attn_bias = torch.cat(
        [pad_attn_bias_unsqueeze(i, max_node_num + 1) for i in attn_biases]
    )
    attn_edge_type = torch.cat(
        [pad_edge_type_unsqueeze(i, max_node_num) for i in attn_edge_types]
    )
    spatial_pos = torch.cat(
        [pad_spatial_pos_unsqueeze(i, max_node_num) for i in spatial_poses]
    )





    return dict(
        attn_bias=attn_bias,
        attn_edge_type=attn_edge_type,
        spatial_pos=spatial_pos,



        x=x,
        edge_input=edge_input,
        y=y
    )


def accuracy_SBM(scores, targets):

    S = targets.cpu().numpy()
    if scores.size(-1) == 1:
        C = torch.sigmoid(scores) < 0.5
        C = (torch.where(C, 0, 1)).cpu().numpy()
        C = C.squeeze(-1)
    else:
        C = scores.argmax(-1).cpu().numpy()
    CM = confusion_matrix(S, C).astype(np.float32)

    nb_classes = CM.shape[0]
    targets = targets.cpu().detach().numpy()
    nb_non_empty_classes = 0
    pr_classes = np.zeros(nb_classes)
    for r in range(nb_classes):
        cluster = np.where(targets==r)[0]
        if cluster.shape[0] != 0:
            pr_classes[r] = CM[r,r] / float(cluster.shape[0])
            if CM[r,r] > 0:
                nb_non_empty_classes += 1
        else:
            pr_classes[r] = 0.0
    print("pre classes acc:", pr_classes)
    acc = 100.* np.sum(pr_classes)/ float(nb_classes)
    return torch.tensor(acc, device=scores.device)

class NativeScalerWithGradNormCount:
    state_dict_key = 'amp_scaler'

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                self._scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            d = optimizer.param_groups[0]
            optimizer.param_groups[0]['lr'] = float(d['lr'])
            d = optimizer.param_groups[1]
            optimizer.param_groups[1]['lr'] = float(d['lr'])

            self._scaler.step(optimizer)
            self._scaler.update()

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)



if __name__ == "__main__":
    path = "data/10x"
    name = "tissue_positions_list.csv"
    test_ck_spot_position(os.path.join(path, "mouse_brain_serial_section2_anterior"), "filtered_feature_bc_matrix.h5")
