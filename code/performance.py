from scipy.stats import pearsonr, spearmanr
import torch
import numpy as np

def get_R(data1: torch.Tensor, data2: torch.Tensor, dim: int=1, func=pearsonr):
    r1, p1 = [], []
    for g in range(data1.shape[dim]):

        if dim == 1:
            r, pv = func(data1[:, g], data2[:, g])
        elif dim == 0:
            r, pv = func(data1[g, :], data2[g, :])
        r1.append(r)
        p1.append(pv)
    r1 = np.array(r1)
    p1 = np.array(p1)
    return r1, p1


def get_metrics(pred, truth):
    gene_pcc, _ = get_R(pred, truth, dim=1, func=pearsonr)
    spot_pcc, _ = get_R(pred, truth, dim=0, func=pearsonr)
    gene_spearman, _ = get_R(pred, truth, dim=1, func=spearmanr)
    rmse = np.sqrt(np.mean((pred - truth) ** 2))
    return {
        'gene_pcc': gene_pcc,
        'spot_pcc': spot_pcc,
        'gene_spearman': gene_spearman,
        'rmse': rmse,
        'mean_gene_pcc': np.nanmean(gene_pcc),
        'mean_spot_pcc': np.nanmean(spot_pcc),
        'mean_gene_spearman': np.nanmean(gene_spearman),
    }