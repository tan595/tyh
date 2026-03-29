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
    """
    Compute multiple evaluation metrics for ST prediction.
    Args:
        pred: [N, G] predicted gene expression
        truth: [N, G] ground truth gene expression
    Returns:
        dict with keys: gene_pcc, spot_pcc, gene_spearman, rmse
    """
    # Gene-wise PCC (per gene, averaged over spots)
    gene_pcc, _ = get_R(pred, truth, dim=1, func=pearsonr)
    # Spot-wise PCC (per spot, averaged over genes)
    spot_pcc, _ = get_R(pred, truth, dim=0, func=pearsonr)
    # Gene-wise Spearman
    gene_spearman, _ = get_R(pred, truth, dim=1, func=spearmanr)
    # RMSE
    rmse = np.sqrt(np.mean((pred - truth) ** 2))

    return {
        'gene_pcc': gene_pcc,           # [G] per-gene PCC
        'spot_pcc': spot_pcc,           # [N] per-spot PCC
        'gene_spearman': gene_spearman, # [G] per-gene Spearman
        'rmse': rmse,                   # scalar
        'mean_gene_pcc': np.nanmean(gene_pcc),
        'mean_spot_pcc': np.nanmean(spot_pcc),
        'mean_gene_spearman': np.nanmean(gene_spearman),
    }