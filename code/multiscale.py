import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleNeighborFusion(nn.Module):

    def __init__(self, dim, n_heads=4, dropout=0.1):
        super().__init__()
        if dim % n_heads != 0:
            raise ValueError(
                f"dim={dim} must be divisible by n_heads={n_heads}")

        self.scale_weights = nn.Parameter(torch.ones(3))
        self.cross_attn = nn.MultiheadAttention(
            dim, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim * 2, dim))
        self.norm2 = nn.LayerNorm(dim)

    @staticmethod
    def _dist_weighted_mean(i_f, idx, dist):
        feats = i_f[idx]
        w = F.softmax(-dist, dim=-1)
        return (feats * w.unsqueeze(-1)).sum(dim=1)

    def forward(self, i_f, x_emb, y_emb,
                knn_small_idx, knn_small_dist,
                knn_large_idx, knn_large_dist,
                knn_query_idx, knn_query_dist):
        s1 = i_f
        s_s = self._dist_weighted_mean(i_f, knn_small_idx, knn_small_dist)
        s_l = self._dist_weighted_mean(i_f, knn_large_idx, knn_large_dist)
        s_xl = self._dist_weighted_mean(i_f, knn_query_idx, knn_query_dist)
        w = F.softmax(self.scale_weights, dim=0)
        fused = w[0] * s1 + w[1] * s_s + w[2] * s_l
        q = (s_xl + x_emb + y_emb).unsqueeze(1)
        kv = torch.stack([s1, s_s, s_l, fused], dim=1)
        out, _ = self.cross_attn(q, kv, kv)
        out = self.norm1(q + out).squeeze(1)
        out = self.norm2(out + self.ffn(out))
        return out, (s1, s_s, s_l, fused)
