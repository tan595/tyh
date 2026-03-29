"""
Progressive Dynamic Graph modules for PDG-ST.
- EMAFeatureBank: per-section EMA feature cache for stable graph construction
- DynamicGraphBuilder: spatial pre-filter + feature top-k graph construction
- GraphBlock: lightweight graph reasoning with residual + LayerNorm
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors


class EMAFeatureBank:
    """
    Per-section EMA feature bank.
    Uses stop_grad EMA features for graph construction to decouple
    the feature->edge_weight backward path and stabilize training.
    """
    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.bank = {}  # section_id -> ema_tensor (on CPU to save GPU mem)

    @torch.no_grad()
    def update(self, section_id, feat):
        """Update EMA for a specific section."""
        feat_cpu = feat.detach().cpu()
        if section_id not in self.bank:
            self.bank[section_id] = feat_cpu.clone()
        else:
            old = self.bank[section_id]
            if old.shape == feat_cpu.shape:
                self.bank[section_id] = self.momentum * old + (1 - self.momentum) * feat_cpu
            else:
                self.bank[section_id] = feat_cpu.clone()

    @torch.no_grad()
    def get(self, section_id, fallback, device):
        """Get EMA features; fallback to detached current features if not cached."""
        if section_id in self.bank:
            return self.bank[section_id].to(device)
        return fallback.detach()


class DynamicGraphBuilder(nn.Module):
    """
    Two-stage graph construction:
    1. Spatial pre-filter: select candidate neighbors within spatial radius (O(N*K))
    2. Feature top-k: rank candidates by feature similarity, keep top-k

    This avoids O(N^2) full similarity computation while maintaining dynamic topology.
    """
    def __init__(self, spatial_k=30, feature_k=6):
        super().__init__()
        self.spatial_k = spatial_k
        self.feature_k = feature_k

    @torch.no_grad()
    def forward(self, ema_feat, pos):
        """
        Args:
            ema_feat: [N, D] EMA features (stop_grad) for similarity computation
            pos: [N, 2] spatial coordinates (integer grid positions)
        Returns:
            topk_indices: [N, feature_k] neighbor indices
            topk_weights: [N, feature_k] normalized attention weights
        """
        N = ema_feat.shape[0]
        device = ema_feat.device
        k_spatial = min(self.spatial_k, N - 1)
        k_feat = min(self.feature_k, k_spatial)

        # Step 1: spatial pre-filter using KDTree (O(N log N) instead of O(N^2))
        pos_np = pos.detach().cpu().float().numpy()
        nn_model = NearestNeighbors(n_neighbors=k_spatial + 1, algorithm='kd_tree')
        nn_model.fit(pos_np)
        _, indices = nn_model.kneighbors(pos_np)
        # Exclude self (first column)
        spatial_candidates = torch.from_numpy(
            indices[:, 1:k_spatial + 1]
        ).long().to(device)  # [N, k_spatial]

        # Step 2: feature similarity within candidates
        ema_feat_norm = F.normalize(ema_feat, dim=-1)  # [N, D]
        # Gather candidate features
        cand_feat = ema_feat_norm[spatial_candidates]  # [N, k_spatial, D]
        query_feat = ema_feat_norm.unsqueeze(1)  # [N, 1, D]
        sim = (query_feat * cand_feat).sum(-1)  # [N, k_spatial] cosine similarity

        # Top-k by feature similarity
        topk_sim, topk_local_idx = torch.topk(sim, k_feat, dim=-1)  # [N, k_feat]
        # Map local indices back to global indices
        topk_indices = torch.gather(spatial_candidates, 1, topk_local_idx)  # [N, k_feat]
        topk_weights = F.softmax(topk_sim, dim=-1)  # [N, k_feat]

        return topk_indices, topk_weights


class GraphBlock(nn.Module):
    """
    Lightweight graph reasoning block with gated knowledge attention.
    Reuses WiKG's core mechanism but:
    - Only 1 layer (prevents over-smoothing)
    - Residual connection + LayerNorm
    - Accepts external graph topology (topk_indices, topk_weights)
      instead of computing its own

    Args:
        dim: feature dimension
        dropout: dropout rate
    """
    def __init__(self, dim, dropout=0.3):
        super().__init__()
        self.W_head = nn.Linear(dim, dim)
        self.W_tail = nn.Linear(dim, dim)

        # Gated knowledge attention
        self.gate_U = nn.Linear(dim, dim // 2)
        self.gate_V = nn.Linear(dim, dim // 2)
        self.gate_W = nn.Linear(dim // 2, dim)

        # Bi-interaction aggregation
        self.linear1 = nn.Linear(dim, dim)
        self.linear2 = nn.Linear(dim, dim)
        self.activation = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, e_h_input, e_t_input, topk_indices, topk_weights):
        """
        Args:
            x: [N, D] main features (for bi-interaction)
            e_h_input: [N, D] head features (query side)
            e_t_input: [N, D] tail features (neighbor side)
            topk_indices: [N, K] neighbor indices from DynamicGraphBuilder
            topk_weights: [N, K] neighbor attention weights
        Returns:
            [N, D] graph-refined features (residual added outside)
        """
        e_h = self.W_head(e_h_input)  # [N, D]
        e_t = self.W_tail(e_t_input)  # [N, D]
        K = topk_indices.shape[1]

        # Gather neighbor tail features
        Nb_h = e_t[topk_indices]  # [N, K, D]

        # Weighted neighbor features using graph builder weights
        topk_prob = topk_weights.unsqueeze(-1)  # [N, K, 1]
        e_h_expand = e_h.unsqueeze(1).expand(-1, K, -1)  # [N, K, D]
        eh_r = topk_prob * Nb_h + (1 - topk_prob) * e_h_expand  # [N, K, D]

        # Gated knowledge attention
        gate = torch.tanh(e_h_expand + eh_r)  # [N, K, D]
        ka_weight = (Nb_h * gate).sum(-1)  # [N, K]
        ka_prob = F.softmax(ka_weight, dim=-1).unsqueeze(1)  # [N, 1, K]
        e_Nh = torch.bmm(ka_prob, Nb_h).squeeze(1)  # [N, D]

        # Bi-interaction aggregation
        sum_emb = self.activation(self.linear1(x + e_Nh))
        bi_emb = self.activation(self.linear2(x * e_Nh))
        h = self.dropout(sum_emb + bi_emb)
        h = self.norm(h)
        return h


class ChannelGate(nn.Module):
    """
    SE-style channel attention: squeeze (global avg pool) → excite (FC→ReLU→FC→Sigmoid).
    Recalibrates channel-wise features to suppress noisy dimensions.
    """
    def __init__(self, dim, reduction=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim, dim // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(dim // reduction, dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: [N, D]
        scale = self.fc(x.mean(dim=0, keepdim=True))  # [1, D]
        return x * scale


class GatedFusion(nn.Module):
    """
    Learnable gated residual: output = gate * graph_out + (1 - gate) * original.
    Gate is conditioned on both inputs, so the model learns when to trust
    graph reasoning vs. the original features.
    """
    def __init__(self, dim):
        super().__init__()
        self.gate_fc = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )
        self.channel_gate = ChannelGate(dim)

    def forward(self, original, graph_out):
        # Channel gate on graph output first
        graph_out = self.channel_gate(graph_out)
        # Learnable fusion gate
        gate = self.gate_fc(torch.cat([original, graph_out], dim=-1))  # [N, D]
        return gate * graph_out + (1 - gate) * original
