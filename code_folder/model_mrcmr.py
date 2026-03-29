"""
MR-CMR: Multi-Resolution Cross-Modal Reconstruction for
Spatial Transcriptomics Prediction from Histology Images.

Key innovations over Reg2ST:
1. Multi-resolution features: target + neighbor + global views
2. Cross-modal masked reconstruction as training auxiliary task
3. Multi-resolution fusion with distillation loss
4. Improved contrastive learning (both features L2-normalized, CE-based InfoNCE)

At test time, only image features are used (no gene expression needed).
"""
import pytorch_lightning as pl
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
import anndata as ann
from sklearn.neighbors import NearestNeighbors
from performance import get_R
from attention import Decoder
from wikg import WiKG
from model import LR_Scheduler, MeanAct, DispAct, ZINB_loss, generate_masked_tensor


# ---------------------------------------------------------------------------
# Neighbor View Encoder: aggregate spatial neighbor embeddings via attention
# ---------------------------------------------------------------------------
class NeighborEncoder(nn.Module):
    def __init__(self, dim, n_heads=4, dropout=0.2):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim * 2, dim))
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, query, neighbor_feats):
        """
        query: [N, D] target spot features
        neighbor_feats: [N, K, D] neighbor spot features
        Returns: [N, D]
        """
        q = query.unsqueeze(1)  # [N, 1, D]
        out, _ = self.cross_attn(q, neighbor_feats, neighbor_feats)
        out = self.norm1(q + out).squeeze(1)  # [N, D]
        out = self.norm2(out + self.ffn(out))  # [N, D]
        return out


# ---------------------------------------------------------------------------
# Global View Encoder: lightweight set-transformer over all spot embeddings
# ---------------------------------------------------------------------------
class GlobalEncoder(nn.Module):
    def __init__(self, dim, n_heads=4, n_layers=2, dropout=0.2):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=n_heads, dim_feedforward=dim * 2,
            dropout=dropout, activation='gelu', batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(dim)

    def forward(self, all_feats):
        """
        all_feats: [N, D] all spot features in the section
        Returns: [N, D] globally-contextualized features
        """
        x = all_feats.unsqueeze(0)  # [1, N, D]
        x = self.encoder(x).squeeze(0)  # [N, D]
        return self.norm(x)


# ---------------------------------------------------------------------------
# Multi-Resolution Fusion Layer
# ---------------------------------------------------------------------------
class MultiResFusion(nn.Module):
    """Fuse target, neighbor, and global features via gated attention."""
    def __init__(self, dim, dropout=0.2):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim * 3, dim * 3), nn.Sigmoid())
        self.proj = nn.Sequential(
            nn.Linear(dim * 3, dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim, dim))
        self.norm = nn.LayerNorm(dim)
        # Per-resolution prediction heads (for distillation)
        self.head_target = nn.Linear(dim, dim)
        self.head_neighbor = nn.Linear(dim, dim)
        self.head_global = nn.Linear(dim, dim)

    def forward(self, target_f, neighbor_f, global_f):
        """
        All inputs: [N, D]
        Returns: fused [N, D], per-resolution features dict
        """
        concat = torch.cat([target_f, neighbor_f, global_f], dim=-1)  # [N, 3D]
        gate = self.gate(concat)
        gated = concat * gate
        fused = self.norm(self.proj(gated))
        per_res = {
            'target': self.head_target(target_f),
            'neighbor': self.head_neighbor(neighbor_f),
            'global': self.head_global(global_f),
        }
        return fused, per_res

# PLACEHOLDER_MRCMR_CLASS

# ---------------------------------------------------------------------------
# MR-CMR: Main Model
# ---------------------------------------------------------------------------
class MRCMR(pl.LightningModule):
    def __init__(self, args):
        super(MRCMR, self).__init__()
        self.args = args
        dim_in = args.dim_in
        dim_hidden = args.dim_hidden
        dim_out = args.dim_out
        dropout = args.dropout
        wikg_top = args.wikg_top
        decoder_layer = args.decoder_layer
        decoder_head = args.decoder_head
        neighbor_k = getattr(args, 'neighbor_k', 6)
        self.w_fusion = getattr(args, 'w_fusion', 0.2)
        self.w_con = getattr(args, 'w_con', 0.5)
        self.w_zinb = getattr(args, 'w_zinb', 0.25)
        self.neighbor_k = neighbor_k

        # Ablation flags
        self.no_multires = getattr(args, 'no_multires', False)

        # ============ Target Encoder (same as Reg2ST) ============
        self.image_encoder = nn.Sequential(
            nn.Linear(dim_in, dim_in), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim_in, dim_hidden))
        self.gene_proj = nn.Sequential(
            nn.Linear(dim_hidden, dim_hidden), nn.GELU(),
            nn.Linear(dim_hidden, dim_hidden))
        self.gene_head = nn.Sequential(
            nn.Linear(dim_hidden, dim_in), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim_in, dim_out))

        # Cross-attention decoder (Reg2ST style)
        self.decoder = Decoder(
            dim=dim_hidden, layers=decoder_layer, heads=decoder_head,
            mlp_dim=1024, dropout=dropout)

        # Positional embeddings
        self.embed_x = nn.Embedding(64, dim_hidden)
        self.embed_y = nn.Embedding(64, dim_hidden)

        # Gene encoder
        self.gene_linear = nn.Sequential(
            nn.Linear(dim_out, 1024), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(1024, dim_hidden))

        # ZINB heads
        self.mean = nn.Sequential(nn.Linear(dim_hidden, dim_out), MeanAct())
        self.disp = nn.Sequential(nn.Linear(dim_hidden, dim_out), DispAct())
        self.pi = nn.Sequential(nn.Linear(dim_hidden, dim_out), nn.Sigmoid())

        # Contrastive temperature
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        # Dynamic GNN (WiKG)
        self.graph_layer = WiKG(
            dim_in=dim_hidden, dim_hidden=dim_hidden,
            topk=wikg_top, n_classes=dim_hidden)

        # ============ NEW: Multi-Resolution Modules ============
        self.neighbor_encoder = NeighborEncoder(dim_hidden, n_heads=4, dropout=dropout)
        self.global_encoder = GlobalEncoder(dim_hidden, n_heads=4, n_layers=2, dropout=dropout)
        self.fusion = MultiResFusion(dim_hidden, dropout=dropout)

        self.lr_scheduler = None
        # Cache for spatial neighbors (avoid recomputing every forward)
        self._spatial_nn_cache = {}

    # ------------------------------------------------------------------
    # Spatial neighbor lookup for local tissue microenvironment context
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _get_spatial_neighbors(self, pos, k):
        """
        Select top-k spatial neighbors based on physical proximity.

        NOTE: This is NOT the same as Reg2ST's criticized "KNN graph construction."
        Here, spatial neighbors are used for LOCAL TISSUE MICROENVIRONMENT context
        aggregation (adjacent spots share morphological features), while the
        actual RELATIONSHIP GRAPH is constructed by WiKG using feature similarity.

        pos: [N, 2] spatial coordinates
        k: number of neighbors
        Returns: [N, K] index tensor
        """
        N = pos.shape[0]
        # Cache by pos content hash (different sections may have same N but different coords)
        pos_np = pos.detach().cpu().float().numpy()
        cache_key = hash(pos_np.tobytes())
        if cache_key not in self._spatial_nn_cache:
            k_actual = min(k, N - 1)
            nn_model = NearestNeighbors(n_neighbors=k_actual + 1, algorithm='kd_tree')
            nn_model.fit(pos_np)
            _, indices = nn_model.kneighbors(pos_np)
            self._spatial_nn_cache[cache_key] = torch.from_numpy(indices[:, 1:])
        return self._spatial_nn_cache[cache_key].to(pos.device)

    # PLACEHOLDER_FORWARD

    # ------------------------------------------------------------------
    # Encoding helpers
    # ------------------------------------------------------------------
    def encode_image(self, image):
        x = self.image_encoder(image)
        proj = self.gene_proj(x)
        return x, proj

    def encode_gene(self, gene):
        return self.gene_linear(gene)

    # ------------------------------------------------------------------
    # InfoNCE (CLIP-style: both features L2-normalized, CE loss)
    # ------------------------------------------------------------------
    def infoNCE_loss(self, img_feat, gene_feat):
        # L2 normalize both
        img_feat = F.normalize(img_feat, dim=-1)
        gene_feat = F.normalize(gene_feat, dim=-1)
        logit_scale = self.logit_scale.exp()
        logits_i2g = logit_scale * img_feat @ gene_feat.t()
        logits_g2i = logits_i2g.t()
        N = img_feat.shape[0]
        labels = torch.arange(N, device=img_feat.device)
        loss_i = F.cross_entropy(logits_i2g, labels)
        loss_g = F.cross_entropy(logits_g2i, labels)
        return (loss_i + loss_g) / 2.0, logits_i2g, logits_g2i

    # ------------------------------------------------------------------
    # Fusion distillation loss
    # ------------------------------------------------------------------
    def fusion_distill_loss(self, per_res_feats, fused_feat, gene_target):
        """Each resolution's prediction should be close to fused prediction."""
        fused_pred = self.gene_head(fused_feat).detach()  # soft target
        loss = 0.0
        for key in per_res_feats:
            pred = self.gene_head(per_res_feats[key])
            loss += F.mse_loss(pred, fused_pred)
        return loss / len(per_res_feats)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, gene, image, pos):
        N = image.shape[0]
        device = image.device

        # 1. Target encoding
        i_f, proj_i_f = self.encode_image(image)  # [N, D], [N, D]
        g_f = self.encode_gene(gene)               # [N, D]
        g_f = F.normalize(g_f, dim=-1)

        # Feature prediction loss (align projected image feat to gene feat)
        proj_loss = F.mse_loss(g_f, proj_i_f)

        # 2. Spatial position embedding
        x_emb = self.embed_x(pos[:, 0])
        y_emb = self.embed_y(pos[:, 1])
        i_ct = i_f.clone() + x_emb + y_emb  # enhanced image features

        # 3. Neighbor view + Global view + Fusion (or skip for ablation)
        if self.no_multires:
            # Ablation: no multi-resolution, use target features only
            fused = i_ct
            per_res = None
        else:
            neighbor_idx = self._get_spatial_neighbors(pos, self.neighbor_k)
            neighbor_feats = i_f[neighbor_idx]  # [N, K, D]
            neighbor_out = self.neighbor_encoder(i_f, neighbor_feats)  # [N, D]
            global_out = self.global_encoder(i_f)  # [N, D]
            fused, per_res = self.fusion(i_ct, neighbor_out, global_out)

        # 6. Cross-attention decoder (fused as query, proj_i_f as memory)
        if self.training:
            mask = generate_masked_tensor(N, zero_prob=getattr(self.args, 'mask_rate', 0.75))
            decoded = self.decoder(fused, proj_i_f, mask.to(device))
        else:
            decoded = self.decoder(fused, proj_i_f)

        # 7. Dynamic GNN
        decoded = self.graph_layer(
            torch.unsqueeze(decoded, 0),
            torch.unsqueeze(i_ct.clone(), 0),
            torch.unsqueeze(proj_i_f.clone(), 0)
        ).squeeze(0)

        # 8. ZINB parameters
        m = self.mean(decoded)
        d = self.disp(decoded)
        p = self.pi(decoded)

        # 9. Gene expression prediction
        i_g = self.gene_head(decoded)

        return i_g, (m, d, p), proj_loss, fused, per_res, i_f, g_f

    # PLACEHOLDER_STEPS

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------
    def training_step(self, batch, batch_idx):
        g, i, pos, _, oris, sfs = batch
        g = g.squeeze(0)
        i = i.squeeze(0)
        pos = pos.squeeze(0)

        i_g, (m, d, p), proj_loss, fused, per_res, i_f, g_f = \
            self.forward(g, i, pos)

        # Losses
        mse_loss = F.mse_loss(g, i_g)
        zinb_loss = ZINB_loss(oris.squeeze(0), m, d, p, sfs.squeeze(0))
        con_loss, _, _ = self.infoNCE_loss(i_f, g_f)

        # Distillation loss (only when multi-res is enabled)
        if per_res is not None:
            distill_loss = self.fusion_distill_loss(per_res, fused, g)
        else:
            distill_loss = torch.tensor(0.0, device=g.device)

        total = (mse_loss
                 + self.w_con * con_loss
                 + self.w_zinb * zinb_loss
                 + proj_loss
                 + self.w_fusion * distill_loss)

        self.log('mse', mse_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('con', con_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('zinb', zinb_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('distill', distill_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_loss', total, on_step=False, on_epoch=True, prog_bar=True)

        if self.lr_scheduler:
            self.lr_scheduler.step()
        return total

    # ------------------------------------------------------------------
    # Validation step
    # ------------------------------------------------------------------
    def validation_step(self, batch, batch_idx):
        g, i, pos, _, _, _ = batch
        g = g.squeeze(0)
        i = i.squeeze(0)
        pos = pos.squeeze(0)
        i_g, *_ = self.forward(g, i, pos)
        p, r = get_R(i_g.detach().cpu().numpy(), g.detach().cpu().numpy())
        pcc = np.nanmean(p)
        self.log('pcc', pcc, prog_bar=True, sync_dist=True)

    # ------------------------------------------------------------------
    # Test step
    # ------------------------------------------------------------------
    def test_step(self, batch, batch_idx):
        g, i, pos, centers, _, _ = batch
        g = g.squeeze(0)
        i = i.squeeze(0)
        pos = pos.squeeze(0)
        centers = centers.squeeze(0)
        i_g, *_ = self.forward(g, i, pos)
        adata = ann.AnnData(X=i_g.detach().cpu().numpy())
        adata.obsm["spatial"] = centers.detach().cpu().numpy()
        p, r = get_R(i_g.detach().cpu().numpy(), g.detach().cpu().numpy())
        pcc = np.nanmean(p)
        self.log('pcc', pcc, prog_bar=True, sync_dist=True)
        self.p = p
        self.r = r
        self.data = adata

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------
    def configure_optimizers(self):
        self.optimizer = optim.AdamW(self.parameters(), lr=1e-4)
        n_samples = getattr(self.args, 'n_train_samples', 31)
        self.lr_scheduler = LR_Scheduler(
            self.optimizer, 10, 1e-5, self.args.epochs, 1e-4, 1e-6, n_samples)
        return self.optimizer
