"""
Unified ablation model supporting 4 configurations:
  1. baseline           (--model final)
  2. baseline + MR      (--model final --use_multires)
  3. baseline + HECLIP  (--model final --use_heclip)
  4. baseline + MR + HECLIP  (--model final --use_multires --use_heclip)

Multi-resolution: NeighborEncoder + GlobalEncoder + MultiResFusion
HECLIP mix: (1-lambda)*hard_InfoNCE + lambda*image_centric_soft_target
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
# Multi-Resolution Modules (from MRCMR)
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
        q = query.unsqueeze(1)
        out, _ = self.cross_attn(q, neighbor_feats, neighbor_feats)
        out = self.norm1(q + out).squeeze(1)
        out = self.norm2(out + self.ffn(out))
        return out


class GlobalEncoder(nn.Module):
    def __init__(self, dim, n_heads=4, n_layers=2, dropout=0.2):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=n_heads, dim_feedforward=dim * 2,
            dropout=dropout, activation='gelu', batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(dim)

    def forward(self, all_feats):
        x = all_feats.unsqueeze(0)
        x = self.encoder(x).squeeze(0)
        return self.norm(x)


class MultiResFusion(nn.Module):
    def __init__(self, dim, dropout=0.2):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(dim * 3, dim * 3), nn.Sigmoid())
        self.proj = nn.Sequential(
            nn.Linear(dim * 3, dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim, dim))
        self.norm = nn.LayerNorm(dim)
        self.head_target = nn.Linear(dim, dim)
        self.head_neighbor = nn.Linear(dim, dim)
        self.head_global = nn.Linear(dim, dim)

    def forward(self, target_f, neighbor_f, global_f):
        concat = torch.cat([target_f, neighbor_f, global_f], dim=-1)
        gate = self.gate(concat)
        gated = concat * gate
        fused = self.norm(self.proj(gated))
        per_res = {
            'target': self.head_target(target_f),
            'neighbor': self.head_neighbor(neighbor_f),
            'global': self.head_global(global_f),
        }
        return fused, per_res


# ---------------------------------------------------------------------------
# Main Model
# ---------------------------------------------------------------------------
class FinalModel(pl.LightningModule):
    def __init__(self, args):
        super(FinalModel, self).__init__()
        self.args = args
        dim_in = args.dim_in
        dim_hidden = args.dim_hidden
        dim_out = args.dim_out
        dropout = args.dropout
        wikg_top = args.wikg_top
        decoder_layer = args.decoder_layer
        decoder_head = args.decoder_head

        # Ablation flags
        self.use_multires = getattr(args, 'use_multires', False)
        self.use_heclip = getattr(args, 'use_heclip', False)

        # Loss weights
        self.w_con = getattr(args, 'w_con', 0.5)
        self.w_zinb = getattr(args, 'w_zinb', 0.25)
        self.w_fusion = getattr(args, 'w_fusion', 0.2)

        # HECLIP settings
        self.heclip_target_temp = getattr(args, 'heclip_target_temp', 0.5)
        self.heclip_mix_lambda = getattr(args, 'heclip_mix_lambda', 0.2)

        # Multi-resolution settings
        self.neighbor_k = getattr(args, 'neighbor_k', 6)

        # ============ Baseline Modules (same as Reg2ST) ============
        self.image_encoder = nn.Sequential(
            nn.Linear(dim_in, dim_in), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim_in, dim_hidden))
        self.gene_proj = nn.Sequential(
            nn.Linear(dim_hidden, dim_hidden), nn.GELU(),
            nn.Linear(dim_hidden, dim_hidden))
        self.gene_head = nn.Sequential(
            nn.Linear(dim_hidden, dim_in), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim_in, dim_out))
        self.decoder = Decoder(
            dim=dim_hidden, layers=decoder_layer, heads=decoder_head,
            mlp_dim=1024, dropout=dropout)
        self.embed_x = nn.Embedding(64, dim_hidden)
        self.embed_y = nn.Embedding(64, dim_hidden)
        self.gene_linear = nn.Sequential(
            nn.Linear(dim_out, 1024), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(1024, dim_hidden))
        self.mean = nn.Sequential(nn.Linear(dim_hidden, dim_out), MeanAct())
        self.disp = nn.Sequential(nn.Linear(dim_hidden, dim_out), DispAct())
        self.pi = nn.Sequential(nn.Linear(dim_hidden, dim_out), nn.Sigmoid())
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.graph_layer = WiKG(
            dim_in=dim_hidden, dim_hidden=dim_hidden,
            topk=wikg_top, n_classes=dim_hidden)
        self.lr_scheduler = None

        # ============ Multi-Resolution Modules (optional) ============
        if self.use_multires:
            self.neighbor_encoder = NeighborEncoder(dim_hidden, n_heads=4, dropout=dropout)
            self.global_encoder = GlobalEncoder(dim_hidden, n_heads=4, n_layers=2, dropout=dropout)
            self.fusion = MultiResFusion(dim_hidden, dropout=dropout)
            self._spatial_nn_cache = {}

    # ------------------------------------------------------------------
    # Spatial neighbor lookup
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _get_spatial_neighbors(self, pos, k):
        N = pos.shape[0]
        pos_np = pos.detach().cpu().float().numpy()
        cache_key = hash(pos_np.tobytes())
        if cache_key not in self._spatial_nn_cache:
            k_actual = min(k, N - 1)
            nn_model = NearestNeighbors(n_neighbors=k_actual + 1, algorithm='kd_tree')
            nn_model.fit(pos_np)
            _, indices = nn_model.kneighbors(pos_np)
            self._spatial_nn_cache[cache_key] = torch.from_numpy(indices[:, 1:])
        return self._spatial_nn_cache[cache_key].to(pos.device)

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------
    def encode_image(self, image):
        x = self.image_encoder(image)
        proj = self.gene_proj(x)
        return x, proj

    def encode_gene(self, gene):
        return self.gene_linear(gene)

    # ------------------------------------------------------------------
    # Loss functions
    # ------------------------------------------------------------------
    def infoNCE_loss(self, logits_per_image, logits_per_gene):
        batch_size = logits_per_image.shape[0]
        labels = torch.eye(batch_size, device=logits_per_image.device)
        loss_i = F.binary_cross_entropy_with_logits(logits_per_image, labels)
        loss_g = F.binary_cross_entropy_with_logits(logits_per_gene, labels)
        return (loss_i + loss_g) / 2.0

    def image_centric_loss(self, logits_per_image, i_f):
        with torch.no_grad():
            img_feat = F.normalize(i_f, dim=1)
            img_sim = img_feat @ img_feat.t()
            logit_scale = self.logit_scale.exp().detach()
            temp = max(float(self.heclip_target_temp), 1e-6)
            target_scale = logit_scale / temp
            targets = F.softmax(img_sim * target_scale, dim=-1)
        log_probs = F.log_softmax(logits_per_image, dim=-1)
        return (-(targets * log_probs).sum(dim=-1)).mean()

    def fusion_distill_loss(self, per_res_feats, fused_feat):
        fused_pred = self.gene_head(fused_feat).detach()
        loss = 0.0
        for key in per_res_feats:
            pred = self.gene_head(per_res_feats[key])
            loss += F.mse_loss(pred, fused_pred)
        return loss / len(per_res_feats)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, gene, image, pos):
        # 1. Target encoding
        i_f, proj_i_f = self.encode_image(image)
        x_emb = self.embed_x(pos[:, 0])
        y_emb = self.embed_y(pos[:, 1])
        i_ct = i_f.clone() + x_emb + y_emb

        # 2. Gene encoding + proj loss
        g_f = self.encode_gene(gene)
        g_f = g_f / g_f.norm(dim=1, keepdim=True)
        proj_loss = F.mse_loss(g_f, proj_i_f)

        # 3. Multi-resolution (optional)
        per_res = None
        if self.use_multires:
            neighbor_idx = self._get_spatial_neighbors(pos, self.neighbor_k)
            neighbor_feats = i_f[neighbor_idx]
            neighbor_out = self.neighbor_encoder(i_f, neighbor_feats)
            global_out = self.global_encoder(i_f)
            fused, per_res = self.fusion(i_ct, neighbor_out, global_out)
        else:
            fused = i_ct

        # 4. Decoder
        if self.training:
            mask = generate_masked_tensor(
                g_f.shape[0],
                zero_prob=getattr(self.args, 'mask_rate', 0.75))
            decoded = self.decoder(fused, proj_i_f, mask.to(fused.device))
        else:
            decoded = self.decoder(fused, proj_i_f)

        # 5. WiKG
        decoded = self.graph_layer(
            torch.unsqueeze(decoded, 0),
            torch.unsqueeze(i_ct.clone(), 0),
            torch.unsqueeze(proj_i_f.clone(), 0)
        ).squeeze(0)

        # 6. Heads
        m = self.mean(decoded)
        d = self.disp(decoded)
        p = self.pi(decoded)
        i_g = self.gene_head(decoded)

        # 7. Contrastive logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * i_f @ g_f.t()
        logits_per_gene = logits_per_image.t()

        return (logits_per_image, logits_per_gene, i_g, (m, d, p),
                proj_loss, per_res, fused, i_f)

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------
    def training_step(self, batch, batch_idx):
        g, i, pos, _, oris, sfs = batch
        g = torch.squeeze(g)
        i = torch.squeeze(i)
        pos = torch.squeeze(pos)

        (logits_img, logits_gene, i_g, (m, d, p),
         proj_loss, per_res, fused, i_f) = self.forward(g, i, pos)

        mse_loss = F.mse_loss(g, i_g)
        zinb_loss = ZINB_loss(oris.squeeze(0), m, d, p, sfs.squeeze(0))

        # Contrastive loss
        hard_con = self.infoNCE_loss(logits_img, logits_gene)
        if self.use_heclip:
            heclip_con = self.image_centric_loss(logits_img, i_f)
            lam = self.heclip_mix_lambda
            con_loss = (1.0 - lam) * hard_con + lam * heclip_con
        else:
            con_loss = hard_con

        # Fusion distillation loss (only with multi-res)
        if self.use_multires and per_res is not None:
            distill_loss = self.fusion_distill_loss(per_res, fused)
        else:
            distill_loss = torch.tensor(0.0, device=g.device)

        total = (mse_loss
                 + proj_loss
                 + self.w_con * con_loss
                 + self.w_zinb * zinb_loss
                 + self.w_fusion * distill_loss)

        self.log('mse', mse_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('con', con_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('proj', proj_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('zinb', zinb_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        if self.use_multires:
            self.log('distill', distill_loss, on_step=False, on_epoch=True, prog_bar=True)
        if self.use_heclip:
            self.log('heclip', heclip_con, on_step=False, on_epoch=True, prog_bar=False)
            self.log('hard_con', hard_con, on_step=False, on_epoch=True, prog_bar=False)
        self.log('train_loss', total, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        if self.lr_scheduler:
            self.lr_scheduler.step()
        return total

    # ------------------------------------------------------------------
    # Validation / Test
    # ------------------------------------------------------------------
    def validation_step(self, batch, batch_idx):
        g, i, pos, _, _, _ = batch
        g = torch.squeeze(g)
        i = torch.squeeze(i)
        pos = torch.squeeze(pos)
        _, _, i_g, _, _, _, _, _ = self.forward(g, i, pos)
        p, _ = get_R(i_g.detach().cpu().numpy(), g.detach().cpu().numpy())
        pcc = np.nanmean(p)
        self.log('pcc', pcc, prog_bar=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        g, i, pos, centers, _, _ = batch
        g = torch.squeeze(g)
        i = torch.squeeze(i)
        pos = torch.squeeze(pos)
        centers = torch.squeeze(centers)
        _, _, i_g, _, _, _, _, _ = self.forward(g, i, pos)
        adata = ann.AnnData(X=i_g.detach().cpu().numpy())
        adata.obsm['spatial'] = centers.detach().cpu().numpy()
        p, r = get_R(i_g.detach().cpu().numpy(), g.detach().cpu().numpy())
        pcc = np.nanmean(p)
        self.log('pcc', pcc, prog_bar=True, sync_dist=True)
        self.p = p
        self.r = r
        self.data = adata

    def configure_optimizers(self):
        self.optimizer = optim.AdamW(self.parameters(), lr=1e-4)
        n_samples = getattr(self.args, 'n_train_samples', 31)
        self.lr_scheduler = LR_Scheduler(
            self.optimizer, 10, 1e-5, self.args.epochs, 1e-4, 1e-6, n_samples)
        return self.optimizer
