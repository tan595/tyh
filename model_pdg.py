"""
PDG-ST: Progressive Dynamic Graph for Spatial Transcriptomics Prediction.

Key innovations over Reg2ST:
1. Progressive Dynamic Graph Reasoning: graph topology evolves across encoding
   and fusion stages (not just a post-hoc correction).
2. Dual-stage Contrastive Alignment: aligns image↔gene at both visual-space
   and multimodal-space stages.
3. EMA-based stable graph construction: decouples feature→edge_weight gradient
   path to prevent topology oscillation.
4. Spatial pre-filter + feature top-k: O(N·K) complexity for scalability.
"""
import pytorch_lightning as pl
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
import anndata as ann

from performance import get_R
from attention import Decoder
from progressive_graph import EMAFeatureBank, DynamicGraphBuilder, GraphBlock, GatedFusion

# Reuse from original model.py
from model import LR_Scheduler, MeanAct, DispAct, ZINB_loss, generate_masked_tensor


class PDG_ST(pl.LightningModule):
    def __init__(self, args):
        super(PDG_ST, self).__init__()
        self.args = args
        dim_in = args.dim_in
        dim_hidden = args.dim_hidden
        dim_out = args.dim_out
        dropout = args.dropout
        decoder_layer = args.decoder_layer
        decoder_head = args.decoder_head

        # ============ Encoding (same as Reg2ST) ============
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

        # Positional embeddings
        self.embed_x = nn.Embedding(64, dim_hidden)
        self.embed_y = nn.Embedding(64, dim_hidden)

        # Gene encoder (expression → hidden)
        self.gene_linear = nn.Sequential(
            nn.Linear(dim_out, 1024), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(1024, dim_hidden))

        # ZINB heads
        self.mean = nn.Sequential(nn.Linear(dim_hidden, dim_out), MeanAct())
        self.disp = nn.Sequential(nn.Linear(dim_hidden, dim_out), DispAct())
        self.pi = nn.Sequential(nn.Linear(dim_hidden, dim_out), nn.Sigmoid())

        # ============ Progressive Dynamic Graph ============
        spatial_k = getattr(args, 'spatial_k', 30)
        feature_k = getattr(args, 'feature_k', 6)
        ema_momentum = getattr(args, 'ema_momentum', 0.99)

        # Stage 1: visual-space graph (after encoding, before decoder)
        self.graph_block_1 = GraphBlock(dim_hidden, dropout)
        # Stage 2: multimodal-space graph (after decoder)
        self.graph_block_2 = GraphBlock(dim_hidden, dropout)
        # Shared graph builder (same spatial_k / feature_k)
        self.graph_builder = DynamicGraphBuilder(spatial_k, feature_k)
        # Per-stage EMA banks (not nn.Module, just dict containers)
        self.ema_bank_1 = EMAFeatureBank(ema_momentum)
        self.ema_bank_2 = EMAFeatureBank(ema_momentum)

        # Gated Fusion (replaces simple residual)
        self.gated_fusion_1 = GatedFusion(dim_hidden)
        self.gated_fusion_2 = GatedFusion(dim_hidden)

        # ============ Dual-stage Contrastive ============
        # Stage 1 projection: visual-space alignment (image ↔ gene)
        self.proj_head_1 = nn.Sequential(
            nn.Linear(dim_hidden, dim_hidden), nn.GELU(),
            nn.Linear(dim_hidden, dim_hidden))
        # Stage 2 projection: multimodal-space alignment (refined ↔ gene)
        self.proj_head_2 = nn.Sequential(
            nn.Linear(dim_hidden, dim_hidden), nn.GELU(),
            nn.Linear(dim_hidden, dim_hidden))

        # Learnable temperature (one per stage)
        self.logit_scale_1 = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.logit_scale_2 = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.lr_scheduler = None

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
    # InfoNCE (CLIP-style: CE on logits, labels = arange)
    # ------------------------------------------------------------------
    def infoNCE_loss(self, logits_per_image, logits_per_gene):
        N = logits_per_image.shape[0]
        labels = torch.arange(N, device=logits_per_image.device)
        loss_i = F.cross_entropy(logits_per_image, labels)
        loss_g = F.cross_entropy(logits_per_gene, labels)
        return (loss_i + loss_g) / 2.0

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, gene, image, pos, section_id="default"):
        """
        Args:
            gene: [N, dim_out] log-normalized expression
            image: [N, dim_in] phikonv2 embeddings
            pos: [N, 2] integer grid positions
            section_id: str, used to key EMA banks per section
        Returns:
            i_g: [N, dim_out] predicted expression
            extra: (mean, disp, pi) for ZINB
            con_loss_1: stage-1 contrastive loss
            con_loss_2: stage-2 contrastive loss
            proj_loss: MSE alignment loss
        """
        # --- Encoding ---
        i_f, proj_i_f = self.encode_image(image)  # [N, D], [N, D]
        x_emb = self.embed_x(pos[:, 0])
        y_emb = self.embed_y(pos[:, 1])
        i_ct = i_f + x_emb + y_emb  # [N, D] position-aware visual features

        g_f = self.encode_gene(gene)  # [N, D]
        g_f_norm = F.normalize(g_f, dim=-1)

        # --- proj_loss: align gene embedding with image projection ---
        proj_i_f_norm = F.normalize(proj_i_f, dim=-1)
        proj_loss = F.mse_loss(g_f_norm, proj_i_f_norm)

        # ============================================================
        # Stage 1: Visual-space graph reasoning
        # ============================================================
        ema_feat_1 = self.ema_bank_1.get(section_id, i_ct, i_ct.device)
        topk_idx_1, topk_w_1 = self.graph_builder(ema_feat_1, pos)
        # GraphBlock: x=i_ct, head=i_ct, tail=proj_i_f
        i_ct_enh = self.graph_block_1(i_ct, i_ct, proj_i_f, topk_idx_1, topk_w_1)
        i_ct_enh = self.gated_fusion_1(i_ct, i_ct_enh)  # gated residual

        # Stage 1 contrastive: image-enhanced ↔ gene
        z_img_1 = F.normalize(self.proj_head_1(i_ct_enh), dim=-1)
        z_gene_1 = F.normalize(self.proj_head_1(g_f), dim=-1)
        scale_1 = self.logit_scale_1.exp()
        logits_i1 = scale_1 * z_img_1 @ z_gene_1.t()
        con_loss_1 = self.infoNCE_loss(logits_i1, logits_i1.t())

        # ============================================================
        # Decoder (cross-attention fusion)
        # ============================================================
        if self.training:
            mask = generate_masked_tensor(i_ct_enh.shape[0])
            i_g = self.decoder(i_ct_enh, proj_i_f, mask.to(i_ct.device))
        else:
            i_g = self.decoder(i_ct_enh, proj_i_f)

        # ============================================================
        # Stage 2: Multimodal-space graph reasoning
        # ============================================================
        # Use proj_head_2 output for EMA → captures multimodal semantics
        z_multi = self.proj_head_2(i_g)
        ema_feat_2 = self.ema_bank_2.get(section_id, z_multi, i_g.device)
        topk_idx_2, topk_w_2 = self.graph_builder(ema_feat_2, pos)
        # GraphBlock: x=i_g, head=i_g, tail=i_ct_enh (cross-stage)
        i_g_enh = self.graph_block_2(i_g, i_g, i_ct_enh, topk_idx_2, topk_w_2)
        i_g = self.gated_fusion_2(i_g, i_g_enh)  # gated residual

        # Stage 2 contrastive: multimodal-refined ↔ gene
        z_img_2 = F.normalize(self.proj_head_2(i_g), dim=-1)
        z_gene_2 = F.normalize(self.proj_head_2(g_f), dim=-1)
        scale_2 = self.logit_scale_2.exp()
        logits_i2 = scale_2 * z_img_2 @ z_gene_2.t()
        con_loss_2 = self.infoNCE_loss(logits_i2, logits_i2.t())

        # ============================================================
        # Prediction heads
        # ============================================================
        m = self.mean(i_g)
        d = self.disp(i_g)
        p = self.pi(i_g)
        extra = (m, d, p)
        i_g_pred = self.gene_head(i_g)

        return i_g_pred, extra, con_loss_1, con_loss_2, proj_loss, i_ct_enh, z_multi

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------
    def training_step(self, batch, batch_idx):
        g, i, pos, _, oris, sfs = batch
        g = g.squeeze(0)
        i = i.squeeze(0)
        pos = pos.squeeze(0)

        section_id = f"train_{batch_idx}"
        i_g, extra, con1, con2, proj_loss, i_ct_enh, z_multi = self.forward(
            g, i, pos, section_id)

        # EMA bank updates (per-section, after forward)
        self.ema_bank_1.update(section_id, i_ct_enh)
        self.ema_bank_2.update(section_id, z_multi)

        # Losses
        w_con1 = getattr(self.args, 'w_con1', 0.3)
        w_con2 = getattr(self.args, 'w_con2', 0.2)
        w_zinb = getattr(self.args, 'w_zinb', 0.25)

        m, d, p = extra
        zinb_loss = ZINB_loss(oris.squeeze(0), m, d, p, sfs.squeeze(0))
        mse_loss = F.mse_loss(g, i_g)

        train_loss = (mse_loss
                      + w_con1 * con1
                      + w_con2 * con2
                      + proj_loss
                      + w_zinb * zinb_loss)

        self.log('mse', mse_loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('con1', con1, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('con2', con2, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('proj', proj_loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('zinb', zinb_loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('train_loss', train_loss, on_epoch=True, prog_bar=True, sync_dist=True)

        if self.lr_scheduler:
            self.lr_scheduler.step()
        return train_loss

    # ------------------------------------------------------------------
    # Validation step
    # ------------------------------------------------------------------
    def validation_step(self, batch, batch_idx):
        g, i, pos, _, _, _ = batch
        g = g.squeeze(0)
        i = i.squeeze(0)
        pos = pos.squeeze(0)

        i_g, *_ = self.forward(g, i, pos, section_id=f"val_{batch_idx}")
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

        i_g, *_ = self.forward(g, i, pos, section_id=f"test_{batch_idx}")
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
