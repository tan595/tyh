import warnings
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
from multiscale import MultiScaleNeighborFusion
from utils import LR_Scheduler, MeanAct, DispAct, ZINB_loss, generate_masked_tensor

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


        self.w_con    = getattr(args, 'w_con',    0.5)
        self.w_zinb   = getattr(args, 'w_zinb',   0.25)
        self.w_branch = getattr(args, 'w_branch', 0.1)


        self.heclip_target_temp = getattr(args, 'heclip_target_temp', 0.5)
        self.heclip_mix_lambda  = getattr(args, 'heclip_mix_lambda',  0.2)


        self.multiscale_k_small = getattr(args, 'multiscale_k_small', 4)
        self.multiscale_k_large = getattr(args, 'multiscale_k_large', 7)
        self.multiscale_k_query = getattr(args, 'multiscale_k_query', 14)
        self.multiscale_heads   = getattr(args, 'multiscale_heads',   4)


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
        self.embed_x = nn.Embedding(256, dim_hidden)
        self.embed_y = nn.Embedding(256, dim_hidden)
        self.gene_linear = nn.Sequential(
            nn.Linear(dim_out, 1024), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(1024, dim_hidden))
        self.mean = nn.Sequential(nn.Linear(dim_hidden, dim_out), MeanAct())
        self.disp = nn.Sequential(nn.Linear(dim_hidden, dim_out), DispAct())
        self.pi   = nn.Sequential(nn.Linear(dim_hidden, dim_out), nn.Sigmoid())
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.graph_layer = WiKG(
            dim_in=dim_hidden, dim_hidden=dim_hidden,
            topk=wikg_top, n_classes=dim_hidden)
        self.lr_scheduler = None
        self._spatial_nn_cache = {}
        self._kquery_warned = False


        if dim_hidden % self.multiscale_heads != 0:
            raise ValueError(
                f"dim_hidden={dim_hidden} must be divisible by "
                f"multiscale_heads={self.multiscale_heads}")
        if not (self.multiscale_k_small < self.multiscale_k_large
                < self.multiscale_k_query):
            raise ValueError(
                f"Required: k_small({self.multiscale_k_small}) < "
                f"k_large({self.multiscale_k_large}) < "
                f"k_query({self.multiscale_k_query})")
        self.multiscale_fusion = MultiScaleNeighborFusion(
            dim=dim_hidden,
            n_heads=self.multiscale_heads,
            dropout=dropout)

        self.branch_head = nn.Linear(dim_hidden, dim_out)
        self.blend_alpha = nn.Parameter(torch.tensor(0.0))
        self.blend_norm  = nn.LayerNorm(dim_hidden)
        self.feat_norm   = nn.LayerNorm(dim_hidden)




    @torch.no_grad()
    def _get_spatial_neighbors(self, pos, k):

        pos_np = pos.detach().cpu().float().numpy()
        cache_key = (hash(pos_np.tobytes()), k)
        if cache_key not in self._spatial_nn_cache:
            N = pos_np.shape[0]
            k_actual = min(k, N - 1)
            nn_model = NearestNeighbors(
                n_neighbors=k_actual + 1, algorithm='kd_tree')
            nn_model.fit(pos_np)
            dists, indices = nn_model.kneighbors(pos_np)

            self._spatial_nn_cache[cache_key] = (
                torch.from_numpy(indices[:, 1:k_actual + 1]),
                torch.from_numpy(dists[:,   1:k_actual + 1].astype(np.float32)))
        idx, dist = self._spatial_nn_cache[cache_key]
        return idx.to(pos.device), dist.to(pos.device)




    def encode_image(self, image):
        x = self.image_encoder(image)
        proj = self.gene_proj(x)
        return x, proj

    def encode_gene(self, gene):
        return self.gene_linear(gene)




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

    def branch_supervision_loss(self, scales, gene_target):

        s1, _, _, fused = scales
        loss = (F.mse_loss(self.branch_head(s1),   gene_target)
              + F.mse_loss(self.branch_head(fused), gene_target))
        return loss / 2.0




    def forward(self, gene, image, pos):

        i_f, proj_i_f = self.encode_image(image)
        x_emb = self.embed_x(pos[:, 0])
        y_emb = self.embed_y(pos[:, 1])



        idx_q, dist_q = self._get_spatial_neighbors(pos, self.multiscale_k_query)

        avail = idx_q.shape[1]
        k_l = min(self.multiscale_k_large, avail)
        k_s = min(self.multiscale_k_small, k_l)
        if avail < self.multiscale_k_query and not self._kquery_warned:
            warnings.warn(
                f"MultiScale: slide has only {avail + 1} spots; "
                f"k_query clipped {self.multiscale_k_query}->{avail}, "
                f"k_large->{k_l}, k_small->{k_s}. "
                f"Asymmetric Q/KV separation may be reduced.",
                RuntimeWarning, stacklevel=2)
            self._kquery_warned = True
        idx_l, dist_l = idx_q[:, :k_l],  dist_q[:, :k_l]
        idx_s, dist_s = idx_q[:, :k_s],  dist_q[:, :k_s]

        ccf_out, scales = self.multiscale_fusion(
            i_f, x_emb, y_emb,
            idx_s, dist_s,
            idx_l, dist_l,
            idx_q, dist_q)

        alpha = torch.sigmoid(self.blend_alpha)
        i_ct = (alpha         * self.blend_norm(ccf_out)
                + (1 - alpha) * self.feat_norm(i_f + x_emb + y_emb))


        g_f = self.encode_gene(gene)
        g_f = F.normalize(g_f, dim=-1, eps=1e-6)
        proj_loss = F.mse_loss(g_f, proj_i_f)


        if self.training:
            mask = generate_masked_tensor(
                g_f.shape[0],
                zero_prob=getattr(self.args, 'mask_rate', 0.75))
            decoded = self.decoder(i_ct, proj_i_f, mask.to(i_ct.device))
        else:
            decoded = self.decoder(i_ct, proj_i_f)


        decoded = self.graph_layer(
            torch.unsqueeze(decoded, 0),
            torch.unsqueeze(i_ct.clone(), 0),
            torch.unsqueeze(proj_i_f.clone(), 0)
        ).squeeze(0)


        m   = self.mean(decoded)
        d   = self.disp(decoded)
        p   = self.pi(decoded)
        i_g = self.gene_head(decoded)


        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * i_f @ g_f.t()
        logits_per_gene  = logits_per_image.t()

        return (logits_per_image, logits_per_gene, i_g, (m, d, p),
                proj_loss, i_f, scales)




    def training_step(self, batch, batch_idx):
        g, i, pos, _, oris, sfs = batch
        g   = g.squeeze(0)
        i   = i.squeeze(0)
        pos = pos.squeeze(0)

        (logits_img, logits_gene, i_g, (m, d, p),
         proj_loss, i_f, scales) = self.forward(g, i, pos)

        mse_loss  = F.mse_loss(g, i_g)
        zinb_loss = ZINB_loss(oris.squeeze(0), m, d, p, sfs.squeeze(0))

        hard_con = self.infoNCE_loss(logits_img, logits_gene)
        heclip_con = self.image_centric_loss(logits_img, i_f)
        lam = self.heclip_mix_lambda
        con_loss = (1.0 - lam) * hard_con + lam * heclip_con
        branch_loss = self.branch_supervision_loss(scales, g)

        total = (mse_loss
                 + proj_loss
                 + self.w_con    * con_loss
                 + self.w_zinb   * zinb_loss
                 + self.w_branch * branch_loss)

        self.log('mse',    mse_loss,  on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('con',    con_loss,  on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('proj',   proj_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('zinb',   zinb_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('branch', branch_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        w = F.softmax(self.multiscale_fusion.scale_weights, dim=0)
        self.log('dw_s1', w[0].item(), on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log('dw_ss', w[1].item(), on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log('dw_sl', w[2].item(), on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log('blend_a', torch.sigmoid(self.blend_alpha).item(),
                 on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log('heclip',   heclip_con, on_step=False, on_epoch=True, prog_bar=False)
        self.log('hard_con', hard_con,   on_step=False, on_epoch=True, prog_bar=False)
        self.log('train_loss', total, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        if self.lr_scheduler:
            self.lr_scheduler.step()
        return total




    def validation_step(self, batch, batch_idx):
        g, i, pos, _, _, _ = batch
        g   = g.squeeze(0)
        i   = i.squeeze(0)
        pos = pos.squeeze(0)
        _, _, i_g, _, _, _, _ = self.forward(g, i, pos)
        p, _ = get_R(i_g.detach().cpu().numpy(), g.detach().cpu().numpy())
        pcc = np.nanmean(p)
        self.log('pcc', pcc, prog_bar=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        g, i, pos, centers, _, _ = batch
        g       = g.squeeze(0)
        i       = i.squeeze(0)
        pos     = pos.squeeze(0)
        centers = centers.squeeze(0)
        _, _, i_g, _, _, _, _ = self.forward(g, i, pos)
        adata = ann.AnnData(X=i_g.detach().cpu().numpy())
        adata.obsm['spatial'] = centers.detach().cpu().numpy()
        p, r = get_R(i_g.detach().cpu().numpy(), g.detach().cpu().numpy())
        pcc = np.nanmean(p)
        self.log('pcc', pcc, prog_bar=True, sync_dist=True)
        self.p    = p
        self.r    = r
        self.data = adata

    def configure_optimizers(self):
        self.optimizer = optim.AdamW(self.parameters(), lr=1e-4)
        n_samples = getattr(self.args, 'n_train_samples', 31)
        self.lr_scheduler = LR_Scheduler(
            self.optimizer, 10, 1e-5, self.args.epochs, 1e-4, 1e-6, n_samples)
        return self.optimizer
