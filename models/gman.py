from torch import nn
import torch
from layers.spatial import SpatialAttention
from layers.temporal import TemporalAttention
import torch.nn.functional as F
import lightning as pl
from layers.positional import STEmbedding
from utils.metrics import *
from torch.optim import AdamW

class GatedFusion(nn.Module):
    """
    gated fusion for spatial attention and temporal attention
    HS: (B, T, N, D)
    HT: (B, T, N, D)
    in_channels: hidden channels in hidden spatial attention and hidden temporal attention
    out_channels: hidden channels in output fused attention, which is the same as in_channels D
    """
    def __init__(self, in_channels, out_channels):
        super(GatedFusion, self).__init__()
        self.hs_fc = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), padding=(0, 0), bias=True)
        self.ht_fc = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), padding=(0, 0), bias=False)
        self.fc = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=(1, 1), padding=(0, 0),),
        )

    def forward(self, hs, ht):
        # hs: (B, T, N, D)
        # ht: (B, T, N, D)
        hs = hs.permute(0, 3, 2, 1) # (B, D, N, T)
        ht = ht.permute(0, 3, 2, 1)
        z = torch.sigmoid(self.hs_fc(hs) + self.ht_fc(ht))
        h = z * hs + (1 - z) * ht
        h = self.fc(h)
        h = h.permute(0, 3, 2, 1) # (B, T, N, D)
        return h

class STAttBlock(nn.Module):
    """
    ST-Attention block
    K: number of attention heads
    d: hidden channels in each attention head, D = K * d
    STE: spatial and temporal embedding
    """
    def __init__(self, K, d):
        super(STAttBlock, self).__init__()
        D = K*d
        self.spatialAttn = SpatialAttention(K=K, d=d)
        self.temporalAttn = TemporalAttention(K=K, d=d, mask=True)
        self.gatedFusion = GatedFusion(in_channels=D, out_channels=D)

    def forward(self, x, ste):
        hs = self.spatialAttn(x, ste)
        ht = self.temporalAttn(x, ste)
        h = self.gatedFusion(hs, ht)
        return h+x

class TransformAttn(nn.Module):
    """
    Transform-Attention block
    Args:
        K (int): number of attention heads
        d: hidden channels in each attention head, D = K * d
    Inputs:
        x: input tensor, containing P timesteps
        ste_p: spatial temporal embedding of previous P timesteps
        ste_q: temporal temporal embedding of prediction Q timesteps
    Outputs:
        h: output tensor, containing Q timesteps
    """
    def __init__(self, K, d):
        super(TransformAttn, self).__init__()
        # key: STE_P; query: STE_Q; value: X
        in_channels = K*d
        out_channels = K*d
        self.K = K
        self.d = d
        self.q_fc = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.v_fc = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.k_fc = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        self.out_fc = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True),
        )
    def forward(self, x, ste_p, ste_q):
        # x, step_p, step_q: (B, T, N, F)
        x = x.permute(0, 3, 2, 1)
        ste_p = ste_p.permute(0, 3, 2, 1)
        ste_q = ste_q.permute(0, 3, 2, 1)
        q = self.q_fc(ste_q)
        k = self.k_fc(ste_p)
        v = self.v_fc(x)

        # split into K heads
        q = q.view(q.size(0), self.K, self.d, q.size(2), q.size(3)).permute(0, 1, 3, 4, 2)  # (B, K, N, Q, d)
        v = v.view(v.size(0), self.K, self.d, v.size(2), v.size(3)).permute(0, 1, 3, 4, 2)  # (B, K, N, P, d)
        k = k.view(k.size(0), self.K, self.d, k.size(2), k.size(3)).permute(0, 1, 3, 4, 2)  # (B, K, N, P, d)

        attention = (q @ k.transpose(-2, -1)) / (self.d ** 0.5)
        attention = F.softmax(attention, dim=-1)
        out = attention @ v  # (B, K, N, Q, d)

        out = out.permute(0, 1, 4, 2, 3)  # (B, K, d, N, Q)
        out = out.reshape(out.size(0), -1, out.size(3), out.size(4))  # (B, D, N, Q)

        # FC before output
        out = self.out_fc(out)
        out = out.permute(0, 3, 2, 1)
        return out


class GMAN(pl.LightningModule):
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 se,
                 K,
                 d,
                 L,
                 dropout,
                 T=None,
                 num_steps=None,
                 scaler=None,
                 lr=1e-3,
                 weight_decay=1e-4,
                 feature_names=None):
        super(GMAN, self).__init__()
        self.save_hyperparameters()
        self.save_hyperparameters(ignore=['se', 'scaler'])
        self.input_fc = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True),
        )

        self.encoder = nn.ModuleList([STAttBlock(K=K, d=d) for _ in range(L)])
        self.decoder = nn.ModuleList([STAttBlock(K=K, d=d) for _ in range(L)])

        self.transform = TransformAttn(K=K, d=d)

        self.output_fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv2d(hidden_channels, in_channels, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True),
        )

        self.stembedding = STEmbedding(se, hidden_channels, T, num_steps)


    def forward(self, x, te):
        # x: (B, P, N, F)
        # ste_p: (B, P, N, F)
        # ste_q: (B, Q, N, F)
        ste = self.stembedding(te)
        if ste.size(0) == 1:
            ste = ste.expand(x.size(0), -1, -1, -1)
        B, P, N, dim = x.shape
        ste_p = ste[:, :P]
        ste_q = ste[:, P:]

        x = self.input_fc(x.permute(0, 3, 2, 1))  # (B, D, N, P)
        x = x.permute(0, 3, 2, 1)  # (B, P, N, D)

        for layer in self.encoder:
            x = layer(x, ste_p)

        x = self.transform(x, ste_p, ste_q)

        for layer in self.decoder:
            x = layer(x, ste_q)

        x = self.output_fc(x.permute(0, 3, 2, 1))  # (B, F, N, Q)
        x = x.permute(0, 3, 2, 1)  # (B, Q, N, F)
        return x

    @staticmethod
    def _unpack(batch):
        if len(batch) == 3:
            return batch
        x, y = batch
        return x, y, None

    def _shared_step(self, batch, tag):
        x, y, te = GMAN._unpack(batch)
        y_hat = self(x, te)
        loss = mae(y_hat, y)
        self.log(f"{tag}_loss", loss, on_step=False, on_epoch=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        x, y, te = GMAN._unpack(batch)
        pred = self(x, te)
        if self.scaler is not None:
            pred = self.scaler.inverse(pred)
            y = self.scaler.inverse(y)
        self.log('test_loss', mse(pred, y))
        self.log('test_mae', mae(pred, y))
        self.log('test_rmse', rmse(pred, y))
        self.log('test_mape', mape(pred, y))
        # per-horizon prediction loss
        for h in range(2, pred.shape[1], 3):
            self.log(f'test_mae_h{h + 1}', mae(pred[:, h], y[:, h]))
            for j, name in enumerate(self.hparams.feature_names):
                self.log(f'test_mae_{name}_h{h + 1}', mae(pred[:, h, :, j], y[:, h, :, j]))

    def configure_optimizers(self):
        optim = AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='min', factor=0.5, patience=3)
        return {"optimizer": optim, "lr_scheduler": {"scheduler": sch, "monitor": "val_loss"}}


