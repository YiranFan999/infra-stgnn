from torch import nn
import torch
from layers.spatial import SpatialAttention
from layers.temporal import TemporalAttention
import torch.nn.functional as F

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


class GMAN(nn.Module):
    def __init__(self):
        super(GMAN, self).__init__()
