from torch import nn
import torch
from layers.spatial import SpatialAttention
from layers.temporal import TemporalAttention

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



class GMAN(nn.Module):
    def __init__(self):
        super(GMAN, self).__init__()
