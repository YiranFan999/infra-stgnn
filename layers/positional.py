from torch import nn
from torch.nn.functional import one_hot
import torch

class STEmbedding(nn.Module):
    def __init__(self, se, D, T=None, num_steps=None):
        # se: [N, d_se]
        # te: [batch_size, P + Q, 2] (dayofweek, timeofday)
        super(STEmbedding, self).__init__()
        self.register_buffer('se', se, persistent=False) # spatial embedding is already in data folder
        self.T = T

        d_se = se.shape[1]
        self.se_fc = nn.Sequential(
            nn.Conv2d(d_se, D, kernel_size=[1,1], bias=True),
            nn.BatchNorm2d(D),
            nn.ReLU(),
            nn.Conv2d(D, D, kernel_size=[1,1], bias=True)
        )

        if T is not None:
            self.te_fc = nn.Sequential(
                nn.Conv2d(7 + T, D, 1), nn.BatchNorm2d(D), nn.ReLU(),
                nn.Conv2d(D, D, 1),
            )
        else:
            self.pos_emb = nn.Embedding(num_steps, D)

    def forward(self, te=None):
        se = self.se[None, None] # se: (1, 1, N, d_se)
        se = se.permute(0, 3, 2, 1) # (1, d_se, N, 1)
        se = self.se_fc(se)
        se = se.permute(0, 3, 2, 1)  # (1, 1, N, D)

        if self.T is not None:
            dayofweek = one_hot(te[..., 0], num_classes=7).float()
            timeofday = one_hot(te[..., 1], num_classes=self.T).float()
            te = torch.cat([dayofweek, timeofday], dim=-1) # (B, P+Q, 295)
            te = te.unsqueeze(2) # (B, P+Q, 1, 295)
            te = te.permute(0, 3, 2, 1)
            te = self.te_fc(te)
            te = te.permute(0, 3, 2, 1)

        else:
            pos = torch.arange(self.pos_emb.num_embeddings, device=se.device)
            te = self.pos_emb(pos)[None, :, None, :]

        return se+te

