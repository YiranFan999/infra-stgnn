import torch
from torch import nn


class nconv(nn.Module):
    """
    Diffusion Layer
    Input:
        x (torch.Tensor): Input tensor of shape (B, C, N, T)
        A (torch.Tensor): Forward/Backword/Self-adaptive adjacency matrix of shape (N, N)
    """
    def __init__(self):
        super(nconv, self).__init__()

    def forward(self, x, A):
        # x: (B, C, N, T), A: (N, N)
        x = torch.einsum('ncvl,vw->ncwl', (x, A))
        return x.contiguous()

class gcn(nn.Module):
    """
    Graph Convolution Layer
    Args:
        in_channels (int): Number of features in the nodes
        out_channels (int): Number of features in the output
        dropout (float): Dropout rate
        support_len: Number of support matrices
        order: Order of diffusion

    Inputs:
        x (torch.Tensor): Input tensor of shape (B, T, N, C)
        support (list): List of support matrices of shape (N, N)
    """
    def __init__(self, in_channels, out_channels, dropout, support_len=3, order=2):
        super(gcn, self).__init__()
        self.order = order
        self.support_len = support_len
        self.nconv = nconv()
        in_channels = (support_len * order + 1) * in_channels
        self.mlp = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True)
        self.dropout = dropout

    def forward(self, x, support):
        # (B, T, N, F) -> (B, F, N, T)
        x = x.permute(0, 3, 2, 1)
        out = [x]
        for a in support:
            x1 = self.nconv(x, a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1, a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out, dim=1)
        h = self.mlp(h)
        h = nn.functional.dropout(h, self.dropout, training=self.training)
        h = h.permute(0, 3, 2, 1)
        return h

