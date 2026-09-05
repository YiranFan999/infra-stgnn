import torch
from torch import nn
from torch_geometric import edge_index
from torch_geometric.nn import GATv2Conv
import torch.nn.functional as F

"---------------------------------------- Diffusion Layer ----------------------------------------"
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

"----------------------------------------- Graph Convolution Layer ---------------------------------"
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


"--------------------------------------- Graph Attention Layer -------------------------------------------"
class GAT(nn.Module):
    """
    GAT Layer
    Args:
        hidden_channels (int): Number of hidden channels output from temporal attention layer
        out_channels (int): Number of output channels
        num_heads (int): Number of heads
        dropout (float): Dropout rate
    Inputs:
        x (torch.Tensor): Input tensor of shape (B, N, F) at last timestep
        edge_index (torch.Tensor): Edge indices of shape (2, E)
    """
    def __init__(self, hidden_channels, out_channels, num_heads, dropout=0):
        super(GAT, self).__init__()
        self.gat = GATv2Conv(hidden_channels, out_channels, num_heads, concat=False, dropout=dropout)

    @staticmethod
    def _batch_edge_index(batch_size, edge_index, num_nodes):
        # copy edge_index B times, edge_index: (2, E)
        offset = torch.arange(batch_size, device=edge_index.device).view(-1, 1, 1) * num_nodes # (B, 1, 1)
        support = edge_index.unsqueeze(0) + offset # (B, 2, E)
        support = support.permute(1, 0, 2) # (2, B, E)
        return support.reshape(2, -1)

    def forward(self, x, edge_index):
        # (B, N, F)
        B, N, F = x.shape
        x = x.reshape(-1, F)
        support = self._batch_edge_index(B, edge_index, N)
        x = self.gat(x, support)
        x = x.reshape(B, N, F)
        return x


"------------------------------------- Spatial Attention Layer --------------------------------------"
class SpatialAttention(nn.Module):
    """
    Spatial Attention Layer
    Args:
        K (int): Number of attention heads
        d (int): Number of channels in each attention head
        in_channels are implicitly calculated, which is 2D (concatenate x and ste)
        out_channels are implicitly calculated, which is D
    Inputs:
        x: (B, T, N, F)
        ste: spatial temporal embedding (B, T, N, F)
    Outputs:
        h: (B, T, N, F), hidden spatial attention
    """
    def __init__(self, K, d):
        super(SpatialAttention, self).__init__()
        self.K = K
        self.d = d
        in_channels = 2 * K * d
        out_channels = K * d
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

    def forward(self, x, ste):
        x = torch.cat([x, ste], dim=-1) # (B, T, N, 2D)
        x = x.permute(0, 3, 2, 1) # (B, 2D, N, P)
        q = self.q_fc(x) # (B, D, N, P)
        v = self.v_fc(x)
        k = self.k_fc(x)

        # split into K heads
        q = q.view(q.size(0), self.K, self.d, q.size(2), q.size(3)).permute(0, 1, 4, 3, 2) # (B, K, T, N, d)
        v = v.view(v.size(0), self.K, self.d, v.size(2), v.size(3)).permute(0, 1, 4, 3, 2)
        k = k.view(k.size(0), self.K, self.d, k.size(2), k.size(3)).permute(0, 1, 4, 3, 2)

        attention = (q @ k.transpose(-2, -1)) / (self.d ** 0.5)
        attention = F.softmax(attention, dim=-1)
        out = attention @ v # (B, K, T, N, d)

        out = out.permute(0, 1, 4, 3, 2) # (B, K, d, N, T)
        out = out.reshape(out.size(0), -1, out.size(3), out.size(4)) # (B, D, N, T)

        # FC before output
        out = self.out_fc(out)
        out = out.permute(0, 3, 2, 1)
        return out
