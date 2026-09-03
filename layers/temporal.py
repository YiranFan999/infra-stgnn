import torch
from torch import nn
from torch.nn.functional import sigmoid
import torch.nn.functional as F

class TemporalConv(nn.Module):
    """ Temporal Convolution Layer with GLU gating

    Args:
        in_channels (int): Number of channels in the node
        out_channels (int): Number of channels in the output
        kernel_size (int or tuple): Size of the temporal conv kernel

    Inputs:
        x: (N, C_in, H, W)

    Outputs:
        (N, C_out, H, W)
    """
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(TemporalConv, self).__init__()
        # input: (N, C, H, W)
        self.conv = nn.Conv2d(in_channels, out_channels * 2, (1, kernel_size), padding=(0, kernel_size // 2))
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1) \
            if in_channels != out_channels else nn.Identity()

    def forward(self, X):
        # (B, T, N, F) -> (B, F, N, T)
        X = X.permute(0, 3, 2, 1)  # (N, C_in, H, W)
        out = self.conv(X)
        P, Q = out.chunk(2, dim=1)
        res = self.residual(X)
        out = P * sigmoid(Q) + res  # GLU gating + residual connection
        out = out.permute(0, 3, 2, 1)
        return out


class DilatedTemporalConv(nn.Module):
    """
    Dilated Casual Temporal Convolution
    Args:
        in_channels (int): Number of features in the node
        out_channels (int): Number of feature in the output
        kernel_size (int or tuple): Size of the temporal conv kernel
        dilation (int): Dilation factor, increase with the layers
    Input:
        x: (N, W, H, C_in), corresponds to (batch_size, time_step, num_nodes, in_channels)
    """

    def __init__(self, in_channels, out_channels, kernel_size=2, dilation=1, padding=0):
        super(DilatedTemporalConv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, (1, kernel_size), dilation=dilation, padding=padding)
        self.conv2 = nn.Conv2d(in_channels, out_channels, (1, kernel_size), dilation=dilation, padding=padding)
        self.g = nn.Tanh()
        self.sigmoid = nn.Sigmoid()


    def forward(self, X):
        # (B, T, N, F) -> (B, F, N, T)
        X = X.permute(0, 3, 2, 1)  # (N, C_in, H, W)
        temporal1 = self.conv1(X)
        temporal2 = self.conv2(X)
        out = self.g(temporal1) * self.sigmoid(temporal2)
        out = out.permute(0, 3, 2, 1)
        return out


class GRU(nn.Module):
    """
    GRU Layer
    Args:
        in_channels (int): Number of features in the node
        hidden_channels (int): Number of channels in the GRU hidden state
        num_layers (int): Number of layers in the GRU
        dropout (float): Dropout rate
        bidirectional (bool): Whether the GRU is bidirectional
        num_heads (int): Number of heads in the GRU

    Input:
        x: (B, T, N, F)
    Output:
        h: (B, N, F), hidden state at the last time step
    """
    def __init__(self, in_channels, hidden_channels, num_layers, dropout, num_heads):
        super(GRU, self).__init__()
        self.gru = nn.GRU(input_size=in_channels, hidden_size=hidden_channels, num_layers=num_layers, batch_first=True, dropout=dropout)
        # batch_first: (batch, seq, feature)
        self.attention = nn.MultiheadAttention(embed_dim=hidden_channels, num_heads=num_heads, batch_first=True)
        self.w_a = nn.Linear(hidden_channels, hidden_channels)

    def forward(self, x):
        B, T, N, F = x.shape
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        x = x.permute(0, 2, 1, 3) # (B, T, N, F) -> (B, N, T, F)
        x = x.reshape(-1, x.shape[-2], x.shape[-1]) # (B*N, T, F)
        out, h = self.gru(x)
        h_a, _ = self.attention(out, out, out, need_weights=False, attn_mask=mask)
        H = self.w_a(out * h_a)
        H = H.reshape(B, -1, H.shape[1], H.shape[2])
        H = H.permute(0, 2, 1, 3)
        return H[:, -1] # the last step has already contained all history information per node


class TemporalAttention(nn.Module):
    def __init__(self, in_channels, K, d, mask=True):
        super(TemporalAttention, self).__init__()
        self.K = K
        self.d = d
        self.mask = mask
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
        x = x.permute(0, 3, 2, 1) # (B, 2D, N, P+Q)
        q = self.q_fc(x) # (B, D, N, P+Q)
        v = self.v_fc(x)
        k = self.k_fc(x)

        # split into K heads
        q = q.view(q.size(0), self.K, self.d, q.size(2), q.size(3)).permute(0, 1, 3, 4, 2) # (B, K, N, T, d)
        v = v.view(v.size(0), self.K, self.d, v.size(2), v.size(3)).permute(0, 1, 3, 4, 2)
        k = k.view(k.size(0), self.K, self.d, k.size(2), k.size(3)).permute(0, 1, 3, 4, 2)

        attention = (q @ k.transpose(-2, -1)) / (self.d ** 0.5)
        if self.mask:
            causal = torch.triu(torch.ones(attention.size(-1), attention.size(-1), dtype=torch.bool, device=attention.device), diagonal=1)
            attention = attention.masked_fill(causal, float('-inf'))
        attention = F.softmax(attention, dim=-1)
        out = attention @ v # (B, K, N, T, d)

        out = out.permute(0, 1, 4, 2, 3) # (B, K, d, N, T)
        out = out.reshape(out.size(0), -1, out.size(3), out.size(4)) # (B, D, N, T)

        # FC before output
        out = self.out_fc(out)
        out = out.permute(0, 3, 2, 1)
        return out
