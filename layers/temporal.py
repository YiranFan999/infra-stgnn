from torch import nn
from torch.nn.functional import sigmoid


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

    