from torch import nn


class FC(nn.Module):
    def __init__(self, in_channels, out_channels, activation=True, norm=True):
        # (B, T, N, F) -> (B, F, N, T)
        super(FC, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=[1, 1])
        self.bn = nn.BatchNorm2d(out_channels) if (activation and norm) else None
        self.ac = nn.ReLU() if activation else None


    def forward(self, x):
        # (B, T, N, F) -> (B, F, N, T)
        x = x.permute(0, 3, 2, 1)
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.ac is not None:
            x = self.ac(x)
        x = x.permute(0, 3, 2, 1)
        return x
