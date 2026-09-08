import torch
from torch.nn.functional import relu

from layers.spatial import gcn
from layers.temporal import DilatedTemporalConv
import lightning as pl
from torch import nn
import torch.nn.functional as F

from utils.metrics import *
from torch.optim import AdamW


class wavenet_layer(nn.Module):
    def __init__(self,
                 res_channels,
                 dilation_channels,
                 dropout,
                 kernel_size,
                 dilation,
                 supports_len,
                 skip_channels):
        super(wavenet_layer, self).__init__()

        self.tcn = DilatedTemporalConv(in_channels=res_channels,
                                                 out_channels=dilation_channels,
                                                 kernel_size=kernel_size,
                                                 dilation=dilation)

        self.gcn = gcn(in_channels=dilation_channels,
                       out_channels=res_channels,
                       dropout=dropout,
                       support_len=supports_len)
        # skip connection is after gated TCN
        self.skip_connection = nn.Conv2d(dilation_channels,
                                         skip_channels,
                                         kernel_size=(1, 1))


    def forward(self, x, supports):
        # input = self.input_transform(x)
        # x: (B, T, N, F)
        temporal_output = self.tcn(x)
        spatial_output = self.gcn(temporal_output, supports)
        output = spatial_output + x[:, -spatial_output.shape[1]:, :, :]
        skip_output = self.skip_connection(temporal_output.permute(0, 3, 2, 1))
        # skip_output shape: (B, F, N, T)
        # output shape: (B, T, N, F)
        return output, skip_output


class GWaveNet(pl.LightningModule):
    def __init__(self,
                 device,
                 num_nodes,
                 dilation = [1, 2],
                 in_dim=2,
                 res_channels=32,
                 dilation_channels=32,
                 dropout=0.3,
                 supports=None,
                 adpadj=True,
                 adpinit=None,
                 kernel_size=2,
                 skip_channels=256,
                 end_channels=512,
                 out_dim=12,
                 blocks=4,
                 layers=2,
                 lr=1e-3,
                 weight_decay=1e-4,
                 scaler=None,
                 feature_names=None
                 ):
        super(GWaveNet, self).__init__()
        if (len(dilation) != layers):
            raise ValueError("Length of dilation list must be equal to the number of layers.")
        self.input_transform = nn.Conv2d(in_dim, res_channels, kernel_size=(1, 1))
        # self.layer = wavenet_layer(res_channels=res_channels,
        #                            dilation_channels=dilation_channels,
        #                            dropout=dropout,
        #                            kernel_size=kernel_size,
        #                            dilation=dilation,
        #                            skip_channels=skip_channels)
        self.wavenet_layer = nn.ModuleList()

        self.end_conv1 = nn.Conv2d(in_channels=skip_channels,
                                   out_channels=end_channels,
                                   kernel_size=(1, 1))

        self.end_conv2 = nn.Conv2d(in_channels=end_channels,
                                   out_channels=out_dim*in_dim, # predict each time step for each feature
                                   kernel_size=(1, 1))

        self.supports = supports
        self.supports_len = 0
        if supports is not None:
            self.supports_len = len(supports)

        # configure the self-adaptive adjacency matrix
        if adpadj:
            if adpinit is None:
                # randomly initialise E1 and E2
                if supports is None:
                    self.supports = []
                self.e1 = nn.Parameter(torch.randn(num_nodes, 10), requires_grad=True).to(device)
                self.e2 = nn.Parameter(torch.randn(10, num_nodes), requires_grad=True).to(device)
                self.supports_len += 1

            else:
                if supports is None:
                    self.supports = []
                u, s, v = torch.svd(adpinit)
                self.e1 = nn.Parameter(u[:, :10] @ torch.diag(s[:10]**0.5))
                self.e2 = nn.Parameter(torch.diag(s[:10]**0.5) @ v[:, :10].t())
                self.supports_len += 1
        self.scaler = scaler
        self.save_hyperparameters(ignore=['adpadj', 'adpinit', 'device', 'supports', 'scaler'])


        # calculate receptive field, which will be used to be compared with window in the forward function
        self.receptive_field = 1
        for block in range(blocks):
            for layer in range(layers):
                self.receptive_field += dilation[layer] * (kernel_size - 1)
                self.wavenet_layer.append(wavenet_layer(res_channels=res_channels,
                                                        dilation_channels=dilation_channels,
                                                        dropout=dropout,
                                                        kernel_size=kernel_size,
                                                        dilation=dilation[layer],
                                                        skip_channels=skip_channels,
                                                        supports_len=self.supports_len))



    def forward(self, x):
        # x: (B, T, N, F)
        in_len = x.shape[1]
        if in_len < self.receptive_field:
            x = F.pad(x, (0, 0, 0, 0, self.receptive_field - in_len, 0))

        adp = F.softmax(F.relu(self.e1 @ self.e2), dim=1)
        supports = self.supports + [adp]

        x = x.permute(0, 3, 2, 1)
        x = self.input_transform(x) # x: (B, F, N, T)
        x = x.permute(0, 3, 2, 1) # x: (B, T, N, F)

        skip = 0
        for i in range(self.hparams.blocks * self.hparams.layers):
            x, s = self.wavenet_layer[i](x, supports)
            try:
                skip = skip[:, :, :, -s.size(3):] + s
            except:
                skip += s
        # x: (B, T, N, F), skip, s: (B, F, N, T)
        skip_output = relu(skip)
        skip_output = self.end_conv1(skip_output)
        skip_output = relu(skip_output)
        skip_output = self.end_conv2(skip_output)
        # skip_output: (B, out_dim*in_dim, N, 1), out_dim is the predicted timestep, in_dim is number of features
        skip_output = skip_output.squeeze(-1) # (B, out_dim*in_dim, N)
        skip_output = skip_output.permute(0, 2, 1) # (B, N, out_dim*in_dim)
        skip_output = skip_output.reshape(skip_output.size(0), skip_output.size(1), self.hparams.out_dim, self.hparams.in_dim) # (B, N, out_dim, in_dim)
        skip_output = skip_output.permute(0, 2, 1, 3)
        return skip_output

    def _shared_step(self, batch, tag):
        x, y = batch
        y_hat = self(x)
        loss = mae(y_hat, y)
        self.log(f"{tag}_loss", loss, on_step=False, on_epoch=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
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
                self.log(f'test_mae_{name}_h{h+1}', mae(pred[:, h, :, j], y[:, h, :, j]))

    def configure_optimizers(self):
        optim = AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='min', factor=0.5, patience=3)
        return {"optimizer": optim, "lr_scheduler": {"scheduler": sch, "monitor": "val_loss"}}