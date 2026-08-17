import lightning as pl
from torch import nn
from torch.nn.functional import relu
from torch.optim import AdamW
from torch_geometric.nn import ChebConv

from layers.temporal import TemporalConv
from utils.datamodule import FeatureScaler
from utils.graph_utils import synthetic_graph
from utils.metrics import *


class STBlock(nn.Module):
    """
    Temporal -> Spatial -> Temporal block, single sandwich structure in STGCN
    Due to ChebConv requires the adjacency matrix to be symmetric, we use Chebyshev polynomial
    approximation for spatial convolution.
    Args:
        in_channels (int): Number of channels in the input graph
        out_channels (int): Number of channels in the output graph
        kernel_size (int): Size of temporal convolution kernel
        K (int): Chebyshev polynomial order for spatial convolution

    Inputs:
        x: (B, T, N, F)
        edge_index: (2, E) tensor of edge indices
        edge_weight: (E,) tensor of edge weights

    Outputs:
        x: (B, T, N, F)
    """
    def __init__(self, in_channels, out_channels, kernel_size= 3, K=3):
        super(STBlock, self).__init__()
        self.temporal1 = TemporalConv(in_channels, out_channels, kernel_size= kernel_size)
        self.spatial = ChebConv(out_channels, out_channels, K)
        self.temporal2 = TemporalConv(out_channels, out_channels, kernel_size= kernel_size)
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x, edge_index, edge_weight):
        # x: (B, T, N, F)
        x = self.temporal1(x)

        B, T, N, F = x.shape
        # ChebConv requires x shape of (N, F)
        x = x.reshape(B * T, N, F)
        x = self.spatial(x, edge_index, edge_weight)
        # convert x shape back for temporal convolution
        x = x.reshape(B, T, N, -1)
        x = relu(x)

        x = self.temporal2(x)

        x = self.norm(x)
        return x

class STGCN(pl.LightningModule):
    """
    Two Sptatio-Temporal blocks + Fully connected layer for single-step prediction.
    Args:
        in_channels (int): Number of channels in the input graph
        out_channels (int): hidden layers
        kernel_size (int): Size of temporal convolution kernel
        K (int): Chebyshev polynomial order for spatial convolution
        lr (float): learning rate
        weight_decay (float): Weight decay for optimizer
    Inputs:
        x: (B, T, N, F)
    Outputs:
        x: (B, T, N, F)
    The edge_index and edge_weight are defined in utils/graph_utils.py
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 K=3,
                 lr=1e-3,
                 weight_decay=1e-4,
                 edge_index=None,
                 edge_weight=None,
                 scaler=None):
        super(STGCN, self).__init__()
        self.scaler = scaler
        self.save_hyperparameters(ignore=['scaler', 'edge_index', 'edge_weight'])

        SYN_G = synthetic_graph()
        self.register_buffer('edge_index', edge_index if edge_index is not None else SYN_G.get_edge_index())
        self.register_buffer('edge_weight', edge_weight if edge_weight is not None else SYN_G.get_edge_weight())

        self.block1 = STBlock(in_channels, out_channels, kernel_size, K)
        self.block2 = STBlock(out_channels, out_channels, kernel_size, K)
        self.output = nn.Linear(out_channels, in_channels)



    def forward(self, x):
        x = self.block1(x, self.edge_index, self.edge_weight)
        x = self.block2(x, self.edge_index, self.edge_weight)
        x = x[:, -1, :, :]  # take the last time step
        x = self.output(x)
        return x

    def _shared_step(self, batch, tag: str):
        x, y = batch
        output = self(x)
        mse_loss = mse(output, y)
        self.log(f"{tag}_loss", mse_loss, on_step=False, on_epoch=True)
        return mse_loss

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

        # FEATURES = ['queue_length', 'utilization', 'arrival_rate', 'avg_delay']
        self.log('test_loss', mse(pred, y))
        self.log('test_mae', mae(pred, y))
        self.log('test_rmse', rmse(pred, y))
        self.log('test_mape', mape(pred, y))
        for j in range(pred.shape[-1]):
            self.log(f'test_mae_feature_{j}', mae(pred[:, :, j], y[:, :, j]))

    def configure_optimizers(self):
        optim = AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='min', factor=0.5, patience=3)
        return {"optimizer": optim, "lr_scheduler": {"scheduler": sch, "monitor": "val_loss"}}




