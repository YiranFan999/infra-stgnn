from torch import nn
from layers.temporal import GRU
from layers.spatial import GAT
import lightning as pl
from utils.metrics import *
from torch.optim import AdamW


class STGAT(pl.LightningModule):
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 out_channels,
                 num_layers=2,
                 dropout_tem=0,
                 num_heads_tem=2,
                 num_heads_spa=2,
                 dropout_spa=0,
                 dropout_mlp=0,
                 horizon=1,
                 edge_index=None,
                 scaler=None,
                 lr=1e-4,
                 weight_decay=1e-5,
                 feature_names=None
                 ):
        super(STGAT, self).__init__()
        self.save_hyperparameters(ignore=['edge_index', 'scaler'])
        self.register_buffer('edge_index', edge_index, persistent=False) # don't store in state_dict in case the graph structure is changed
        self.scaler = scaler

        self.temporal = GRU(in_channels=in_channels,
                       hidden_channels=hidden_channels,
                       num_layers=num_layers,
                       dropout=dropout_tem,
                       num_heads=num_heads_tem)
        self.spatial = GAT(hidden_channels=hidden_channels,
                           out_channels=out_channels,
                           num_heads=num_heads_spa,
                           dropout=dropout_spa)

        self.mlp = nn.Sequential(nn.Linear(out_channels, hidden_channels),
                                 nn.LeakyReLU(),
                                 nn.Dropout(dropout_mlp),
                                 nn.Linear(hidden_channels, in_channels*horizon),)



    def forward(self, x):
        B, T, N, F = x.shape
        temporal_out = self.temporal(x)
        spatial_out = self.spatial(temporal_out, self.edge_index)
        mlp_out = self.mlp(spatial_out) # (B, N, F*horizon)
        out = mlp_out.reshape(B, N, -1, F)
        return out.permute(0, 2, 1, 3).contiguous()

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