from pathlib import Path

import lightning.pytorch as pl
import torch
from lightning.pytorch.loggers import TensorBoardLogger

from models.stgcn import STGCN
from utils.datamodule import DSPDataModule
from utils.graph_utils import AbileneGraph
import yaml

ROOT = Path(__file__).parent.parent
PATH = Path(__file__).parent.parent / 'data' / 'abilene_data.npy'
NUM_FEATURES = 2
dm = DSPDataModule(str(PATH), scaling='zscore')
dm.setup()

graph = AbileneGraph()
with open(ROOT / 'results' / 'best_params_stgcn_abilene.yaml', 'r') as f:
    params = yaml.safe_load(f)
model = STGCN(in_channels=NUM_FEATURES,
              out_channels=params['out_channels'],
              lr=params['lr'],
              weight_decay=params['weight_decay'],
              edge_index=graph.get_edge_index(),
              edge_weight=graph.get_edge_weight(),
              scaler=dm.scaler # inverse test dataset to see loss on original dataset
              )

logger_dir = ROOT / 'logs'
logger_dir.mkdir(exist_ok=True)
trainer = pl.Trainer(max_epochs=100,
                     # accelerator='gpu' if torch.cuda.is_available() else 'cpu', # use this when deploying when GPU is available
                     accelerator='mps' if torch.backends.mps.is_available() else 'cpu', # use this for local testing
                     callbacks=[pl.callbacks.EarlyStopping(monitor='val_loss', patience=5),
                                pl.callbacks.ModelCheckpoint(monitor='val_loss', save_top_k=1)],
                     logger=TensorBoardLogger(save_dir=str(logger_dir), name='stgcn_abilene'),
                     gradient_clip_val=0.5)

if __name__ == '__main__':
    trainer.fit(model, dm)
    trainer.test(model, dm)