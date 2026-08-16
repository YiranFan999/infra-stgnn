
from pathlib import Path

import lightning.pytorch as pl
import torch
from lightning.pytorch.loggers import TensorBoardLogger

from models.stgcn import STGCN
from utils.datamodule import DSPDataModule


ROOT = Path(__file__).parent.parent
PATH = Path(__file__).parent.parent / 'data' / 'mock_data.npy'
NUM_FEATURES = 4
dm = DSPDataModule(str(PATH), scaling='zscore')
dm.setup()


model = STGCN(in_channels=NUM_FEATURES,
              out_channels=32,
              lr=0.001,
              weight_decay=1e-4,
              scaler=dm.scaler)

logger_dir = ROOT / 'logs'
logger_dir.mkdir(exist_ok=True)
trainer = pl.Trainer(max_epochs=100,
                     accelerator='gpu' if torch.cuda.is_available() else 'cpu',
                     callbacks=[pl.callbacks.EarlyStopping(monitor='val_loss', patience=5),
                                pl.callbacks.ModelCheckpoint(monitor='val_loss', save_top_k=1)],
                     logger=TensorBoardLogger(save_dir=str(logger_dir), name='stgcn'),
                     gradient_clip_val=0.5)

if __name__ == '__main__':
    trainer.fit(model, dm)
    trainer.test(model, dm)