import os
from pathlib import Path

import lightning.pytorch as pl
import torch
from lightning.pytorch.loggers import TensorBoardLogger

from models.stgcn import STGCN
from utils.datamodule import DSPDataModule
import yaml

NUM_FEATURES = 4
HORIZON = 6
KERNEL_SIZE = 3
ROOT = Path(__file__).parent.parent
PATH = Path(__file__).parent.parent / 'data' / 'mock_data.npy'
OUT_PATH = ROOT / 'results' / f'stgcn_synthetic_h{HORIZON}.yaml'
OUT_PATH.parent.mkdir(exist_ok=True)


dm = DSPDataModule(str(PATH), scaling='zscore', horizon=HORIZON)
dm.setup()

# with open(ROOT / 'results' / 'best_params_stgcn.yaml', 'r') as f:
#     params = yaml.safe_load(f)
model = STGCN(in_channels=NUM_FEATURES,
              out_channels=16,
              kernel_size=KERNEL_SIZE,
              lr=0.005868,
              weight_decay=0.00047536,
              horizon=HORIZON,
              scaler=dm.scaler)

logger_dir = ROOT / 'logs'
logger_dir.mkdir(exist_ok=True)
trainer = pl.Trainer(max_epochs=50,
                     accelerator='gpu' if torch.cuda.is_available() else 'cpu',
                     callbacks=[pl.callbacks.EarlyStopping(monitor='val_loss', patience=5),
                                pl.callbacks.ModelCheckpoint(monitor='val_loss', save_top_k=1)],
                     logger=TensorBoardLogger(save_dir=str(logger_dir), name=f'stgcn_h{HORIZON}_kernel{KERNEL_SIZE}'),
                     gradient_clip_val=0.5)

if __name__ == '__main__':
    trainer.fit(model, dm)
    trainer.test(model, dm, ckpt_path='best')
    results = {k: float(v) for k, v in trainer.callback_metrics.items() if 'test' in k}
    results['horizon'] = HORIZON
    with open(OUT_PATH, 'w') as f:
        yaml.dump(results, f)
    print(f'Saved to {OUT_PATH}')