from pathlib import Path

import lightning.pytorch as pl
import torch
from lightning.pytorch.loggers import TensorBoardLogger

from models import gman
from utils.datamodule import DSPDataModule
import yaml



ROOT = Path(__file__).parent.parent

try:
    with open(ROOT / 'config' / 'best_params_gman.yaml', 'r') as f:
        config_file = yaml.safe_load(f)
        datasource = config_file['datasource']
        config = config_file[datasource]
except FileNotFoundError:
    with open(ROOT / 'config' / 'gman.yaml') as f:
        config_file = yaml.safe_load(f)
        datasource = config_file['datasource']
        config = config_file[datasource]

if datasource not in ['synthetic', 'abilene']:
    raise ValueError('datasource must be one of [synthetic, abilene]')

# Define the dataset file, output file, and tensorboard log file
if datasource == 'synthetic':
    se = torch.load(ROOT / 'data' / 'node2vec_synthetic.pt', weights_only=True)
    PATH = ROOT / 'data' / 'mock_data.npy'
    OUT_PATH = ROOT / 'results' / 'gman_synthetic.yaml'
    OUT_PATH.parent.mkdir(exist_ok=True)
    LOG_NAME = 'gman_synthetic'
elif datasource == 'abilene':
    se = torch.load(ROOT / 'data' / 'node2vec_abilene.pt', weights_only=True)
    PATH = ROOT / 'data' / 'abilene_data.npy'
    OUT_PATH = ROOT / 'results' / 'gman_abilene.yaml'
    OUT_PATH.parent.mkdir(exist_ok=True)
    LOG_NAME = 'gman_abilene'

if torch.cuda.is_available():
    acc = 'gpu'
elif torch.backends.mps.is_available():
    acc = 'mps'
else:
    acc = 'cpu'
logger_dir = ROOT / 'logs'
logger_dir.mkdir(exist_ok=True)


if __name__ == '__main__':
    dm = DSPDataModule(str(PATH),
                       scaling='zscore',
                       horizon=config['horizon'],
                       window=config['window'],
                       batch_size=config['batch_size'],
                       steps_per_day=config.get('steps_per_day'))
    dm.setup()


    model = gman.GMAN(in_channels=config['num_features'],
                      hidden_channels=config['K'] * config['d'],
                      se=se,
                      K=config['K'],
                      d=config['d'],
                      L=config['L'],
                      dropout=config['dropout'],
                      T=config.get('steps_per_day'),
                      num_steps=None if config.get('steps_per_day') else config['window'] + config['horizon'],
                      lr=config['lr'],
                      weight_decay=config['weight_decay'],
                      feature_names=config['feature_names'],
                      scaler=dm.scaler)


    trainer = pl.Trainer(max_epochs=config['max_epochs'],
                         accelerator=acc,
                         callbacks=[pl.callbacks.EarlyStopping(monitor='val_loss', patience=5),
                                    pl.callbacks.ModelCheckpoint(monitor='val_loss', save_top_k=1)],
                         logger=TensorBoardLogger(save_dir=str(logger_dir), name=LOG_NAME),
                         gradient_clip_val=0.5)


    trainer.fit(model, dm)
    trainer.test(model, dm, ckpt_path='best')
    results = {k: float(v) for k, v in trainer.callback_metrics.items() if 'test' in k}
    results['horizon'] = config['horizon']
    with open(OUT_PATH, 'w') as f:
        yaml.dump(results, f)
    print(f'Saved to {OUT_PATH}')
