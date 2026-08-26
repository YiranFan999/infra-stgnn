from pathlib import Path

import lightning.pytorch as pl
import torch
from lightning.pytorch.loggers import TensorBoardLogger

from models import stgat
from utils.datamodule import DSPDataModule
import yaml



ROOT = Path(__file__).parent.parent

try:
    with open(ROOT / 'results' / 'best_params_stgat.yaml', 'r') as f:
        config_file = yaml.safe_load(f)
        datasource = config_file['datasource']
        config = config_file[datasource]
except FileNotFoundError:
    with open(ROOT / 'config' / 'stgat.yaml') as f:
        config_file = yaml.safe_load(f)
        datasource = config_file['datasource']
        config = config_file[datasource]

if datasource not in ['synthetic', 'abilene']:
    raise ValueError('datasource must be one of [synthetic, abilene]')

# Define the dataset file, output file, and tensorboard log file
if datasource == 'synthetic':
    from utils.graph_utils import synthetic_graph
    g = synthetic_graph()
    PATH = ROOT / 'data' / 'mock_data.npy'
    OUT_PATH = ROOT / 'results' / 'stgat_synthetic.yaml'
    OUT_PATH.parent.mkdir(exist_ok=True)
    LOG_NAME = 'stgat_synthetic'
elif datasource == 'abilene':
    from utils.graph_utils import AbileneGraph
    g = AbileneGraph()
    PATH = ROOT / 'data' / 'abilene_data.npy'
    OUT_PATH = ROOT / 'results' / 'stgat_abilene.yaml'
    OUT_PATH.parent.mkdir(exist_ok=True)
    LOG_NAME = 'stgat_abilene'

if config['num_layers'] == 1 and config['dropout_tem'] > 0:
    raise ValueError('num_layers must be greater than 1 when applying dropout on the GRU block')


if torch.cuda.is_available():
    acc = 'gpu'
elif torch.backends.mps.is_available():
    acc = 'mps'
else:
    acc = 'cpu'
logger_dir = ROOT / 'logs'
logger_dir.mkdir(exist_ok=True)


if __name__ == '__main__':
    dm = DSPDataModule(str(PATH), scaling='zscore', horizon=config['horizon'], batch_size=config['batch_size'])
    dm.setup()


    model = stgat.STGAT(feature_names=config['feature_names'],
                        in_channels=config['num_features'],
                        hidden_channels=config['hidden_channels'],
                        out_channels=config['hidden_channels'],
                        num_layers=config['num_layers'],
                        dropout_tem=config['dropout_tem'],
                        num_heads_tem=config['num_heads_tem'],
                        num_heads_spa=config['num_heads_spa'],
                        dropout_spa=config['dropout_spa'],
                        dropout_mlp=config['dropout_mlp'],
                        horizon=config['horizon'],
                        edge_index=g.get_edge_index(),
                        lr=config['lr'],
                        weight_decay=config['weight_decay'],
                        scaler=dm.scaler,)


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