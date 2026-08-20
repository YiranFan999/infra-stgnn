from pathlib import Path

import lightning.pytorch as pl
import torch
from lightning.pytorch.loggers import TensorBoardLogger

from models import wavenet
from utils.datamodule import DSPDataModule
import yaml



ROOT = Path(__file__).parent.parent
# PATH = Path(__file__).parent.parent / 'data' / 'mock_data.npy'

try:
    with open(ROOT / 'results' / 'best_params_wavenet.yaml', 'r') as f:
        config_file = yaml.safe_load(f)
        datasource = config_file['datasource']
        config = config_file[datasource]
except FileNotFoundError:
    with open(ROOT / 'config' / 'wavenet.yaml') as f:
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
    OUT_PATH = ROOT / 'results' / 'wavenet_synthetic.yaml'
    OUT_PATH.parent.mkdir(exist_ok=True)
    LOG_NAME = 'wavenet_synthetic'
elif datasource == 'abilene':
    from utils.graph_utils import AbileneGraph
    g = AbileneGraph()
    PATH = ROOT / 'data' / 'abilene_data.npy'
    OUT_PATH = ROOT / 'results' / 'wavenet_abilene.yaml'
    OUT_PATH.parent.mkdir(exist_ok=True)
    LOG_NAME = 'wavenet_abilene'



device = None
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

SUPPORTS = [g.get_forward_matrix().to(device), g.get_backward_matrix().to(device)]


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


    model = wavenet.GWaveNet(device=device,
                             num_nodes=config['num_nodes'],
                             dilation=config['dilation'],
                             in_dim=config['num_features'],
                             res_channels=config['res_channels'],
                             dilation_channels=config['dilation_channels'],
                             dropout=config['dropout'],
                             supports=SUPPORTS,
                             adpadj=config['adpadj'],
                             adpinit=config['adpinit'],
                             kernel_size=config['kernel_size'],
                             skip_channels=config['skip_channels'],
                             end_channels=config['end_channels'],
                             out_dim=config['horizon'],
                             blocks=config['blocks'],
                             layers=config['layers'],
                             lr=config['lr'],
                             weight_decay=config['weight_decay'])


    trainer = pl.Trainer(max_epochs=config['max_epochs'],
                         accelerator=acc,
                         callbacks=[pl.callbacks.EarlyStopping(monitor='val_loss', patience=5),
                                    pl.callbacks.ModelCheckpoint(monitor='val_loss', save_top_k=1)],
                         logger=TensorBoardLogger(save_dir=str(logger_dir), name=f'{LOG_NAME}'),
                         gradient_clip_val=0.5)


    trainer.fit(model, dm)
    trainer.test(model, dm, ckpt_path='best')
    results = {k: float(v) for k, v in trainer.callback_metrics.items() if 'test' in k}
    results['horizon'] = config['horizon']
    with open(OUT_PATH, 'w') as f:
        yaml.dump(results, f)
    print(f'Saved to {OUT_PATH}')