import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import optuna
import torch
from models import wavenet
from utils.datamodule import DSPDataModule
import lightning.pytorch as pl
from optuna_integration import PyTorchLightningPruningCallback
import yaml
from utils.graph_utils import synthetic_graph, AbileneGraph



ROOT = Path(__file__).parent.parent


def objective(trial: optuna.trial.Trial, datasource):
    res_channels = trial.suggest_categorical('res_channels', [16, 32, 64])
    dilation_channels = trial.suggest_categorical('dilation_channels', [16, 32, 64])
    skip_channels = trial.suggest_categorical('skip_channels', [128, 256])
    dropout = trial.suggest_float('dropout', 0.2, 0.5)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
    blocks = trial.suggest_categorical('blocks', [2, 4])

    if datasource == 'synthetic':
        g = synthetic_graph()
        path = ROOT / 'data' / 'mock_data.npy'
        num_nodes, in_dim = 5, 4
    elif datasource == 'abilene':
        g = AbileneGraph()
        path = ROOT / 'data' / 'abilene_data.npy'
        num_nodes, in_dim = 12, 2

    device = torch.device('cuda' if torch.cuda.is_available() else
                          'mps' if torch.backends.mps.is_available() else 'cpu')

    supports = [g.get_forward_matrix().to(device), g.get_backward_matrix().to(device)]

    dm = DSPDataModule(str(path), scaling='zscore', horizon=12, batch_size=64, max_samples=3000)
    dm.setup()

    model = wavenet.GWaveNet(device=device,
                             num_nodes=num_nodes,
                             in_dim=in_dim,
                             res_channels=res_channels,
                             dilation_channels=dilation_channels,
                             dropout=dropout,
                             skip_channels=skip_channels,
                             supports=supports,
                             adpinit=None,
                             adpadj=True,
                             kernel_size=2,
                             end_channels=512,
                             dilation=[1, 2],
                             out_dim=12,
                             blocks=blocks,
                             lr=lr,
                             weight_decay=weight_decay,)


    trainer = pl.Trainer(max_epochs=20,
                         accelerator="gpu" if torch.cuda.is_available() else
                         "mps" if torch.backends.mps.is_available() else "cpu",
                         callbacks=[pl.callbacks.EarlyStopping(monitor="val_loss", patience=5),
                                    PyTorchLightningPruningCallback(trial, monitor='val_loss')],
                         enable_checkpointing=False,
                         logger=False,
                         )
    trainer.fit(model, dm)

    ckpt_dir = Path('results/wavenet_search/checkpoints')
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"{trial.number}.ckpt"
    trainer.save_checkpoint(ckpt_path)
    return trainer.callback_metrics["val_loss"].item()

def main():
    datasource = 'synthetic'
    pruner = optuna.pruners.MedianPruner()
    study = optuna.create_study(direction="minimize",
                                pruner=pruner,
                                study_name="wavenet_search",
                                storage="sqlite:///results/tune_wavenet.db",
                                load_if_exists=True,)
    study.optimize(lambda trial: objective(trial, datasource), n_trials=20)



    best = study.best_params
    best['datasource'] = datasource
    out = {'datasource': datasource,
           datasource: {**best,
                        'horizon': 12,
                        'num_nodes': 5 if datasource == 'synthetic' else 12,
                        'num_features': 4 if datasource == 'synthetic' else 2,
                        'batch_size': 64,
                        'kernel_size': 2,
                        'dilation': [1, 2],
                        'end_channels': 512,
                        'adpinit': None,
                        'adpadj': True,
                        'max_epochs': 50,
                        }}
    (ROOT / 'results').mkdir(exist_ok=True)
    out_path = ROOT / 'results' / 'best_params_wavenet.yaml'
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, 'w') as f:
        yaml.dump(out, f)
    print('Saved to', out_path)
    print('Best params:', best)
    print('Best val_loss:', study.best_value)
    print('Best trial:', study.best_trial.number)


if __name__ == "__main__":
    main()
