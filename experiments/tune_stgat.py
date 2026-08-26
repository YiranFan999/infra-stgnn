import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import optuna
import torch
from models import stgat
from utils.datamodule import DSPDataModule
import lightning.pytorch as pl
from optuna_integration import PyTorchLightningPruningCallback
import yaml
from utils.graph_utils import synthetic_graph, AbileneGraph



ROOT = Path(__file__).parent.parent
FEATURES = {
    'synthetic': ['queue_length', 'utilization', 'arrival_rate', 'avg_delay'],
    'abilene':   ['in_flow', 'out_flow'],
}

def objective(trial: optuna.trial.Trial, datasource):
    hidden_channels = trial.suggest_categorical('hidden_channels', [32, 64, 128])
    num_layers = trial.suggest_int('num_layers', 1, 2)
    num_heads_tem = trial.suggest_categorical('num_heads_tem', [2, 4, 8])
    num_heads_spa = trial.suggest_categorical('num_heads_spa', [2, 4, 8])
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
    if num_layers > 1:
        dropout_tem = trial.suggest_float('dropout_tem', 0.0, 0.3)
    else:
        dropout_tem = 0.0

    dropout_spa = trial.suggest_categorical('dropout_spa', [0.0, 0.1, 0.2])
    dropout_mlp = trial.suggest_float('dropout_mlp', 0.0, 0.4)

    feature_names = FEATURES[datasource]
    if datasource == 'synthetic':
        g = synthetic_graph()
        path = ROOT / 'data' / 'mock_data.npy'
        num_nodes, in_dim = 5, 4

    elif datasource == 'abilene':
        g = AbileneGraph()
        path = ROOT / 'data' / 'abilene_data.npy'
        num_nodes, in_dim = 12, 2


    dm = DSPDataModule(str(path), scaling='zscore', horizon=12, batch_size=64, max_samples=3000)
    dm.setup()

    model = stgat.STGAT(in_channels=in_dim,
                        hidden_channels=hidden_channels,
                        out_channels=hidden_channels,
                        num_layers=num_layers,
                        dropout_tem=dropout_tem,
                        num_heads_tem=num_heads_tem,
                        num_heads_spa=num_heads_spa,
                        dropout_spa=dropout_spa,
                        dropout_mlp=dropout_mlp,
                        horizon=12,
                        edge_index=g.get_edge_index(),
                        scaler=dm.scaler,
                        lr=lr,
                        weight_decay=weight_decay,
                        feature_names=feature_names
                        )


    trainer = pl.Trainer(max_epochs=50,
                         accelerator="gpu" if torch.cuda.is_available() else
                         "mps" if torch.backends.mps.is_available() else "cpu",
                         callbacks=[pl.callbacks.EarlyStopping(monitor="val_loss", patience=5),
                                    PyTorchLightningPruningCallback(trial, monitor='val_loss')],
                         enable_checkpointing=False,
                         logger=False,
                         )
    trainer.fit(model, dm)

    ckpt_dir = ROOT / 'results' / 'stgat_search' / 'checkpoints'
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"{trial.number}.ckpt"
    trainer.save_checkpoint(ckpt_path)
    return trainer.callback_metrics["val_loss"].item()

def main():
    datasource = 'synthetic'
    (ROOT / 'results').mkdir(exist_ok=True)
    pruner = optuna.pruners.MedianPruner()
    study = optuna.create_study(direction="minimize",
                                pruner=pruner,
                                study_name="stgat_search",
                                storage=f"sqlite:///{ROOT / 'results' / 'tune_stgat.db'}",
                                load_if_exists=True,)
    study.optimize(lambda trial: objective(trial, datasource), n_trials=50)



    best = study.best_params
    out = {'datasource': datasource,
           datasource: {**best,
                        'feature_names': FEATURES[datasource],
                        'horizon': 12,
                        'num_nodes': 5 if datasource == 'synthetic' else 12,
                        'num_features': 4 if datasource == 'synthetic' else 2,
                        'batch_size': 64,
                        'max_epochs': 50,
                        'dropout_tem': best.get('dropout_tem', 0.0),
                        }}

    out_path = ROOT / 'results' / 'best_params_stgat.yaml'
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, 'w') as f:
        yaml.dump(out, f)
    print('Saved to', out_path)
    print('Best params:', best)
    print('Best val_loss:', study.best_value)
    print('Best trial:', study.best_trial.number)


if __name__ == "__main__":
    main()
