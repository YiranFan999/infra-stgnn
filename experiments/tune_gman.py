import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import optuna
import torch
from models import gman
from utils.datamodule import DSPDataModule
import lightning.pytorch as pl
from optuna_integration import PyTorchLightningPruningCallback
import yaml



ROOT = Path(__file__).parent.parent
FEATURES = {
    'synthetic': ['queue_length', 'utilization', 'arrival_rate', 'avg_delay'],
    'abilene':   ['in_flow', 'out_flow'],
}

def objective(trial: optuna.trial.Trial, datasource):
    K = trial.suggest_int('K', 1, 4)
    d = trial.suggest_categorical('d', [4, 8, 16])
    L = trial.suggest_int('L', 1, 3)
    dropout = trial.suggest_float('dropout', 0, 0.5)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)

    feature_names = FEATURES[datasource]
    if datasource == 'synthetic': 
        path = ROOT / 'data' / 'mock_data.npy'
        se = torch.load(ROOT / 'data' / 'node2vec_synthetic.pt', weights_only=True)
        num_nodes, in_dim = 5, 4
    elif datasource == 'abilene':
        path = ROOT / 'data' / 'abilene_data.npy'
        se = torch.load(ROOT / 'data' / 'node2vec_abilene.pt', weights_only=True)
        num_nodes, in_dim = 12, 2

    steps_per_day = 288 if datasource == 'abilene' else None
    dm = DSPDataModule(str(path), scaling='zscore', horizon=12, window=12, batch_size=64, max_samples=3000, steps_per_day=steps_per_day)
    dm.setup()
    horizon = 12
    window = 12
    model = gman.GMAN(in_channels=in_dim,
                      hidden_channels=K*d,
                      se=se,
                      K=K,
                      d=d,
                      L=L,
                      dropout=dropout,
                      T=288 if datasource == 'abilene' else None,
                      num_steps=horizon + window if datasource == 'synthetic' else None,
                      scaler=dm.scaler,
                      lr=lr,
                      weight_decay=weight_decay,
                      feature_names=feature_names,
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

    ckpt_dir = ROOT / 'results' / 'gman_search' / 'checkpoints'
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
                                study_name="gman_search",
                                storage=f"sqlite:///{ROOT / 'results' / 'tune_gman.db'}",
                                load_if_exists=True, )
    study.optimize(lambda trial: objective(trial, datasource), n_trials=50)

    best = study.best_params
    out = {'datasource': datasource,
           datasource: {**best,
                        'feature_names': FEATURES[datasource],
                        'window': 12,
                        'horizon': 12,
                        'steps_per_day': 288 if datasource == 'abilene' else None,
                        'num_nodes': 5 if datasource == 'synthetic' else 12,
                        'num_features': 4 if datasource == 'synthetic' else 2,
                        'batch_size': 64,
                        'max_epochs': 50,
                        }}

    out_path = ROOT / 'results' / 'best_params_gman.yaml'
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, 'w') as f:
        yaml.dump(out, f)
    print('Saved to', out_path)
    print('Best params:', best)
    print('Best val_loss:', study.best_value)
    print('Best trial:', study.best_trial.number)


if __name__ == "__main__":
    main()
