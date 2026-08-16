from pathlib import Path

import optuna
import torch

from models.stgcn import STGCN
from utils.datamodule import DSPDataModule
import lightning as pl
from optuna.integration import PyTorchLightningPruningCallback

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

ROOT = Path(__file__).parent.parent
path = ROOT / 'data' / 'mock_data.npy'

def objective(trial: optuna.trial.Trial):
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
    out_channels = trial.suggest_categorical("out_channels", [16, 32, 64, 128])

    dm = DSPDataModule(str(path))
    dm.setup()
    model = STGCN(in_channels=4, out_channels=out_channels, lr=lr, weight_decay=weight_decay, scaler=dm.scaler)

    trainer = pl.Trainer(max_epochs=30,
                         accelerator="gpu" if torch.cuda.is_available() else "cpu",
                         callbacks=[pl.callbacks.EarlyStopping(monitor="val_loss", patience=5),
                                    PyTorchLightningPruningCallback(trial, monitor='val_loss')],
                         enable_checkpointing=False,
                         logger=False,
                         )
    trainer.fit(model, dm)

    ckpt_dir = Path('results/stgcn_search/checkpoints')
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"{trial.number}.ckpt"
    trainer.save_checkpoint(ckpt_path)
    return trainer.callback_metrics["val_loss"].item()

def main():
    pruner = optuna.pruners.MedianPruner()
    study = optuna.create_study(direction="minimize",
                                pruner=pruner,
                                study_name="stgcn_search",
                                storage="sqlite:///results/optuna_study.db",
                                load_if_exists=True,)
    study.optimize(objective, n_trials=20)

    print('Best params:', study.best_params)
    print('Best val_loss:', study.best_value)
    print('Best trial:', study.best_trial.number)

if __name__ == "__main__":
    main()