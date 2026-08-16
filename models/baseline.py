import numpy as np
import torch
from utils.datamodule import DSPDataModule
from utils.metrics import mae, rmse, mse
from pathlib import Path

ROOT = Path(__file__).parent.parent
path = ROOT / 'data' / 'mock_data.npy'

dm = DSPDataModule(str(path), scaling='zscore')
dm.setup()

def naive_baseline():
    """naive baseline: using the value at the last timestep as the prediction"""
    FEATURES = ['queue_length', 'utilization', 'arrival_rate', 'avg_delay']
    all_mae = {f: [] for f in FEATURES}

    for x, y in dm.test_dataloader():
        pred = x[:, -1, :, :]
        pred = dm.scaler.inverse(pred)
        y = dm.scaler.inverse(y)
        for j, feat in enumerate(FEATURES):
            all_mae[feat].append(mae(pred[:, :, j], y[:, :, j]).item())

    print('Naive baseline MAE:')
    for feat in FEATURES:
        print(f'  {feat}: {np.mean(all_mae[feat]):.4f}')


if __name__ == '__main__':
    naive_baseline()