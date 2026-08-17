import numpy as np
import torch
from utils.datamodule import DSPDataModule
from utils.metrics import mae, rmse, mape
from pathlib import Path
import yaml

ROOT = Path(__file__).parent.parent
path_syn = ROOT / 'data' / 'mock_data.npy'
path_abi = ROOT / 'data' / 'abilene_data.npy'
OUT_PATH = ROOT / 'results' / 'baseline_results.yaml'


def naive_baseline(horizon=1):
    """naive baseline: using the value at the last timestep as the prediction"""
    dm = DSPDataModule(str(path_syn), scaling='zscore', horizon=horizon)
    dm.setup()
    FEATURES = ['queue_length', 'utilization', 'arrival_rate', 'avg_delay']

    all_mae = {f: [] for f in FEATURES}
    all_mape = {f: [] for f in FEATURES}
    all_rmse = {f: [] for f in FEATURES}

    all_mae_h = [[] for _ in range(horizon)]

    for x, y in dm.test_dataloader():
        pred = x[:, -1:, :, :].expand(-1, horizon, -1, -1)
        # pred = dm.scaler.inverse(pred)
        # y = dm.scaler.inverse(y)
        for j, feat in enumerate(FEATURES):
            all_mae[feat].append(mae(pred[:, :, :, j], y[:, :, :, j]).item())
            all_rmse[feat].append(rmse(pred[:, :, :, j], y[:, :, :, j]).item())
            all_mape[feat].append(mape(pred[:, :, :, j], y[:, :, :, j]).item())
        for h in range(horizon):
            all_mae_h[h].append(mae(pred[:, h], y[:, h]).item())

    print('Per-horizon MAE:')
    for h in range(horizon):
        print(f'  h{h + 1}: {np.mean(all_mae_h[h]):.4f}')
    print('Per-feature MAE/RMSE/MAPE:')
    for feat in FEATURES:
        print(
            f'  {feat}: MAE={np.mean(all_mae[feat]):.4f}  RMSE={np.mean(all_rmse[feat]):.4f}  MAPE={np.mean(all_mape[feat]):.4f}')

    print('Naive baseline MAE:')
    for feat in FEATURES:
        print(f'  {feat}: {np.mean(all_mae[feat]):.4f}')
    data = np.load(path_syn)
    print('synthetic autocorrelation lag-1:', np.corrcoef(data[:-1].flatten(), data[1:].flatten())[0, 1])
    print('synthetic autocorrelation lag-6:', np.corrcoef(data[:-6].flatten(), data[6:].flatten())[0, 1])

def naive_abilene(horizon=1):
    """naive baseline for abilene dataset: using the value at the last timestep as the prediction"""
    dm = DSPDataModule(str(path_abi), scaling='zscore', horizon=horizon)
    dm.setup()
    FEATURES = ['in_flow', 'out_flow']
    all_mae  = {f: [] for f in FEATURES}
    all_mape = {f: [] for f in FEATURES}
    all_rmse = {f: [] for f in FEATURES}

    all_mae_h = [[] for _ in range(horizon)]

    for x, y in dm.test_dataloader():
        pred = x[:, -1:, :, :].expand(-1, horizon, -1, -1)
        # pred = dm.scaler.inverse(pred)
        # y = dm.scaler.inverse(y)
        for j, feat in enumerate(FEATURES):
            all_mae[feat].append(mae(pred[:, :, :, j], y[:, :, :, j]).item())
            all_rmse[feat].append(rmse(pred[:, :, :, j], y[:, :, :, j]).item())
            all_mape[feat].append(mape(pred[:, :, :, j], y[:, :, :, j]).item())
        for h in range(horizon):
            all_mae_h[h].append(mae(pred[:, h], y[:, h]).item())

    print('Per-horizon MAE:')
    for h in range(horizon):
        print(f'  h{h + 1}: {np.mean(all_mae_h[h]):.4f}')
    print('Per-feature MAE/RMSE/MAPE:')
    for feat in FEATURES:
        print(
            f'  {feat}: MAE={np.mean(all_mae[feat]):.4f}  RMSE={np.mean(all_rmse[feat]):.4f}  MAPE={np.mean(all_mape[feat]):.4f}')

    results = {
        'naive_abilene': {
            feat: {
                'mae': float(np.mean(all_mae[feat])),
                'rmse': float(np.mean(all_rmse[feat])),
                'mape': float(np.mean(all_mape[feat])),
            }
            for feat in FEATURES
        }
    }


    OUT_PATH.parent.mkdir(exist_ok=True)
    with open(OUT_PATH, 'w') as f:
        yaml.dump(results, f)
    print(f'Saved to {OUT_PATH}')
    data = np.load(path_abi)
    print('autocorrelation lag-1:', np.corrcoef(data[:-1].flatten(), data[1:].flatten())[0, 1])
    print('autocorrelation lag-3:', np.corrcoef(data[:-3].flatten(), data[3:].flatten())[0, 1])


if __name__ == '__main__':
     # naive_baseline(horizon=6)
      naive_abilene(horizon=6)