""" DataModule: scaling, sliding-window, train/val/test split for (T, N, F) simulation data. """
import numpy as np
import lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset


class FeatureScaler:
    """Scaler with multiple scaling methods."""
    def __init__(self, method: str, cube: np.ndarray): # cube shape: (T, N, F), F=4
        self.method = method.lower()
        self.SCALE = np.array([1., 1., 1., 1.], dtype=np.float32)

        if self.method == "none" or self.method == "simple":
            self.shift = np.zeros(4, np.float32)
            self.scale = self.SCALE
        elif self.method == "minmax":
            # minmax: (X - min) / (max - min)
            self.shift = cube.min(axis=(0, 1)).astype(np.float32)
            scale_val = cube.max(axis=(0, 1)).astype(np.float32) - self.shift
            self.scale = np.where(scale_val > 1e-8, scale_val, 1.0).astype(np.float32)
        elif self.method == "zscore" or self.method == "standardize":
            # Standardization: (X - mean) / std
            self.shift = cube.mean(axis=(0, 1)).astype(np.float32)
            std_val = cube.std(axis=(0, 1))
            self.scale = np.where(std_val > 1e-8, std_val, 1.0).astype(np.float32)
        elif self.method == "robust":
            # Robust: (X - median) / (Q75 - Q25)
            self.shift = np.median(cube, axis=(0, 1)).astype(np.float32)
            q75, q25 = np.percentile(cube, (75, 25), axis=(0, 1))
            scale_val = q75 - q25
            self.scale = np.where(scale_val > 1e-8, scale_val, 1.0).astype(np.float32)
        else:
            raise ValueError("Unknown method: {}".format(self.method))

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward transformation."""
        return (x - self.shift) / self.scale

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        """Inverse transformation."""
        return x * torch.from_numpy(self.scale).to(x.device) + torch.from_numpy(self.shift).to(x.device)


class DSPDataset(Dataset):
    """ Windowed dataset for DSP application. """
    def __init__(self, data: np.ndarray, window: int, horizon=1): # data: [T, N, F]
        X, y = [], []
        for t in range(window, len(data) - horizon + 1):
            X.append(data[t - window : t])
            y.append(data[t : t + horizon]) # multi-step prediction
        self.X = torch.from_numpy(np.stack(X)).float()  # shape: (num_samples, window, N, F)
        self.y = torch.from_numpy(np.stack(y)).float()

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


class DSPDataModule(pl.LightningDataModule):
    """ Lightning wrapper for DSP application (handling scaling and splitting). """
    def __init__(self,
                 data_path: str,
                 window: int = 12,
                 batch_size: int = 256,
                 num_workers: int = 0,
                 horizon: int = 1,
                 scaling: str = "zscore",
                 pin_memory: bool = True,
                 max_samples=None
                 ):
        super().__init__()
        self.scaler = None
        self.data_path = data_path
        self.save_hyperparameters(ignore=["data_path"])

        self.train_ds, self.val_ds, self.test_ds = None, None, None # defined in `setup`
        self.max_samples = max_samples

    def setup(self, stage=None):
        # Load data
        data = np.load(self.data_path)  # shape: (T, N, F)
        if self.max_samples is not None:
            data = data[:self.max_samples]
        self.scaler = FeatureScaler(self.hparams.scaling, data)
        scaled_data = self.scaler.forward(data)

        # Split into train/val/test
        T = scaled_data.shape[0]
        train_end = int(T * 0.7)
        val_end = int(T * 0.85)

        train_data = scaled_data[:train_end]
        val_data = scaled_data[train_end:val_end]
        test_data = scaled_data[val_end:]

        self.train_ds = DSPDataset(train_data, self.hparams.window, self.hparams.horizon)
        self.val_ds = DSPDataset(val_data, self.hparams.window, self.hparams.horizon)
        self.test_ds = DSPDataset(test_data, self.hparams.window, self.hparams.horizon)

    def train_dataloader(self):
        return DataLoader(self.train_ds,
                          batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.num_workers,
                          pin_memory=self.hparams.pin_memory,
                          shuffle=True,
                          drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds,
                          batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.num_workers,
                          pin_memory=self.hparams.pin_memory,
                          shuffle=False,
                          drop_last=True)

    def test_dataloader(self):
        return DataLoader(self.test_ds,
                          batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.num_workers,
                          pin_memory=self.hparams.pin_memory,
                          shuffle=False,
                          drop_last=True)
