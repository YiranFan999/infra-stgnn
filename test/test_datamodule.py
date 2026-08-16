from pathlib import Path
from utils.datamodule import DSPDataModule

ROOT_DIR = Path(__file__).parent.parent

dm = DSPDataModule(data_path=ROOT_DIR / "data" / "mock_data.npy", window=12, batch_size=256, num_workers=0, scaling="zscore")
dm.setup()

batch = next(iter(dm.train_dataloader()))
X, y = batch
print(X.shape, y.shape)
# X has shape (batch_size, window, num_nodes, num_features)


batch = next(iter(dm.val_dataloader()))
X, y = batch
print(X.shape, y.shape)
