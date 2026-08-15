import numpy as np
import matplotlib.pyplot as plt
import os,sys


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

arr = np.load(os.path.join(ROOT, 'data', 'mock_data.npy'))
print('shape:', arr.shape)

NODES = ['parser', 'counter', 'matcher', 'node1', 'node2']
FEATURES = ['queue_length', 'utilization', 'arrival_rate', 'avg_delay']

for i, node in enumerate(NODES):
    print(f"\n{node}:")
    for j, feat in enumerate(FEATURES):
        print(f"  {feat}: mean={arr[:, i, j].mean():.3f}, max={arr[:, i, j].max():.3f}")

fig, axes = plt.subplots(4, 5, figsize=(16, 10))
for j, feat in enumerate(FEATURES):
    for i, node in enumerate(NODES):
        axes[j, i].plot(arr[:100, i, j]) # check the first 100 datapoints
        axes[j, i].set_title(f'{node}\n{feat}', fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(ROOT, 'scripts', 'data_check.png'))
plt.show()