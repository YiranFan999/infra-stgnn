import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns

ROOT = Path(__file__).parent.parent
data = np.load(ROOT / 'data' / 'abilene_data.npy')  # (T, 12, 2)

# temporal autocorrelation curve
max_H = 20
rho = []
for H in range(1, max_H + 1):
    r = np.corrcoef(data[:-H].flatten(), data[H:].flatten())[0, 1]
    rho.append(r)

plt.plot(range(1, max_H + 1), rho, marker='o')
plt.xlabel('Horizon H (x5 min)')
plt.ylabel('Autocorrelation')
plt.title('Abilene Temporal Autocorrelation')
plt.grid(True)
plt.savefig(ROOT / 'results' / 'abilene_autocorr.png')
plt.show()

for H, r in enumerate(rho, 1):
    print(f'H={H:2d} ({H*5:3d} min): {r:.4f}')

# spatial correlation matrix
# data shape: (T, 12, 2), average in/out flow as node feature
node_data = data.mean(axis=2)  # (T, 12)

corr_matrix = np.corrcoef(node_data.T)  # (12, 12)

NODES = [
    'ATLAM5', 'ATLAng', 'CHINng', 'DNVRng', 'HSTNng', 'IPLSng',
    'KSCYng', 'LOSAng', 'NYCMng', 'SNVAng', 'STTLng', 'WASHng'
]

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, xticklabels=NODES, yticklabels=NODES,
            annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Abilene Spatial Correlation Matrix')
plt.tight_layout()
plt.savefig(ROOT / 'results' / 'abilene_spatial_corr.png')
plt.show()

print(f'Mean spatial correlation: {corr_matrix[np.triu_indices(12, k=1)].mean():.4f}')