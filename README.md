# infra-stgnn

Comparing spatial-temporal GNN architectures for multi-step forecasting of operator-level metrics (delay, utilisation, queue length, arrival rate) in a data stream processing pipeline, and of link traffic in a real backbone network.

## Models

| Model | Temporal | Spatial | Status |
|---|---|---|---|
| **STGCN** | gated temporal conv (GLU) | ChebConv, fixed graph | done |
| **Graph WaveNet** | dilated causal conv | diffusion conv, fixed + adaptive adjacency | done |
| **GRU-GAT** | GRU + causal temporal attention | GATv2, fixed graph | in progress |
| **GMAN** | temporal attention | spatial attention, encoder–decoder | planned |

All models take `(B, T, N, F)` and predict `(B, H, N, F)`, where `H` is the forecast horizon.

### References

- **STGCN** — Yu, Yin & Zhu (2018), *Spatio-Temporal Graph Convolutional Networks: A Deep
  Learning Framework for Traffic Forecasting*, IJCAI. [arXiv:1709.04875](https://arxiv.org/abs/1709.04875)
- **Graph WaveNet** — Wu, Pan, Long, Jiang & Zhang (2019), *Graph WaveNet for Deep
  Spatial-Temporal Graph Modeling*, IJCAI. [arXiv:1906.00121](https://arxiv.org/abs/1906.00121)
- **GRU-GAT** — the spatial-temporal encoder of Wang & Leung (2025), *Spatial-Temporal
  Reinforcement Learning for Network Routing with Non-Markovian Traffic*,
  [arXiv:2507.22174](https://arxiv.org/abs/2507.22174), with the RL policy head replaced by a
  multi-step forecasting head. Uses GATv2 (Brody et al., 2022) in place of the original GAT.
- **GMAN** — Zheng, Fan, Wang & Qi (2020), *GMAN: A Graph Multi-Attention Network for Traffic
  Prediction*, AAAI. [arXiv:1911.08415](https://arxiv.org/abs/1911.08415)

## Data

- **Synthetic** — SimPy simulation of a DSP queueing network (source → parser → counter/matcher → node1/node2), bursty Poisson arrivals. 3000 steps, 5 nodes, 4 features: queue length, utilisation, arrival rate, avg delay.
- **Abilene** — US academic backbone traffic (12 nodes, 15 undirected links, 5-min intervals). 3000 steps, 2 features: per-node in/out flow.

Both are stored as `(T, N, F)` arrays and are **not tracked by git** — regenerate with the
scripts in `data/`.

The two datasets differ sharply in persistence: Abilene has lag-1 autocorrelation 0.979
(still 0.941 at lag-20) and mean pairwise spatial correlation 0.669, so the naive baseline is
hard to beat there. The synthetic traces are burstier by construction (lag-1 0.882).
`scripts/analyse_abilene.py` produces the autocorrelation curve and spatial correlation
heatmap.

## Structure

```
config/
  wavenet.yaml            per-dataset hyperparameters (synthetic / abilene)
data/
  mock_dataset.py         SimPy queueing network simulation
  abilene_loader.py       Abilene XML parser
  abilene/                raw XML files (not tracked)
layers/
  temporal.py             gated temporal conv (GLU), dilated causal conv
  spatial.py              diffusion graph conv (nconv, gcn)
models/
  baseline.py             naive last-step baseline
  stgcn.py                STGCN
  wavenet.py              Graph WaveNet
utils/
  datamodule.py           sliding-window dataset, scaler, train/val/test split
  graph_utils.py          synthetic_graph, AbileneGraph (edge_index, forward/backward matrices)
  metrics.py              MAE, MSE, RMSE, MAPE
experiments/
  train_stgcn.py          STGCN on synthetic
  train_abliene.py        STGCN on Abilene
  train_wavenet.py        Graph WaveNet, dataset chosen in config/wavenet.yaml
  tune_stgcn.py           Optuna search (synthetic)
  tune_stgcn_abilene.py   Optuna search (Abilene)
  tune_wavenet.py         Optuna search, resumable via SQLite storage
scripts/
  check_data.py           visualise synthetic data
  check_abilene.py        verify Abilene graph structure
  analyse_abilene.py      autocorrelation curve, spatial correlation heatmap
results/                  metrics and best hyperparameters (yaml)
test/
  test_datamodule.py
```

## Quickstart

```bash
pip install -r requirements.txt
```

Generate data:

```bash
python data/mock_dataset.py
```

Train:

```bash
python experiments/train_wavenet.py
```

The dataset, horizon and all model hyperparameters come from `config/wavenet.yaml`. Switch
datasets by editing the top-level `datasource` key (`synthetic` or `abilene`); the script
picks the matching graph, data path and output file automatically. If
`results/best_params_wavenet.yaml` exists it is loaded in preference to the config.

Baselines and hyperparameter search:

```bash
python models/baseline.py
python experiments/tune_wavenet.py
```

`tune_wavenet.py` writes to `results/tune_wavenet.db`, so an interrupted study resumes where
it left off (`load_if_exists=True`).

## Notes

- Device is selected automatically (CUDA → MPS → CPU); no flags needed.
- Metrics are logged per horizon (`test_mae_h1` … `test_mae_hH`) as well as overall, since
  error growth with horizon is the main thing that distinguishes the models.
- `mape` masks targets below a threshold (default 1.0) to avoid division by near-zero on
  scaled data.
