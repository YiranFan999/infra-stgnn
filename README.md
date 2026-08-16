# infra-stgnn

Comparing spatial-temporal GNN architectures for predicting operator-level metrics (delay, utilisation, queue length, arrival rate) in a data stream processing pipeline.

## Models

- **STGCN** — Chebyshev graph conv + gated temporal conv
- **TCN + GAT** — dilated causal conv + graph attention
- **Graph WaveNet** — adaptive adjacency + dilated conv
- **GMAN** — encoder-decoder with spatial-temporal attention

## Data

- **Synthetic** — SimPy simulation of a DSP queueing network (source → parser → counter/matcher → node1/node2), with bursty Poisson arrivals. Features: queue length, utilisation, arrival rate, avg delay.
- **Abilene** — Real-world US academic network traffic (12 nodes, 5-min intervals). Features: per-node in/out flow (Mbit/s).

## Structure

```
data/
  mock_dataset.py       SimPy queueing network simulation
  abilene_loader.py     Abilene XML parser
  abilene/              raw XML files (not tracked by git)
layers/
  temporal.py           gated temporal convolution (GLU)
models/
  stgcn.py              STGCN implementation
utils/
  datamodule.py         sliding window dataset, scaler, train/val/test split
  graph_utils.py        DSP and Abilene graph topology
  metrics.py            MAE, MSE, RMSE, MAPE
experiments/
  train_stgcn.py              STGCN training on synthetic data
  tune_stgcn.py         Optuna hyperparameter search
scripts/
  check_data.py         visualise synthetic data
  check_abilene.py      verify Abilene graph structure
  baseline.py           naive last-step baseline
```

## Quickstart

```bash
pip install -r requirements.txt
python data/mock_dataset.py        # generate synthetic data
python experiments/train_stgcn.py        # train STGCN
```
