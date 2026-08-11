# infra-stgnn

Comparing spatial-temporal GNN architectures for predicting operator-level metrics (delay, utilisation, queue length) in a data stream processing pipeline.

## Models

- **STGCN** — Chebyshev graph conv + gated temporal conv
- **TCN + GAT** — dilated causal conv + graph attention
- **Graph WaveNet** — adaptive adjacency + dilated conv
- **GMAN** — encoder-decoder with spatial-temporal attention

## Data

- **Synthetic** — SimPy simulation of a DSP queueing network (source → parser → counter/matcher → node1/node2), with bursty Poisson arrivals
- **Real** — GEANT network dataset (23 nodes, 15-min intervals)

## Structure

```
data/          simulation and dataset loading
models/        STGCN, TCN+GAT, WaveNet, GMAN
utils/         graph construction, datamodule, metrics
experiments/   training scripts
notebooks/     results and visualisation
```

## Quickstart

```bash
pip install -r requirements.txt
python data/mock_dataset.py        # generate synthetic data
python experiments/train.py --model stgcn
```