"""
Node2Vec
For synthetic data, the adjacency matrix is directed, node 3/4 starts random walk will end up context size=1.
If we flip the directed graph, feed to pretrain_node2vec, and concat it with the original embedding, the source/sink
feature of the node can be learned.
"""
from pathlib import Path

import torch
from torch_geometric.nn import Node2Vec

from utils.graph_utils import synthetic_graph, AbileneGraph
import torch_cluster   # noqa: F401


def pretrain_node2vec(edge_index, embed_dim, walk_length, context_size, walks_per_node, p, q, N, num_neg=1,
                      lr=0.01, epochs=100, device='cpu'):
    model = Node2Vec(edge_index=edge_index,
                     embedding_dim=embed_dim,
                     walk_length=walk_length,
                     context_size=context_size,
                     walks_per_node=walks_per_node,
                     num_negative_samples=num_neg,
                     p=p,
                     q=q,
                     num_nodes=N,
                     sparse=True).to(device)
    model.random_walk_fn = torch.ops.torch_cluster.random_walk # this enables biased deep walk
    loader = model.loader(batch_size=4, shuffle=True)
    optim = torch.optim.SparseAdam(list(model.parameters()), lr=lr)
    # for pos_rw, neg_rw in loader:
    #     print(pos_rw[:5])
    #     break
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for pos_sam, neg_sam in loader:
            optim.zero_grad()
            loss = model.loss(pos_sam.to(device), neg_sam.to(device))
            loss.backward()
            optim.step()
            total_loss += loss.item()
        if epoch % 10 == 0:
            print(f'epoch {epoch:3d}  loss {total_loss / len(loader):.4f}')

    return model.embedding.weight.detach().cpu()

datasource = 'abilene'
ROOT = Path(__file__).parent.parent
if datasource == 'synthetic':
    g = synthetic_graph()
    N = 5
elif datasource == 'abilene':
    g = AbileneGraph()
    N = 12
else:
    raise ValueError(f'datasource {datasource} is not supported')
edge_index = g.get_edge_index()


emb_forward = pretrain_node2vec(edge_index,
                        embed_dim=32,
                        walk_length=10,
                        context_size=5,
                        walks_per_node=20,
                        p=1,
                        q=2,
                        N=N)
emb_backward = pretrain_node2vec(edge_index.flip(0),
                        embed_dim=32,
                        walk_length=10,
                        context_size=5,
                        walks_per_node=20,
                        p=1,
                        q=2,
                        N=N)
emb = torch.cat([emb_forward, emb_backward], dim=1)
print(emb.shape)

torch.save(emb, ROOT / 'data' / f'node2vec_{datasource}.pt')

