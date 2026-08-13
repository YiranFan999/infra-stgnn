##          -> counter -> node1
## -> parser
##          -> matcher -> node2
import torch

def get_edge_index() -> torch.Tensor:
    return torch.tensor([[0, 1, 0, 2],
                              [1, 3, 2, 4]], dtype=torch.long)


def get_edge_weight(weighted=None) -> torch.Tensor:
    if weighted is None:
        return torch.ones(4, dtype=torch.float32)
    return torch.tensor(weighted, dtype=torch.float32)

