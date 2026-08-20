
import torch

class synthetic_graph:
    ##          -> counter -> node1
    ## -> parser
    ##          -> matcher -> node2
    def __init__(self):
        self.edge_index = self.get_edge_index()
        self.edge_weight = self.get_edge_weight()

    def get_edge_index(self) -> torch.Tensor:
        return torch.tensor([[0, 1, 0, 2],
                                  [1, 3, 2, 4]], dtype=torch.long)


    def get_edge_weight(self, weighted=None) -> torch.Tensor:
        if weighted is None:
            return torch.ones(4, dtype=torch.float32)
        return torch.tensor(weighted, dtype=torch.float32)

    def get_adj(self, weighted=None) -> torch.Tensor:
        N = 5
        A = torch.zeros(N, N, dtype=torch.float32)
        edges = [(0, 1), (1, 3), (0, 2), (2, 4)]
        for i, j in edges:
            A[i, j] = 1.0 if weighted is None else weighted[(i, j)]
        return A

    def get_forward_matrix(self):
        A = self.get_adj()
        D = A.sum(dim=1, keepdim=True).clamp(min=1)
        return A/D
    def get_backward_matrix(self):
        AT = self.get_adj().t()
        DT = AT.sum(dim=1, keepdim=True).clamp(min=1)
        return AT/DT




class AbileneGraph:
    """
    Abilene network physical topology.
    12 nodes, 15 undirected links (30 directed edges).
    Node order matches abilene_loader.py NODES list.
    """
    NODES = [
        'ATLAM5', 'ATLAng', 'CHINng', 'DNVRng', 'HSTNng', 'IPLSng',
        'KSCYng', 'LOSAng', 'NYCMng', 'SNVAng', 'STTLng', 'WASHng'
    ]

    # Physical links (undirected)
    LINKS = [
        ('ATLAM5', 'ATLAng'),
        ('ATLAng', 'HSTNng'),
        ('ATLAng', 'IPLSng'),
        ('ATLAng', 'WASHng'),
        ('CHINng', 'IPLSng'),
        ('CHINng', 'NYCMng'),
        ('DNVRng', 'KSCYng'),
        ('DNVRng', 'SNVAng'),
        ('DNVRng', 'STTLng'),
        ('HSTNng', 'LOSAng'),
        ('HSTNng', 'KSCYng'),
        ('IPLSng', 'KSCYng'),
        ('LOSAng', 'SNVAng'),
        ('NYCMng', 'WASHng'),
        ('SNVAng', 'STTLng'),
    ]

    NODE_INDEX = {n: i for i, n in enumerate(NODES)}

    def get_edge_index(self) -> torch.Tensor:
        edges = []
        for src, tgt in self.LINKS:
            i, j = self.NODE_INDEX[src], self.NODE_INDEX[tgt]
            edges.append([i, j])
            edges.append([j, i])
        return torch.tensor(edges, dtype=torch.long).t().contiguous()  # (2, 30)

    def get_edge_weight(self) -> torch.Tensor:
        return torch.ones(len(self.LINKS) * 2, dtype=torch.float32)

    def get_adj(self, weighted=None) -> torch.Tensor:
        N = 12
        A = torch.zeros(N, N, dtype=torch.float32)
        edges = [(self.NODE_INDEX[src], self.NODE_INDEX[tgt]) for src, tgt in self.LINKS]
        for i, j in edges:
            A[i, j] = 1.0 if weighted is None else weighted[(i, j)]
            A[j, i] = 1.0 if weighted is None else weighted[(j, i)]
        return A

    def get_forward_matrix(self):
        A = self.get_adj()
        D = A.sum(dim=1, keepdim=True).clamp(min=1)
        return A / D

    def get_backward_matrix(self):
        AT = self.get_adj().t()
        DT = AT.sum(dim=1, keepdim=True).clamp(min=1)
        return AT / DT
