import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.graph_utils import AbileneGraph

g = AbileneGraph()
print('edge_index shape:', g.get_edge_index().shape)
print('edge_weight shape:', g.get_edge_weight().shape)
print('num edges:', g.get_edge_index().shape[1])
print(g.get_edge_index().t())