from typing import Dict, List, Optional, Tuple 

import torch 
from torch import Tensor
from torch.nn import Embedding
from torch.utils.data import DataLoader

from torch_geometric.utils import EdgeType, NodeType, OptTensor
from torch_geometric.utils import sort_edge_index
from torch_geometric.utils.sparse import index2ptr

EPS = 1e-15

class MetaPath2Vec(torch.nn.Module):
    def __init__(self,
                 edge_index_dict: Dict[EdgeType, Tensor],
                 embedding_dim: int,
                 metapath: List[EdgeType],
                 walk_length: int,
                 context_size: int,
                 walks_per_node: int = 1,
                 num_negative_samples: int = 1,
                 num_nodes_dict: Optional[Dict[NodeType, int]] = None,
                 sparse: bool = False,):
        super().__init__()

        if num_nodes_dict is None:
            num_nodes_dict = {}
            for keys, edge_index in edge_index_dict.items():
                num_nodes_dict[keys] = max(N, edge_)