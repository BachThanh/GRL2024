# Corrected version of GCNConv and full training pipeline
import torch 
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils import add_self_loops, degree, subgraph
import torch_geometric.nn

import random
import copy

# GCNConv
class GCNConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNConv, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels)
        
    def forward(self, x, edge_index):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        x = self.linear(x)
        out = torch.zeros_like(x)
        for i in range(edge_index.size(1)):
            out[row[i]] += norm[i] * x[col[i]]

        return out

# Graph Encoder using GCN
class GCNEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return torch_geometric.nn.global_mean_pool(x, batch)

# Graph Augmentation: Node Dropping and Edge Perturbation
def node_drop(data, drop_prob=0.2):
    node_mask = torch.rand(data.num_nodes, device=data.x.device) > drop_prob
    if node_mask.sum() == 0:
        return data 
    new_edge_index, _ = subgraph(node_mask, data.edge_index, relabel_nodes=True, num_nodes=data.num_nodes)
    data.x = data.x[node_mask]
    data.edge_index = new_edge_index
    if hasattr(data, 'batch') and data.batch is not None:
        data.batch = data.batch[node_mask]
    data.num_nodes = node_mask.sum().item()
    return data

def edge_perturb(data, perturb_ratio=0.2):
    edge_index = data.edge_index
    num_edges = edge_index.size(1)
    num_perturb = int(num_edges * perturb_ratio)
    perm = torch.randperm(num_edges)
    edge_index = edge_index[:, perm[:num_edges - num_perturb]]
    data.edge_index = edge_index
    return data

# Contrastive loss: InfoNCE
class InfoNCE(nn.Module):
    def __init__(self, temperature=0.5):
        super(InfoNCE, self).__init__()
        self.temperature = temperature

    def forward(self, z1, z2):
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)
        sim_matrix = torch.mm(z1, z2.t()) / self.temperature
        labels = torch.arange(z1.size(0)).to(z1.device)
        return F.cross_entropy(sim_matrix, labels)

# Training GraphCL

data = TUDataset(root='/tmp/PROTEINS', name='PROTEINS', transform=NormalizeFeatures())
dataloader = DataLoader(data, batch_size=32, shuffle=True)

model = GCNEncoder(in_channels=data.num_features, hidden_channels=64, out_channels=64)
criterion = InfoNCE()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

model.train()
for epoch in range(1, 101):
    total_loss = 0
    for batch in dataloader:
        batch = batch.to(device)

        data1 = node_drop(copy.deepcopy(batch))
        data2 = edge_perturb(copy.deepcopy(batch))

        z1 = model(data1.x, data1.edge_index, data1.batch)
        z2 = model(data2.x, data2.edge_index, data2.batch)

        loss = criterion(z1, z2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    print(f"Epoch {epoch}, Loss: {total_loss:.4f}")
