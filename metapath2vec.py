import torch
import torch_geometric
from torch_geometric.nn.models import metapath2vec

# Create a graph dataset
data = torch_geometric.datasets.Planetoid(root='/tmp/Cora', name='Cora')

# Define the metapath2vec model
model = metapath2vec.MetaPath2Vec(data.num_features, embedding_dim=128)

# Train the model
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
model.train()
for epoch in range(200):
    optimizer.zero_grad()
    loss = model.loss(data.x, data.edge_index)
    loss.backward()
    optimizer.step()

# Get the learned node embeddings
embeddings = model.get_embeddings()

# Use the embeddings for downstream tasks
# ...
