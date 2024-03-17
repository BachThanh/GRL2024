import networkx as nx
from gensim.models import Word2Vec

# Step 1: Load the graph
G = nx.read_edgelist("path/to/your/graph.txt")

# Step 2: Preprocess the graph (e.g., add missing nodes, remove isolated nodes)

# Step 3: Generate random walks
walks = []
num_walks = 10
walk_length = 80

for _ in range(num_walks):
    for node in G.nodes():
        walk = [node]
        for _ in range(walk_length - 1):
            neighbors = list(G.neighbors(walk[-1]))
            if len(neighbors) > 0:
                walk.append(np.random.choice(neighbors))
            else:
                break
        walks.append(walk)

# Step 4: Train the Word2Vec model
model = Word2Vec(walks, size=128, window=10, min_count=0, sg=1, workers=8, iter=1)

# Step 5: Use the learned embeddings for downstream tasks
node_embeddings = model.wv

# Example: Get the embedding of a specific node
embedding = node_embeddings["your_node_id"]