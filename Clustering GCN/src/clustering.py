import metis
import torch
import random
import numpy as np
import networkx as nx
from sklearn.model_selection import train_test_split

class ClusteringMachine(object):
    """
    Clustering the graph, feature set and target.
    """
    def __init__(self, args, graph, features, target):
        """
        :param args: Arguments object with parameters.
        :param graph: Networkx Graph.
        :param features: Feature matrix (ndarray).
        :param target: Target vector (ndarray).
        """
        self.args = args
        self.graph = graph
        self.features = features
        self.target = target
        self._set_sizes()

    def _set_sizes(self):
        """
        Setting the feature and class count.
        """
        self.feature_count = self.features.shape[1] 
        self.class_count = np.max(self.target)+1

    def decompose(self):
        """
        Decomposing the graph, partitioning the features and target, creating Torch arrays.
        """
        if self.args.clustering_method == "metis":
            print("\nMetis graph clustering started.\n")
            self.metis_clustering()
        else:
            print("\nRandom graph clustering started.\n")
            self.random_clustering()
        self.general_data_partitioning()
        self.transfer_edges_and_nodes()

    def random_clustering(self):
        """
        Random clustering the nodes.
        """
        self.clusters = [cluster for cluster in range(self.args.cluster_number)]
        self.cluster_membership = {node: random.choice(self.clusters) for node in self.graph.nodes()}

    def metis_clustering(self):
        """
        Clustering the graph with Metis. For details see:
        """
        (st, parts) = metis.part_graph(self.graph, self.args.cluster_number)
        self.clusters = list(set(parts))
        self.cluster_membership = {node: membership for node, membership in enumerate(parts)}

    def general_data_partitioning(self):
        """
        Creating data partitions and train-test splits.
        """
        self.sg_nodes = {}
        self.sg_edges = {}
        self.sg_train_nodes = {}
        self.sg_test_nodes = {}
        self.sg_features = {}
        self.sg_targets = {}
        for cluster in self.clusters:
            # Create a subgraph for the current cluster
            subgraph = self.graph.subgraph([node for node in sorted(self.graph.nodes()) if self.cluster_membership[node] == cluster])
            # Store sorted nodes of the subgraph
            self.sg_nodes[cluster] = [node for node in sorted(subgraph.nodes())]
            # Create a mapping from original node IDs to new sequential IDs (0 to n-1) for the subgraph
            mapper = {node: i for i, node in enumerate(sorted(self.sg_nodes[cluster]))}
            # Store edges of the subgraph, mapping original node IDs to new sequential IDs
            # Edges are duplicated to represent an undirected graph (e.g., [0,1] and [1,0])
            self.sg_edges[cluster] = [[mapper[edge[0]], mapper[edge[1]]] for edge in subgraph.edges()] +  [[mapper[edge[1]], mapper[edge[0]]] for edge in subgraph.edges()]
            # Split subgraph nodes into training and testing sets based on new sequential IDs
            self.sg_train_nodes[cluster], self.sg_test_nodes[cluster] = train_test_split(list(mapper.values()), test_size = self.args.test_ratio)
            # Sort the test and train node lists
            self.sg_test_nodes[cluster] = sorted(self.sg_test_nodes[cluster])
            self.sg_train_nodes[cluster] = sorted(self.sg_train_nodes[cluster])
            # Extract features for the nodes in the current subgraph
            self.sg_features[cluster] = self.features[self.sg_nodes[cluster],:]
            # Extract targets for the nodes in the current subgraph
            self.sg_targets[cluster] = self.target[self.sg_nodes[cluster],:]

    def transfer_edges_and_nodes(self):
        """
        Transfering the data to PyTorch format.
        """
        for cluster in self.clusters:
            # Convert subgraph nodes to PyTorch LongTensor
            self.sg_nodes[cluster] = torch.LongTensor(self.sg_nodes[cluster])
            # Convert subgraph edges to PyTorch LongTensor and transpose it
            self.sg_edges[cluster] = torch.LongTensor(self.sg_edges[cluster]).t()
            # Convert training nodes to PyTorch LongTensor
            self.sg_train_nodes[cluster] = torch.LongTensor(self.sg_train_nodes[cluster])
            # Convert testing nodes to PyTorch LongTensor
            self.sg_test_nodes[cluster] = torch.LongTensor(self.sg_test_nodes[cluster])
            # Convert subgraph features to PyTorch FloatTensor
            self.sg_features[cluster] = torch.FloatTensor(self.sg_features[cluster])
            # Convert subgraph targets to PyTorch LongTensor
            self.sg_targets[cluster] = torch.LongTensor(self.sg_targets[cluster])