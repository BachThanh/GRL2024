import torch 
import random 
import numpy as np
from tqdm import trange, tqdm  
from layers import StackedGCN  
from torch.autograd import Variable
from sklearn.metrics import f1_score, accuracy_score  

class ClusterGCNTrainer(object):
    """
    Trainer for Cluster-GCN model. Performs mini-batch training using graph clusters.
    """

    def __init__(self, args, clustering_machine):
        """
        Initializes the trainer with model parameters and data handler.

        :param args: Argument object (contains epochs, learning rate, etc.)
        :param clustering_machine: Object that holds clusters, edges, features, labels, etc.
        """
        self.args = args
        self.clustering_machine = clustering_machine
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Use GPU if available
        self.create_model()

    def create_model(self):
        """
        Creates a StackedGCN model and moves it to the designated device (GPU/CPU).
        """
        self.model = StackedGCN(
            self.args,
            self.clustering_machine.feature_count,  # Input feature size
            self.clustering_machine.class_count     # Number of output classes
        )
        self.model = self.model.to(self.device)

    def do_forward_pass(self, cluster):
        """
        Perform a forward pass using the subgraph (cluster).

        :param cluster: Index of the cluster to process
        :return average_loss: Computed loss on this cluster
        :return node_count: Number of training nodes in this cluster
        """
        # Extract and move data to the device
        edges = self.clustering_machine.sg_edges[cluster].to(self.device)
        macro_nodes = self.clustering_machine.sg_nodes[cluster].to(self.device)
        train_nodes = self.clustering_machine.train_nodes[cluster].to(self.device)
        features = self.clustering_machine.features[macro_nodes].to(self.device)
        target = self.clustering_machine.target[macro_nodes].to(self.device).squeeze()

        # Forward pass
        predictions = self.model(edges, features)

        # Compute NLL loss on training nodes
        average_loss = torch.nn.functional.nll_loss(predictions[train_nodes], target[train_nodes])
        node_count = train_nodes.shape[0]
        return average_loss, node_count

    def update_average_loss(self, batch_average_loss, node_count):
        """
        Accumulates and computes average loss over the epoch.

        :param batch_average_loss: Loss for the current cluster
        :param node_count: Number of nodes contributing to that loss
        :return average_loss: Updated epoch-wide average loss
        """
        self.accumulated_train_loss += batch_average_loss.item() * node_count
        self.node_count_seen += node_count
        average_loss = self.accumulated_train_loss / self.node_count_seen
        return average_loss

    def train(self):
        """
        Trains the model for a number of epochs using cluster-wise mini-batching.
        """
        epochs = tqdm(range(self.args.epochs), desc="Train loss") 

        # Optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        self.model.train()

        # Training loop
        for epoch in epochs:
            random.shuffle(self.clustering_machine.clusters)  # Shuffle cluster order
            self.node_count_seen = 0
            self.accumulated_train_loss = 0

            # Train over each cluster
            for cluster in self.clustering_machine.clusters:
                self.optimizer.zero_grad()  # Reset gradients
                batch_average_loss, node_count = self.do_forward_pass(cluster)  # Forward
                batch_average_loss.backward()  # Backprop
                self.optimizer.step()  # Optimizer step
                average_loss = self.update_average_loss(batch_average_loss, node_count)  # Track loss

            # Update progress bar with current loss
            epochs.set_description(f"Train loss: {average_loss:.4f}")

    def test(self):
        """
        Runs inference on all clusters and prints micro-F1 score.
        """
        self.model.eval()  # Set model to evaluation mode
        self.predictions = []
        self.targets = []

        # Perform prediction cluster by cluster
        for cluster in self.clustering_machine.clusters:
            prediction, target = self.do_prediction(cluster)  
            self.predictions.append(prediction.cpu().detach().numpy())
            self.targets.append(target.cpu().detach().numpy())

        # Compute evaluation metrics
        self.targets = np.concatenate(self.targets).argmax(1)
        score = f1_score(self.targets, self.predictions, average='micro')
        print(f"F1 score: {score:.4f}")
