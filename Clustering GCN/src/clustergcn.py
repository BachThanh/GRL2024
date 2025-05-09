import torch 
import random 
import numpy as np
from tqdm import trange, tqdm
from layers import StackedGCN
from torch.autograd import Variable
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

class ClusterGCNTrainer(object):
    """
    Train a Cluster GCN
    """

    def __init__(self, args, clustering_machine):
        """
        :param args: Arguments objects
        :param clustering_machine: Clustering machine object
        """
        self.args = args
        self.clustering_machine = clustering_machine
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.create_model()

    def create_model(self):
        """
        Create a StackedGCN and tranfer to GPU
        """
        self.model = StackedGCN(self.args, self.clustering_machine.feature_count, self.clustering_machine.class_count)
        self.model = self.model.to(self.device)

    def do_forward_pass(self, cluster):
        """
        Make a forward pass with data from a given partition
        :param cluster: Cluster index
        :return average_loss: average loss on the cluster
        :return node_count: number of nodes
        """
        
        edges = self.clustering_machine.sg_edges[cluster].to(self.device)
        macro_nodes = self.clustering_machine.sg_nodes[cluster].to(self.device)
        train_nodes = self.clustering_machine.train_nodes[cluster].to(self.device)
        features = self.clustering_machine.features[macro_nodes].to(self.device)
        target = self.clustering_machine.target[macro_nodes].to(self.device).squeeze()
        preditions = self.model(edges, features)
        average_loss = torch.nn.functional.nll_loss(preditions[train_nodes], target[train_nodes])
        node_count = train_nodes.shape[0]
        return average_loss, node_count
    
    def update_average_loss(self, batch_average_loss, node_count):
        """
        Update the average loss in the epoch
        :param batch_average_loss: loss of the cluster
        :param node_count: Number of nodes in currently processed cluster
        :return average_loss: average loss in the epoch
        """

        self.accumulated_train_loss = self.accumulated_train_loss + batch_average_loss.item() * node_count
        self.node_count_seen = self.node_count_seen + node_count
        average_loss = self.accumulated_train_loss / self.node_count_seen
        return average_loss
    
    def train(self):
        """
        Train model
        """

        epochs = self.args.epochs, desc="Train loss"
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = self.args.learning_rate)
        self.model.train()

        for epoch in epochs:
            random.shuffle(self.clustering_machine.clusters)
            self.node_count_seen = 0
            self.accumulated_training_loss = 0
            for cluster in self.clustering_machine.clusters:
                self.optimizer.zero_grad()
                batch_average_loss, node_count = self.do_forward_pass(cluster)
                batch_average_loss.backward()
                self.optimizer.step()
                average_loss = self.update_average_loss(batch_average_loss, node_count)
            epochs.set_description(f"Train loss: {average_loss:.4f}")

    def test(self):

        """
        Score the test and print F1 score
        """

        self.model.eval()
        self.predictions = []
        self.targets = []

        for cluster in self.clustering_machine.clusters:
            prediction, target = self.do_prediction(cluster)
            self.predictions.append(prediction.cpu().detach().numpy())
            self.targets.append(target.cpu().detach().numpy())
        
        self.targets = np.concatenate(self.targets).argmax(1)
        score = f1_score(self.targets, self.predictions, average='micro')
        print(f"F1 score: {score:.4f}")