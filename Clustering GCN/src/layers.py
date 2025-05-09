import torch
from torch_geometric.nn import GCNConv

class StackedGCN(torch.nn.Module):
    """Multi-layer Graph Convolutional Network (GCN) model."""

    def __init__(self, args, input_channels, output_channels):
        """
        Initializes the StackedGCN model.
        
        :param args: Arguments object containing configurations like hidden layers and dropout.
        :param input_channels: Number of input features per node.
        :param output_channels: Number of classes for output (e.g., node classification).
        """
        super(StackedGCN, self).__init__()
        self.args = args
        self.input_channels = input_channels
        self.output_channels = output_channels

        self.setup_layers()  # Initialize layers based on the args

    def setup_layers(self):
        """
        Constructs GCN layers dynamically based on args.layers configuration.
        Includes input, hidden, and output layers.
        """
        self.layers = []

        # Complete layer dimensions from input to output
        self.args.layers = [self.input_channels] + self.args.layers + [self.output_channels]

        # Create GCNConv layers for each pair of consecutive dimensions
        for i, _ in enumerate(self.args.layers[:-1]):
            self.layers.append(GCNConv(self.args.layers[i], self.args.layers[i + 1]))

        # Wrap the list of layers in a ListModule for easy indexing and iteration
        self.layers = ListModule(*self.layers)

    def forward(self, edges, features):
        """
        Forward pass through the GCN layers.
        
        :param edges: Edge list tensor (shape: [2, num_edges]).
        :param features: Node feature tensor (shape: [num_nodes, num_features]).
        :return: Log-softmax predictions (for classification).
        """

        # Pass through all but the last layer, applying ReLU and dropout
        for i, _ in enumerate(self.args.layers[:-2]):
            features = torch.nn.functional.relu(self.layers[i](features, edges))
            if i > 1:  # Apply dropout after first two layers
                features = torch.nn.functional.dropout(features, p=self.args.dropout, training=self.training)

        # Final layer: linear output without activation
        features = self.layers[i + 1](features, edges)

        # Apply log-softmax for probabilistic output over classes
        predictions = torch.nn.functional.log_softmax(features, dim=1)
        return predictions


class ListModule(torch.nn.ModuleList):
    """
    A wrapper class to treat a list of layers like a single PyTorch module.
    Provides access via indexing and iteration.
    """

    def __init__(self, *args):
        """
        Initializes with any number of layer modules.
        
        :param args: Variable number of modules (e.g., GCN layers).
        """
        super(ListModule, self).__init__(args)
        idx = 0
        for module in args:
            self.add_module(str(idx), module)  # Add each module with string key
            idx += 1

    def __getitem__(self, idx):
        """
        Retrieve a module by index.
        
        :param idx: Index of the module.
        :return: The corresponding layer/module.
        """
        if idx < 0 or idx >= len(self.__modules):
            raise IndexError('Index {} is out of range'.format(idx))

        it = iter(self.__modules.values())  # Manual iterator over modules
        for i in range(idx):
            next(it)
        return next(it)

    def __iter__(self):
        """
        Iterate over all modules (layers).
        """
        return iter(self.__modules.values())

    def __len__(self):
        """
        Get the number of layers/modules.
        """
        return len(self.__modules)
