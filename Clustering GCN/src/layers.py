import torch
from torch_geometric.nn import GCNConv

class StackedGCN(torch.nn.Module):
    """Multi-layer GCN model."""

    def __init__(self, args, input_channels, output_channels):
        """
        :param args: Arguments objects
        :input_channels: number of features
        :output_channels: number of target features
        """

        super(StackedGCN, self).__init__()
        self.args = args
        self.input_channels = input_channels
        self.output_channels = output_channels

        self.setup_layers()

    def setup_layers(self):

        """
        Create the layers based on the args
        """

        self.layers = []
        self.args.layers = [self.input_channels] + self.args.layers + [self.output_channels]
        for i, _ in enumerate(self.args.layers[:-1]):
            self.layer.append(GCNConv(self.args.layers[i], self.args.layers[i + 1]))
        self.layers = ListModule(*self.layers)

    def forward(self, edges, features):

        """"
        Make a forward pass
        :param edges: Edge list LongTensor
        :param features: Feature matrix input FloatTensor
        :return predictions: Prediction matrix output FloatTensor
        """

        for i, _ in enumerate(self.args.layers[:-2]):
            features = torch.nn.functional.relu(self.layers[i](features, edges))
            if i>1:
                features = torch.nn.functional.dropout(features, p=self.args.dropout, training=self.training)
        features = self.layers[i+1](features, edges)
        predictions = torch.nn.functional.log_softmax(features, dim=1)
        return predictions
    
class ListModule(torch.nn.ModuleList):
    """"
    Abstract list layer class
    """

    def __init__(self, *args):
        """"
        Module initializing
        """

        super(ListModule, self).__init__(args)
        idx = 0
        for module in args:
            self.add_module(str(idx), module)
            idx += 1

    def __getitem__(self, idx):
        """"
        Get item from indexed layer
        """

        if idx < 0 or idx >= len(self.__modules):
            raise IndexError('Index {} is out of range'.format(idx))
        it = iter(self.__modules.values())
        for i in range(idx):
            next(it)
        return next(it)
    
    def __iter__(self):
        """
        Iterate on the layers
        """

        return iter(self.__modules.values())
    
    def __len__(self):
        """
        Number of the layers
        """

        return len(self.__modules)