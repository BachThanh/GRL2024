import os
import pickle

import torch
from torch_geometric.utils import get_laplacian, to_scipy_sparse_matrix, to_dense_adj
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import expm


class PositionEncoding(object):
    """
    Base class for computing and applying position encodings to graph datasets.
    It handles saving and loading precomputed encodings.
    """
    def __init__(self, savepath=None, zero_diag=False):
        """
        Initializes the PositionEncoding class.

        Args:
            savepath (str, optional): Path to save/load precomputed encodings. Defaults to None.
            zero_diag (bool, optional): Whether to zero out the diagonal of the position encoding matrix. Defaults to False.
        """
        self.savepath = savepath
        self.zero_diag = zero_diag

    def apply_to(self, dataset, split='train'):
        """
        Applies position encoding to each graph in the dataset.
        It first tries to load precomputed encodings if a savepath is provided.
        If not found, it computes them and saves them.

        Args:
            dataset: A dataset object where each item is a graph.
            split (str, optional): The dataset split (e.g., 'train', 'val', 'test'). Defaults to 'train'.

        Returns:
            The dataset with an added 'pe_list' attribute containing position encodings for each graph.
        """
        saved_pos_enc = self.load(split)
        all_pe = []
        dataset.pe_list = []
        for i, g in enumerate(dataset):
            if saved_pos_enc is None:
                # Compute position encoding if not loaded
                pe = self.compute_pe(g)
                all_pe.append(pe)
            else:
                # Use loaded position encoding
                pe = saved_pos_enc[i]
            if self.zero_diag:
                # Zero out the diagonal if specified
                pe = pe.clone()
                pe.diagonal()[:] = 0
            dataset.pe_list.append(pe)

        # Save computed encodings if a savepath is provided and encodings were computed
        if saved_pos_enc is None:
            self.save(all_pe, split)

        return dataset

    def save(self, pos_enc, split):
        """
        Saves the computed position encodings to a file using pickle.

        Args:
            pos_enc (list): A list of position encoding tensors.
            split (str): The dataset split (e.g., 'train', 'val', 'test').
        """
        if self.savepath is None:
            return
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.savepath), exist_ok=True)
        if not os.path.isfile(self.savepath + "." + split):
            with open(self.savepath + "." + split, 'wb') as handle:
                pickle.dump(pos_enc, handle)

    def load(self, split):
        """
        Loads precomputed position encodings from a file.

        Args:
            split (str): The dataset split (e.g., 'train', 'val', 'test').

        Returns:
            list: A list of position encoding tensors, or None if the file doesn't exist or savepath is None.
        """
        if self.savepath is None:
            return None
        if not os.path.isfile(self.savepath + "." + split):
            return None
        with open(self.savepath + "." + split, 'rb') as handle:
            pos_enc = pickle.load(handle)
        return pos_enc

    def compute_pe(self, graph):
        """
        Placeholder method for computing position encoding for a single graph.
        This should be implemented by subclasses.

        Args:
            graph: A graph object.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        pass


class DiffusionEncoding(PositionEncoding):
    """
    Computes position encoding based on the heat kernel (diffusion process) of the graph Laplacian.
    PE = exp(-beta * L)
    """
    def __init__(self, savepath, beta=1., use_edge_attr=False, normalization=None, zero_diag=False):
        """
        Initializes DiffusionEncoding.

        Args:
            savepath (str): Path to save/load precomputed encodings.
            beta (float, optional): The diffusion time parameter. Defaults to 1.0.
            use_edge_attr (bool, optional): Whether to use edge attributes for Laplacian computation. Defaults to False.
            normalization (str, optional): Normalization type for the Laplacian ('sym', 'rw', or None). Defaults to None.
            zero_diag (bool, optional): Whether to zero out the diagonal of the PE matrix. Defaults to False.
        """
        super().__init__(savepath, zero_diag)
        self.beta = beta
        self.normalization = normalization
        self.use_edge_attr = use_edge_attr

    def compute_pe(self, graph):
        """
        Computes the diffusion-based position encoding for a graph.

        Args:
            graph: A PyTorch Geometric graph data object.

        Returns:
            torch.Tensor: The position encoding matrix.
        """
        edge_attr = graph.edge_attr if self.use_edge_attr else None
        # Get the graph Laplacian
        edge_index, edge_weight = get_laplacian(
                graph.edge_index, edge_attr, normalization=self.normalization,
                num_nodes=graph.num_nodes)
        # Convert to SciPy sparse matrix
        L = to_scipy_sparse_matrix(edge_index, edge_weight, num_nodes=graph.num_nodes).tocsc()
        # Compute the matrix exponential (heat kernel)
        L_expm = expm(-self.beta * L)
        return torch.from_numpy(L_expm.toarray())


class PStepRWEncoding(PositionEncoding):
    """
    Computes position encoding based on a p-step random walk transition matrix.
    PE = (I - beta * L)^p
    """
    def __init__(self, savepath, p=1, beta=0.5, use_edge_attr=False, normalization=None, zero_diag=False):
        """
        Initializes PStepRWEncoding.

        Args:
            savepath (str): Path to save/load precomputed encodings.
            p (int, optional): The number of random walk steps. Defaults to 1.
            beta (float, optional): Scaling factor for the Laplacian. Defaults to 0.5.
            use_edge_attr (bool, optional): Whether to use edge attributes for Laplacian computation. Defaults to False.
            normalization (str, optional): Normalization type for the Laplacian ('sym', 'rw', or None). Defaults to None.
            zero_diag (bool, optional): Whether to zero out the diagonal of the PE matrix. Defaults to False.
        """
        super().__init__(savepath, zero_diag)
        self.p = p
        self.beta = beta
        self.normalization = normalization
        self.use_edge_attr = use_edge_attr

    def compute_pe(self, graph):
        """
        Computes the p-step random walk based position encoding for a graph.

        Args:
            graph: A PyTorch Geometric graph data object.

        Returns:
            torch.Tensor: The position encoding matrix.
        """
        edge_attr = graph.edge_attr if self.use_edge_attr else None
        # Get the graph Laplacian
        edge_index, edge_weight = get_laplacian(
            graph.edge_index, edge_attr, normalization=self.normalization,
            num_nodes=graph.num_nodes)
        # Convert to SciPy sparse matrix
        L = to_scipy_sparse_matrix(edge_index, edge_weight, num_nodes=graph.num_nodes).tocsc()
        # Compute (I - beta * L)
        TransitionMatrix = sp.identity(L.shape[0], dtype=L.dtype, format='csc') - self.beta * L
        # Compute (I - beta * L)^p
        P_step_TransitionMatrix = TransitionMatrix
        for _ in range(self.p - 1):
            P_step_TransitionMatrix = P_step_TransitionMatrix.dot(TransitionMatrix)
        return torch.from_numpy(P_step_TransitionMatrix.toarray())


class AdjEncoding(PositionEncoding):
    """
    Computes position encoding using the (optionally normalized) adjacency matrix of the graph.
    """
    def __init__(self, savepath, normalization=None, zero_diag=False):
        """
        Initializes AdjEncoding.

        Args:
            savepath (str): Path to save/load precomputed encodings.
            normalization (str, optional): Normalization type for the adjacency matrix (currently not directly used by to_dense_adj in this way,
                                         but kept for consistency or future use with a normalized adjacency). Defaults to None.
            zero_diag (bool, optional): Whether to zero out the diagonal of the PE matrix. Defaults to False.
        """
        super().__init__(savepath, zero_diag)
        self.normalization = normalization # Note: to_dense_adj doesn't directly use this for 'sym' or 'rw' like get_laplacian

    def compute_pe(self, graph):
        """
        Computes the adjacency-based position encoding for a graph.

        Args:
            graph: A PyTorch Geometric graph data object.

        Returns:
            torch.Tensor: The dense adjacency matrix as the position encoding.
        """
        # to_dense_adj returns a [num_nodes, num_nodes] tensor
        # The normalization argument here is not directly used by to_dense_adj in the same way as get_laplacian.
        # If normalization is needed, it should be applied to graph.edge_index or graph.edge_attr before this call,
        # or the result of to_dense_adj should be post-processed.
        return to_dense_adj(graph.edge_index, edge_attr=graph.edge_attr, max_num_nodes=graph.num_nodes)[0]

class FullEncoding(PositionEncoding):
    """
    Computes a position encoding matrix of all ones.
    This can be a baseline or used in models that learn to ignore PE.
    """
    def __init__(self, savepath, zero_diag=False):
        """
        Initializes FullEncoding.

        Args:
            savepath (str): Path to save/load precomputed encodings.
            zero_diag (bool, optional): Whether to zero out the diagonal of the PE matrix. Defaults to False.
        """
        super().__init__(savepath, zero_diag)

    def compute_pe(self, graph):
        """
        Computes the full (all ones) position encoding for a graph.

        Args:
            graph: A PyTorch Geometric graph data object.

        Returns:
            torch.Tensor: A matrix of ones with shape (num_nodes, num_nodes).
        """
        return torch.ones((graph.num_nodes, graph.num_nodes))

## Absolute position encoding
class LapEncoding(PositionEncoding):
    """
    Computes absolute position encoding using the eigenvectors of the graph Laplacian.
    These are often referred to as Laplacian Eigenmaps.
    """
    def __init__(self, dim, use_edge_attr=False, normalization=None):
        """
        Initializes LapEncoding.
        Note: This encoding does not use the `savepath` and `zero_diag` from the base class constructor
              as it has its own `apply_to` method and typically isn't saved/loaded in the same way.

        Args:
            dim (int): The number of smallest (non-trivial) eigenvectors to use as features.
            use_edge_attr (bool, optional): Whether to use edge attributes for Laplacian computation. Defaults to False.
            normalization (str, optional): Normalization type for the Laplacian ('sym', 'rw', or None). Defaults to None.
        """
        # Does not call super().__init__() as it has a different apply_to signature and behavior regarding saving.
        self.pos_enc_dim = dim
        self.normalization = normalization
        self.use_edge_attr = use_edge_attr

    def compute_pe(self, graph):
        """
        Computes the Laplacian eigenvector position encoding for a graph.

        Args:
            graph: A PyTorch Geometric graph data object.

        Returns:
            torch.Tensor: A tensor of shape (num_nodes, dim) containing the selected eigenvectors.
        """
        edge_attr = graph.edge_attr if self.use_edge_attr else None
        # Get the graph Laplacian
        edge_index, edge_weight = get_laplacian(
            graph.edge_index, edge_attr, normalization=self.normalization, num_nodes=graph.num_nodes)
        # Convert to SciPy sparse matrix
        L = to_scipy_sparse_matrix(edge_index, edge_weight, num_nodes=graph.num_nodes).tocsc()
        # Compute eigenvalues and eigenvectors
        EigVal, EigVec = np.linalg.eig(L.toarray())
        idx = EigVal.argsort() # Sort eigenvalues in increasing order
        EigVal, EigVec = EigVal[idx], np.real(EigVec[:,idx])
        # Select the eigenvectors corresponding to the smallest non-trivial eigenvalues
        # EigVec[:, 0] usually corresponds to the trivial eigenvector (all ones for connected graph, eigenvalue 0)
        return torch.from_numpy(EigVec[:, 1:self.pos_enc_dim+1]).float()

    def apply_to(self, dataset):
        """
        Applies Laplacian position encoding to each graph in the dataset.
        Stores the encodings in `dataset.lap_pe_list`.

        Args:
            dataset: A dataset object where each item is a graph.

        Returns:
            The dataset with an added 'lap_pe_list' attribute.
        """
        dataset.lap_pe_list = []
        for i, g in enumerate(dataset):
            pe = self.compute_pe(g)
            dataset.lap_pe_list.append(pe)

        return dataset


# Dictionary mapping encoding names to their respective classes
POSENCODINGS = {
    "diffusion": DiffusionEncoding,
    "pstep": PStepRWEncoding,
    "adj": AdjEncoding,
    # "full": FullEncoding, # FullEncoding is available but not in this default map
    # "lap": LapEncoding,   # LapEncoding is available but not in this default map (different application style)
}