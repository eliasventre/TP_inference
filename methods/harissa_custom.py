import numpy as np
from harissa import NetworkModel 

class NetworkInference:
    def __init__(self, G):
        self.G = G
        self.inter = np.zeros((G, G))  # adjacency matrix / MI scores

    def fit(self, X_rna, X_prot, d_rna, d_prot, time):
        """
        X_rna, X_prot: (cells × genes) expression matrices
        d_rna, d_prot: degradation rates (can be ignored for harissa)
        time: timepoints
        """
        model_harissa = NetworkModel()
        data = X_rna.copy()
        data[:, 0] = time
        model_harissa.fit(data)
        self.inter = model_harissa.inter

