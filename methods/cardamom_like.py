import numpy as np
import torch
import torch.nn as nn

class GRNInference(nn.Module):
    def __init__(self, G, k0=0.0, k1=1.0, bias_init=0.0, theta_init=0.0):
        super().__init__()
        self.G = G
        self.k0 = torch.tensor(k0, dtype=torch.float32) if np.isscalar(k0) else torch.tensor(k0, dtype=torch.float32)
        self.k1 = torch.tensor(k1, dtype=torch.float32) if np.isscalar(k1) else torch.tensor(k1, dtype=torch.float32)

        # paramètres appris : θ et biais
        if np.isscalar(theta_init):
            theta_init_tensor = torch.full((G, G), theta_init, dtype=torch.float32)
        else:
            theta_init_tensor = torch.tensor(theta_init, dtype=torch.float32)
        self.theta = nn.Parameter(theta_init_tensor)

        if np.isscalar(bias_init):
            bias_init_tensor = torch.full((G,), bias_init, dtype=torch.float32)
        else:
            bias_init_tensor = torch.tensor(bias_init, dtype=torch.float32)
        self.bias = nn.Parameter(bias_init_tensor)

    def forward(self, X_prot):
        """
        Calcule kon^θ(X_prot)
        X_prot: (batch_size, G)
        Retourne: (batch_size, G)
        """
        pre_activation = X_prot @ self.theta.T + self.bias
        sigmoid_output = torch.sigmoid(pre_activation)
        kon = self.k0 + (self.k1 - self.k0) * sigmoid_output
        return kon


class NetworkInference:
    def __init__(self, G):
        self.G = G
        self.inter = np.zeros((G, G))
        self.model = None

    def fit(self, X_rna, X_prot, d_rna, d_prot, time,
            k0=0.0, k1=1.0,
            n_epochs=500, 
            lr=1e-2,
            weight_decay=0.0,
            l1_lambda=0.0,
            verbose=1, 
            print_every=100,
            bias_init=0.0,  # Bias initialization
            theta_init=0.0):  # Theta initialization
        """
        Apprend θ en minimisant MSE(kon^θ(X_prot), X_rna)
        
        Args:
            X_rna: (n_cells, G) numpy (expected binary)
            X_prot: (n_cells, G) numpy (features)
        """

        device = "cpu"
        X_prot_tensor = torch.tensor(X_prot, dtype=torch.float32, device=device)
        X_rna_tensor = torch.tensor(X_rna, dtype=torch.float32, device=device)

        # modèle
        self.model = GRNInference(self.G, k0, k1, bias_init, theta_init).to(device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        mse = nn.MSELoss(reduction='mean')

        old_loss = 1e16
        for epoch in range(1, n_epochs+1):
            optimizer.zero_grad()
            kon_pred = self.model(X_prot_tensor)
            loss = mse(kon_pred, X_rna_tensor)

            # régularisation L1
            if l1_lambda is not None and l1_lambda > 0.0:
                loss = loss + l1_lambda * torch.norm(self.model.theta, p=1)

            loss.backward()
            optimizer.step()

            if verbose and (epoch % print_every == 0 or epoch == 1 or epoch == n_epochs):
                print(f"[Epoch {epoch:4d}/{n_epochs}] loss = {loss.item():.6e}")
                if abs(loss.item() - old_loss) < 1e-6:
                    print(f"Converged at epoch {epoch}")
                    break
                old_loss = loss.item()

        # stocke les paramètres appris
        self.inter = self.model.theta.detach().cpu().numpy()
        self._bias = self.model.bias.detach().cpu().numpy()
        self._k0 = np.array(k0 if np.isscalar(k0) else k0)
        self._k1 = np.array(k1 if np.isscalar(k1) else k1)

    def predict(self, X_prot):
        """
        Retourne kon^θ(X_prot)
        """
        if self.model is None:
            raise ValueError("Model not fitted yet. Call fit() first.")

        device = next(self.model.parameters()).device
        X_prot_tensor = torch.tensor(X_prot, dtype=torch.float32, device=device)
        with torch.no_grad():
            kon_pred = self.model(X_prot_tensor)
        return kon_pred.cpu().numpy()
