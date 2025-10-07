import numpy as np
import torch
import torch.nn as nn
from torchdiffeq import odeint

class GeneRegulatoryODE(nn.Module):

    def __init__(self, G, d_prot, k0, k1, bias_init=0.0, theta_init=0.0):
        super().__init__()
        self.G = G
        self.d_prot = torch.tensor(d_prot, dtype=torch.float32)
        self.k0 = torch.tensor(k0, dtype=torch.float32)
        self.k1 = torch.tensor(k1, dtype=torch.float32)
        
        # Learnable parameters
        # Initialize theta with the specified values
        if np.isscalar(theta_init):
            theta_init_tensor = torch.full((G, G), theta_init, dtype=torch.float32)
        else:
            theta_init_tensor = torch.tensor(theta_init, dtype=torch.float32)
        self.theta = nn.Parameter(theta_init_tensor)
        
        # Initialize bias with the specified value
        if np.isscalar(bias_init):
            bias_init_tensor = torch.full((G,), bias_init, dtype=torch.float32)
        else:
            bias_init_tensor = torch.tensor(bias_init, dtype=torch.float32)
        self.bias = nn.Parameter(bias_init_tensor)
        

    def forward(self, t, X):
        """
        ODE function: dX/dt = d * (kon - X)
        X: (batch_size, G) or (G,) if single sample
        """
        # Ensure X is 2D (batch_size, G)
        if X.dim() == 1:
            X = X.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        # CRITICAL: Force stimulus (gene 0) to remain at 1
        X = X.clone()  # Don't modify input tensor
        X[:, 0] = 1.0
            
        # Compute kon via neural network
        # X @ theta.T + bias -> (batch_size, G)
        pre_activation = X @ self.theta.T + self.bias
        sigmoid_output = torch.sigmoid(pre_activation)
        kon = self.k0 + (self.k1 - self.k0) * sigmoid_output
        
        # ODE: 
        dXdt = ...
        
        # CRITICAL: Force stimulus derivative to 0 (no change)
        dXdt[:, 0] = 0.0
        
        if squeeze_output:
            dXdt = dXdt.squeeze(0)
            
        return dXdt

class NetworkInference:
    def __init__(self, G):
        self.G = G
        self.inter = np.zeros((G, G))  # adjacency matrix / MI scores
        self.ode_func = None

    def fit(self,
            X_rna, X_prot, d_rna, d_prot, time,
            k0=0, k1=1,
            n_epochs=500, lr=1e-1,
            weight_decay=0.0,
            l1_lambda=0.0,
            verbose=1,
            print_every=100,
            method='dopri5',  # ODE solver method
            rtol=1e-7,
            atol=1e-9,
            bias_init=0.0,  # Bias initialization
            theta_init=0.0):  # Theta initialization
        """
        X_rna, X_prot : numpy arrays (n_cells, G)
        d_prot : array-like (G,)
        time : array-like (n_cells,) with discrete time points (must be orderable)
        k0, k1 : array-like (G,) or scalars -> default heuristics used
        bias_init : scalar or array-like (G,) -> initial bias values
        theta_init : scalar or array-like (G, G) -> initial theta matrix values
        method : ODE solver method ('dopri5', 'rk4', 'euler', etc.)
        rtol, atol : tolerances for adaptive solvers
        """
        # --- Convert to numpy and prepare data ---
        device = "cpu"
        X_prot_np = np.asarray(X_prot, dtype=float)
        d_prot_np = np.asarray(d_prot, dtype=float)
        d_prot_np[0] = 1  # Keep this constraint from original

        # --- Prepare k0 / k1 ---
        if np.isscalar(k0) or k0 is None:
            if k0 is None:
                k0_np = np.zeros(self.G)
            else:
                k0_np = np.full(self.G, k0)
        else:
            k0_np = np.asarray(k0, dtype=float)
            
        if np.isscalar(k1) or k1 is None:
            if k1 is None:
                # heuristic: max observed (plus small epsilon)
                k1_np = np.maximum(1.0, X_prot_np.max(axis=0))
            else:
                k1_np = np.full(self.G, k1)
        else:
            k1_np = np.asarray(k1, dtype=float)

        # --- Build time pairs and corresponding X0, X1 batches ---
        unique_times = np.unique(time)
        pairs = []
        for idx in range(len(unique_times)-1):
            t0 = unique_times[idx]
            t1 = unique_times[idx+1]

            X0_np = ...
            X1_np = ...

            pairs.append((float(t0), float(t1), X0_np, X1_np))

        # --- Initialize ODE function ---
        self.ode_func = GeneRegulatoryODE(self.G, d_prot_np, k0_np, k1_np, bias_init, theta_init).to(device)
        
        # --- Optimizer and loss ---
        optimizer = torch.optim.Adam(self.ode_func.parameters(), lr=lr, weight_decay=weight_decay)
        loss_custom = ...

        # --- Training loop ---
        old_loss = 1e16
        for epoch in range(1, n_epochs + 1):
            optimizer.zero_grad()
            total_loss = 0.0
            total_count = 0
            
            for (t0, t1, X0_batch, X1_batch) in pairs:
                # Convert to tensors
                X0_tensor = torch.tensor(X0_batch, dtype=torch.float32, device=device)
                X1_tensor = torch.tensor(X1_batch, dtype=torch.float32, device=device)
                
                # CRITICAL: Ensure stimulus stays at 1 for both initial and target states
                X0_tensor[:, 0] = 1.0
                X1_tensor[:, 0] = 1.0
                
                # Time span for integration
                t_span = torch.tensor([t0, t1], dtype=torch.float32, device=device)
                
                # Solve ODE
                X_pred_trajectory = odeint(
                    self.ode_func, 
                    X0_tensor, 
                    t_span,
                    method=method,
                    rtol=rtol,
                    atol=atol
                )
                
                # Extract final state (at t1)
                X_pred = X_pred_trajectory[-1]  # Shape: (batch_size, G)
                
                # CRITICAL: Ensure stimulus stays at 1 in predictions
                X_pred[:, 0] = 1.0
                
                # Compute loss
                loss_batch = ...
                total_loss += loss_batch * X0_batch.shape[0]
                total_count += X0_batch.shape[0]
            
            loss = total_loss / total_count
            
            # L1 penalty on theta
            if l1_lambda is not None and l1_lambda > 0.0:
                loss = loss + l1_lambda * torch.norm(self.ode_func.theta, p=1)
            
            loss.backward()
            optimizer.step()

            if verbose and (epoch % print_every == 0 or epoch == 1 or epoch == n_epochs):
                print(f"[Epoch {epoch:4d}/{n_epochs}] loss = {loss.item():.6e}")
                if abs(loss.item() - old_loss) < 1e-6:
                    print(f"Converged at epoch {epoch}")
                    break
                old_loss = loss.item()

        # --- Store learned parameters ---
        self.inter = self.ode_func.theta.detach().cpu().numpy()
        self._bias = self.ode_func.bias.detach().cpu().numpy()
        self._k0 = k0_np
        self._k1 = k1_np
        
    def predict(self, X0, t_span, method='dopri5', rtol=1e-7, atol=1e-9):
        """
        Predict trajectory from initial state X0 over time span t_span
        
        Args:
            X0: Initial state (batch_size, G) or (G,)
            t_span: Time points to evaluate (n_times,)
            method: ODE solver method
            
        Returns:
            Trajectory (n_times, batch_size, G) or (n_times, G) if single sample
        """
        if self.ode_func is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
            
        device = next(self.ode_func.parameters()).device
        X0_tensor = torch.tensor(X0, dtype=torch.float32, device=device)
        t_span_tensor = torch.tensor(t_span, dtype=torch.float32, device=device)
        
        # CRITICAL: Ensure stimulus starts at 1
        if X0_tensor.dim() == 1:
            X0_tensor[0] = 1.0
        else:
            X0_tensor[:, 0] = 1.0
        
        with torch.no_grad():
            trajectory = odeint(
                self.ode_func,
                X0_tensor,
                t_span_tensor,
                method=method,
                rtol=rtol,
                atol=atol
            )
            
            # CRITICAL: Force stimulus to 1 at all time points in trajectory
            trajectory[:, :, 0] = 1.0  # All times, all samples, gene 0 = 1
        
        return trajectory.cpu().numpy()
    
    
    def compare_trajectories_umap(self, X_prot, time, 
                                 n_neighbors=15, min_dist=0.1, 
                                 figsize=(12, 5), method='dopri5'):
        """
        Compare real vs predicted trajectories using UMAP visualization
        
        Args:
            X_prot: Original protein data (n_cells, G)
            time: Time points corresponding to X_prot (n_cells,)
            n_neighbors, min_dist: UMAP parameters
            figsize: Figure size for plots
            method: ODE solver method for predictions
        """
        import matplotlib.pyplot as plt
        import umap
        from matplotlib.colors import ListedColormap
        
        if self.ode_func is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        
        # Prepare data
        X_prot_np = np.asarray(X_prot, dtype=float)
        time_np = np.asarray(time)
        unique_times = np.unique(time_np)
        
        # Generate predictions for all time points
        X_pred_full = []
        time_pred_full = []
        
        for i, t in enumerate(unique_times):
            mask = time_np == t
            X_at_t = X_prot_np[mask]
            
            if i < len(unique_times) - 1:
                # Predict next time point
                t_next = unique_times[i + 1]
                t_span = np.array([t, t_next])
                
                # Get predictions
                traj = self.predict(X_at_t, t_span, method=method)
                X_pred_next = traj[-1]  # Final state
                
                # Store predictions
                X_pred_full.append(X_pred_next)
                time_pred_full.extend([t_next] * X_pred_next.shape[0])
        
        if len(X_pred_full) == 0:
            raise ValueError("Need at least 2 time points for predictions")
            
        X_pred_concat = np.vstack(X_pred_full)
        time_pred_concat = np.array(time_pred_full)
        
        # Combine real and predicted data for UMAP
        X_combined = np.vstack([X_prot_np, X_pred_concat])
        
        # Create labels: 0 for real, 1 for predicted
        labels_combined = np.concatenate([
            np.zeros(len(X_prot_np)), 
            np.ones(len(X_pred_concat))
        ])
        
        # Create time labels for coloring
        time_combined = np.concatenate([time_np, time_pred_concat])
        
        # Fit UMAP
        print("Fitting UMAP")
        umap_model = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=42
        )
        embedding = umap_model.fit_transform(X_combined)
        
        # Create plots
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Plot 1: Real vs Predicted
        real_mask = labels_combined == 0
        pred_mask = labels_combined == 1
        
        axes[0].scatter(embedding[real_mask, 0], embedding[real_mask, 1], 
                       c='blue', alpha=0.6, s=30, label='Real data')
        axes[0].scatter(embedding[pred_mask, 0], embedding[pred_mask, 1], 
                       c='red', alpha=0.6, s=30, label='Predicted')
        axes[0].set_title('Real vs Predicted Trajectories')
        axes[0].legend()
        axes[0].set_xlabel('UMAP 1')
        axes[0].set_ylabel('UMAP 2')
        
        # Plot 2: Colored by time
        scatter = axes[1].scatter(embedding[:, 0], embedding[:, 1], 
                                c=time_combined, cmap='viridis', 
                                alpha=0.7, s=30)
        axes[1].set_title('Trajectories Colored by Time')
        axes[1].set_xlabel('UMAP 1')
        axes[1].set_ylabel('UMAP 2')
        plt.colorbar(scatter, ax=axes[1], label='Time')
        
        plt.tight_layout()
        plt.show()
        
        # Print some statistics
        print(f"\nData summary:")
        print(f"Real data points: {np.sum(real_mask)}")
        print(f"Predicted data points: {np.sum(pred_mask)}")
        print(f"Time points: {len(unique_times)} ({unique_times.min():.2f} to {unique_times.max():.2f})")
        
        return embedding, labels_combined, time_combined
    
    def plot_gene_trajectories(self, X_prot, time, gene_indices=None, 
                              figsize=(15, 10), method='dopri5'):
        """
        Plot individual gene expression trajectories: real vs predicted
        
        Args:
            X_prot: Original protein data (n_cells, G)
            time: Time points (n_cells,)
            gene_indices: List of gene indices to plot (default: first 6)
            figsize: Figure size
            method: ODE solver method
        """
        import matplotlib.pyplot as plt
        
        if self.ode_func is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        
        X_prot_np = np.asarray(X_prot, dtype=float)
        time_np = np.asarray(time)
        unique_times = np.unique(time_np)
        
        if gene_indices is None:
            gene_indices = list(range(min(6, self.G)))
        
        # Generate predictions
        predictions_by_time = {}
        real_by_time = {}
        
        for t in unique_times:
            mask = time_np == t
            real_by_time[t] = X_prot_np[mask]
        
        for i, t in enumerate(unique_times[:-1]):
            t_next = unique_times[i + 1]
            X_at_t = real_by_time[t]
            
            traj = self.predict(X_at_t, [t, t_next], method=method)
            predictions_by_time[t_next] = traj[-1]
            
            # CRITICAL: Ensure stimulus stays at 1 in predictions
            predictions_by_time[t_next][:, 0] = 1.0
        
        # Create subplots
        n_genes = len(gene_indices)
        rows = int(np.ceil(n_genes / 3))
        cols = min(3, n_genes)
        
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        if n_genes == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes.reshape(1, -1)
        
        for idx, gene_idx in enumerate(gene_indices):
            row = idx // cols
            col = idx % cols
            ax = axes[row, col] if rows > 1 else axes[col]
            
            # Plot real data
            for t in unique_times:
                real_vals = real_by_time[t][:, gene_idx]
                ax.scatter([t] * len(real_vals), real_vals, 
                          alpha=0.6, c='blue', s=30, label='Real' if t == unique_times[0] else "")
            
            # Plot predictions
            for t in unique_times[1:]:
                if t in predictions_by_time:
                    pred_vals = predictions_by_time[t][:, gene_idx]
                    ax.scatter([t] * len(pred_vals), pred_vals, 
                              alpha=0.6, c='red', s=30, marker='^', 
                              label='Predicted' if t == unique_times[1] else "")
            
            ax.set_title(f'Gene {gene_idx}')
            ax.set_xlabel('Time')
            ax.set_ylabel('Expression')
            if idx == 0:
                ax.legend()
        
        # Hide empty subplots
        for idx in range(n_genes, rows * cols):
            if rows > 1:
                axes[idx // cols, idx % cols].set_visible(False)
            else:
                axes[idx].set_visible(False)
        
        plt.tight_layout()
        plt.show()