# =====================================================================
# Data Simulation Script for aNetwork (Harissa)
#
# This script generates single-cell gene expression data 
# from a predefined gene regulatory network using Harissa.
# The output consists of simulated expression data stored in 
# AnnData objects, and the true network topology (adjacency matrix).
# =====================================================================

# outfile='Network4'
outfile='Network8'












import sys
sys.path += ['../']  # add Harissa to the Python path

import numpy as np
import pandas as pd
import anndata as ad
from harissa import NetworkModel

# ---------------------------------------------------------------------
# Simulation settings
# ---------------------------------------------------------------------
C = 1000   # number of cells
N = 3     # number of independent simulation runs

# Time points (hours)
t = np.linspace(0, 100, 10) 
# Assign time points to cells (C cells distributed over len(t) time points)
k = np.linspace(0, C, len(t)+1, dtype=int)
time = np.zeros(C, dtype=int)
for i in range(len(t)):
    time[k[i]:k[i+1]] = t[i]


if outfile == 'Network4':
    G = 4  # number of genes
    # ---------------------------------------------------------------------
    # Initialize network model
    # ---------------------------------------------------------------------
    model = NetworkModel(G)
    # Gene degradation rates
    model.d[0] = .2
    model.d[1] = 0.04
    # Basal expression levels
    model.basal[1:] = -5
    # Regulatory interactions
    # Format: model.inter[regulator, target] = strength
    model.inter[0,1] = 10   # Stimulus → gene1
    model.inter[1,2] = 10   # gene1 → gene2
    model.inter[1,3] = 10   # gene1 → gene3
    model.inter[4,1] = -10  # gene4 → gene1 
    model.inter[3,4] = 10  # gene3 → gene4 
    model.inter[2,2] = 10   # gene2 self-activation
    model.inter[3,3] = 10   # gene3 self-activation

if outfile == 'Network8':
    G = 8 # number of genes
    # ---------------------------------------------------------------------
    # Initialize network model
    # ---------------------------------------------------------------------
    model = NetworkModel(G)
    # Gene degradation rates
    model.d[0] = 0.4
    model.d[1] = 0.08
    # Basal expression levels
    model.basal[1:] = -5
    # Regulatory interactions
    # Format: model.inter[regulator, target] = strength
    model.inter[0, 1] = 10
    model.inter[1, 2] = 10
    model.inter[2, 3] = 10
    model.inter[3, 4] = 10
    model.inter[3, 5] = 10
    model.inter[3, 6] = 10
    model.inter[4, 1] = -10
    model.inter[5, 1] = -10
    model.inter[6, 1] = -10
    model.inter[4, 4] = 10
    model.inter[5, 5] = 10
    model.inter[6, 6] = 10
    model.inter[4, 8] = -10
    model.inter[4, 7] = -10
    model.inter[6, 7] = 10
    model.inter[7, 6] = 10
    model.inter[8, 8] = 10

# Save true network topology as adjacency matrix
inter = (abs(model.inter) > 0).astype(int)
np.save(f"{outfile}/true/inter", inter)
np.save(f"{outfile}/true/inter_signed", model.inter)

# ---------------------------------------------------------------------
# Run multiple simulations
# ---------------------------------------------------------------------
for r in range(N):
    print(f"Run {r+1}...")
    
    ### Simulate trajectories

    data_rna = np.zeros((C, G+1), dtype=float)
    data_prot = np.zeros((C, G+1), dtype=float)
    data_rna[time > 0, 0] = 100 # Stimulus
    data_prot[time > 0, 0] = 1 # Stimulus
    
    # Simulated expression data
    n_cells_atorigin = np.sum(time==0)
    for c in range(C):
        if c < n_cells_atorigin:
            sim = model.simulate(time[c], burnin=5)
        else:
            sim = model.simulate(time[c] - time[c-n_cells_atorigin], M0=data_rna[c-n_cells_atorigin], P0=data_prot[c-n_cells_atorigin])
        data_rna[c, 1:] = sim.m[-1]
        data_prot[c, 1:] = sim.p[-1]

    # Create AnnData object
    adata_rna = ad.AnnData(X=np.random.poisson(data_rna))
    adata_prot = ad.AnnData(X=data_prot)

    # Add time information to observations
    adata_rna.obs['time'] = time.astype(int)
    adata_rna.var['d0'] = model.d[0]
    adata_prot.obs['time'] = time.astype(int)
    adata_prot.var['d1'] = model.d[1]

    # Save dataset
    adata_rna.write(f"{outfile}/data_rna/data_traj_{r+1}.h5ad")
    adata_prot.write(f"{outfile}/data_prot/data_traj_{r+1}.h5ad")

    ### Simulate distributions

    data_rna = np.zeros((C, G+1), dtype=float)
    data_prot = np.zeros((C, G+1), dtype=float)
    data_rna[time > 0, 0] = 100 # Stimulus
    data_prot[time > 0, 0] = 1 # Stimulus

    # Simulated expression data
    for c in range(C):
        sim = model.simulate(time[c], burnin=5)
        data_rna[c, 1:] = sim.m[-1]
        data_prot[c, 1:] = sim.p[-1]


    # Create AnnData object
    adata_rna = ad.AnnData(X=np.random.poisson(data_rna))
    adata_prot = ad.AnnData(X=data_prot)

    # Add time information to observations
    adata_rna.obs['time'] = time.astype(int)
    adata_rna.var['d0'] = model.d[0]
    adata_prot.obs['time'] = time.astype(int)
    adata_prot.var['d1'] = model.d[1]

    # Save dataset
    adata_rna.write(f"{outfile}/data_rna/data_distrib_{r+1}.h5ad")
    adata_prot.write(f"{outfile}/data_prot/data_distrib_{r+1}.h5ad")

