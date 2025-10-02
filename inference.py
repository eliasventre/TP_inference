# =====================================================================
# Script to infer gene regulatory networks from AnnData objects
# with a chosen method (or all methods in ./methods/)
# =====================================================================


import importlib
import argparse
import glob
import numpy as np
import anndata as ad

from align_data import align_proteins
from binarize_data import binarize_mrnas

import os
import sys
sys.path.append("methods")  # allow importing user-defined methods

# outfile = 'Network8'
outfile = 'Network4'
type_data = 'traj'
# type_data = 'distrib'

# ---------------------------------------------------------------------
# Command-line arguments
# ---------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Infer GRNs from single-cell data")
parser.add_argument(
    "-m", "--method",
    type=str,
    default="all",
    help='Method to use. Either a module name in methods/ or "all" for all methods'
)
args = parser.parse_args()
method_arg = args.method

# ---------------------------------------------------------------------
# Discover available methods
# ---------------------------------------------------------------------
methods = [
    os.path.splitext(os.path.basename(f))[0]
    for f in glob.glob("methods/*.py")
    if not f.endswith("__init__.py")
]

if method_arg.lower() == "all":
    selected_methods = methods
else:
    if method_arg not in methods:
        raise ValueError(f"Method {method_arg} not found in available methods or missing results folder")
    selected_methods = [method_arg]

print("Selected methods:", selected_methods)

# ---------------------------------------------------------------------
# General settings
# ---------------------------------------------------------------------
N = 5     # number of simulation runs
verb = 1   # verbosity

# ---------------------------------------------------------------------
# Loop over selected methods
# ---------------------------------------------------------------------
for method in selected_methods:
    if verb: print(f"=== Inferring with method {method} ===")
    
    # Import dynamically
    module = importlib.import_module(method)
    NetworkInference = getattr(module, "NetworkInference")
    
    outdir = os.path.join(outfile, method)
    os.makedirs(outdir, exist_ok=True)


    # ---------------------------------------------------------------------
    # Loop over datasets
    # ---------------------------------------------------------------------
    for r in range(1, N+1):
        if verb: print(f"--- Run {r} with method {method} ---")

        # Load AnnData object
        fname_rna = f"{outfile}/data_rna/data_{type_data}_{r}.h5ad"
        adata_rna = ad.read_h5ad(fname_rna)
        fname_prot = f"{outfile}/data_prot/data_{type_data}_{r}.h5ad"
        adata_prot = ad.read_h5ad(fname_prot)

        # Extract expression matrix (cells × genes) and time
        X_rna = adata_rna.X              # shape (cells, genes)
        X_prot = adata_prot.X              # shape (cells, genes)
        time = adata_rna.obs['time'] # shape (cells,)

        # Transform data if needed
        if type_data == 'distrib' and method == 'neuralODEs':
            X_prot = align_proteins(X_prot, time)

        if method == 'cardamom_like':
            X_rna = binarize_mrnas(X_rna, time)

        # Number of genes
        G = X_rna.shape[1]

        # Degradation rates
        d0 = adata_rna.var['d0']
        d1 = adata_prot.var['d1']

        # Initialize model
        model_inference = NetworkInference(G)

        # Fit the model on data : THIS IS THE CORE OF THIS TP
        model_inference.fit(X_rna, X_prot, d0, d1, time)

        # if method == 'neuralODEs':
        #     print(model_inference.inter, model_inference._bias)
        #     # Visualisation UMAP
        #     embedding, labels, time_labels = model_inference.compare_trajectories_umap(
        #         X_prot, time, 
        #         n_neighbors=15, 
        #         min_dist=0.1
        #     )

        #     # Trajectoires de gènes
        #     model_inference.plot_gene_trajectories(
        #         X_prot, time, 
        #         gene_indices=np.arange(G), 
        #         figsize=(15, 8)
        #     )

        # Get inferred network (adjacency / scores)
        score = model_inference.inter

        # Build output path
        outdir = f"{outfile}/{method}"
        os.makedirs(outdir, exist_ok=True)   # create folder if it does not exist

        outname = os.path.join(outdir, f"score_{r}.npy")
        np.save(outname, score)

        if verb: print(f"Saved inferred network to {outname}")

