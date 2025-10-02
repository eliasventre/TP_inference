# =====================================================================
# outfile undirected inference methods (AUPR curves)
#
# Methods are discovered automatically from Python files in ./methods/
# Only keep those with results in ./network/{method}/
# True network is loaded from network/true/inter.npy
# Inferred scores are loaded from network/{method}/score_{r}.npy
#
# Output: AUPR boxplots per method
# =====================================================================

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc

# ---------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------
# outfile = "Network8"   # which dataset to evaluate
outfile = "Network4" # which dataset to evaluate
methods_path = "methods"  # path to methods
N = 5  # number of runs to take into account

# ---------------------------------------------------------------------
# Discover available methods = all .py files in methods/
# Keep only those that have results in outfile/
# ---------------------------------------------------------------------
all_methods = [
    os.path.splitext(os.path.basename(f))[0]
    for f in glob.glob(os.path.join(methods_path, "*.py"))
    if not f.endswith("__init__.py")
]

methods = [m for m in all_methods if os.path.isdir(os.path.join(outfile, m))]
print("Methods with results found:", methods)

# ---------------------------------------------------------------------
# Load true network and prepare edge list
# ---------------------------------------------------------------------
inter = abs(np.load(f"{outfile}/true/inter.npy"))
G = inter.shape[0]
edges = [(i, j) for i in range(G) for j in range(i+1, G)]
y_true = np.array([max(inter[i, j], inter[j, i]) for (i, j) in edges])

# ---------------------------------------------------------------------
# Compute AUPR for each method
# ---------------------------------------------------------------------
aupr = {m: [] for m in methods}

for r in range(1, N+1):
    for m in methods:
        try:
            score = abs(np.load(f"{outfile}/{m}/score_{r}.npy"))
        except FileNotFoundError:
            print(f"⚠️ Missing results for {m}, run {r}")
            continue

        # Symmetrize scores
        y_score = np.array([max(score[i, j], score[j, i]) for (i, j) in edges])

        # Compute precision-recall and AUPR
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        aupr[m].append(auc(recall, precision))

# Baseline random predictor = mean edge density
random_baseline = np.mean(y_true)

# ---------------------------------------------------------------------
# Plot results
# ---------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 4))

# Colors for methods
cmap = plt.get_cmap("tab20")
colors = {m: cmap(i) for i, m in enumerate(methods)}

# Draw Random as a horizontal line
ax.axhline(random_baseline, color="lightgray", ls="--", lw=1, label="Random")

# Draw boxplots for each method
positions = range(1, len(methods) + 1)
for i, m in enumerate(methods):
    box = ax.boxplot(
        aupr[m],
        positions=[i + 1],
        widths=0.6,
        patch_artist=False,  # no fill
    )
    # Customize colors
    plt.setp(box["boxes"], color=colors[m], lw=1.5)
    plt.setp(box["whiskers"], color=colors[m], lw=1)
    plt.setp(box["caps"], color=colors[m], lw=1)
    plt.setp(box["medians"], color=colors[m], lw=2)

# Axis settings
ax.set_xticks(positions)
ax.set_xticklabels(methods, rotation=45, ha="right")
ax.set_ylabel("AUPR")
ax.set_ylim(0, 1)
ax.set_title(f"AUPR outfile ({outfile})")

plt.tight_layout()
plt.savefig(f"{outfile}_aupr.pdf", dpi=300)
plt.show()
