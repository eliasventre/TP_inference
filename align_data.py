# align_proteins.py
"""
Protein trajectory alignment using Optimal Transport (OT).

This script provides a function to align protein expression data (X_prot)
across consecutive time points when the natural pairing between cells/samples
is missing.

The alignment is done with Optimal Transport (Earth Mover's Distance).
"""

import numpy as np
import ot  # POT library: pip install pot


def align_proteins(X_prot, time):
    """
    Align protein trajectories between consecutive time points
    using Optimal Transport (OT).

    Parameters
    ----------
    X_prot : np.ndarray
        Protein data of shape (n_samples, n_features).
        It contains cells/samples from all time points concatenated together.

    time : np.ndarray
        Array of shape (n_samples,) containing the time label of each sample.
        Example: [0,0,0,1,1,1] means 3 cells at t=0 and 3 cells at t=1.

    Returns
    -------
    X_aligned : np.ndarray
        Same shape as X_prot, but with samples at later time points
        aligned to their "parents" at the previous time points.
    """

    unique_times = np.sort(np.unique(time))
    X_aligned = X_prot.copy()

    for t1, t2 in zip(unique_times[:-1], unique_times[1:]):
        # Extract subsets at consecutive times
        X1 = X_aligned[time == t1]
        X2 = X_aligned[time == t2]

        n1, n2 = X1.shape[0], X2.shape[0]
        if n1 != n2:
            raise ValueError(f"Number of samples must match: "
                             f"{n1} at t={t1} vs {n2} at t={t2}")

        # Compute squared Euclidean distance matrix
        M = ot.dist(X1, X2, metric='euclidean')**2

        # Uniform weights for both distributions
        a, b = np.ones(n1) / n1, np.ones(n2) / n2

        # Optimal Transport plan (Earth Mover’s Distance)
        P = ot.emd(a, b, M)

        # Align: each point in X1 is matched to its descendant in X2
        # Here we compute barycenters, but with uniform cardinality
        # it reduces to a permutation-like mapping.
        X2_aligned = P @ X2 * n2  # rescale because sum(b) = 1

        # Replace X2 with its aligned version
        X_aligned[time == t2] = X2_aligned

    return X_aligned


if __name__ == "__main__":
    # Toy example
    np.random.seed(0)

    # Two time points with 5 samples each
    time = np.repeat([0, 1], 5)
    X_prot = np.vstack([
        np.random.randn(5, 3),       # t=0
        np.random.randn(5, 3) + 1.0  # t=1 (shifted)
    ])

    print("Original X_prot (t=0 on top, t=1 below):")
    print(X_prot)

    X_aligned = align_proteins(X_prot, time)

    print("\nAligned X_prot:")
    print(X_aligned)
