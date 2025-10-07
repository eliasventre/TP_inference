# binarize_data.py

"""
Core functions for ZiNB mixture inference, mainly used in fit_mixture
"""
import numpy as np
from scipy.special import psi, polygamma, gammaln
import scipy.stats as stats
from scipy.optimize import minimize

def zinb_pmf(x, k, c, pi):
    """Zero-Inflated Negative Binomial PMF."""
    nb_pmf = stats.nbinom.pmf(x, k, c / (c + 1))
    zinb_pmf = np.where(x == 0, pi + (1 - pi) * nb_pmf, (1 - pi) * nb_pmf)
    return zinb_pmf


def log_likelihood_zinb(pi, X_i, k_i, c_i):
    """Log-vraisemblance sous une ZINB pour un gene i."""
    pi = np.clip(pi, 1e-6, 1 - 1e-6)  # Évite les valeurs extrêmes
    likelihoods = zinb_pmf(X_i, k_i, c_i, pi)
    return -np.sum(np.log(likelihoods + 1e-16))  # Évite log(0)


def estimate_pi(X, K, C):
    """Estime pi optimal pour chaque gène."""
    G = X.shape[1]
    pi_opt = np.zeros(G)
    
    for i in range(G):
        X_i = X[:, i]  # Expression réelle du gène i
        k_i = K[:, i]  # Paramètre k_i pour chaque cellule
        c_i = C[i]      # Paramètre c_i (fixe par gène)
        
        # Optimisation de la log-vraisemblance
        res = minimize(log_likelihood_zinb, x0=0.5, 
                       args=(X_i, k_i, c_i), 
                       method='L-BFGS-B', 
                       tol = 1e-9,
                       bounds=[(0, 1)])
        pi_opt[i] = res.x[0]
    
    return pi_opt


def estim_gamma_poisson(x, mod=0, a_init=0, b_init=0):
    """
    Estimate parameters a and b of the Gamma-Poisson(a,b) distribution,
    a.k.a. negative binomial distribution, using the method of moments.
    """
    m = np.mean(x)
    v = np.var(x)
    if m == 0: return 0, 1
    if v == 0: return m, 1
    r = v - m
    if r > 0: 
        a = (m**2) / r
        b = a / m
    else:
        if a_init and b_init:
            if a_init/b_init < v:
                a = ((a_init/b_init)**2) / (v - a_init/b_init)
                b = a / m
            else:
                a = a_init
                b = b_init
        else:
            a = (m**2) / v
            b = a / m
    if mod == -1 and a_init:
        if a_init/b_init < a/b:
            a, b = a_init, b_init
    elif mod == 1 and a_init:
        if a_init/b_init > a/b:
            a, b = a_init, b_init
    return a, b


def infer_kinetics_temporal(x, times, seuil, a_init=np.ones(100), b_init=1, tol=1e-5, max_iter=100, verb=False):
    """
    Infer parameters a[0], ..., a[m-1] and b of a Gamma-Poisson model
    with time-dependant a and constant b for a given gene at m time points.

    Parameters
    ----------
    x[k] = gene expression in cell k
    times[k] = time point of cell k
    """

    t = np.sort(list(set(times)))
    m = t.size
    n = np.zeros(m) # Number of cells for each time point
    a = np.zeros(m)
    b = np.zeros(m)
    # Initialization of a and b
    for i in range(m):
        cells = (times == t[i])
        n[i] = np.sum(cells)
        a[i], b[i] = estim_gamma_poisson(x[cells], mod=(i==m)-(i==0), a_init=a_init[i], b_init=b_init)
    b = np.mean(b)
    # Newton-like method
    k, c = 0, 0
    sx = np.sum(x)
    while (k == 0) or (k < max_iter and c > tol):
        da = np.zeros(m)
        for i in range(m):
            if a[i] > 0:
                cells = (times == t[i])
                z = a[i] + x[cells]
                p0 = np.sum(psi(z))
                p1 = np.sum(polygamma(1, z))
                d = n[i]*(np.log(b)-np.log(b+1)-psi(a[i])) + p0
                h = p1 - n[i]*polygamma(1, a[i])
                da[i] = -d/h
        # Ensure that we don't move forward too fast
        tmp = np.maximum(.1, a/10)
        if np.sum(np.abs(da) > tmp) > 0:
            for i in range(m):
                if abs(da[i]) > tmp[i]:
                    da[i] = (1 - 2*(da[i] < 0)) * tmp[i]
        if np.sum((a + da) < 0) == 0: a += da
        else:
            for i in range(m):
                if (a + da)[i] < 0: a[i] = 0
                else: a[i] += da[i]
        b = np.sum(n*a)/(sx + 1e-16)
        c = np.max(np.abs(da))
        k += 1
    if (k == max_iter) and (c > tol):
        print('Warning: bad convergence (b = {})'.format(b), k/max_iter, c, b)
        if b > 1: a, b = a/b, 1
    if np.sum(a < 0) > 0: print('WARNING: a < 0')
    if b < 0: print('WARNING: b < 0')
    if np.all(a == 0): print('WARNING: a == 0')
    b = max(b, seuil)
    a_max = np.max(a)
    if a_max/b > 50:
        a = np.maximum(a, b)
    else:
        a = np.maximum(a, a_max/50)

    return a, b


class NegativeBinomialMixtureEM:
    def __init__(self, min_components=1, max_components=10, tol=1e-3, max_iter_loopsrefining=50, max_iter_kinetics=1000):
        self.max_components = max_components
        self.min_components = min_components
        self.tol = tol
        self.max_iter_loopsrefining = max_iter_loopsrefining
        self.best_model = None
        self.scale = 100
        self.max_iter_kinetics=max_iter_kinetics
    
    def negative_binomial_pmf(self, x, k, c):
        return stats.nbinom.pmf(x, k, c/(c+1))
    
    def refine_kinetics(self, data, n_components, ks, c, seuil):

        vect_binom = np.array([self.negative_binomial_pmf(data, ks[z], c) for z in range(0, n_components)]).T
        basins = np.argmax(vect_binom, 1)
        weights = np.sum(vect_binom, 0) / np.sum(vect_binom)
        if len(np.unique(basins)) < n_components:
            ks = np.linspace(np.min(seuil + data*c), np.max(seuil + data*c), n_components)
            vect_binom = np.array([self.negative_binomial_pmf(data, ks[z], c) for z in range(0, n_components)]).T
            weights = np.sum(vect_binom, 0) / np.sum(vect_binom)
            basins = np.argmax(vect_binom, 1)
            if len(np.unique(basins)) < n_components:
                return weights, ks, c, -np.inf
        
        log_likelihoods = []
        
        n_iter=0

        for _ in range(self.max_iter_loopsrefining):

            ks_new, c_new = infer_kinetics_temporal(data, basins, seuil, a_init=ks, b_init=c, max_iter=self.max_iter_kinetics)
            # Calcul du log-vraisemblance
            for z in range(0, n_components):
                vect_binom[:, z] = self.negative_binomial_pmf(data, ks_new[z], c_new)
            basins = np.argmax(vect_binom, 1)
            weights = np.sum(vect_binom, 0) / np.sum(vect_binom)
            log_likelihood = np.sum(np.log(np.max(vect_binom, 1) + 1e-16))
            log_likelihoods.append(log_likelihood)

            if len(log_likelihoods) > 1:
                if len(np.unique(basins)) < n_components:
                    return weights, ks, c, -np.inf
                if (log_likelihoods[-1] - log_likelihoods[-2]) < 0: 
                    ks_new, c_new = np.linspace(np.min(ks)/2, np.max(ks)*2, n_components), c*2
                    n_iter += 1
                else:
                    n_iter = 0
                if (n_iter > 2) or (log_likelihoods[-1] - log_likelihoods[-2] < self.tol):
                    break
            ks, c = ks_new[:], c_new
        return weights, ks, c, log_likelihoods[-1]
    

    def fit(self, data, vect_t, seuil):

        best_bic = np.inf
        best_params = None
        kt, c_init = infer_kinetics_temporal(data, vect_t, seuil, max_iter=self.max_iter_kinetics)
        tmp = np.argsort(kt)
        kt = kt[tmp]
        T = len(np.unique(vect_t))

        for n_components in range(self.min_components, self.max_components+1):
            n = 1/T
            m, M = np.min(kt), np.max(kt)
            ks_init = np.ones(n_components)
            while (len(np.unique((ks_init/c_init).astype(int))) < n_components) and (n > 1/data.size):
                mn = min(m, np.quantile(seuil + data*c_init, n))
                Mn = max(M, np.quantile(seuil + data*c_init, 1-n))
                ks_init = np.linspace(mn, Mn, n_components)
                n *= .5
            ks, c = ks_init.copy(), c_init
            weights, ks, c, log_likelihood = self.refine_kinetics(data, n_components, ks, c, seuil)
            
            num_params = n_components + 1
            bic = num_params*np.log(data.size) - 2*log_likelihood
            
            if bic < best_bic:
                best_bic = bic
                best_params = (weights, ks, c, n_components)
        
        self.best_model = best_params
        return best_params
    

    def predict_proba(self, data, seuil):
        if self.best_model is None:
            print("WARNING : inference did not work")
            return np.ones(2), np.ones((len(data), 2)), np.ones(2), 1
        
        weights, ks, c, n_components = self.best_model
        tmp = np.argsort(ks)
        ks = ks[tmp]
        weights = weights[tmp]

        ### Check that we have no useless basins and infer again if it's the case
        z, modif = 0, []
        ks_new = ks.copy()
        while z < n_components-1:
            if abs(ks[z] - ks[z+1]) < 2*c:
                # Modify ks_new for new inference
                ks_new[z] = np.mean(ks[z:z+2])
                if z+2 <= n_components-1:
                    if abs(ks[z+2] - ks[z+1]) < c:
                        ks_new[z+2] = np.mean(ks[z+1:z+3])
                ks_new = np.delete(ks_new, z+1)
                ks = np.delete(ks, z+1)
                weights = np.delete(weights, z+1)
                weights /= np.sum(weights)
                n_components -= 1
                modif = 1
            else: z += 1
        if modif:
            weights, ks, c, _ = self.refine_kinetics(data, n_components, ks_new, c, seuil)
        
        ### Double if one single component for technical reasons
        if n_components == 1:
            print("WARNING : one component optimal")
            weight_tmp, ks_tmp, c, n_components = weights[0], ks[0], c, 2
            weights, ks = weight_tmp * np.ones(n_components), ks_tmp * np.ones(n_components)

        ### Compute final probabilities
        probabilities = np.zeros((len(data), n_components))
        for z in range(n_components):
            probabilities[:, z] = self.negative_binomial_pmf(data, ks[z], c)
        probabilities /= (probabilities.sum(axis=1, keepdims=True) + 1e-16)

        return weights, probabilities, ks, c
    

def binarize_mrnas(data_rna, vect_t, max_iter_kinetics=2000, seuil=1e-3, verb=True):

        # Get kinetic parameters
        N_cells, G_tot = data_rna.shape
        frequency_modes = np.ones_like(data_rna, dtype="float")
        ks = []
        c = np.ones(G_tot)
        n_components = 0
        kinetics = NegativeBinomialMixtureEM(min_components=2, max_components=2, max_iter_kinetics=max_iter_kinetics)
        for g in range(1, G_tot):
            x = data_rna[:, g]
            kinetics.fit(x, vect_t, seuil)
            weightg, probag, kg, cg = kinetics.predict_proba(x, seuil)
            ks.append(kg)
            ## Compute associated modes
            frequency_modes[:, g] = np.sum(kg * probag, axis=1)
            if verb: print('Gene {} calibrated...'.format(g), kg, cg)
            c[g] = cg
            if len(kg) > n_components:
                n_components = len(kg)
        
        return frequency_modes