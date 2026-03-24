import numpy as np
from scipy import linalg

def ppca_em(X, k, max_iters=100, tol=1e-5, min_sigma2=1e-8):
    """
    PPCA using a simple EM-style iterative procedure with missing values.

    Parameters
    ----------
    X : np.ndarray
        Data matrix of shape (n_samples, n_features), missing values as np.nan
    k : int
        Number of latent dimensions
    max_iters : int
        Maximum number of iterations
    tol : float
        Convergence tolerance
    min_sigma2 : float
        Lower bound for sigma^2 to avoid numerical issues

    Returns
    -------
    X_filled : np.ndarray
        Imputed data matrix
    W : np.ndarray
        Loading matrix of shape (d, k)
    sigma2 : float
        Isotropic noise variance
    mu : np.ndarray
        Mean vector of shape (d,)
    """
    X = np.asarray(X, dtype=float)
    n, d = X.shape

    missing_mask = np.isnan(X)
    observed_mask = ~missing_mask

    # ---- Initialization ----
    X_filled = X.copy()
    mu = np.nanmean(X, axis=0)

    # If a whole column is NaN, fallback to zero
    mu = np.where(np.isnan(mu), 0.0, mu)

    X_filled[missing_mask] = np.take(mu, np.where(missing_mask)[1])

    X_centered = X_filled - mu

    # SVD initialization
    U, s, Vt = linalg.svd(X_centered, full_matrices=False)

    k_eff = min(k, d, len(s))
    W = Vt.T[:, :k_eff] * s[:k_eff]

    if k_eff < k:
        W_pad = np.zeros((d, k))
        W_pad[:, :k_eff] = W
        W = W_pad
    else:
        W = W[:, :k]

    if k < len(s):
        sigma2 = np.mean(s[k:]**2) / max(n, 1)
    else:
        sigma2 = np.var(X_centered) * 0.01

    sigma2 = max(float(sigma2), min_sigma2)

    # ---- EM iterations ----
    for _ in range(max_iters):
        W_prev = W.copy()
        sigma2_prev = sigma2
        mu_prev = mu.copy()

        # update mean from filled data
        mu = X_filled.mean(axis=0)
        X_centered = X_filled - mu

        # E-step
        M = W.T @ W + sigma2 * np.eye(k)
        M_inv = linalg.inv(M)

        Ez = (M_inv @ W.T @ X_centered.T).T                     # (n, k)
        Ezz = sigma2 * M_inv[np.newaxis, :, :] + \
              Ez[:, :, np.newaxis] * Ez[:, np.newaxis, :]      # (n, k, k)

        # M-step: update W
        sum_Ezz = np.sum(Ezz, axis=0)                          # (k, k)
        sum_xz = X_centered.T @ Ez                             # (d, k)
        W = sum_xz @ linalg.inv(sum_Ezz)

        # update sigma2
        term1 = np.sum(X_centered**2)
        term2 = 2.0 * np.sum(Ez * (X_centered @ W))
        term3 = np.sum([np.trace(Ezz[i] @ (W.T @ W)) for i in range(n)])
        sigma2 = (term1 - term2 + term3) / (n * d)
        sigma2 = max(float(sigma2), min_sigma2)

        # update missing values using conditional mean in data space
        X_model_mean = mu + Ez @ W.T
        X_filled[missing_mask] = X_model_mean[missing_mask]

        # convergence check
        dW = np.linalg.norm(W - W_prev)
        ds = abs(sigma2 - sigma2_prev)
        dm = np.linalg.norm(mu - mu_prev)

        if dW < tol and ds < tol and dm < tol:
            break

    return X_filled, W, sigma2, mu


def ppca_posterior_mean(row, W, sigma2, mu):
    """
    Posterior mean of latent variable for one sample with missing values.

    E[z | t_obs] = (W_o^T W_o + sigma2 I)^(-1) W_o^T (t_obs - mu_obs)
    """
    row = np.asarray(row, dtype=float)
    mu = np.asarray(mu, dtype=float)

    d, k = W.shape
    obs_idx = ~np.isnan(row)

    if not np.any(obs_idx):
        return np.zeros(k), obs_idx

    Wo = W[obs_idx, :]
    xo = row[obs_idx] - mu[obs_idx]
    M_i = Wo.T @ Wo + sigma2 * np.eye(k)
    Ez = linalg.solve(M_i, Wo.T @ xo, assume_a='pos')

    return Ez, obs_idx


def ppca_reconstruct_latent_mean(row, W, sigma2, mu):
    """
    Reconstruction using:
        mu + W E[z | t]
    This is the skewed PPCA latent-mean reconstruction discussed in the paper.
    """
    Ez, _ = ppca_posterior_mean(row, W, sigma2, mu)
    x_recon = mu + W @ Ez
    return x_recon, Ez


def ppca_reconstruct_optimal(row, W, sigma2, mu):
    """
    Optimal reconstruction from the conditional latent mean
    according to Tipping & Bishop (1999), Sec. 3.4:

        t_hat = W (W^T W)^(-1) M E[z|t] + mu

    For missing-data case, use observed-part posterior:
        E[z|t_obs] = M_i^(-1) W_o^T (t_obs - mu_obs)
        M_i = W_o^T W_o + sigma2 I

    Then:
        t_hat = mu + W (W^T W)^(-1) M_i E[z|t_obs]
              = mu + W (W^T W)^(-1) W_o^T (t_obs - mu_obs)
    """
    row = np.asarray(row, dtype=float)
    mu = np.asarray(mu, dtype=float)

    d, k = W.shape
    obs_idx = ~np.isnan(row)

    if not np.any(obs_idx):
        return mu.copy(), np.zeros(k)

    Wo = W[obs_idx, :]
    xo = row[obs_idx] - mu[obs_idx]

    M_i = Wo.T @ Wo + sigma2 * np.eye(k)
    Ez = linalg.solve(M_i, Wo.T @ xo, assume_a='pos')

    WT_W = W.T @ W
    WT_W_inv = linalg.pinv(WT_W)

    # skew-corrected optimal reconstruction
    x_recon = mu + W @ (WT_W_inv @ (M_i @ Ez))

    return x_recon, Ez


def ppca_reconstruct_matrix(X_missing, W, sigma2, mu, method='optimal', return_latent=False):
    """
    Reconstruct matrix row-wise.

    method:
        - 'latent_mean' : mu + W E[z|t_obs]
        - 'optimal'     : skew-corrected PPCA reconstruction
    """
    X_missing = np.asarray(X_missing, dtype=float)
    n, d = X_missing.shape
    k = W.shape[1]

    X_recon = np.empty((n, d), dtype=float)
    Z_mean = np.zeros((n, k), dtype=float)

    for i in range(n):
        if method == 'latent_mean':
            x_rec, ez = ppca_reconstruct_latent_mean(X_missing[i], W, sigma2, mu)
        elif method == 'optimal':
            x_rec, ez = ppca_reconstruct_optimal(X_missing[i], W, sigma2, mu)
        else:
            raise ValueError("method must be 'latent_mean' or 'optimal'")

        X_recon[i] = x_rec
        Z_mean[i] = ez

    if return_latent:
        return X_recon, Z_mean
    return X_recon
