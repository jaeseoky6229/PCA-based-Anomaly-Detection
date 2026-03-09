import numpy as np
from scipy.stats import chi2, genpareto


# =============================================================================
# Residual statistic
# =============================================================================
def compute_residual_statistic(X_missing, X_recon, mode='rspe', return_df=False):
    """
    Compute residual-based monitoring statistic.

    Parameters
    ----------
    X_missing : np.ndarray, shape (n_samples, n_features)
        Input data with possible NaNs.
    X_recon : np.ndarray, shape (n_samples, n_features)
        Reconstructed data.
    mode : str
        'rspe' : RSPE = sum(residual^2) / n_obs
        'spe'  : SPE  = sum(residual^2)
    return_df : bool
        Whether to return observed dimension for each sample.

    Returns
    -------
    stats : np.ndarray, shape (n_samples,)
        Monitoring statistic for each sample.
    dfs : np.ndarray, optional
        Observed dimensions for each sample.
    """
    X_missing = np.asarray(X_missing, dtype=float)
    X_recon = np.asarray(X_recon, dtype=float)

    n_samples = X_missing.shape[0]
    stats = np.full(n_samples, np.nan)
    dfs = np.zeros(n_samples, dtype=int)

    for i in range(n_samples):
        obs = ~np.isnan(X_missing[i])
        n_obs = int(obs.sum())
        dfs[i] = n_obs

        if n_obs == 0:
            continue

        residual = X_missing[i, obs] - X_recon[i, obs]
        spe = np.sum(residual ** 2)

        if mode == 'rspe':
            stats[i] = spe / n_obs
        elif mode == 'spe':
            stats[i] = spe
        else:
            raise ValueError("mode must be 'rspe' or 'spe'")

    if return_df:
        return stats, dfs
    return stats


def compute_single_statistic(row_missing, row_recon, mode='rspe'):
    """
    Compute monitoring statistic for a single sample.
    """
    row_missing = np.asarray(row_missing, dtype=float)
    row_recon = np.asarray(row_recon, dtype=float)

    obs = ~np.isnan(row_missing)
    n_obs = int(obs.sum())

    if n_obs == 0:
        return np.nan, 0

    residual = row_missing[obs] - row_recon[obs]
    spe = np.sum(residual ** 2)

    if mode == 'rspe':
        stat = spe / n_obs
    elif mode == 'spe':
        stat = spe
    else:
        raise ValueError("mode must be 'rspe' or 'spe'")

    return stat, n_obs


# =============================================================================
# Threshold fitting helpers
# =============================================================================
def _valid_stats(train_stats):
    train_stats = np.asarray(train_stats, dtype=float)
    valid = train_stats[~np.isnan(train_stats)]
    valid = valid[np.isfinite(valid)]
    return valid


def scaled_chi2_thresholds(train_stats, alphas=(0.95, 0.99)):
    """
    Fit scaled chi-square by moment matching:
        Y ~ a * chi2(nu)

    E[Y] = a * nu
    Var[Y] = 2 * a^2 * nu
    """
    valid = _valid_stats(train_stats)
    valid = valid[valid > 0]

    if len(valid) < 2:
        thresholds = {a: np.nan for a in alphas}
        params = {'nu': np.nan, 'scale': np.nan, 'mean': np.nan, 'var': np.nan}
        return thresholds, params

    m = np.mean(valid)
    v = np.var(valid, ddof=1)

    if m <= 0 or v <= 0:
        thresholds = {a: np.nan for a in alphas}
        params = {'nu': np.nan, 'scale': np.nan, 'mean': m, 'var': v}
        return thresholds, params

    nu = 2 * (m ** 2) / v
    scale = v / (2 * m)

    thresholds = {a: scale * chi2.ppf(a, nu) for a in alphas}
    params = {'nu': nu, 'scale': scale, 'mean': m, 'var': v}
    return thresholds, params


def empirical_thresholds(train_stats, alphas=(0.95, 0.99)):
    """
    Empirical quantile thresholds.
    """
    valid = _valid_stats(train_stats)

    if len(valid) < 2:
        thresholds = {a: np.nan for a in alphas}
        params = {'n_valid': len(valid)}
        return thresholds, params

    thresholds = {a: np.quantile(valid, a) for a in alphas}
    params = {'n_valid': len(valid)}
    return thresholds, params


def evt_thresholds(train_stats, alphas=(0.95, 0.99), tail_fraction=0.1, min_exceedances=15):
    """
    EVT-POT thresholds using Generalized Pareto Distribution (GPD).

    Parameters
    ----------
    train_stats : array-like
    alphas : tuple
    tail_fraction : float
        Upper tail fraction used to define threshold u.
    min_exceedances : int
        Minimum number of exceedances required for stable fitting.

    Returns
    -------
    thresholds : dict
    params : dict
    """
    valid = _valid_stats(train_stats)
    valid = valid[valid > 0]

    if len(valid) < max(30, min_exceedances + 5):
        thresholds, params_emp = empirical_thresholds(valid, alphas=alphas)
        params = {
            'fallback': 'empirical',
            'tail_fraction': tail_fraction,
            'n_valid': len(valid),
            'params_empirical': params_emp
        }
        return thresholds, params

    u = np.quantile(valid, 1 - tail_fraction)
    exceedances = valid[valid > u] - u
    n_exc = len(exceedances)
    n_total = len(valid)

    if n_exc < min_exceedances:
        thresholds, params_emp = empirical_thresholds(valid, alphas=alphas)
        params = {
            'fallback': 'empirical',
            'tail_fraction': tail_fraction,
            'u': u,
            'n_valid': n_total,
            'n_exceedances': n_exc,
            'params_empirical': params_emp
        }
        return thresholds, params

    try:
        c, loc, scale = genpareto.fit(exceedances, floc=0)

        thresholds = {}
        for alpha in alphas:
            q_tail = 1.0 - (1.0 - alpha) / tail_fraction

            if q_tail <= 0 or q_tail >= 1:
                thresholds[alpha] = np.quantile(valid, alpha)
            else:
                thresholds[alpha] = u + genpareto.ppf(q_tail, c, loc=loc, scale=scale)

        params = {
            'tail_fraction': tail_fraction,
            'u': u,
            'shape': c,
            'loc': loc,
            'scale': scale,
            'n_valid': n_total,
            'n_exceedances': n_exc,
            'fallback': None
        }
        return thresholds, params

    except Exception:
        thresholds, params_emp = empirical_thresholds(valid, alphas=alphas)
        params = {
            'fallback': 'empirical_exception',
            'tail_fraction': tail_fraction,
            'u': u,
            'n_valid': n_total,
            'n_exceedances': n_exc,
            'params_empirical': params_emp
        }
        return thresholds, params


def conformal_thresholds(calib_stats, alphas=(0.95, 0.99)):
    """
    Inductive conformal thresholds using calibration residuals.

    For target coverage alpha:
        threshold = sorted_scores[ ceil((n + 1) * alpha) - 1 ]
    """
    valid = _valid_stats(calib_stats)

    if len(valid) < 2:
        thresholds = {a: np.nan for a in alphas}
        params = {'n_calib': len(valid)}
        return thresholds, params

    sorted_scores = np.sort(valid)
    n = len(sorted_scores)

    thresholds = {}
    for alpha in alphas:
        k = int(np.ceil((n + 1) * alpha))
        idx = min(max(k - 1, 0), n - 1)
        thresholds[alpha] = sorted_scores[idx]

    params = {'n_calib': n}
    return thresholds, params


# =============================================================================
# Unified threshold interface
# =============================================================================
def fit_thresholds(train_stats, method='scaled_chi2', alphas=(0.95, 0.99), **kwargs):
    """
    Unified threshold fitting interface.

    Supported methods
    -----------------
    - 'scaled_chi2'
    - 'empirical'
    - 'evt'
    - 'conformal'
    """
    if method == 'scaled_chi2':
        thresholds, params = scaled_chi2_thresholds(train_stats, alphas=alphas)

    elif method == 'empirical':
        thresholds, params = empirical_thresholds(train_stats, alphas=alphas)

    elif method == 'evt':
        thresholds, params = evt_thresholds(
            train_stats,
            alphas=alphas,
            tail_fraction=kwargs.get('tail_fraction', 0.1),
            min_exceedances=kwargs.get('min_exceedances', 15)
        )

    elif method == 'conformal':
        thresholds, params = conformal_thresholds(train_stats, alphas=alphas)

    else:
        raise ValueError(
            "method must be one of: 'scaled_chi2', 'empirical', 'evt', 'conformal'"
        )

    return {
        'method': method,
        'thresholds': thresholds,
        'fit_params': params
    }


# =============================================================================
# Detection helpers
# =============================================================================
def detect_with_thresholds(stats, thresholds):
    """
    Compare monitoring statistics with threshold(s).

    Parameters
    ----------
    stats : np.ndarray
    thresholds : scalar or np.ndarray

    Returns
    -------
    flags : np.ndarray of bool
    """
    stats = np.asarray(stats, dtype=float)

    if np.isscalar(thresholds):
        flags = stats > thresholds
        flags[np.isnan(stats)] = False
        return flags

    thresholds = np.asarray(thresholds, dtype=float)
    flags = np.full(stats.shape, False, dtype=bool)
    valid = ~np.isnan(stats) & ~np.isnan(thresholds)
    flags[valid] = stats[valid] > thresholds[valid]
    return flags


def single_sample_stat_and_threshold(row_missing, row_recon,
                                     statistic_mode='rspe',
                                     threshold_info=None,
                                     alphas=(0.95, 0.99)):
    """
    Compute statistic and thresholds for a single sample.

    Parameters
    ----------
    row_missing : np.ndarray
    row_recon : np.ndarray
    statistic_mode : str
        'rspe' or 'spe'
    threshold_info : dict
        Output of fit_thresholds(...)
    alphas : tuple

    Returns
    -------
    result : dict
        {
            'stat': float,
            'df': int,
            'thresholds': {alpha: value},
            'flags': {alpha: bool}
        }
    """
    stat, df = compute_single_statistic(row_missing, row_recon, mode=statistic_mode)

    if threshold_info is None:
        raise ValueError("threshold_info is required")

    thresholds = {}
    for a in alphas:
        thresholds[a] = threshold_info['thresholds'].get(a, np.nan)

    flags = {a: (False if np.isnan(stat) or np.isnan(thresholds[a]) else stat > thresholds[a])
             for a in alphas}

    return {
        'stat': stat,
        'df': df,
        'thresholds': thresholds,
        'flags': flags
    }
