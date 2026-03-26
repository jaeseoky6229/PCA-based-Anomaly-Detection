import numpy as np
from dataclasses import dataclass
from scipy.stats import chi2


@dataclass
class PCTResult:
    q_values: np.ndarray
    ucls: np.ndarray
    alarms: np.ndarray
    dfs: np.ndarray
    scales: np.ndarray
    n_obs_vars: np.ndarray


class PCTChart:
    """
    PCT chart for multivariate monitoring with missing values
    based on Kim, Cho, and Lim (2022), Journal of the Korean Statistical Society.

    Optional covariance shrinkage is applied to the Phase I covariance estimator:
        Sigma_shrunk = lam * Sigma_hat + (1 - lam) * mu * I

    Parameters
    ----------
    alpha : float
        Significance level for control limit.
    min_n1 : int
        Minimum number of observed Phase I values per variable.
    eps : float
        Small numerical constant.
    use_shrinkage : bool
        Whether to apply covariance shrinkage to the Phase I covariance matrix.
    shrink_lambda : float
        Shrinkage intensity in [0, 1].
        1.0 means no shrinkage, smaller values mean stronger shrinkage.
    shrink_target : str or float
        If 'trace', use mu = trace(Sigma_hat)/p.
        If float, use that fixed value as mu.
    enforce_pd : bool
        If True, add a small diagonal inflation when the shrunk covariance is not PD.
    pd_tol : float
        Minimum eigenvalue threshold used when enforce_pd=True.
    """

    def __init__(
        self,
        alpha=0.05,
        min_n1=6,
        eps=1e-12,
        use_shrinkage=False,
        shrink_lambda=0.9,
        shrink_target="trace",
        enforce_pd=True,
        pd_tol=1e-8,
    ):
        self.alpha = alpha
        self.min_n1 = min_n1
        self.eps = eps

        self.use_shrinkage = use_shrinkage
        self.shrink_lambda = shrink_lambda
        self.shrink_target = shrink_target
        self.enforce_pd = enforce_pd
        self.pd_tol = pd_tol

        self.fitted_ = False

    def fit(self, X):
        X = self._to_numpy(X)
        n, p = X.shape

        self.n_phase1_ = n
        self.p_ = p

        # Per-variable counts and means
        self.n1j_ = np.sum(~np.isnan(X), axis=0).astype(int)
        self.mean1_ = np.nanmean(X, axis=0)

        # Per-variable sample variances s_j^{*2}, eq. (5)
        self.var1_ = np.full(p, np.nan, dtype=float)
        for j in range(p):
            xj = X[:, j]
            obs = ~np.isnan(xj)
            nj = obs.sum()
            if nj >= 2:
                self.var1_[j] = np.var(xj[obs], ddof=1)

        # Pairwise complete counts and pairwise covariance s^*_{ij}, eq. (6)
        self.n1ij_ = np.zeros((p, p), dtype=int)
        self.cov1_ = np.full((p, p), np.nan, dtype=float)

        for i in range(p):
            for j in range(i, p):
                obs = ~np.isnan(X[:, i]) & ~np.isnan(X[:, j])
                nij = int(obs.sum())
                self.n1ij_[i, j] = self.n1ij_[j, i] = nij

                if i == j:
                    self.cov1_[i, j] = self.var1_[i]
                else:
                    if nij >= 2:
                        xi = X[obs, i]
                        xj = X[obs, j]
                        cov_ij = np.cov(xi, xj, ddof=1)[0, 1]
                        self.cov1_[i, j] = self.cov1_[j, i] = cov_ij

        # Basic validity checks
        invalid = np.where(self.n1j_ < self.min_n1)[0]
        if len(invalid) > 0:
            print(
                f"[Warning] Some variables have n1j < {self.min_n1}: "
                f"indices={invalid.tolist()}. "
                "Moment approximation may be unstable."
            )

        # Build covariance actually used in scoring
        self._prepare_covariance_for_scoring()

        self.fitted_ = True
        return self

    def _prepare_covariance_for_scoring(self):
        """
        Prepare covariance matrix used in scoring.
        - Fill NaN entries conservatively
        - Symmetrize
        - Optionally apply linear shrinkage
        - Optionally enforce positive definiteness
        """
        p = self.p_
        S = np.array(self.cov1_, dtype=float, copy=True)

        # Diagonal fallback from var1_
        for j in range(p):
            if not np.isfinite(S[j, j]):
                S[j, j] = self.var1_[j]

        # Off-diagonal NaN -> 0
        nan_mask = ~np.isfinite(S)
        S[nan_mask] = 0.0

        # Symmetrize
        S = 0.5 * (S + S.T)

        # Diagonal safeguard
        for j in range(p):
            if (not np.isfinite(S[j, j])) or (S[j, j] <= self.eps):
                S[j, j] = self.eps

        self.cov1_raw_filled_ = S.copy()

        if self.use_shrinkage:
            lam = float(self.shrink_lambda)
            lam = min(max(lam, 0.0), 1.0)

            if isinstance(self.shrink_target, str):
                if self.shrink_target.lower() == "trace":
                    mu = np.trace(S) / p
                else:
                    raise ValueError("shrink_target must be 'trace' or a float.")
            else:
                mu = float(self.shrink_target)

            if not np.isfinite(mu) or mu <= 0:
                mu = max(np.nanmean(np.diag(S)), self.eps)

            S_used = lam * S + (1.0 - lam) * mu * np.eye(p)
        else:
            S_used = S.copy()

        # Optional PD enforcement
        if self.enforce_pd:
            eigvals = np.linalg.eigvalsh(S_used)
            min_eig = np.min(eigvals)
            if min_eig < self.pd_tol:
                bump = self.pd_tol - min_eig
                S_used = S_used + bump * np.eye(p)

        self.cov1_used_ = S_used
        self.var1_used_ = np.diag(S_used).copy()

    def score_samples(self, X):
        self._check_is_fitted()
        X = self._to_numpy(X)

        q_values = []
        ucls = []
        alarms = []
        dfs = []
        scales = []
        n_obs_vars = []

        for x in X:
            q, c_star, d_star, ucl, nobs = self._score_one(x)
            q_values.append(q)
            ucls.append(ucl)
            alarms.append(q > ucl if np.isfinite(ucl) else False)
            dfs.append(d_star)
            scales.append(c_star)
            n_obs_vars.append(nobs)

        return PCTResult(
            q_values=np.array(q_values, dtype=float),
            ucls=np.array(ucls, dtype=float),
            alarms=np.array(alarms, dtype=bool),
            dfs=np.array(dfs, dtype=float),
            scales=np.array(scales, dtype=float),
            n_obs_vars=np.array(n_obs_vars, dtype=int),
        )

    def predict(self, X):
        return self.score_samples(X).alarms.astype(int)

    def decision_function(self, X):
        return self.score_samples(X).q_values

    def _score_one(self, x):
        obs = ~np.isnan(x)
        idx = np.where(obs)[0]
        p_obs = len(idx)

        if p_obs == 0:
            return np.nan, np.nan, np.nan, np.nan, 0

        # Exclude variables with unusable variance or too-small n1j
        valid_idx = []
        for j in idx:
            if (
                np.isfinite(self.mean1_[j])
                and np.isfinite(self.var1_used_[j])
                and self.var1_used_[j] > self.eps
                and self.n1j_[j] > 5
            ):
                valid_idx.append(j)

        idx = np.array(valid_idx, dtype=int)
        p_obs = len(idx)

        if p_obs == 0:
            return np.nan, np.nan, np.nan, np.nan, 0

        # Individual observation monitoring => n2j = 1
        n1 = self.n1j_[idx].astype(float)
        a_star = n1 / (n1 + 1.0)

        qj = ((self.mean1_[idx] - x[idx]) ** 2) / self.var1_used_[idx]
        q_star = np.mean(a_star * qj)  # eq. (7)

        # E(Q*), eq. (9)
        E_q = np.mean((n1 - 1.0) / (n1 - 3.0))

        # Var(Q*), eq. (14)
        term1 = np.sum(
            2.0 * ((n1 - 1.0) / (n1 - 3.0)) ** 2 * ((n1 - 2.0) / (n1 - 5.0))
        ) / (p_obs ** 2)

        term2_sum = 0.0
        for a in range(p_obs):
            i = idx[a]
            for b in range(a + 1, p_obs):
                j = idx[b]

                s2i = self.var1_used_[i]
                s2j = self.var1_used_[j]
                covij = self.cov1_used_[i, j]

                if not np.isfinite(covij):
                    covij = 0.0

                n1i = self.n1j_[i]
                n1j = self.n1j_[j]
                n1ij = self.n1ij_[i, j]
                n2ij = 1  # both observed in current sample

                tau_hat = covij * (n1ij / (n1i * n1j) + n2ij / 1.0)

                pair_term = (
                    (1.0 + (2.0 * a_star[a] * a_star[b] * (tau_hat ** 2)) / (s2i * s2j + self.eps))
                    * ((n1i + 1.0) / (n1i - 1.0))
                    * ((n1j + 1.0) / (n1j - 1.0))
                    - ((n1i - 1.0) / (n1i - 3.0)) * ((n1j - 1.0) / (n1j - 3.0))
                )
                term2_sum += pair_term

        term2 = 2.0 * term2_sum / (p_obs ** 2)
        Var_q = term1 + term2

        if (not np.isfinite(E_q)) or (not np.isfinite(Var_q)) or Var_q <= self.eps:
            return q_star, np.nan, np.nan, np.nan, p_obs

        c_star = Var_q / (2.0 * E_q)
        d_star = 2.0 * (E_q ** 2) / Var_q

        if c_star <= 0 or d_star <= 0 or (not np.isfinite(c_star)) or (not np.isfinite(d_star)):
            return q_star, np.nan, np.nan, np.nan, p_obs

        ucl = c_star * chi2.ppf(1.0 - self.alpha, d_star)
        return q_star, c_star, d_star, ucl, p_obs

    @staticmethod
    def _to_numpy(X):
        if hasattr(X, "values"):
            X = X.values
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return X

    def _check_is_fitted(self):
        if not self.fitted_:
            raise RuntimeError("PCTChart is not fitted yet. Call fit(X_phase1) first.")
