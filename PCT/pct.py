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

    Phase I:
        fit(X_phase1)

    Phase II:
        score_samples(X_phase2)
        predict(X_phase2)

    Notes
    -----
    - Missing values must be np.nan.
    - This implementation is designed for individual monitoring observations.
      Thus for each phase-II sample, n2j = 1 for observed variables.
    - The control limit depends on the missing pattern of each incoming sample.
    """

    def __init__(self, alpha=0.05, min_n1=6, eps=1e-12):
        self.alpha = alpha
        self.min_n1 = min_n1
        self.eps = eps
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

        self.fitted_ = True
        return self

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
                and np.isfinite(self.var1_[j])
                and self.var1_[j] > self.eps
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

        qj = ((self.mean1_[idx] - x[idx]) ** 2) / self.var1_[idx]
        q_star = np.mean(a_star * qj)  # eq. (7), averaged over observed variables

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

                s2i = self.var1_[i]
                s2j = self.var1_[j]
                covij = self.cov1_[i, j]

                if not np.isfinite(covij):
                    covij = 0.0

                # tau_hat_ij from paper (for individual monitoring n2i=n2j=1)
                # tau_hat_ij = covij/(n1i*n1j)*n1ij + covij/(n2i*n2j)*n2ij
                # for one observation in phase II, if both i,j are observed, n2ij = 1
                n1i = self.n1j_[i]
                n1j = self.n1j_[j]
                n1ij = self.n1ij_[i, j]
                n2ij = 1  # both observed in current sample

                tau_hat = covij * (n1ij / (n1i * n1j) + n2ij / (1.0 * 1.0))

                pair_term = (
                    (1.0 + (2.0 * a_star[a] * a_star[b] * (tau_hat ** 2)) / (s2i * s2j + self.eps))
                    * ((n1i + 1.0) / (n1i - 1.0))
                    * ((n1j + 1.0) / (n1j - 1.0))
                    - ((n1i - 1.0) / (n1i - 3.0)) * ((n1j - 1.0) / (n1j - 3.0))
                )
                term2_sum += pair_term

        term2 = 2.0 * term2_sum / (p_obs ** 2)

        Var_q = term1 + term2

        # Numerical guard
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


"""
# Phase I residuals (normal baseline)
X_phase1 = np.array([
    [0.01, -0.02,  0.03,  np.nan],
    [0.00, -0.01,  0.02,  0.01],
    [0.02,  np.nan, 0.01, -0.01],
    [0.01, -0.03,  0.00,  0.00],
    [0.00, -0.02,  0.01,  0.02],
    [0.01, -0.01,  0.02,  0.01],
    [0.02, -0.02,  np.nan, 0.00],
    [0.01, -0.01,  0.01,  0.01],
])

# Phase II residuals to monitor
X_phase2 = np.array([
    [0.01, -0.02,  0.03, 0.00],
    [0.20, -0.25,  0.30, 0.10],
    [np.nan, -0.02, 0.01, 0.00],
])

pct = PCTChart(alpha=0.05)
pct.fit(X_phase1)

result = pct.score_samples(X_phase2)

print("Q*      :", result.q_values)
print("UCL     :", result.ucls)
print("Alarm   :", result.alarms)
print("df      :", result.dfs)
print("scale   :", result.scales)
print("n_obs   :", result.n_obs_vars)
"""