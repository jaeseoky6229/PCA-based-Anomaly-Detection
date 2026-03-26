import pandas as pd
import numpy as np
from random_nans import random_nans
from inject_random_missing_two_modes import inject_random_missing_two_modes
from sklearn.preprocessing import StandardScaler
from ppca import ppca_em, ppca_reconstruct_matrix
from select_retained_nPC import select_retained_nPC
from pct import PCTChart


"""
================================================================================
PCT summary by missing ratio(alpha=0.05, b=0.001, MCAR, noise=seed, missing=seed)
================================================================================
missing_ratio=0.1 | FAR = 0.0533 ± 0.0186 | DR = 1.0000 ± 0.0000
missing_ratio=0.2 | FAR = 0.0725 ± 0.0266 | DR = 0.9981 ± 0.0061
missing_ratio=0.3 | FAR = 0.0867 ± 0.0312 | DR = 0.9881 ± 0.0118
missing_ratio=0.4 | FAR = 0.0912 ± 0.0287 | DR = 0.9681 ± 0.0196
missing_ratio=0.5 | FAR = 0.0983 ± 0.0303 | DR = 0.9256 ± 0.0299
missing_ratio=0.6 | FAR = 0.0958 ± 0.0339 | DR = 0.8525 ± 0.0427


================================================================================
PCT summary by missing ratio(alpha=0.05, b=0.001, MCAR, noise=constant, missing=seed)
================================================================================
missing_ratio=0.1 | FAR = 0.0579 ± 0.0280 | DR = 1.0000 ± 0.0000
missing_ratio=0.2 | FAR = 0.0829 ± 0.0334 | DR = 0.9981 ± 0.0046
missing_ratio=0.3 | FAR = 0.0963 ± 0.0309 | DR = 0.9887 ± 0.0107
missing_ratio=0.4 | FAR = 0.0983 ± 0.0234 | DR = 0.9694 ± 0.0174
missing_ratio=0.5 | FAR = 0.1046 ± 0.0253 | DR = 0.9244 ± 0.0331
missing_ratio=0.6 | FAR = 0.0892 ± 0.0254 | DR = 0.8588 ± 0.0398


================================================================================
PCT summary by missing ratio(alpha=0.05, b=0.001, MCAR, noise=seed, missing=constant)
================================================================================
missing_ratio=0.1 | FAR = 0.0679 ± 0.0197 | DR = 1.0000 ± 0.0000
missing_ratio=0.2 | FAR = 0.0800 ± 0.0241 | DR = 1.0000 ± 0.0000
missing_ratio=0.3 | FAR = 0.0896 ± 0.0274 | DR = 1.0000 ± 0.0000
missing_ratio=0.4 | FAR = 0.1108 ± 0.0285 | DR = 0.9631 ± 0.0111
missing_ratio=0.5 | FAR = 0.1229 ± 0.0366 | DR = 0.9437 ± 0.0095
missing_ratio=0.6 | FAR = 0.1146 ± 0.0302 | DR = 0.8956 ± 0.0142


================================================================================
PCT summary by missing ratio(alpha=0.05, b=0.0001, MCAR, noise=seed, missing=seed)
================================================================================
missing_ratio=0.1 | FAR = 0.1663 ± 0.0383 | DR = 1.0000 ± 0.0000
missing_ratio=0.2 | FAR = 0.1492 ± 0.0283 | DR = 1.0000 ± 0.0000
missing_ratio=0.3 | FAR = 0.1725 ± 0.0358 | DR = 0.9944 ± 0.0076
missing_ratio=0.4 | FAR = 0.1692 ± 0.0327 | DR = 0.9775 ± 0.0126
missing_ratio=0.5 | FAR = 0.1829 ± 0.0338 | DR = 0.9456 ± 0.0223
missing_ratio=0.6 | FAR = 0.1508 ± 0.0246 | DR = 0.8775 ± 0.0401

================================================================================
PCT summary by missing ratio(alpha=0.05, b=0.01, MCAR, noise=seed, missing=seed)
================================================================================
missing_ratio=0.1 | FAR = 0.0508 ± 0.0209 | DR = 0.1512 ± 0.0851
missing_ratio=0.2 | FAR = 0.0567 ± 0.0238 | DR = 0.1581 ± 0.0752
missing_ratio=0.3 | FAR = 0.0683 ± 0.0308 | DR = 0.1337 ± 0.0899
missing_ratio=0.4 | FAR = 0.0633 ± 0.0233 | DR = 0.1575 ± 0.0868
missing_ratio=0.5 | FAR = 0.0583 ± 0.0179 | DR = 0.1656 ± 0.0539
missing_ratio=0.6 | FAR = 0.0604 ± 0.0195 | DR = 0.1562 ± 0.0481
"""

def transform_SNR(X, b=0.0, rms_input=None, random_state=None):
    rng = np.random.default_rng(random_state)
    X = np.asarray(X, dtype=float)
    
    if rms_input is None:
        rms_cols = np.sqrt(np.nanmean(X**2, axis=0))
    else:
        rms_cols = rms_input
        
    noise = b * rms_cols * rng.normal(0, 1, size=X.shape)
    X_noisy = np.where(np.isnan(X), np.nan, X + noise)
    
    return X_noisy, rms_cols


def missing_pattern(X, method="MCAR", fraction=0.2, random_state=42):
    if method == "MCAR":
        return random_nans(X.copy(), fraction=fraction, random_state=random_state)
    elif method == "mode_selected":
        return inject_random_missing_two_modes(X.copy(), fraction=fraction, random_state=random_state)
    else:
        raise ValueError("method must be 'MCAR' or 'mode_selected'")


path = 'data/df_result(Quasi-linear,0.9,simply-supported,6).csv'
X = pd.read_csv(path, encoding='utf-8')

n_train = 200
n_damaged = 320
n_repeat = 20
b = 0.001
missing_method = 'MCAR'    # MCAR or mode_selected
recon_method = 'optimal'   # 'optimal' or 'latent_mean'

missing_ratio_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

X_train = X.iloc[:n_train, :].copy()
X_test = X.iloc[n_train:, :].copy()

summary_records = []

for missing_ratio in missing_ratio_list:
    for seed in range(n_repeat):
        scaler = StandardScaler()

        # --------------------------------------------------
        # 1) Noise injection
        # --------------------------------------------------
        X_noise_train, rms_train = transform_SNR(X_train, b=b, random_state=seed)
        X_noise_damage, _ = transform_SNR(X_test, b=b, rms_input=rms_train,random_state=seed)
        X_noise = np.vstack([X_noise_train, X_noise_damage])

        # --------------------------------------------------
        # 2) Missing injection
        # --------------------------------------------------
        X_missing = missing_pattern(
            X_noise,
            method=missing_method,
            fraction=missing_ratio,
            random_state=seed
        )

        train_ = X_missing[:n_train, :]
        test_ = X_missing[n_train:, :]

        # --------------------------------------------------
        # 3) Scaling
        # --------------------------------------------------
        scaler.fit(train_)
        train_scaled = scaler.transform(train_)
        test_scaled = scaler.transform(test_)

        n_sample, p = train_scaled.shape
        max_components = min(n_sample - 1, p - 1)

        # --------------------------------------------------
        # 4) Full PPCA for optimal k selection
        # --------------------------------------------------
        _, W_full, sigma2_full, mu_full = ppca_em(train_scaled.copy(), k=max_components)

        top_eigs = np.sort(np.linalg.eigvalsh(W_full.T @ W_full))[::-1] + sigma2_full
        if max_components < p:
            model_eigenvalues = np.concatenate(
                [top_eigs, np.full(p - max_components, sigma2_full)]
            )
        else:
            model_eigenvalues = top_eigs

        model_eigenvalues = np.sort(model_eigenvalues)[::-1]

        k_opt = select_retained_nPC(
            model_eigenvalues,
            nPC_select_algorithm='eigengap'
        )

        if isinstance(k_opt, (list, tuple, np.ndarray)):
            k_opt = int(k_opt[0])
        else:
            k_opt = int(k_opt)

        k_opt = max(1, min(k_opt, max_components))

        # --------------------------------------------------
        # 5) Refit PPCA with optimal dimension
        # --------------------------------------------------
        _, W_opt, sigma2_opt, mu_opt = ppca_em(train_scaled.copy(), k=k_opt)

        # --------------------------------------------------
        # 6) Reconstruction
        # --------------------------------------------------
        train_recon = ppca_reconstruct_matrix(
            train_scaled.copy(),
            W_opt,
            sigma2_opt,
            mu_opt,
            method=recon_method
        )

        test_recon = ppca_reconstruct_matrix(
            test_scaled.copy(),
            W_opt,
            sigma2_opt,
            mu_opt,
            method=recon_method
        )

        # --------------------------------------------------
        # 7) Residual
        # --------------------------------------------------
        train_residual = train_scaled - train_recon
        test_residual = test_scaled - test_recon

        # --------------------------------------------------
        # 8) Split residual data for PCT
        #    train:   0~199
        #    normal:  200~319
        #    damaged: 320~
        # --------------------------------------------------
        n_normal = n_damaged - n_train   # 120

        cols_keep = [0, 2, 3, 4]

        X_res_train = train_residual[:, cols_keep]
        X_res_normal = test_residual[:n_normal, :][:, cols_keep]
        X_res_damaged = test_residual[n_normal:, :][:, cols_keep]

        # --------------------------------------------------
        # 9) PCT fit / score
        # --------------------------------------------------
        pct = PCTChart(alpha=0.05)
        pct.fit(X_res_train)

        res_normal = pct.score_samples(X_res_normal)
        res_damaged = pct.score_samples(X_res_damaged)

        normal_alarm = np.asarray(res_normal.alarms, dtype=bool)
        damaged_alarm = np.asarray(res_damaged.alarms, dtype=bool)

        far = normal_alarm.mean()
        detection_rate = damaged_alarm.mean()

        summary_records.append({
            "missing_ratio": missing_ratio,
            "seed": seed,
            "k_opt": k_opt,
            "far": far,
            "detection_rate": detection_rate,
            "fp": int(normal_alarm.sum()),
            "tn": int((~normal_alarm).sum()),
            "tp": int(damaged_alarm.sum()),
            "fn": int((~damaged_alarm).sum()),
        })

        print(
            f"missing_ratio={missing_ratio:.1f}, seed={seed:02d}, "
            f"k_opt={k_opt}, FAR={far:.4f}, DR={detection_rate:.4f}"
        )

# --------------------------------------------------
# 10) Aggregate mean / std by missing ratio
# --------------------------------------------------
summary_df = pd.DataFrame(summary_records)

agg_df = (
    summary_df
    .groupby("missing_ratio")
    .agg(
        far_mean=("far", "mean"),
        far_std=("far", "std"),
        dr_mean=("detection_rate", "mean"),
        dr_std=("detection_rate", "std"),
        k_opt_mean=("k_opt", "mean"),
        k_opt_std=("k_opt", "std"),
    )
    .reset_index()
)

print("\n" + "=" * 80)
print("PCT summary by missing ratio")
print("=" * 80)

for _, row in agg_df.iterrows():
    print(
        f"missing_ratio={row['missing_ratio']:.1f} | "
        f"FAR = {row['far_mean']:.4f} ± {row['far_std']:.4f} | "
        f"DR = {row['dr_mean']:.4f} ± {row['dr_std']:.4f}"
    )

# 필요시 저장
# summary_df.to_csv("pct_by_seed_summary.csv", index=False)
# agg_df.to_csv("pct_by_missing_ratio_summary.csv", index=False)
