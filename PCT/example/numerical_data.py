import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False

from scipy.stats import probplot


# =========================================================
# 1. Mean shift
# =========================================================
def make_mean_vector(p, mean_type="M0", c=10):
    mu = np.zeros(p)

    if mean_type == "M0":
        return mu
    elif mean_type == "M1":
        mu = c * (1 / np.sqrt(p)) * np.ones(p)
    elif mean_type == "M2":
        mu[0] = c
    elif mean_type == "M3":
        k = p // 2
        mu[:k] = c * np.sqrt(2 / p)
    else:
        raise ValueError(f"Unknown mean_type: {mean_type}")

    return mu


# =========================================================
# 2. Covariance matrix
# =========================================================
def make_covariance_matrix(p, cov_type="C1", rho=0.3, delta=0.3):
    if p < 3:
        raise ValueError("p must be >= 3.")

    base_diag = np.ones(p)
    base_diag[0] = 4
    base_diag[1] = 3
    base_diag[2] = 2
    D = np.diag(base_diag)

    if cov_type == "C1":
        Sigma = D

    elif cov_type == "C2":
        A = np.fromfunction(lambda i, j: rho ** np.abs(i - j), (p, p), dtype=int)
        Sigma = D + (A - np.eye(p))

    elif cov_type == "C3":
        J = np.ones((p, p))
        Sigma = D + delta * J

    else:
        raise ValueError(f"Unknown cov_type: {cov_type}")

    return Sigma


# =========================================================
# 3. Missing injection
# =========================================================
def inject_missing_values(X, missing_rate=0.0, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    X_miss = X.copy()

    if missing_rate <= 0:
        return X_miss

    n, p = X_miss.shape
    total_entries = n * p
    n_missing = int(np.floor(total_entries * missing_rate))

    flat_idx = rng.choice(total_entries, size=n_missing, replace=False)
    rows, cols = np.unravel_index(flat_idx, (n, p))
    X_miss[rows, cols] = np.nan

    return X_miss


# =========================================================
# 4. Single dataset generation
# =========================================================
def generate_single_dataset(
    n1=50,
    n2=50,
    gamma=0.1,
    mean_type="M0",
    cov_type="C1",
    rho=0.3,
    delta=0.3,
    c=10,
    missing_rate=0.0,
    seed=None
):
    rng = np.random.default_rng(seed)

    n = n1 + n2
    p = int(n * gamma)

    mu = make_mean_vector(p=p, mean_type=mean_type, c=c)
    Sigma = make_covariance_matrix(p=p, cov_type=cov_type, rho=rho, delta=delta)

    X1 = rng.multivariate_normal(mean=np.zeros(p), cov=Sigma, size=n1)
    X2 = rng.multivariate_normal(mean=mu, cov=Sigma, size=n2)

    X = np.vstack([X1, X2])
    X = inject_missing_values(X, missing_rate=missing_rate, rng=rng)

    phase = np.array([0] * n1 + [1] * n2)

    return {
        "X": X,
        "phase": phase,
        "mu": mu,
        "Sigma": Sigma,
        "p": p,
        "n1": n1,
        "n2": n2,
        "gamma": gamma,
        "mean_type": mean_type,
        "cov_type": cov_type,
        "missing_rate": missing_rate,
    }


# =========================================================
# 5. Dataset -> DataFrame
# =========================================================
def dataset_to_dataframe(dataset):
    X = dataset["X"]
    phase = dataset["phase"]

    p = X.shape[1]
    columns = [f"x{i+1}" for i in range(p)]
    df = pd.DataFrame(X, columns=columns)
    df.insert(0, "phase", phase)

    return df


# =========================================================
# 6. Summary table
# =========================================================
def summarize_dataframe(df):
    feature_cols = [col for col in df.columns if col != "phase"]

    rows = []
    for col in feature_cols:
        s = df[col]
        rows.append({
            "column": col,
            "n_total": len(s),
            "n_missing": s.isna().sum(),
            "missing_ratio": s.isna().mean(),
            "n_observed": s.notna().sum(),
            "mean": s.mean(),
            "std": s.std(),
            "min": s.min(),
            "25%": s.quantile(0.25),
            "50%": s.quantile(0.50),
            "75%": s.quantile(0.75),
            "max": s.max(),
        })

    return pd.DataFrame(rows)


# =========================================================
# 7. Plot one column
# =========================================================
def plot_single_column_distribution(df, column, save_path=None):
    data_all = df[column].dropna()
    data_phase0 = df.loc[df["phase"] == 0, column].dropna()
    data_phase1 = df.loc[df["phase"] == 1, column].dropna()

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    ax1, ax2, ax3, ax4 = axes.flatten()

    # -------------------------
    # Histogram + KDE
    # -------------------------
    if SEABORN_AVAILABLE:
        sns.histplot(data_all, bins=20, kde=True, color="steelblue", ax=ax1)
    else:
        ax1.hist(data_all, bins=20, color="steelblue", alpha=0.7, edgecolor="black")

    ax1.set_title(f"{column} - Overall Distribution")
    ax1.set_xlabel(column)
    ax1.set_ylabel("Count")

    # -------------------------
    # Phase-wise histogram
    # -------------------------
    bins = 20
    ax2.hist(data_phase0, bins=bins, alpha=0.6, label="Phase 1", color="tab:blue", edgecolor="black")
    ax2.hist(data_phase1, bins=bins, alpha=0.6, label="Phase 2", color="tab:orange", edgecolor="black")
    ax2.set_title(f"{column} - Phase-wise Distribution")
    ax2.set_xlabel(column)
    ax2.set_ylabel("Count")
    ax2.legend()

    # -------------------------
    # Boxplot
    # -------------------------
    box_data = [data_phase0.values, data_phase1.values]
    ax3.boxplot(box_data, labels=["Phase 1", "Phase 2"])
    ax3.set_title(f"{column} - Boxplot by Phase")
    ax3.set_ylabel(column)

    # -------------------------
    # Q-Q plot
    # -------------------------
    if len(data_all) >= 3:
        probplot(data_all, dist="norm", plot=ax4)
        ax4.set_title(f"{column} - Q-Q Plot")
    else:
        ax4.text(0.5, 0.5, "Not enough data for Q-Q plot",
                 ha="center", va="center", transform=ax4.transAxes)
        ax4.set_title(f"{column} - Q-Q Plot")

    missing_count = df[column].isna().sum()
    total_count = len(df[column])
    observed_count = total_count - missing_count

    fig.suptitle(
        f"{column} | total={total_count}, observed={observed_count}, missing={missing_count}",
        fontsize=14
    )
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()
    plt.close()


# =========================================================
# 8. Plot all columns individually
# =========================================================
def plot_all_columns(df, output_dir=None):
    feature_cols = [col for col in df.columns if col != "phase"]

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    for col in feature_cols:
        save_path = None
        if output_dir is not None:
            save_path = os.path.join(output_dir, f"{col}_distribution.png")
        plot_single_column_distribution(df, col, save_path=save_path)


# =========================================================
# 9. Missingness visualization
# =========================================================
def plot_missing_pattern(df, save_path=None):
    feature_df = df.drop(columns=["phase"])

    plt.figure(figsize=(12, 6))
    plt.imshow(feature_df.isna(), aspect="auto", cmap="gray_r", interpolation="nearest")
    plt.colorbar(label="Missing (1=True, 0=False)")
    plt.xlabel("Columns")
    plt.ylabel("Samples")
    plt.title("Missing Data Pattern")
    plt.xticks(ticks=np.arange(feature_df.shape[1]), labels=feature_df.columns, rotation=90)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()
    plt.close()


# =========================================================
# 10. Grid histogram for all columns
# =========================================================
def plot_histograms_grid(df, bins=20, cols_per_row=4, save_path=None):
    feature_cols = [col for col in df.columns if col != "phase"]
    n_features = len(feature_cols)
    n_rows = math.ceil(n_features / cols_per_row)

    fig, axes = plt.subplots(n_rows, cols_per_row, figsize=(4 * cols_per_row, 3 * n_rows))
    axes = np.array(axes).reshape(-1)

    for i, col in enumerate(feature_cols):
        ax = axes[i]
        data = df[col].dropna()

        ax.hist(data, bins=bins, color="steelblue", alpha=0.7, edgecolor="black")
        ax.set_title(col)
        ax.set_xlabel("")
        ax.set_ylabel("")

        missing_ratio = df[col].isna().mean()
        ax.text(
            0.98, 0.95,
            f"miss={missing_ratio:.2f}",
            transform=ax.transAxes,
            ha="right", va="top",
            fontsize=9,
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray")
        )

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    fig.suptitle("Histogram of All Columns", fontsize=16)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()
    plt.close()


# =========================================================
# 11. Grid boxplot for all columns by phase
# =========================================================
def plot_boxplots_grid_by_phase(df, cols_per_row=4, save_path=None):
    feature_cols = [col for col in df.columns if col != "phase"]
    n_features = len(feature_cols)
    n_rows = math.ceil(n_features / cols_per_row)

    fig, axes = plt.subplots(n_rows, cols_per_row, figsize=(4 * cols_per_row, 3.5 * n_rows))
    axes = np.array(axes).reshape(-1)

    for i, col in enumerate(feature_cols):
        ax = axes[i]
        p0 = df.loc[df["phase"] == 0, col].dropna()
        p1 = df.loc[df["phase"] == 1, col].dropna()

        ax.boxplot([p0.values, p1.values], labels=["P1", "P2"])
        ax.set_title(col)

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    fig.suptitle("Boxplots of All Columns by Phase", fontsize=16)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()
    plt.close()


# =========================================================
# 12. Run full analysis for one missing rate
# =========================================================
def run_distribution_analysis_for_missing_rate(
    missing_rate,
    n1=50,
    n2=50,
    gamma=0.3,
    mean_type="M1",
    cov_type="C2",
    rho=0.5,
    delta=0.3,
    c=10,
    seed=42,
    output_root="distribution_results",
    save_figures=True
):
    dataset = generate_single_dataset(
        n1=n1,
        n2=n2,
        gamma=gamma,
        mean_type=mean_type,
        cov_type=cov_type,
        rho=rho,
        delta=delta,
        c=c,
        missing_rate=missing_rate,
        seed=seed
    )

    df = dataset_to_dataframe(dataset)

    scenario_name = (
        f"gamma_{gamma}_mean_{mean_type}_cov_{cov_type}"
        f"_miss_{int(missing_rate*100)}"
    )

    if cov_type == "C2":
        scenario_name += f"_rho_{rho}"
    elif cov_type == "C3":
        scenario_name += f"_delta_{delta}"

    output_dir = os.path.join(output_root, scenario_name)
    if save_figures:
        os.makedirs(output_dir, exist_ok=True)

    # Summary statistics
    summary_df = summarize_dataframe(df)
    print("\n" + "=" * 80)
    print(f"Scenario: {scenario_name}")
    print("=" * 80)
    print(summary_df)

    summary_csv_path = os.path.join(output_dir, "summary_statistics.csv")
    data_csv_path = os.path.join(output_dir, "generated_data.csv")

    summary_df.to_csv(summary_csv_path, index=False)
    df.to_csv(data_csv_path, index=False)

    # Missing pattern
    plot_missing_pattern(
        df,
        save_path=os.path.join(output_dir, "missing_pattern.png") if save_figures else None
    )

    # Grid plots
    plot_histograms_grid(
        df,
        bins=20,
        cols_per_row=4,
        save_path=os.path.join(output_dir, "all_histograms.png") if save_figures else None
    )

    plot_boxplots_grid_by_phase(
        df,
        cols_per_row=4,
        save_path=os.path.join(output_dir, "all_boxplots_by_phase.png") if save_figures else None
    )

    # Individual column plots
    indiv_dir = os.path.join(output_dir, "individual_columns") if save_figures else None
    plot_all_columns(df, output_dir=indiv_dir)

    return df, summary_df, dataset


# =========================================================
# 13. Main
# =========================================================
if __name__ == "__main__":
    # -----------------------------------------------------
    # Example scenario:
    # p = n * gamma = 100 * 0.3 = 30
    # mean shift = M1
    # covariance = C2 with rho=0.5
    # missing rates = 10%, 30%
    # -----------------------------------------------------

    for miss_rate in [0.10, 0.30]:
        df, summary_df, dataset = run_distribution_analysis_for_missing_rate(
            missing_rate=miss_rate,
            n1=50,
            n2=50,
            gamma=0.1,
            mean_type="M0",
            cov_type="C2",
            rho=0.7,
            delta=0.3,
            c=10,
            seed=42,
            output_root="distribution_results",
            save_figures=True
        )
