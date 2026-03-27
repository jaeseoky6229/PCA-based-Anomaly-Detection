import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm


# =========================================================
# 1. 설정
# =========================================================
DATA_FILES = {
    0.1: "residual_features_frac_0.1.csv",
    0.2: "residual_features_frac_0.2.csv",
    0.3: "residual_features_frac_0.3.csv",
    0.4: "residual_features_frac_0.4.csv",
    0.5: "residual_features_frac_0.5.csv",
    0.6: "residual_features_frac_0.6.csv",
}

OUTPUT_DIR = "missing_rate_normalfit_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================================================
# 2. CSV 로드
# =========================================================
def load_csv_safe(file_path):
    if not os.path.exists(file_path):
        print(f"[경고] 파일이 없습니다: {file_path}")
        return None

    try:
        df = pd.read_csv(file_path, encoding="utf-8-sig")
    except Exception:
        df = pd.read_csv(file_path)

    return df


# =========================================================
# 3. 수치형 feature 컬럼 추출
# =========================================================
def get_numeric_feature_columns(df, exclude_cols=None):
    if exclude_cols is None:
        exclude_cols = ["sample_idx", "status"]

    numeric_cols = []
    for col in df.columns:
        if col in exclude_cols:
            continue

        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.notna().sum() > 0:
            numeric_cols.append(col)

    return numeric_cols


def convert_numeric_columns(df, numeric_cols):
    df = df.copy()
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# =========================================================
# 4. 요약 통계
# =========================================================
def summarize_features(df, feature_cols):
    rows = []
    n_rows = len(df)

    for col in feature_cols:
        s = df[col]
        rows.append({
            "column": col,
            "n_total": n_rows,
            "n_missing": s.isna().sum(),
            "missing_ratio_actual": s.isna().mean(),
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
# 5. 결측 패턴 시각화
# =========================================================
def plot_missing_pattern(df, feature_cols, title="", save_path=None):
    plt.figure(figsize=(12, 6))
    plt.imshow(df[feature_cols].isna(), aspect="auto", cmap="gray_r", interpolation="nearest")
    plt.colorbar(label="Missing (1=True, 0=False)")
    plt.xticks(ticks=np.arange(len(feature_cols)), labels=feature_cols, rotation=90)
    plt.xlabel("Feature")
    plt.ylabel("Sample")
    plt.title(title if title else "Missing Data Pattern")
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()


# =========================================================
# 6. Grid histogram + Normal fit
# =========================================================
def plot_grid_histograms_with_normal_fit(
    df,
    feature_cols,
    missing_frac=None,
    bins=20,
    cols_per_row=3,
    save_path=None
):
    n_features = len(feature_cols)
    n_rows = math.ceil(n_features / cols_per_row)

    fig, axes = plt.subplots(n_rows, cols_per_row, figsize=(5 * cols_per_row, 4 * n_rows))
    axes = np.array(axes).reshape(-1)

    for i, col in enumerate(feature_cols):
        ax = axes[i]
        data = df[col].dropna().values

        if len(data) == 0:
            ax.set_title(f"{col} (all missing)")
            ax.axis("off")
            continue

        # 히스토그램 density=True 로 정규 pdf와 스케일 맞춤
        ax.hist(
            data,
            bins=bins,
            density=True,
            alpha=0.7,
            color="steelblue",
            edgecolor="black"
        )

        # 정규분포 피팅
        if len(data) >= 2 and np.std(data, ddof=1) > 0:
            mu, sigma = norm.fit(data)
            x = np.linspace(np.min(data), np.max(data), 200)
            y = norm.pdf(x, loc=mu, scale=sigma)
            ax.plot(x, y, color="red", linewidth=2, label="Normal fit")
            ax.legend(fontsize=8)

            ax.text(
                0.98, 0.95,
                f"miss={df[col].isna().mean():.2f}\nμ={mu:.3f}\nσ={sigma:.3f}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=8,
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray")
            )
        else:
            ax.text(
                0.98, 0.95,
                f"miss={df[col].isna().mean():.2f}\nfit not available",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=8,
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray")
            )

        ax.set_title(str(col))
        ax.set_xlabel("")
        ax.set_ylabel("Density")

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    title = "Feature Histograms with Normal Fit"
    if missing_frac is not None:
        title += f" - missing frac {missing_frac}"
    fig.suptitle(title, fontsize=16)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()


# =========================================================
# 7. 결측률별 비교용 normal fit histogram
# =========================================================
def plot_feature_comparison_across_missing_rates_with_fit(
    data_dict,
    feature_name,
    bins=20,
    save_path=None
):
    n_cases = len(data_dict)
    fig, axes = plt.subplots(n_cases, 1, figsize=(8, 3.2 * n_cases), sharex=True)

    if n_cases == 1:
        axes = [axes]

    for ax, (frac, df) in zip(axes, sorted(data_dict.items())):
        if df is None or feature_name not in df.columns:
            ax.set_title(f"{feature_name} - frac={frac:.1f} (file missing)")
            ax.axis("off")
            continue

        data = df[feature_name].dropna().values
        if len(data) == 0:
            ax.set_title(f"{feature_name} - frac={frac:.1f} (all missing)")
            ax.axis("off")
            continue

        ax.hist(
            data,
            bins=bins,
            density=True,
            alpha=0.7,
            color="steelblue",
            edgecolor="black"
        )

        if len(data) >= 2 and np.std(data, ddof=1) > 0:
            mu, sigma = norm.fit(data)
            x = np.linspace(np.min(data), np.max(data), 200)
            y = norm.pdf(x, loc=mu, scale=sigma)
            ax.plot(x, y, color="red", linewidth=2)

            ax.text(
                0.98, 0.95,
                f"frac={frac:.1f}\nμ={mu:.3f}\nσ={sigma:.3f}\nobs={len(data)}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=8,
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray")
            )

        ax.set_ylabel("Density")
        ax.set_title(f"{feature_name} - frac={frac:.1f}")

    axes[-1].set_xlabel(feature_name)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()


# =========================================================
# 8. 한 파일 처리
# =========================================================
def process_one_file(missing_frac, file_path, output_root=OUTPUT_DIR, bins=20):
    df = load_csv_safe(file_path)
    if df is None:
        return None, None

    # 훈련데이터만 사용: sample_idx 0~199
    df = df[df["sample_idx"].between(0, 199)].copy()

    feature_cols = get_numeric_feature_columns(df, exclude_cols=["sample_idx", "status"])
    df = convert_numeric_columns(df, feature_cols)

    frac_dir = os.path.join(output_root, f"frac_{missing_frac:.1f}")
    os.makedirs(frac_dir, exist_ok=True)

    # 요약 통계
    summary_df = summarize_features(df, feature_cols)
    summary_df.to_csv(os.path.join(frac_dir, "summary_statistics.csv"), index=False)

    # 결측 패턴
    plot_missing_pattern(
        df,
        feature_cols,
        title=f"Missing Pattern - frac {missing_frac:.1f} (train only)",
        save_path=os.path.join(frac_dir, "missing_pattern.png")
    )

    # grid histogram + normal fit
    plot_grid_histograms_with_normal_fit(
        df,
        feature_cols,
        missing_frac=missing_frac,
        bins=bins,
        cols_per_row=3,
        save_path=os.path.join(frac_dir, "all_feature_histograms_with_normal_fit.png")
    )

    print(f"[완료] frac={missing_frac:.1f}, file={file_path}")
    return df, feature_cols


# =========================================================
# 9. 전체 feature 비교
# =========================================================
def compare_all_features_across_missing_rates(data_dict, feature_cols, output_root=OUTPUT_DIR, bins=20):
    compare_dir = os.path.join(output_root, "compare_across_missing_rates")
    os.makedirs(compare_dir, exist_ok=True)

    for col in feature_cols:
        plot_feature_comparison_across_missing_rates_with_fit(
            data_dict=data_dict,
            feature_name=col,
            bins=bins,
            save_path=os.path.join(compare_dir, f"{col}_compare_normal_fit.png")
        )


# =========================================================
# 10. 메인
# =========================================================
def main():
    loaded_data = {}
    common_feature_cols = None

    for frac, file_path in DATA_FILES.items():
        df, feature_cols = process_one_file(
            missing_frac=frac,
            file_path=file_path,
            output_root=OUTPUT_DIR,
            bins=20
        )

        loaded_data[frac] = df

        if feature_cols is not None:
            if common_feature_cols is None:
                common_feature_cols = feature_cols
            else:
                common_feature_cols = [c for c in common_feature_cols if c in feature_cols]

    if common_feature_cols is not None and len(common_feature_cols) > 0:
        compare_all_features_across_missing_rates(
            data_dict=loaded_data,
            feature_cols=common_feature_cols,
            output_root=OUTPUT_DIR,
            bins=20
        )

    print("\n모든 작업이 완료되었습니다.")


if __name__ == "__main__":
    main()
