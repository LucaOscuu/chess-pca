"""
figure_h1.py

Figure H1: Scree plot + null models for EVR(PC1+PC2).

What it does:
- fits PCA on standardized features (full components for scree),
- computes participation ratio (PR),
- builds two null distributions for EVR(PC1+PC2):
  (i) shuffled-columns null, (ii) Gaussian null matching mean/std.

Outputs:
- CSV/JSON summaries in out_dir/figure_h1_data/
- PNG: h1_scree_plots.png, h1_null_comparisons.png
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def compute_scree_data(X: np.ndarray, features_count: int, perspective_name: str, random_state: int = 42):
    """Compute full scree plot data (all components) + participation ratio."""
    print(f"🔬 Computing scree data for {perspective_name}...")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    means = np.abs(X_scaled.mean(axis=0))
    stds = np.abs(X_scaled.std(axis=0) - 1)
    print(f"   ✓ Mean(|μ|) = {means.mean():.2e}")
    print(f"   ✓ Mean(|σ-1|) = {stds.mean():.2e}")

    pca_full = PCA(n_components=features_count, random_state=random_state)
    pca_full.fit(X_scaled)

    eigenvalues = pca_full.explained_variance_
    evr = pca_full.explained_variance_ratio_
    cumulative_evr = np.cumsum(evr)

    PR = (eigenvalues.sum() ** 2) / (eigenvalues ** 2).sum()

    print(f"   ✓ PR = {PR:.2f}, EVR(PC1+PC2) = {evr[0] + evr[1]:.4f}")

    return eigenvalues, evr, cumulative_evr, float(PR)


def compute_null_models_ultrafast(X: np.ndarray, perspective_name: str, M: int = 500, random_state: int = 42):
    """Ultra-fast null models using eigenvalues of covariance matrices."""
    print(f"\n🎲 Computing null models for {perspective_name} (M={M})...")

    rng = np.random.default_rng(random_state)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    N, d = X_scaled.shape

    # Observed EVR(PC1+PC2)
    pca_obs = PCA(n_components=2, random_state=random_state)
    pca_obs.fit(X_scaled)
    observed_evr = float(pca_obs.explained_variance_ratio_.sum())
    print(f"   📌 Observed EVR(PC1+PC2) = {observed_evr:.4f}")

    means = X_scaled.mean(axis=0)
    stds = X_scaled.std(axis=0)

    shuffle_evr = np.zeros(M, dtype=float)
    gaussian_evr = np.zeros(M, dtype=float)

    X_temp = np.empty_like(X_scaled)

    print(f"   ⏳ Running {M} simulations...")
    for i in tqdm(range(M), desc=f"   {perspective_name}", leave=False):
        # SHUFFLED: permute each column independently
        for j in range(d):
            X_temp[:, j] = rng.permutation(X_scaled[:, j])

        cov_shuf = np.cov(X_temp, rowvar=False)
        eigvals = np.sort(np.linalg.eigvalsh(cov_shuf))[::-1]
        shuffle_evr[i] = (eigvals[0] + eigvals[1]) / eigvals.sum()

        # GAUSSIAN with same mean/std per feature
        X_gauss = rng.normal(loc=means, scale=stds, size=(N, d))
        cov_gauss = np.cov(X_gauss, rowvar=False)
        eigvals = np.sort(np.linalg.eigvalsh(cov_gauss))[::-1]
        gaussian_evr[i] = (eigvals[0] + eigvals[1]) / eigvals.sum()

    def stats(arr: np.ndarray, obs: float):
        return {
            "mean": float(arr.mean()),
            "sd": float(arr.std()),
            "p_emp": float((arr >= obs).mean()),
            "p025": float(np.percentile(arr, 2.5)),
            "p975": float(np.percentile(arr, 97.5)),
        }

    summary = {
        "observed_evr_pc1_pc2": observed_evr,
        **{f"shuffle_{k}": v for k, v in stats(shuffle_evr, observed_evr).items()},
        **{f"gaussian_{k}": v for k, v in stats(gaussian_evr, observed_evr).items()},
    }

    print(f"   ✓ Shuffle:  μ={summary['shuffle_mean']:.4f}, p={summary['shuffle_p_emp']:.3f}")
    print(f"   ✓ Gaussian: μ={summary['gaussian_mean']:.4f}, p={summary['gaussian_p_emp']:.3f}")

    return shuffle_evr, gaussian_evr, summary


def run_figure_h1(
    white_data: pd.DataFrame,
    black_data: pd.DataFrame,
    features: list[str],
    out_dir: str | Path,
    M_null: int = 500,
    random_state: int = 42,
    bins: int = 30,
    scree_first_k: int = 10,
):
    """
    Generates Figure H1:
      - scree plots (white/black)
      - null model histograms (shuffle + gaussian, white/black)
      - saves CSV/JSON summaries

    Outputs in: out_dir/figure_h1_data + PNGs in out_dir
    """
    print("\n" + "=" * 80)
    print("📊 GENERATING FIGURE H1: SCREE + NULL MODELS")
    print("=" * 80)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data_dir = out_dir / "figure_h1_data"
    data_dir.mkdir(exist_ok=True)

    print(f"⚡ Configuration: M={M_null} null iterations\n")

    # --- WHITE ---
    print("\n" + "=" * 80)
    print("⚪ WHITE PERSPECTIVE")
    print("=" * 80)
    t0 = time.time()

    X_white = white_data[features].to_numpy()
    eig_w, evr_w, cum_w, PR_w = compute_scree_data(X_white, len(features), "white", random_state=random_state)
    shuf_w, gauss_w, null_w = compute_null_models_ultrafast(X_white, "white", M=M_null, random_state=random_state)

    t_white = time.time() - t0
    print(f"⏱️  White completed in {t_white:.1f}s")

    # --- BLACK ---
    print("\n" + "=" * 80)
    print("⚫ BLACK PERSPECTIVE")
    print("=" * 80)
    t0 = time.time()

    X_black = black_data[features].to_numpy()
    eig_b, evr_b, cum_b, PR_b = compute_scree_data(X_black, len(features), "black", random_state=random_state)
    shuf_b, gauss_b, null_b = compute_null_models_ultrafast(X_black, "black", M=M_null, random_state=random_state)

    t_black = time.time() - t0
    print(f"⏱️  Black completed in {t_black:.1f}s")

    # --- SAVE DATA ---
    print("\n💾 Saving data files...")

    pd.DataFrame(
        {
            "component": range(1, len(eig_w) + 1),
            "eigenvalue": eig_w,
            "explained_variance_ratio": evr_w,
            "cumulative_evr": cum_w,
        }
    ).to_csv(data_dir / "scree_white.csv", index=False)

    pd.DataFrame(
        {
            "component": range(1, len(eig_b) + 1),
            "eigenvalue": eig_b,
            "explained_variance_ratio": evr_b,
            "cumulative_evr": cum_b,
        }
    ).to_csv(data_dir / "scree_black.csv", index=False)

    pd.DataFrame({"evr_pc1_pc2": shuf_w}).to_csv(data_dir / "null_shuffle_white.csv", index=False)
    pd.DataFrame({"evr_pc1_pc2": gauss_w}).to_csv(data_dir / "null_gaussian_white.csv", index=False)
    pd.DataFrame({"evr_pc1_pc2": shuf_b}).to_csv(data_dir / "null_shuffle_black.csv", index=False)
    pd.DataFrame({"evr_pc1_pc2": gauss_b}).to_csv(data_dir / "null_gaussian_black.csv", index=False)

    with open(data_dir / "summary_white.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "PR": PR_w,
                "EVR_PC1": float(evr_w[0]),
                "EVR_PC2": float(evr_w[1]),
                "EVR_PC1_PC2": float(evr_w[0] + evr_w[1]),
            },
            f,
            indent=2,
        )

    with open(data_dir / "summary_black.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "PR": PR_b,
                "EVR_PC1": float(evr_b[0]),
                "EVR_PC2": float(evr_b[1]),
                "EVR_PC1_PC2": float(evr_b[0] + evr_b[1]),
            },
            f,
            indent=2,
        )

    with open(data_dir / "null_summary_white.json", "w", encoding="utf-8") as f:
        json.dump(null_w, f, indent=2)

    with open(data_dir / "null_summary_black.json", "w", encoding="utf-8") as f:
        json.dump(null_b, f, indent=2)

    print("✅ Data files saved")

    # --- VISUALIZATION ---
    print("\n📊 Creating visualizations...")

    # FIGURE 1: SCREE PLOTS
    k = min(scree_first_k, len(evr_w), len(evr_b))
    x = np.arange(1, k + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    ax.bar(x, evr_w[:k], alpha=0.7, edgecolor="black", linewidth=0.5)
    ax.plot(x, cum_w[:k], marker="o", linewidth=2.5, markersize=6, label="Cumulative EVR")
    ax.axhline(evr_w[0] + evr_w[1], linestyle="--", linewidth=1.5, label=f"PC1+PC2 = {evr_w[0] + evr_w[1]:.3f}")
    ax.set_xlabel("Principal Component", fontweight="bold")
    ax.set_ylabel("Explained Variance Ratio", fontweight="bold")
    ax.set_title(f"White Scree Plot (PR={PR_w:.2f})", fontweight="bold")
    ax.set_xticks(x)
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.bar(x, evr_b[:k], alpha=0.7, edgecolor="black", linewidth=0.5)
    ax.plot(x, cum_b[:k], marker="o", linewidth=2.5, markersize=6, label="Cumulative EVR")
    ax.axhline(evr_b[0] + evr_b[1], linestyle="--", linewidth=1.5, label=f"PC1+PC2 = {evr_b[0] + evr_b[1]:.3f}")
    ax.set_xlabel("Principal Component", fontweight="bold")
    ax.set_ylabel("Explained Variance Ratio", fontweight="bold")
    ax.set_title(f"Black Scree Plot (PR={PR_b:.2f})", fontweight="bold")
    ax.set_xticks(x)
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    scree_png = out_dir / "h1_scree_plots.png"
    fig.savefig(scree_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Saved: {scree_png}")

    # FIGURE 2: NULL MODEL COMPARISONS (CORRECT / INFORMATIVE)
    # - show the null distribution on its natural support (zoom)
    # - keep observed as an "out-of-range" marker so it’s still visible
    # - use density to avoid “frequency vs binning” artifacts

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    def plot_null(ax, arr, obs, mu, title):
        # histogram (density is more stable than counts when zooming)
        ax.hist(arr, bins=bins, density=True, alpha=0.6, edgecolor="black")

        # null mean
        ax.axvline(mu, linestyle="--", linewidth=1.8, label=f"Null μ = {mu:.4f}")

        # zoom to the null support (with a small margin)
        lo, hi = np.quantile(arr, [0.001, 0.999])
        pad = 0.10 * (hi - lo)
        x0, x1 = lo - pad, hi + pad
        ax.set_xlim(x0, x1)

        # observed: if outside, draw at border with an arrow label
# observed: if outside, draw at border with an arrow label
        if obs < x0:
            ax.axvline(x0, linewidth=2.5, color="C0", label=f"Observed = {obs:.4f} (outside)")
            ax.autoscale_view()
            ymax = ax.get_ylim()[1]
            ax.annotate(
                f"Observed\n= {obs:.4f}",
                xy=(x0, ymax * 0.80),
                xytext=(x0 + (x1 - x0) * 0.25, ymax * 0.80),
                fontsize=9,
                ha="left",
                va="center",
                color="C0",
                fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="C0", lw=1.5),
            )
        elif obs > x1:
            ax.axvline(x1, linewidth=2.5, color="C0", label=f"Observed = {obs:.4f} (outside)")
            ax.autoscale_view()
            ymax = ax.get_ylim()[1]
            ax.annotate(
                f"Observed\n= {obs:.4f}",
                xy=(x1, ymax * 0.80),
                xytext=(x1 - (x1 - x0) * 0.25, ymax * 0.80),
                fontsize=9,
                ha="right",
                va="center",
                color="C0",
                fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="C0", lw=1.5),
            )
        else:
            ax.axvline(obs, linewidth=2.5, color="C0", label=f"Observed = {obs:.4f}")

        ax.set_xlabel("EVR(PC1+PC2)", fontweight="bold")
        ax.set_ylabel("Density", fontweight="bold")
        ax.set_title(title, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plot_null(
        axes[0, 0],
        shuf_w,
        null_w["observed_evr_pc1_pc2"],
        null_w["shuffle_mean"],
        f"White - Shuffled Null (p={null_w['shuffle_p_emp']:.3f})",
    )

    plot_null(
        axes[0, 1],
        gauss_w,
        null_w["observed_evr_pc1_pc2"],
        null_w["gaussian_mean"],
        f"White - Gaussian Null (p={null_w['gaussian_p_emp']:.3f})",
    )

    plot_null(
        axes[1, 0],
        shuf_b,
        null_b["observed_evr_pc1_pc2"],
        null_b["shuffle_mean"],
        f"Black - Shuffled Null (p={null_b['shuffle_p_emp']:.3f})",
    )

    plot_null(
        axes[1, 1],
        gauss_b,
        null_b["observed_evr_pc1_pc2"],
        null_b["gaussian_mean"],
        f"Black - Gaussian Null (p={null_b['gaussian_p_emp']:.3f})",
    )

    fig.tight_layout()
    null_png = out_dir / "h1_null_comparisons.png"
    fig.savefig(null_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Saved: {null_png}")

    print("\n" + "=" * 80)
    print("✅ FIGURE H1 COMPLETE")
    print("=" * 80)
    print(f"\n⏱️  Total time: {t_white + t_black:.1f}s (~{(t_white + t_black)/60:.1f} min)")
    print(f"\nWhite: PR={PR_w:.2f}, EVR(PC1+PC2)={evr_w[0]+evr_w[1]:.4f}")
    print(f"  Shuffle p={null_w['shuffle_p_emp']:.3f}, Gaussian p={null_w['gaussian_p_emp']:.3f}")
    print(f"\nBlack: PR={PR_b:.2f}, EVR(PC1+PC2)={evr_b[0]+evr_b[1]:.4f}")
    print(f"  Shuffle p={null_b['shuffle_p_emp']:.3f}, Gaussian p={null_b['gaussian_p_emp']:.3f}")
    print(f"\n📁 Files: {scree_png.name}, {null_png.name} + data in {data_dir}")
    print("=" * 80)

    return {
        "white": {"PR": PR_w, "evr": evr_w, "null": null_w},
        "black": {"PR": PR_b, "evr": evr_b, "null": null_b},
        "paths": {"data_dir": str(data_dir), "scree_png": str(scree_png), "null_png": str(null_png)},
    }

if __name__ == "__main__":
    import argparse
    from pathlib import Path

    from chess_pca.pipeline import build_or_load_dataset
    from chess_pca.features import numeric_features

    parser = argparse.ArgumentParser(description="Run Figure H1 (Scree + Null Models)")
    parser.add_argument("--input_path", type=str, default="data/raw", help="Path to PGN folder or file")
    parser.add_argument("--max_games", type=int, default=10000)
    parser.add_argument("--min_moves", type=int, default=10)
    parser.add_argument("--cache_dir", type=str, default="data/cache")
    parser.add_argument("--out_dir", type=str, default="data/outputs/figure_h1_data")
    parser.add_argument("--m_null", type=int, default=500)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--force_recompute", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("🧪 RUNNING FIGURE H1 FROM CLI")
    print("=" * 80)
    print(f"input_path={args.input_path}")
    print(f"out_dir={out_dir}")
    print(f"max_games={args.max_games} | min_moves={args.min_moves} | M_null={args.m_null}")
    print("=" * 80 + "\n")

    white_data, black_data = build_or_load_dataset(
        input_path=args.input_path,
        max_games=args.max_games,
        min_moves=args.min_moves,
        cache_dir=args.cache_dir,
        force_recompute=args.force_recompute,
    )

    FEATURES = numeric_features(white_data)

    # run_figure_h1 is already imported at top of this file in your project
    run_figure_h1(
        white_data=white_data,
        black_data=black_data,
        features=FEATURES,
        out_dir=out_dir,
        M_null=args.m_null,
        random_state=args.random_state,
    )

    print("\n✅ Done. Outputs saved in:", out_dir.resolve())