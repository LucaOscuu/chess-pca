"""
figure_h2.py

Figure H2: Directional stability of PCA loadings via bootstrap.

Method:
- fit a reference PCA (PC1/PC2) on the full dataset,
- bootstrap sample games (by game_id) and refit PCA,
- measure cosine similarity between reference and bootstrap components
  (absolute value => sign-invariant stability).

Outputs:
- raw CSV + stats JSON in out_dir/figure_h2_data/
- two hist plots: black and white with data-driven x-limits.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cosine


def bootstrap_pca_stability(
    data: pd.DataFrame,
    features: list[str],
    n_boot: int = 80,
    frac: float = 0.8,
    random_state: int = 42,
    perspective_name: str = "",
):
    """
    Bootstrap PCA stability analysis.
    Returns cosine similarities for PC1 and PC2 across bootstraps (absolute, sign-invariant)
    and summary stats for each component.
    """
    print(f"🔬 Computing bootstrap stability for {perspective_name}...")

    rng = np.random.default_rng(random_state)

    if "game_id" not in data.columns:
        raise ValueError("bootstrap_pca_stability requires a 'game_id' column in data.")

    game_ids = data["game_id"].unique()
    n_games = len(game_ids)
    n_sample = max(1, int(n_games * frac))

    print(f"   Total games: {n_games}")
    print(f"   Games per bootstrap: {n_sample}")

    # Fit full PCA (reference)
    X_full = data[features].to_numpy()
    scaler_full = StandardScaler()
    X_scaled_full = scaler_full.fit_transform(X_full)
    pca_full = PCA(n_components=2, random_state=random_state)
    pca_full.fit(X_scaled_full)

    ref_pc1 = pca_full.components_[0]
    ref_pc2 = pca_full.components_[1]

    sim_pc1 = np.zeros(n_boot, dtype=float)
    sim_pc2 = np.zeros(n_boot, dtype=float)

    print(f"   ⏳ Running {n_boot} bootstrap iterations...")

    for i in tqdm(range(n_boot), desc=f"   {perspective_name}", leave=False):
        boot_games = rng.choice(game_ids, size=n_sample, replace=True)
        boot_data = data[data["game_id"].isin(boot_games)]

        X_boot = boot_data[features].to_numpy()
        scaler_boot = StandardScaler()
        X_scaled_boot = scaler_boot.fit_transform(X_boot)
        pca_boot = PCA(n_components=2, random_state=random_state)
        pca_boot.fit(X_scaled_boot)

        boot_pc1 = pca_boot.components_[0]
        boot_pc2 = pca_boot.components_[1]

        cos_sim_pc1 = 1 - cosine(ref_pc1, boot_pc1)
        cos_sim_pc2 = 1 - cosine(ref_pc2, boot_pc2)

        sim_pc1[i] = abs(cos_sim_pc1)
        sim_pc2[i] = abs(cos_sim_pc2)

    def stats(arr: np.ndarray):
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr, ddof=1)),
            "p05": float(np.percentile(arr, 5)),
            "min": float(np.min(arr)),
        }

    stats_pc1 = stats(sim_pc1)
    stats_pc2 = stats(sim_pc2)

    print(f"\n   📊 {perspective_name.upper()} STABILITY:")
    print(
        f"      PC1: mean={stats_pc1['mean']:.4f}, std={stats_pc1['std']:.4f}, "
        f"p05={stats_pc1['p05']:.4f}, min={stats_pc1['min']:.4f}"
    )
    print(
        f"      PC2: mean={stats_pc2['mean']:.4f}, std={stats_pc2['std']:.4f}, "
        f"p05={stats_pc2['p05']:.4f}, min={stats_pc2['min']:.4f}"
    )

    return sim_pc1, sim_pc2, stats_pc1, stats_pc2


def _data_driven_xlim(a: np.ndarray, b: np.ndarray, pad_frac: float = 0.05):
    """Compute xlim tightly around the actual data range, with a small pad."""
    all_vals = np.concatenate([a, b])
    all_vals = all_vals[np.isfinite(all_vals)]
    lo = float(np.min(all_vals))
    hi = float(np.max(all_vals))
    span = max(hi - lo, 1e-6)
    pad = span * pad_frac
    return max(0.0, lo - pad), min(1.0, hi + pad)


def _add_vlines_annotated(ax, stats, colors, linestyles, label_prefix, y_positions, fontsize=8):
    # Mean
    ax.axvline(stats["mean"], linestyle=linestyles[0], linewidth=1.5,
               color=colors[0], alpha=0.85, label=f"{label_prefix} Mean", zorder=10)
    # 5th percentile
    ax.axvline(stats["p05"], linestyle=linestyles[1], linewidth=1.3,
               color=colors[1], alpha=0.85, label=f"{label_prefix} 5th %ile", zorder=10)
    # Minimum
    ax.axvline(stats["min"], linestyle=linestyles[2], linewidth=1.3,
               color=colors[2], alpha=0.85, label=f"{label_prefix} Min", zorder=10)

    ylim = ax.get_ylim()
    y_range = ylim[1] - ylim[0]

    ax.text(stats["mean"], ylim[0] + y_range * y_positions[0],
            f'μ={stats["mean"]:.4f}', rotation=90, fontsize=fontsize,
            color=colors[0], va="bottom", ha="right", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7, edgecolor=colors[0]))

    ax.text(stats["p05"], ylim[0] + y_range * y_positions[1],
            f'5th={stats["p05"]:.4f}', rotation=90, fontsize=fontsize,
            color=colors[1], va="bottom", ha="right", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7, edgecolor=colors[1]))

    ax.text(stats["min"], ylim[0] + y_range * y_positions[2],
            f'min={stats["min"]:.4f}', rotation=90, fontsize=fontsize,
            color=colors[2], va="bottom", ha="right", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7, edgecolor=colors[2]))


def run_figure_h2(
    white_data: pd.DataFrame,
    black_data: pd.DataFrame,
    features: list[str],
    out_dir: str | Path,
    n_bootstrap: int = 80,
    sample_fraction: float = 0.8,
    random_state: int = 42,
):
    """
    Generates Figure H2 (Directional Stability via bootstrap):
      - saves raw CSV + stats JSON + verification table
      - saves two plots: black and white, both with data-driven xlim and bins

    Output goes to:
      out_dir/figure_h2_data/ (CSVs/JSON)
      out_dir/h2_black_fullrange.png
      out_dir/h2_white_zoom.png
    """
    print("\n" + "=" * 80)
    print("🔄 GENERATING DATA FOR FIGURE H2: DIRECTIONAL STABILITY")
    print("=" * 80)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    h2_dir = out_dir / "figure_h2_data"
    h2_dir.mkdir(exist_ok=True)

    print(f"Bootstrap samples: {n_bootstrap}")
    print(f"Sample fraction: {sample_fraction:.0%}")
    print(f"Random seed: {random_state}\n")

    print("\n" + "=" * 80)
    print("⚪ WHITE BOOTSTRAP STABILITY")
    print("=" * 80)
    sim_white_pc1, sim_white_pc2, stats_w_pc1, stats_w_pc2 = bootstrap_pca_stability(
        white_data, features, n_boot=n_bootstrap, frac=sample_fraction,
        random_state=random_state, perspective_name="White"
    )

    print("\n" + "=" * 80)
    print("⚫ BLACK BOOTSTRAP STABILITY")
    print("=" * 80)
    sim_black_pc1, sim_black_pc2, stats_b_pc1, stats_b_pc2 = bootstrap_pca_stability(
        black_data, features, n_boot=n_bootstrap, frac=sample_fraction,
        random_state=random_state, perspective_name="Black"
    )

    # --- SAVE RAW DATA ---
    pd.DataFrame({"pc1": sim_black_pc1, "pc2": sim_black_pc2}).to_csv(h2_dir / "stability_black.csv", index=False)
    pd.DataFrame({"pc1": sim_white_pc1, "pc2": sim_white_pc2}).to_csv(h2_dir / "stability_white.csv", index=False)

    with open(h2_dir / "stability_stats_white.json", "w", encoding="utf-8") as f:
        json.dump({"PC1": stats_w_pc1, "PC2": stats_w_pc2}, f, indent=2)

    with open(h2_dir / "stability_stats_black.json", "w", encoding="utf-8") as f:
        json.dump({"PC1": stats_b_pc1, "PC2": stats_b_pc2}, f, indent=2)

    # --- VERIFICATION TABLE ---
    verification_df = pd.DataFrame({
        "Perspective": ["White", "White", "Black", "Black"],
        "Component": ["PC1", "PC2", "PC1", "PC2"],
        "Mean": [stats_w_pc1["mean"], stats_w_pc2["mean"], stats_b_pc1["mean"], stats_b_pc2["mean"]],
        "5th %ile": [stats_w_pc1["p05"], stats_w_pc2["p05"], stats_b_pc1["p05"], stats_b_pc2["p05"]],
        "Min": [stats_w_pc1["min"], stats_w_pc2["min"], stats_b_pc1["min"], stats_b_pc2["min"]],
        "Std": [stats_w_pc1["std"], stats_w_pc2["std"], stats_b_pc1["std"], stats_b_pc2["std"]],
    })
    verification_df.to_csv(h2_dir / "table_h2_verification.csv", index=False, float_format="%.4f")

    print(f"\n💾 Raw data saved in: {h2_dir}")

    # --- PLOT 1: Black — data-driven range ---
    print("\n📊 Creating Figure H2.1: Black...")
    xmin_b, xmax_b = _data_driven_xlim(sim_black_pc1, sim_black_pc2)
    bins_black = np.linspace(xmin_b, xmax_b, 32)

    fig, ax = plt.subplots(figsize=(11, 7.5))
    ax.hist(sim_black_pc1, bins=bins_black, alpha=0.60, label="PC1 Stability",
            edgecolor="none", zorder=5)
    ax.hist(sim_black_pc2, bins=bins_black, alpha=0.60, label="PC2 Stability",
            edgecolor="none", zorder=5)
    ax.set_xlim(xmin_b, xmax_b)
    ax.set_ylim(bottom=0)

    _add_vlines_annotated(
        ax, stats_b_pc1,
        colors=("darkgreen", "forestgreen", "limegreen"),
        linestyles=("-", "--", ":"),
        label_prefix="PC1",
        y_positions=(0.75, 0.60, 0.45),
        fontsize=8
    )
    _add_vlines_annotated(
        ax, stats_b_pc2,
        colors=("darkred", "crimson", "lightcoral"),
        linestyles=("-", "--", ":"),
        label_prefix="PC2",
        y_positions=(0.85, 0.70, 0.55),
        fontsize=8
    )

    ax.set_title(
        f"H2 Validation: Directional Stability (Black)\n"
        f"N={n_bootstrap} bootstraps  |  x-range: [{xmin_b:.4f}, {xmax_b:.4f}]",
        fontsize=14, fontweight="bold", pad=15
    )
    ax.set_xlabel("Cosine Similarity (1.0 = Perfect Stability)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Frequency", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.5)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:2] + handles[2:8], labels[:2] + labels[2:8],
              loc="upper left", frameon=True, fontsize=9, ncol=2,
              framealpha=0.95, edgecolor="gray")

    black_png = out_dir / "h2_black_fullrange.png"
    plt.tight_layout()
    plt.savefig(black_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Saved: {black_png}  (xlim: {xmin_b:.4f}–{xmax_b:.4f})")

    # --- PLOT 2: White — data-driven range ---
    print("\n📊 Creating Figure H2.2: White...")
    xmin_w, xmax_w = _data_driven_xlim(sim_white_pc1, sim_white_pc2)
    bins_white = np.linspace(xmin_w, xmax_w, 32)

    fig, ax = plt.subplots(figsize=(11, 7.5))
    ax.hist(sim_white_pc1, bins=bins_white, alpha=0.60, label="PC1 Stability",
            edgecolor="none", zorder=5)
    ax.hist(sim_white_pc2, bins=bins_white, alpha=0.60, label="PC2 Stability",
            edgecolor="none", zorder=5)
    ax.set_xlim(xmin_w, xmax_w)
    ax.set_ylim(bottom=0)

    _add_vlines_annotated(
        ax, stats_w_pc1,
        colors=("darkgreen", "forestgreen", "limegreen"),
        linestyles=("-", "--", ":"),
        label_prefix="PC1",
        y_positions=(0.70, 0.55, 0.40),
        fontsize=8
    )
    _add_vlines_annotated(
        ax, stats_w_pc2,
        colors=("darkred", "crimson", "lightcoral"),
        linestyles=("-", "--", ":"),
        label_prefix="PC2",
        y_positions=(0.80, 0.65, 0.50),
        fontsize=8
    )

    ax.set_title(
        f"H2 Validation: Directional Stability (White)\n"
        f"N={n_bootstrap} bootstraps  |  x-range: [{xmin_w:.4f}, {xmax_w:.4f}]",
        fontsize=14, fontweight="bold", pad=15
    )
    ax.set_xlabel("Cosine Similarity (1.0 = Perfect Stability)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Frequency", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.5)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:2] + handles[2:8], labels[:2] + labels[2:8],
              loc="upper left", frameon=True, fontsize=9, ncol=2,
              framealpha=0.95, edgecolor="gray")

    white_png = out_dir / "h2_white_zoom.png"
    plt.tight_layout()
    plt.savefig(white_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Saved: {white_png}  (xlim: {xmin_w:.4f}–{xmax_w:.4f})")

    print("\n" + "=" * 80)
    print("✅ FIGURE H2 GENERATION COMPLETE")
    print("=" * 80)

    return {
        "paths": {
            "h2_dir": str(h2_dir),
            "black_png": str(black_png),
            "white_png": str(white_png),
        },
        "stats": {
            "white": {"pc1": stats_w_pc1, "pc2": stats_w_pc2},
            "black": {"pc1": stats_b_pc1, "pc2": stats_b_pc2},
        }
    }

if __name__ == "__main__":
    import argparse
    from pathlib import Path

    from chess_pca.pipeline import build_or_load_dataset
    from chess_pca.features import numeric_features

    parser = argparse.ArgumentParser(description="Run Figure H2 (Directional Stability via Bootstrap)")
    parser.add_argument("--input_path", type=str, default="data/raw", help="Path to PGN folder or file")
    parser.add_argument("--max_games", type=int, default=10000)
    parser.add_argument("--min_moves", type=int, default=10)
    parser.add_argument("--cache_dir", type=str, default="data/cache")
    parser.add_argument("--out_dir", type=str, default="data/outputs/figure_h2_data")

    parser.add_argument("--n_boot", type=int, default=80)
    parser.add_argument("--sample_frac", type=float, default=0.80)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--force_recompute", action="store_true")

    # opzionale: setta limiti plot se il tuo codice li supporta
    parser.add_argument("--xlim_black_min", type=float, default=0.30)
    parser.add_argument("--xlim_black_max", type=float, default=1.00)
    parser.add_argument("--xlim_white_min", type=float, default=0.993)
    parser.add_argument("--xlim_white_max", type=float, default=1.000)

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("🧪 RUNNING FIGURE H2 FROM CLI")
    print("=" * 80)
    print(f"input_path={args.input_path}")
    print(f"out_dir={out_dir}")
    print(f"max_games={args.max_games} | min_moves={args.min_moves} | n_boot={args.n_boot} | sample_frac={args.sample_frac}")
    print("=" * 80 + "\n")

    white_data, black_data = build_or_load_dataset(
        input_path=args.input_path,
        max_games=args.max_games,
        min_moves=args.min_moves,
        cache_dir=args.cache_dir,
        force_recompute=args.force_recompute,
    )

    FEATURES = numeric_features(white_data)

    # run_figure_h2 deve esistere in figure_h2.py (come per H1)
    # Se la tua firma è diversa, dimmelo e lo adatto al volo.
    run_figure_h2(
        white_data=white_data,
        black_data=black_data,
        features=FEATURES,
        out_dir=out_dir,
        n_boot=args.n_boot,
        sample_fraction=args.sample_frac,
        random_state=args.random_state,
        xlim_black=(args.xlim_black_min, args.xlim_black_max),
        xlim_white=(args.xlim_white_min, args.xlim_white_max),
    )

    print("\n✅ Done. Outputs saved in:", out_dir.resolve())