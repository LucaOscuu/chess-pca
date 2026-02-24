"""
figure_loadings.py

Utility to export and visualize full loading profiles for PC1/PC2.

- plot_loading_profile: barplot sorted by |loading|, annotates top-5 features.
- run_loading_profiles: generates 4 plots (white/black × PC1/PC2) + CSV tables and
  a quick comparison table.

Outputs:
- PNG + PDF for each loading profile
- CSV: white_loadings_table.csv, black_loadings_table.csv, loadings_comparison.csv
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def plot_loading_profile(
    series: pd.Series,
    title: str,
    outpath_png: Path,
    positive_color: str = "steelblue",
):
    """
    Publication-ish loading profile barplot:
    - sorted by |loading|
    - positive bars = positive_color
    - negative bars = tomato
    - labels on top-5 by |loading|
    Saves: PNG + PDF
    """
    s_sorted = series.reindex(series.abs().sort_values(ascending=False).index)

    fig, ax = plt.subplots(figsize=(14, 6))

    colors = [positive_color if v > 0 else "tomato" for v in s_sorted.values]
    ax.bar(range(len(s_sorted)), s_sorted.values, color=colors, edgecolor="black", linewidth=0.5, alpha=0.85)

    ax.axhline(0, color="black", linewidth=1.5, alpha=0.7)
    ax.set_xticks(range(len(s_sorted)))
    ax.set_xticklabels(s_sorted.index, rotation=90, ha="center", fontsize=10)
    ax.set_ylabel("Loading (w)", fontsize=13, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.grid(True, alpha=0.25, axis="y", linestyle="--", linewidth=0.5)

    # annotate top-5 by |loading|
    top_names = s_sorted.abs().nlargest(5).index
    for i, (name, val) in enumerate(s_sorted.items()):
        if name in top_names:
            y_offset = 0.01 if val > 0 else -0.01
            va = "bottom" if val > 0 else "top"
            ax.text(
                i, val + y_offset, f"{val:.3f}",
                ha="center", va=va, fontsize=8, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85, edgecolor="gray")
            )

    fig.tight_layout()
    outpath_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath_png, dpi=300, bbox_inches="tight")
    fig.savefig(outpath_png.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    # console top-5
    print("\n  Top 5 features by |loading|:")
    for name, val in s_sorted.head(5).items():
        sign = "➕" if val > 0 else "➖"
        print(f"    {sign} {name:25s} {val:+.4f}")


def run_loading_profiles(
    white_loadings: pd.DataFrame,
    black_loadings: pd.DataFrame,
    out_dir: str | Path = "data/outputs/loading_profiles",
):
    """
    Expects:
      - white_loadings: DataFrame indexed by feature, columns include ['PCA1','PCA2'] or ['PC1','PC2']
      - black_loadings: same structure
    Produces:
      - 4 plots (white/black × PC1/PC2) + PDF copies
      - white_loadings_table.csv, black_loadings_table.csv
      - loadings_comparison.csv
    """
    out_dir = _ensure_dir(out_dir)

    # accept both naming conventions
    def _normalize(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        cols = {c.lower(): c for c in df.columns}
        if "pca1" in cols and "pca2" in cols:
            df = df.rename(columns={cols["pca1"]: "PC1", cols["pca2"]: "PC2"})
        elif "pc1" in cols and "pc2" in cols:
            df = df.rename(columns={cols["pc1"]: "PC1", cols["pc2"]: "PC2"})
        else:
            raise ValueError(f"Loadings columns not recognized. Found: {list(df.columns)}. Need PC1/PC2 or PCA1/PCA2.")
        return df[["PC1", "PC2"]]

    white = _normalize(white_loadings)
    black = _normalize(black_loadings)

    FEATURES = list(white.index)
    d = len(FEATURES)

    print("\n" + "=" * 80)
    print("📊 GENERATING COMPLETE LOADING PROFILES (PC1 & PC2)")
    print("=" * 80)
    print(f"✅ Features: {d} variables")
    print(f"📁 Output dir: {out_dir.resolve()}")

    # ---- 4 profiles
    print("\n⚪ WHITE - PC1 Loading Profile")
    plot_loading_profile(
        white["PC1"],
        f"White PC1 Complete Loading Profile (d={d}, sorted by |loading|)",
        out_dir / "white_pc1_loading_profile.png",
        positive_color="steelblue",
    )

    print("\n⚪ WHITE - PC2 Loading Profile")
    plot_loading_profile(
        white["PC2"],
        f"White PC2 Complete Loading Profile (d={d}, sorted by |loading|)",
        out_dir / "white_pc2_loading_profile.png",
        positive_color="steelblue",
    )

    print("\n⚫ BLACK - PC1 Loading Profile")
    plot_loading_profile(
        black["PC1"],
        f"Black PC1 Complete Loading Profile (d={d}, sorted by |loading|)",
        out_dir / "black_pc1_loading_profile.png",
        positive_color="purple",
    )

    print("\n⚫ BLACK - PC2 Loading Profile")
    plot_loading_profile(
        black["PC2"],
        f"Black PC2 Complete Loading Profile (d={d}, sorted by |loading|)",
        out_dir / "black_pc2_loading_profile.png",
        positive_color="purple",
    )

    # ---- Save tables
    print("\n💾 Saving loading tables...")
    white_sorted = white.reindex(white["PC1"].abs().sort_values(ascending=False).index)
    black_sorted = black.reindex(black["PC1"].abs().sort_values(ascending=False).index)

    white_sorted.to_csv(out_dir / "white_loadings_table.csv")
    black_sorted.to_csv(out_dir / "black_loadings_table.csv")

    # ---- Comparison
    print("\n" + "=" * 80)
    print("📋 LOADING COMPARISON (White vs Black)")
    print("=" * 80)

    comparison = pd.DataFrame(
        {
            "White_PC1": white["PC1"],
            "Black_PC1": black["PC1"],
            "White_PC2": white["PC2"],
            "Black_PC2": black["PC2"],
        }
    )
    comparison["avg_abs_pc1"] = (comparison["White_PC1"].abs() + comparison["Black_PC1"].abs()) / 2
    comparison = comparison.sort_values("avg_abs_pc1", ascending=False).drop(columns=["avg_abs_pc1"])

    print("\nTop 10 features by average PC1 importance:")
    print(comparison.head(10).to_string(float_format=lambda x: f"{x:+.4f}"))

    comparison.to_csv(out_dir / "loadings_comparison.csv")

    print("\n" + "=" * 80)
    print("✅ COMPLETE LOADING PROFILES GENERATED")
    print("=" * 80)
    print("\n📋 Generated files:")
    print("   🖼️  PNG:")
    print("      • white_pc1_loading_profile.png")
    print("      • white_pc2_loading_profile.png")
    print("      • black_pc1_loading_profile.png")
    print("      • black_pc2_loading_profile.png")
    print("   📄 PDF:")
    print("      • white_pc1_loading_profile.pdf")
    print("      • white_pc2_loading_profile.pdf")
    print("      • black_pc1_loading_profile.pdf")
    print("      • black_pc2_loading_profile.pdf")
    print("   📊 CSV:")
    print("      • white_loadings_table.csv")
    print("      • black_loadings_table.csv")
    print("      • loadings_comparison.csv")

    return {
        "white": white,
        "black": black,
        "comparison": comparison,
        "out_dir": out_dir,
    }