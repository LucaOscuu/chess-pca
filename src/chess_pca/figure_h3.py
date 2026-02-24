"""
figure_h3.py

Figure H3: Geometric drift on PC1 (Δz1) between an initial time T_initial and final state.

Steps:
- project per-move states onto PCA coordinates (z1,z2) using a PCA model per perspective,
- for each game: pick first move >= start_move as T_initial, and last move as T_final,
- compute drift Δz1 = z1_final - z1_init, compare distributions across outcomes.

Statistics:
- descriptive stats (median, IQR),
- Mann-Whitney U test (Win vs Loss) + rank-biserial effect size.

Outputs:
- drift_data.csv + drift_statistics.json
- hist, boxplot, drift-vs-length scatter, and per-perspective boxplots.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from scipy.stats import mannwhitneyu


def _summ(x):
    x = np.asarray(x, dtype=float)
    return {
        "N": int(len(x)),
        "mean": float(np.mean(x)) if len(x) else float("nan"),
        "median": float(np.median(x)) if len(x) else float("nan"),
        "std": float(np.std(x)) if len(x) else float("nan"),
        "iqr_low": float(np.percentile(x, 25)) if len(x) else float("nan"),
        "iqr_high": float(np.percentile(x, 75)) if len(x) else float("nan"),
    }


def prepare_drift_dataframe(
    data: pd.DataFrame,
    features: list[str],
    pca_model,
    perspective_name: str,
):
    """
    Prepare dataframe with PCA coordinates for drift analysis.
    IMPORTANT:
      - uses 'ply' as move index (as in your fixed version)
      - maps result W/L/D to Win/Loss/Draw
      - fits a fresh scaler on the passed data before transform (as in your block)
    """
    if "game_id" not in data.columns or "ply" not in data.columns or "result" not in data.columns:
        raise ValueError("Data must contain columns: game_id, ply, result")

    X = data[features].to_numpy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pc_coords = pca_model.transform(X_scaled)

    df_drift = pd.DataFrame(
        {
            "game_id": data["game_id"].to_numpy(),
            "move_idx": data["ply"].to_numpy(),
            "perspective": perspective_name,
            "outcome": data["result"].to_numpy(),
            "z1": pc_coords[:, 0],
            "z2": pc_coords[:, 1],
        }
    )

    outcome_map = {"W": "Win", "L": "Loss", "D": "Draw"}
    df_drift["outcome"] = df_drift["outcome"].map(outcome_map).fillna(df_drift["outcome"])

    return df_drift


def run_figure_h3(
    white_data: pd.DataFrame,
    black_data: pd.DataFrame,
    features: list[str],
    pca_white,
    pca_black,
    out_dir: str | Path,
    start_move: int = 5,
    exclude_resignation: bool = False,
    bins_main: int = 50,
    save_prefix: str = "h3",
):
    """
    H3: Geometric Drift on PC1 (Δz1) from T_initial (>= start_move) to T_final.

    Outputs:
      - out_dir/figure_h3_data/drift_data.csv
      - out_dir/figure_h3_data/drift_statistics.json
      - out_dir/{save_prefix}_drift_hist.png
      - out_dir/{save_prefix}_drift_boxplot.png
      - out_dir/{save_prefix}_len_scatter.png
      - out_dir/{save_prefix}_perspective_boxplot.png
    """
    print("\n" + "=" * 80)
    print("📊 GENERATING FIGURE H3: GEOMETRIC DRIFT ANALYSIS")
    print("=" * 80)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    h3_dir = out_dir / "figure_h3_data"
    h3_dir.mkdir(exist_ok=True)

    print("Configuration:")
    print(f"  Start move (T_initial): {start_move}")
    print(f"  Exclude resignations: {exclude_resignation}")
    print(f"  Bins for histograms: {bins_main}\n")

    # --- Prepare drift DF (project to PCA for each perspective) ---
    print("🔧 Preparing drift data...")

    print("  ⚪ Processing White perspective...")
    df_white_drift = prepare_drift_dataframe(white_data, features, pca_white, "White")

    print("  ⚫ Processing Black perspective...")
    df_black_drift = prepare_drift_dataframe(black_data, features, pca_black, "Black")

    df = pd.concat([df_white_drift, df_black_drift], ignore_index=True)
    print(f"  ✓ Total states: {len(df):,}")
    print(f"  ✓ Unique games: {df['game_id'].nunique():,}")

    # Placeholder (as in your block)
    df["is_resignation"] = False

    # --- Compute drifts ---
    print("\n📐 Computing geometric drifts (Δz1)...")

    drifts = []

    for persp, gpersp in df.groupby("perspective"):
        print(f"  Processing {persp}...")

        for gid, g in tqdm(gpersp.groupby("game_id"), desc=f"  {persp}", leave=False):
            if g.empty:
                continue

            # outcome for this game/perspective
            if not g["outcome"].mode().empty:
                oc = g["outcome"].mode().iat[0]
            else:
                oc = g["outcome"].iloc[0]

            # T_initial: first move >= start_move
            g_ge = g[g["move_idx"] >= start_move]
            if g_ge.empty:
                continue

            t_init_row = g_ge.sort_values("move_idx").iloc[0]
            z1_init = float(t_init_row["z1"])
            t_init = int(t_init_row["move_idx"])

            # T_final: last move (optionally exclude resignations placeholder)
            if exclude_resignation:
                g_nonres = g[~g["is_resignation"]]
                if g_nonres.empty:
                    continue
                t_final_row = g_nonres.sort_values("move_idx").iloc[-1]
            else:
                t_final_row = g.sort_values("move_idx").iloc[-1]

            z1_final = float(t_final_row["z1"])
            t_final = int(t_final_row["move_idx"])

            drift = z1_final - z1_init

            drifts.append(
                {
                    "game_id": gid,
                    "perspective": persp,
                    "outcome": oc,
                    "t_init": t_init,
                    "t_final": t_final,
                    "z1_init": z1_init,
                    "z1_final": z1_final,
                    "drift_z1": drift,
                    "game_len": t_final - t_init + 1,
                }
            )

    df_h3 = pd.DataFrame(drifts)
    print(f"  ✓ Computed drifts for {len(df_h3):,} games")

    if df_h3.empty:
        raise ValueError("H3: drift dataframe is empty. Check start_move / data columns / filters.")

    # --- Statistics ---
    print("\n" + "=" * 80)
    print("📊 CORE STATISTICS")
    print("=" * 80)

    stats_win = _summ(df_h3.loc[df_h3["outcome"] == "Win", "drift_z1"])
    stats_loss = _summ(df_h3.loc[df_h3["outcome"] == "Loss", "drift_z1"])
    stats_draw = _summ(df_h3.loc[df_h3["outcome"] == "Draw", "drift_z1"])

    print(f"\n  WIN (N={stats_win['N']}):")
    print(f"    Median: {stats_win['median']:+.3f}, IQR: [{stats_win['iqr_low']:+.3f}, {stats_win['iqr_high']:+.3f}]")
    print(f"\n  LOSS (N={stats_loss['N']}):")
    print(f"    Median: {stats_loss['median']:+.3f}, IQR: [{stats_loss['iqr_low']:+.3f}, {stats_loss['iqr_high']:+.3f}]")
    print(f"\n  DRAW (N={stats_draw['N']}):")
    print(f"    Median: {stats_draw['median']:+.3f}, IQR: [{stats_draw['iqr_low']:+.3f}, {stats_draw['iqr_high']:+.3f}]")

    # Mann-Whitney Win vs Loss
    w = df_h3.loc[df_h3["outcome"] == "Win", "drift_z1"].to_numpy()
    l = df_h3.loc[df_h3["outcome"] == "Loss", "drift_z1"].to_numpy()

    if len(w) > 0 and len(l) > 0:
        u_stat, p_val = mannwhitneyu(w, l, alternative="two-sided")
        r_rb = 1.0 - (2.0 * u_stat) / (len(w) * len(l))
    else:
        u_stat, p_val, r_rb = float("nan"), float("nan"), float("nan")

    print(f"\n📈 MANN-WHITNEY: U={u_stat:.3g}, p={p_val:.3e}, r={r_rb:+.3f}")

    # --- Save data ---
    df_h3.to_csv(h3_dir / "drift_data.csv", index=False)

    summary_stats = {
        "Win": stats_win,
        "Loss": stats_loss,
        "Draw": stats_draw,
        "Mann_Whitney": {"U_statistic": float(u_stat), "p_value": float(p_val), "rank_biserial_r": float(r_rb)},
    }
    with open(h3_dir / "drift_statistics.json", "w", encoding="utf-8") as f:
        json.dump(summary_stats, f, indent=2)

    print(f"\n💾 Data saved to: {h3_dir}")

    # --- Visual style (keep yours) ---
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
        }
    )

    # ===== FIGURE 1: HISTOGRAM =====
    print("\n📈 Figure 1: Drift distributions...")

    fig, ax = plt.subplots(figsize=(12, 7))

    # Use default matplotlib colors (avoid hardcoding if you want)
    all_drifts = df_h3["drift_z1"].to_numpy()
    bins = np.linspace(all_drifts.min(), all_drifts.max(), bins_main)

    for outcome in ["Win", "Draw", "Loss"]:
        d = df_h3.loc[df_h3["outcome"] == outcome, "drift_z1"].to_numpy()
        ax.hist(d, bins=bins, alpha=0.5, label=f"{outcome} (N={len(d):,})", edgecolor="white", linewidth=0.5)

    ax.axvline(stats_win["median"], linestyle="--", linewidth=2.5, label=f"Win median = {stats_win['median']:.2f}", zorder=10)
    ax.axvline(stats_loss["median"], linestyle="--", linewidth=2.5, label=f"Loss median = {stats_loss['median']:.2f}", zorder=10)
    ax.axvline(stats_draw["median"], linestyle="--", linewidth=2.5, label=f"Draw median = {stats_draw['median']:.2f}", zorder=10)
    ax.axvline(0, color="black", linestyle="-", linewidth=1.5, alpha=0.3, zorder=5)

    ax.set_xlabel("Δz₁ = z₁(final) − z₁(initial)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Frequency", fontsize=13, fontweight="bold")
    ax.set_title(f"H3: Geometric Drift on PC1\nMann-Whitney: p={p_val:.2e}, rank-biserial r={r_rb:+.3f}", fontsize=14, fontweight="bold", pad=15)
    ax.legend(loc="upper left", fontsize=10, framealpha=0.95, edgecolor="gray", ncol=2)
    ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.5)

    plt.tight_layout()
    p1 = out_dir / f"{save_prefix}_drift_hist.png"
    plt.savefig(p1, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Saved: {p1.name}")

    # ===== FIGURE 2: BOXPLOT =====
    print("\n📊 Figure 2: Boxplot comparison...")

    fig, ax = plt.subplots(figsize=(10, 7))

    draw_vals = df_h3.loc[df_h3["outcome"] == "Draw", "drift_z1"].to_numpy()
    data_box = [w, draw_vals, l]
    labels = ["Win", "Draw", "Loss"]

    ax.boxplot(
        data_box,
        labels=labels,
        showfliers=False,
        patch_artist=True,
        widths=0.6,
        boxprops=dict(alpha=0.7, edgecolor="black", linewidth=1.5),
        medianprops=dict(color="darkred", linewidth=3),
        whiskerprops=dict(color="black", linewidth=1.5),
        capprops=dict(color="black", linewidth=1.5),
    )

    ax.axhline(0.0, linestyle="--", linewidth=2, color="black", alpha=0.4, zorder=5)
    ax.set_ylabel("Δz₁", fontsize=13, fontweight="bold")
    ax.set_xlabel("Outcome", fontsize=13, fontweight="bold")
    ax.set_title("H3: Drift Distribution by Outcome", fontsize=14, fontweight="bold", pad=15)
    ax.grid(True, alpha=0.25, axis="y", linestyle="--", linewidth=0.5)

    y0, y1 = ax.get_ylim()
    y_pos = y0 + (y1 - y0) * 0.05
    for i, (lab, stat) in enumerate(zip(["Win", "Draw", "Loss"], [stats_win, stats_draw, stats_loss]), 1):
        ax.text(
            i,
            y_pos,
            f"N={stat['N']:,}\nMedian={stat['median']:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8, edgecolor="gray"),
        )

    plt.tight_layout()
    p2 = out_dir / f"{save_prefix}_drift_boxplot.png"
    plt.savefig(p2, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Saved: {p2.name}")

    # ===== FIGURE 3: SCATTER =====
    print("\n📊 Figure 3: Drift vs game length...")

    fig, ax = plt.subplots(figsize=(12, 7))

    ax.scatter(df_h3["game_len"], np.abs(df_h3["drift_z1"]), s=15, alpha=0.3, edgecolors="none", rasterized=True)

    short = df_h3["game_len"] < 40
    long = df_h3["game_len"] > 60

    if short.any():
        short_mean = np.abs(df_h3.loc[short, "drift_z1"]).mean()
        ax.axhline(short_mean, linestyle="--", linewidth=2.5, alpha=0.9, label=f"Short games (<40): mean |Δz₁| = {short_mean:.2f}", zorder=10)

    if long.any():
        long_mean = np.abs(df_h3.loc[long, "drift_z1"]).mean()
        ax.axhline(long_mean, linestyle="--", linewidth=2.5, alpha=0.9, label=f"Long games (>60): mean |Δz₁| = {long_mean:.2f}", zorder=10)

    ax.set_xlabel("Game Length (moves)", fontsize=13, fontweight="bold")
    ax.set_ylabel("|Δz₁|", fontsize=13, fontweight="bold")
    ax.set_title("H3 Robustness: Drift Magnitude vs Game Length", fontsize=14, fontweight="bold", pad=15)
    ax.legend(loc="upper right", fontsize=10, framealpha=0.95, edgecolor="gray")
    ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.5)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    p3 = out_dir / f"{save_prefix}_len_scatter.png"
    plt.savefig(p3, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Saved: {p3.name}")

    # ===== FIGURE 4: PERSPECTIVE =====
    print("\n📊 Figure 4: Drift by perspective...")

    fig, axes = plt.subplots(1, 2, figsize=(15, 7))

    for idx, persp in enumerate(["White", "Black"]):
        ax = axes[idx]
        df_p = df_h3[df_h3["perspective"] == persp]

        stats_p = {
            "Win": _summ(df_p.loc[df_p["outcome"] == "Win", "drift_z1"]),
            "Draw": _summ(df_p.loc[df_p["outcome"] == "Draw", "drift_z1"]),
            "Loss": _summ(df_p.loc[df_p["outcome"] == "Loss", "drift_z1"]),
        }

        data_box_p = [
            df_p.loc[df_p["outcome"] == "Win", "drift_z1"].to_numpy(),
            df_p.loc[df_p["outcome"] == "Draw", "drift_z1"].to_numpy(),
            df_p.loc[df_p["outcome"] == "Loss", "drift_z1"].to_numpy(),
        ]

        ax.boxplot(
            data_box_p,
            labels=["Win", "Draw", "Loss"],
            showfliers=False,
            patch_artist=True,
            widths=0.6,
            boxprops=dict(alpha=0.7, edgecolor="black", linewidth=1.5),
            medianprops=dict(color="darkred", linewidth=3),
            whiskerprops=dict(color="black", linewidth=1.5),
            capprops=dict(color="black", linewidth=1.5),
        )

        ax.axhline(0.0, linestyle="--", linewidth=2, color="black", alpha=0.4, zorder=5)
        ax.set_ylabel("Δz₁", fontsize=12, fontweight="bold")
        ax.set_xlabel("Outcome", fontsize=12, fontweight="bold")
        ax.set_title(f"{persp} Perspective", fontsize=13, fontweight="bold")
        ax.grid(True, alpha=0.25, axis="y", linestyle="--", linewidth=0.5)

        y0, y1 = ax.get_ylim()
        y_pos = y0 + (y1 - y0) * 0.05
        for i, lab in enumerate(["Win", "Draw", "Loss"], 1):
            stat = stats_p[lab]
            ax.text(
                i,
                y_pos,
                f"N={stat['N']:,}\n{stat['median']:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.8, edgecolor="gray"),
            )

    plt.suptitle("H3 Robustness: Drift by Perspective", fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout()

    p4 = out_dir / f"{save_prefix}_perspective_boxplot.png"
    plt.savefig(p4, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Saved: {p4.name}")

    print("\n" + "=" * 80)
    print("✅ FIGURE H3 COMPLETE")
    print("=" * 80)
    print("\n📊 KEY FINDINGS:")
    print(f"  Win:  median Δz₁ = {stats_win['median']:+.3f}  (N={stats_win['N']:,})")
    print(f"  Draw: median Δz₁ = {stats_draw['median']:+.3f}  (N={stats_draw['N']:,})")
    print(f"  Loss: median Δz₁ = {stats_loss['median']:+.3f}  (N={stats_loss['N']:,})")
    print(f"\n  Mann-Whitney: p={p_val:.3e}, r={r_rb:+.3f}")
    print("=" * 80)

    return {
        "paths": {
            "h3_dir": str(h3_dir),
            "drift_csv": str(h3_dir / "drift_data.csv"),
            "stats_json": str(h3_dir / "drift_statistics.json"),
            "hist_png": str(p1),
            "box_png": str(p2),
            "scatter_png": str(p3),
            "persp_png": str(p4),
        },
        "stats": summary_stats,
    }

if __name__ == "__main__":
    import argparse
    from pathlib import Path

    from chess_pca.pipeline import build_or_load_dataset
    from chess_pca.features import numeric_features

    parser = argparse.ArgumentParser(description="Run Figure H3 (Geometric Drift on PC1)")
    parser.add_argument("--input_path", type=str, default="data/raw", help="Path to PGN folder or file")
    parser.add_argument("--max_games", type=int, default=10000)
    parser.add_argument("--min_moves", type=int, default=10)
    parser.add_argument("--cache_dir", type=str, default="data/cache")
    parser.add_argument("--out_dir", type=str, default="data/outputs/figure_h3_data")

    parser.add_argument("--start_move", type=int, default=5)
    parser.add_argument("--exclude_resignation", action="store_true")
    parser.add_argument("--bins", type=int, default=50)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--force_recompute", action="store_true")

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("🧪 RUNNING FIGURE H3 FROM CLI")
    print("=" * 80)
    print(f"input_path={args.input_path}")
    print(f"out_dir={out_dir}")
    print(f"max_games={args.max_games} | min_moves={args.min_moves} | start_move={args.start_move} | bins={args.bins}")
    print("=" * 80 + "\n")

    white_data, black_data = build_or_load_dataset(
        input_path=args.input_path,
        max_games=args.max_games,
        min_moves=args.min_moves,
        cache_dir=args.cache_dir,
        force_recompute=args.force_recompute,
    )

    FEATURES = numeric_features(white_data)

    # run_figure_h3 deve esistere in figure_h3.py
    run_figure_h3(
        white_data=white_data,
        black_data=black_data,
        features=FEATURES,
        out_dir=out_dir,
        start_move=args.start_move,
        exclude_resignation=args.exclude_resignation,
        bins=args.bins,
        random_state=args.random_state,
    )

    print("\n✅ Done. Outputs saved in:", out_dir.resolve())