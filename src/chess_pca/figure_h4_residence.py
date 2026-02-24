from __future__ import annotations

"""
Residence-Time Analysis for Attractor Basins (H4) — Linear threshold version
=============================================================================
Per ciascuna traiettoria (game_id) calcola la frazione di tempo trascorsa in:

  PC1:
  - Omega_win  = { z : PC1 < -1.0 }
  - Omega_loss = { z : PC1 > +0.75 }

  PC2:
  - Omega_win2  = { z : PC2 < -0.75 }
  - Omega_loss2 = { z : PC2 > +0.75 }

Le soglie sono derivate a priori dalle traiettorie medie per outcome (Figura 8),
non dai dati di outcome stessi — nessuna circolarità.

Stampa:
  - mediana + IQR (wins vs losses) per PC1 e PC2
  - Spearman tra residence fraction e lunghezza traiettoria
  - Partial Spearman (controlling for trajectory length) tra residence e outcome
  - Salva CSV + JSON + PNG in out_dir
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.linalg import lstsq
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# ── thresholds PC1 ────────────────────────────────────────────────────────────
OMEGA_WIN_THRESHOLD  = -1.00   # PC1 < this  → Omega_win
OMEGA_LOSS_THRESHOLD = +0.75   # PC1 > this  → Omega_loss

# ── thresholds PC2 ────────────────────────────────────────────────────────────
OMEGA_WIN2_THRESHOLD  = -0.75  # PC2 < this  → Omega_win2
OMEGA_LOSS2_THRESHOLD = +0.75  # PC2 > this  → Omega_loss2


# ── helpers ───────────────────────────────────────────────────────────────────

def _normalize_result_series(s: pd.Series) -> pd.Series:
    s2 = s.astype(str).str.strip().str.upper()
    mapping = {
        "1-0": "1-0", "0-1": "0-1",
        "1/2-1/2": "1/2-1/2", "1/2 - 1/2": "1/2-1/2",
        "0.5-0.5": "1/2-1/2", "0.5 - 0.5": "1/2-1/2",
        "W": "1-0", "L": "0-1", "D": "1/2-1/2",
        "WHITE": "1-0", "BLACK": "0-1",
        "WHITE WIN": "1-0", "BLACK WIN": "0-1",
        "DRAW": "1/2-1/2", "TIE": "1/2-1/2",
        "1": "1-0", "-1": "0-1", "0": "1/2-1/2",
    }
    return s2.map(lambda x: mapping.get(x, x))


def _project_to_pcs(
    data: pd.DataFrame,
    features: list[str],
    pca: PCA,
    scaler: StandardScaler,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (pc1_scores, pc2_scores) as plain arrays."""
    X = data[features].to_numpy()
    X_scaled = scaler.transform(X)
    scores = pca.transform(X_scaled)
    return scores[:, 0], scores[:, 1]


def partial_spearman(
    x: np.ndarray,
    y: np.ndarray,
    covariate: np.ndarray,
) -> tuple[float, float]:
    """Partial Spearman correlation between x and y, controlling for covariate."""
    from scipy.stats import rankdata

    xr = rankdata(x).astype(float)
    yr = rankdata(y).astype(float)
    cr = rankdata(covariate).astype(float)

    def _residualize(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        B = np.column_stack([b, np.ones(len(b))])
        coef, *_ = lstsq(B, a, rcond=None)
        return a - B @ coef

    rx = _residualize(xr, cr)
    ry = _residualize(yr, cr)
    return stats.spearmanr(rx, ry)


# ── core computation ──────────────────────────────────────────────────────────

def compute_residence_times(
    data: pd.DataFrame,
    features: list[str],
    pca: PCA,
    scaler: StandardScaler,
    omega_win_thresh: float   = OMEGA_WIN_THRESHOLD,
    omega_loss_thresh: float  = OMEGA_LOSS_THRESHOLD,
    omega_win2_thresh: float  = OMEGA_WIN2_THRESHOLD,
    omega_loss2_thresh: float = OMEGA_LOSS2_THRESHOLD,
) -> pd.DataFrame:
    """
    For each game_id in `data`, compute residence fractions for both PC1 and PC2.
    """
    if "game_id" not in data.columns:
        raise KeyError("Expected column 'game_id' in data.")

    pc1_vals, pc2_vals = _project_to_pcs(data, features, pca, scaler)

    df = data.copy().reset_index(drop=True)
    df["_pc1"] = pc1_vals
    df["_pc2"] = pc2_vals

    if "result" in df.columns:
        df["result"] = _normalize_result_series(df["result"])

    records: list[dict] = []
    for gid, grp in df.groupby("game_id", sort=False):
        n = len(grp)
        if n <= 0:
            continue

        row: dict = {
            "game_id":            gid,
            "trajectory_length":  int(n),
            # PC1 regions
            "frac_win_region":    float((grp["_pc1"] < omega_win_thresh).sum()  / n),
            "frac_loss_region":   float((grp["_pc1"] > omega_loss_thresh).sum() / n),
            # PC2 regions
            "frac_win2_region":   float((grp["_pc2"] < omega_win2_thresh).sum()  / n),
            "frac_loss2_region":  float((grp["_pc2"] > omega_loss2_thresh).sum() / n),
        }

        if "result" in grp.columns:
            row["result"] = grp["result"].iloc[0]

        records.append(row)

    return pd.DataFrame(records).reset_index(drop=True)


# ── stats helpers ─────────────────────────────────────────────────────────────

def _print_region_stats(df_win, df_loss, col, thresh_label):
    print(f"\n  {thresh_label}")
    for label, sub in [("Wins", df_win), ("Losses", df_loss)]:
        if len(sub) == 0:
            print(f"    {label:8s}: N=0")
            continue
        med = sub[col].median()
        q1  = sub[col].quantile(0.25)
        q3  = sub[col].quantile(0.75)
        print(f"    {label:8s}: median={med:.0%}  IQR=[{q1:.0%}, {q3:.0%}]")


def _spearman_block(df_win, df_loss, df_all, col_win, col_loss, name):
    rho, p, rho_p, p_p = None, None, None, None

    if len(df_win) > 5:
        r, pv = stats.spearmanr(df_win["trajectory_length"], df_win[col_win])
        print(f"  Spearman (wins)   — length vs {col_win}:  ρ={r:.2f}, p={pv:.2e}")

    if len(df_loss) > 5:
        r, pv = stats.spearmanr(df_loss["trajectory_length"], df_loss[col_loss])
        print(f"  Spearman (losses) — length vs {col_loss}: ρ={r:.2f}, p={pv:.2e}")

    if len(df_all) > 5:
        rho, p = stats.spearmanr(df_all[col_win], df_all["is_win"])
        print(f"\n  Spearman (all)    — {col_win} vs outcome:       ρ={rho:.2f}, p={p:.2e}")

        rho_p, p_p = partial_spearman(
            df_all[col_win].to_numpy(),
            df_all["is_win"].to_numpy(),
            df_all["trajectory_length"].to_numpy(),
        )
        print(f"  Partial Spearman  — {col_win} vs outcome        ρ={rho_p:.2f}, p={p_p:.2e}")
        print(f"                      (controlling for trajectory length)")

    return rho, p, rho_p, p_p


# ── plots ─────────────────────────────────────────────────────────────────────

def _plot_residence(
    df_win: pd.DataFrame,
    df_loss: pd.DataFrame,
    col_win: str,
    col_loss: str,
    thresh_win: float,
    thresh_loss: float,
    pc_label: str,
    name: str,
    fig_dir: Path,
):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle(
        f"Residence-Time in Attractor Basins — {name.capitalize()} ({pc_label})\n"
        f"Ω_win: {pc_label} < {thresh_win}   |   Ω_loss: {pc_label} > {thresh_loss}",
        fontsize=13, fontweight="bold",
    )

    for ax, col, region_label in zip(
        axes,
        [col_win, col_loss],
        [f"Ω_win ({pc_label} < {thresh_win})",
         f"Ω_loss ({pc_label} > {thresh_loss})"],
    ):
        bins = np.linspace(0, 1, 31)

        ax.hist(df_win[col]  if len(df_win)  else [],
                bins=bins, alpha=0.65, color="steelblue",
                label=f"Wins (N={len(df_win)})", edgecolor="none")
        ax.hist(df_loss[col] if len(df_loss) else [],
                bins=bins, alpha=0.65, color="tomato",
                label=f"Losses (N={len(df_loss)})", edgecolor="none")

        if len(df_win):
            ax.axvline(df_win[col].median(), color="steelblue",
                       linestyle="--", linewidth=1.8,
                       label=f"Win median={df_win[col].median():.0%}")
        if len(df_loss):
            ax.axvline(df_loss[col].median(), color="tomato",
                       linestyle="--", linewidth=1.8,
                       label=f"Loss median={df_loss[col].median():.0%}")

        ax.set_title(region_label, fontsize=11, fontweight="bold")
        ax.set_xlabel("Fraction of trajectory in region", fontsize=10)
        ax.set_ylabel("# Games", fontsize=10)
        ax.legend(fontsize=9, framealpha=0.9)
        ax.set_xlim(0, 1)
        ax.grid(True, alpha=0.25, linestyle="--")

    plt.tight_layout()
    png_path = fig_dir / f"residence_{name}_{pc_label.lower()}.png"
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  💾 Saved: {png_path}")


# ── main analysis ─────────────────────────────────────────────────────────────

def run_residence_analysis(
    white_data: pd.DataFrame,
    black_data: pd.DataFrame,
    features: list[str],
    out_dir: str | Path,
    omega_win_thresh: float   = OMEGA_WIN_THRESHOLD,
    omega_loss_thresh: float  = OMEGA_LOSS_THRESHOLD,
    omega_win2_thresh: float  = OMEGA_WIN2_THRESHOLD,
    omega_loss2_thresh: float = OMEGA_LOSS2_THRESHOLD,
    random_state: int = 42,
) -> dict[str, pd.DataFrame]:
    """
    Full linear-threshold residence-time analysis for H4 (PC1 + PC2).

    Soglie definite a priori dalle traiettorie medie per outcome (Figura 8).
    Nessuna circolarità: la definizione dei basin non dipende dall'outcome
    delle singole partite.

    Produce 4 PNG per colore (PC1 e PC2), CSV e JSON.
    Returns {"white": df_white, "black": df_black}
    """
    out_dir  = Path(out_dir)
    base_dir = out_dir / "h4_residence"
    data_dir = base_dir / "data"
    fig_dir  = base_dir / "figures"
    data_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, pd.DataFrame] = {}

    for name, data in [("white", white_data), ("black", black_data)]:
        print(f"\n{'='*70}")
        print(f"  {name.upper()} — Residence-Time Analysis (H4) — Linear thresholds")
        print(f"{'='*70}")

        # ── fit PCA ──────────────────────────────────────────────────────────
        X        = data[features].to_numpy()
        scaler   = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        pca      = PCA(n_components=2, random_state=random_state)
        pca.fit(X_scaled)

        ev = pca.explained_variance_ratio_
        print(f"  PCA variance explained: PC1={ev[0]:.2%}, PC2={ev[1]:.2%}")

        # ── normalize results ─────────────────────────────────────────────────
        data = data.copy().reset_index(drop=True)
        if "result" in data.columns:
            data["result"] = _normalize_result_series(data["result"])
            print("\n  Result value counts (normalized):")
            print(data["result"].value_counts(dropna=False).head(10).to_string())

        # ── per-game residence fractions ──────────────────────────────────────
        df_res = compute_residence_times(
            data=data,
            features=features,
            pca=pca,
            scaler=scaler,
            omega_win_thresh=omega_win_thresh,
            omega_loss_thresh=omega_loss_thresh,
            omega_win2_thresh=omega_win2_thresh,
            omega_loss2_thresh=omega_loss2_thresh,
        )

        # ── save CSV ──────────────────────────────────────────────────────────
        csv_path = data_dir / f"residence_{name}.csv"
        df_res.to_csv(csv_path, index=False)
        print(f"\n  💾 Saved: {csv_path}")

        if "result" not in df_res.columns:
            print("  ⚠️  No 'result' column — cannot split by outcome.")
            results[name] = df_res
            continue

        # ── outcome masks — built safely on reset index ───────────────────────
        if name == "white":
            win_mask  = df_res["result"] == "1-0"
            loss_mask = df_res["result"] == "0-1"
        else:
            win_mask  = df_res["result"] == "0-1"
            loss_mask = df_res["result"] == "1-0"

        df_win  = df_res[win_mask].copy().reset_index(drop=True)
        df_loss = df_res[loss_mask].copy().reset_index(drop=True)
        df_draw = df_res[~win_mask & ~loss_mask].copy().reset_index(drop=True)

        print(
            f"\n  Games: {len(df_res)} total | "
            f"{len(df_win)} wins | {len(df_loss)} losses | {len(df_draw)} draws"
        )

        # ── df_all with is_win — safe, index-independent ──────────────────────
        df_all = df_res[win_mask | loss_mask].copy().reset_index(drop=True)
        df_all["is_win"] = (
            df_res.loc[win_mask | loss_mask, "result"]
            .reset_index(drop=True)
            .map({"1-0": (1 if name == "white" else 0),
                  "0-1": (0 if name == "white" else 1)})
            .astype(int)
        )

        # ── PC1 stats + Spearman ──────────────────────────────────────────────
        print(f"\n  ── PC1 REGIONS ──")
        _print_region_stats(df_win, df_loss, "frac_win_region",
                            f"Ω_win  (PC1 < {omega_win_thresh})")
        _print_region_stats(df_win, df_loss, "frac_loss_region",
                            f"Ω_loss (PC1 > {omega_loss_thresh})")
        print()
        rho1, p1, rho1p, p1p = _spearman_block(
            df_win, df_loss, df_all,
            "frac_win_region", "frac_loss_region", name
        )

        # ── PC2 stats + Spearman ──────────────────────────────────────────────
        print(f"\n  ── PC2 REGIONS ──")
        _print_region_stats(df_win, df_loss, "frac_win2_region",
                            f"Ω_win2  (PC2 < {omega_win2_thresh})")
        _print_region_stats(df_win, df_loss, "frac_loss2_region",
                            f"Ω_loss2 (PC2 > {omega_loss2_thresh})")
        print()
        rho2, p2, rho2p, p2p = _spearman_block(
            df_win, df_loss, df_all,
            "frac_win2_region", "frac_loss2_region", name
        )

        # ── JSON summary ──────────────────────────────────────────────────────
        def _iqr(s: pd.Series) -> list[float]:
            return [float(s.quantile(0.25)), float(s.quantile(0.75))]

        summary: dict = {
            "perspective":         name,
            "omega_win_thresh":    float(omega_win_thresh),
            "omega_loss_thresh":   float(omega_loss_thresh),
            "omega_win2_thresh":   float(omega_win2_thresh),
            "omega_loss2_thresh":  float(omega_loss2_thresh),
            "n_total": int(len(df_res)),
            "n_win":   int(len(df_win)),
            "n_loss":  int(len(df_loss)),
            "n_draw":  int(len(df_draw)),
            "pca_evr_pc1": float(ev[0]),
            "pca_evr_pc2": float(ev[1]),
        }

        for grp_label, sub in [("win", df_win), ("loss", df_loss)]:
            for reg in ["win", "loss", "win2", "loss2"]:
                col = f"frac_{reg}_region"
                if len(sub) > 0:
                    summary[f"{grp_label}_{col}_median"] = float(sub[col].median())
                    summary[f"{grp_label}_{col}_iqr"]    = _iqr(sub[col])

        for tag, rho, p in [
            ("pc1_spearman",         rho1,  p1),
            ("pc1_partial_spearman", rho1p, p1p),
            ("pc2_spearman",         rho2,  p2),
            ("pc2_partial_spearman", rho2p, p2p),
        ]:
            if rho is not None:
                summary[f"{tag}_rho"] = float(rho)
                summary[f"{tag}_p"]   = float(p)

        json_path = data_dir / f"residence_{name}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"\n  💾 Saved: {json_path}")

        # ── plots: PC1 e PC2 ──────────────────────────────────────────────────
        _plot_residence(
            df_win, df_loss,
            "frac_win_region", "frac_loss_region",
            omega_win_thresh, omega_loss_thresh,
            "PC1", name, fig_dir,
        )
        _plot_residence(
            df_win, df_loss,
            "frac_win2_region", "frac_loss2_region",
            omega_win2_thresh, omega_loss2_thresh,
            "PC2", name, fig_dir,
        )

        results[name] = df_res

    print(f"\n{'='*70}")
    print("✅ RESIDENCE-TIME ANALYSIS COMPLETE (H4 — linear thresholds)")
    print(f"{'='*70}\n")

    return results


# ── standalone entry-point ────────────────────────────────────────────────────
if __name__ == "__main__":
    from chess_pca.features import numeric_features
    from chess_pca.pipeline import build_or_load_dataset

    white_data, black_data = build_or_load_dataset(
        input_path="data/raw",
        max_games=10000,
        min_moves=10,
        cache_dir="data/cache",
    )

    features = numeric_features(white_data)

    run_residence_analysis(
        white_data=white_data,
        black_data=black_data,
        features=features,
        out_dir="data/outputs",
    )