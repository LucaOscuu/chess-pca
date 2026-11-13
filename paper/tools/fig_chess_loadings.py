"""Figura (Sez. 3.2.1): cosa misurano i due assi della mappa di Fishtest.
Loading delle 18 coordinate su PC1 e PC2, sulla STESSA mappa della Figura 3 (meta di training,
stesso seed, funzioni di src/chess_pca/occupancy_grouped.py). Il segno degli assi e arbitrario.
Controllo: la mappa riproduce occupancy_grouped_full.json, altrimenti si ferma.

Uso (dalla cartella principale della repo):  python paper/tools/fig_chess_loadings.py
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from chess_pca.occupancy_grouped import FEATURES, fit_space, residence, select_and_test  # noqa: E402

SEED = 20260925
REF = json.loads((ROOT / "data/outputs/occupancy_grouped/occupancy_grouped_full.json").read_text())
BLUE, ORANGE, INK, MUTED = "#2a78d6", "#eb6834", "#1f1f1e", "#8a8a85"
LABELS = {
    "material": "Material balance", "mobility": "Mobility", "captures_avail": "Captures available",
    "checks_avail": "Checks available", "attack_overlap": "Contested squares", "hanging_diff": "Undefended pieces",
    "center_core": "Center control", "center_ext": "Extended center", "king_safety_diff": "King safety",
    "pawns_doubled_diff": "Doubled pawns", "pawns_isolated_diff": "Isolated pawns",
    "pawns_passed_diff": "Passed pawns", "cnt_pawn": "Pawns", "cnt_knight": "Knights", "cnt_bishop": "Bishops",
    "cnt_rook": "Rooks", "cnt_queen": "Queens", "delta_mobility": "Change in mobility",
}


def load():
    code = {"W": 1, "D": 0, "L": -1}
    parts = []
    for side, f in (("W", "white"), ("B", "black")):
        d = pd.read_parquet(ROOT / f"data/cache/{f}_data_max10000_min10.parquet")
        d["perspective"] = side
        d["result"] = d["result"].map(code)
        parts.append(d)
    df = pd.concat(parts).sort_values(["game_id", "perspective", "ply"]).reset_index(drop=True)
    df["material_cp"] = df["material"]
    return df


def main():
    df = load()
    rng = np.random.default_rng(SEED)
    games = df["game_id"].unique()
    train = set(rng.choice(games, size=len(games) // 2, replace=False))
    feats = [f for f in FEATURES if df[f].std() > 0]
    mu, sd, W, q = fit_space(df[df["game_id"].isin(train)], feats)
    R = residence(df, mu, sd, W, q, feats, True, 5)
    wr = df[df["perspective"] == "W"].groupby("game_id")["result"].first()
    obs = select_and_test(R, wr, train)
    for p in ("W", "B"):
        if abs(obs[p]["signed_gap_test_pp"] - REF["material_matched"][p]["signed_gap_test_pp"]) > 1e-6:
            sys.exit("La mappa non coincide con quella della Figura 3")

    # varianza spiegata dalle due componenti (sulla meta di training)
    Xt = df[df["game_id"].isin(train)][feats].to_numpy(float)
    Z = (Xt - mu) / sd
    ev = np.linalg.svd(Z - Z.mean(0), compute_uv=False) ** 2
    share = ev[:2] / ev.sum() * 100
    L = pd.DataFrame(W, index=feats, columns=["PC1", "PC2"])
    print("controllo OK; varianza spiegata PC1, PC2 (%):", share.round(1))
    print(L.round(3))
    L.round(4).to_csv(ROOT / "paper/figures/fig_chess_loadings.csv")
    (ROOT / "data/outputs/occupancy_grouped/fig_chess_loadings_evr.json").write_text(json.dumps(
        {"evr_pc1_pct": round(float(share[0]), 2), "evr_pc2_pct": round(float(share[1]), 2),
         "evr_pc12_pct": round(float(share[:2].sum()), 2), "nota": "meta di training della mappa di Fishtest (Fig. occupancy)"}, indent=1), encoding="utf-8")

    order = L["PC1"].abs().sort_values().index      # dal piu piccolo al piu grande, dal basso verso l'alto
    y = np.arange(len(order))
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.6), sharey=True)
    for ax, pc, sh in zip(axes, ("PC1", "PC2"), share):
        v = L.loc[order, pc].to_numpy()
        ax.barh(y, v, color=np.where(v >= 0, BLUE, ORANGE), height=0.7)
        ax.axvline(0, color=INK, lw=0.8)
        ax.set_xlim(-0.65, 0.65)
        ax.set_title(f"{'A' if pc == 'PC1' else 'B'}. Weights on {pc} ({sh:.0f}% of variance)", loc="left",
                     fontsize=10, color=INK)
        ax.set_xlabel("Weight (loading)")
        ax.grid(axis="x", color="#e6e5e0", lw=0.6); ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(colors=INK, labelsize=8.5)
    axes[0].set_yticks(y, [LABELS[f] for f in order])
    fig.tight_layout()
    out = ROOT / "paper/figures/fig_chess_loadings.pdf"
    fig.savefig(out); fig.savefig(out.with_suffix(".png"), dpi=200)
    print("scritta", out)


if __name__ == "__main__":
    main()
