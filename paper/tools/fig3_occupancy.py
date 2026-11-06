"""Figura 3 (Sez. 4): mappa di Fishtest con la regione EAB e quote di tempo di vittorie e sconfitte
a materiale appaiato. Riusa SENZA modifiche le funzioni di src/chess_pca/occupancy_grouped.py
(stessa divisione training/test, stesso seed, stesse regioni dai quartili del training).
Controllo: il gap ricalcolato deve coincidere con occupancy_grouped_full.json, altrimenti si ferma.

Uso (dalla cartella principale della repo):  python paper/tools/fig3_occupancy.py
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from chess_pca.occupancy_grouped import FEATURES, REGIONS, fit_space, residence, select_and_test  # noqa: E402

SEED, CP = 20260925, 50
REF = json.loads((ROOT / "data/outputs/occupancy_grouped/occupancy_grouped_full.json").read_text())
BLUE, ORANGE, INK, MUTED = "#2a78d6", "#eb6834", "#1f1f1e", "#8a8a85"


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
        ref = REF["material_matched"][p]["signed_gap_test_pp"]
        if abs(obs[p]["signed_gap_test_pp"] - ref) > 1e-6 or obs[p]["region"] != REF["material_matched"][p]["region"]:
            sys.exit(f"Il ricalcolo non coincide con il file dei risultati ({p}): {obs[p]} vs {ref}")
    print("controllo OK:", {p: round(obs[p]["signed_gap_test_pp"], 2) for p in obs}, obs["W"]["region"])

    # posizioni del test, a materiale appaiato, proiettate sulla mappa del training
    test = df[~df["game_id"].isin(train) & (df["material_cp"].abs() <= CP)]
    S = ((test[feats].to_numpy(float) - mu) / sd) @ W

    fig, (a, b) = plt.subplots(1, 2, figsize=(9.2, 3.9), gridspec_kw={"width_ratios": [1.25, 1]})
    a.hexbin(S[:, 0], S[:, 1], gridsize=55, cmap="Greys", mincnt=1, bins="log", linewidths=0)
    lo, hi = np.percentile(S, [0.5, 99.5], axis=0)
    a.set_xlim(lo[0], hi[0]); a.set_ylim(lo[1], hi[1])
    reg = obs["W"]["region"]                     # PC1_low per entrambe le prospettive
    a.axvspan(lo[0], q[0, 0], color=BLUE, alpha=0.18, lw=0)
    for x in q[:, 0]:
        a.axvline(x, color=MUTED, lw=0.8, ls="--")
    for y in q[:, 1]:
        a.axhline(y, color=MUTED, lw=0.8, ls="--")
    a.text(lo[0] + 0.03 * (hi[0] - lo[0]), hi[1] - 0.06 * (hi[1] - lo[1]), "Selected region\n(lower tail of PC1)",
           color=INK, fontsize=8.5, va="top")
    a.set_xlabel("PC1"); a.set_ylabel("PC2")
    a.set_title("A. Map of test positions, equal material", loc="left", fontsize=10, color=INK)

    # quota di tempo nella regione per vittorie e sconfitte (test), con IC bootstrap sulle partite
    j = REGIONS.index(reg)
    Rt = R[~R["game_id"].isin(train)].copy()
    res = Rt["game_id"].map(wr).to_numpy()
    Rt["res"] = np.where(Rt["perspective"] == "W", res, -res)
    boot = np.random.default_rng(1)
    xs = np.arange(2)
    for k, (lab, val, col) in enumerate((("Wins", 1, BLUE), ("Losses", -1, ORANGE))):
        means, errs = [], []
        for p in ("W", "B"):
            v = Rt.loc[(Rt["perspective"] == p) & (Rt["res"] == val), REGIONS[j]].to_numpy() * 100
            bs = [boot.choice(v, v.size).mean() for _ in range(1000)]
            means.append(v.mean()); errs.append([v.mean() - np.percentile(bs, 2.5), np.percentile(bs, 97.5) - v.mean()])
        b.bar(xs + (k - 0.5) * 0.36, means, width=0.32, color=col, label=lab, yerr=np.array(errs).T,
              error_kw=dict(ecolor=INK, lw=1, capsize=3))
    b.set_xticks(xs, ["White", "Black"])
    b.set_ylabel("Share of plies in region (%)")
    b.set_title("B. Time in the region, test half", loc="left", fontsize=10, color=INK)
    b.legend(frameon=False, fontsize=8.5)
    for ax in (a, b):
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(colors=INK, labelsize=8.5)
    b.grid(axis="y", color="#e6e5e0", lw=0.6); b.set_axisbelow(True)
    fig.tight_layout()
    out = ROOT / "paper/figures/fig3_occupancy.pdf"
    fig.savefig(out); fig.savefig(out.with_suffix(".png"), dpi=200)
    print("scritta", out)


if __name__ == "__main__":
    main()
