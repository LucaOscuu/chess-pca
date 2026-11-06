"""Figura dell'introduzione: come vincitori e perdenti percorrono la mappa di Fishtest.
Stessa mappa della Figura 3 (stessa divisione training/test, stesso seed, stesse funzioni di
src/chess_pca/occupancy_grouped.py). Mostra le traiettorie medie della meta di test, SENZA
appaiare il materiale: e un'illustrazione, la misura vera e la Figura 3.
Controllo: la mappa e i quartili coincidono con quelli che riproducono occupancy_grouped_full.json.

Uso (dalla cartella principale della repo):  python paper/tools/fig0_trajectories.py
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

SEED, MAXMOVE = 20260925, 60
REF = json.loads((ROOT / "data/outputs/occupancy_grouped/occupancy_grouped_full.json").read_text())
BLUE, ORANGE, GREY, INK, MUTED = "#2a78d6", "#eb6834", "#9a9a95", "#1f1f1e", "#8a8a85"


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

    # controllo: stessa mappa della Figura 3
    R = residence(df, mu, sd, W, q, feats, True, 5)
    wr = df[df["perspective"] == "W"].groupby("game_id")["result"].first()
    obs = select_and_test(R, wr, train)
    for p in ("W", "B"):
        if abs(obs[p]["signed_gap_test_pp"] - REF["material_matched"][p]["signed_gap_test_pp"]) > 1e-6:
            sys.exit("La mappa non coincide con quella della Figura 3")
    # il risultato del Nero nella cache e gia dal punto di vista del Nero
    br = df[df["perspective"] == "B"].groupby("game_id")["result"].first()
    if not (br.reindex(wr.index) == -wr).all():
        sys.exit("Convenzione del risultato inattesa nella cache")
    print("controllo OK: mappa della Figura 3; quartili PC1", q[:, 0].round(3))

    te = df[~df["game_id"].isin(train)].copy()
    S = ((te[feats].to_numpy(float) - mu) / sd) @ W
    te["pc1"], te["pc2"] = S[:, 0], S[:, 1]
    te["move"] = te.groupby(["game_id", "perspective"]).cumcount() + 1
    te = te[te["move"] <= MAXMOVE]

    groups = (("Wins", 1, BLUE), ("Draws", 0, GREY), ("Losses", -1, ORANGE))
    fig, (a, b) = plt.subplots(1, 2, figsize=(9.2, 3.9), gridspec_kw={"width_ratios": [1.15, 1]})

    # A. PC1 mossa per mossa, media con IC bootstrap sulle partite
    boot = np.random.default_rng(1)
    lo_all, hi_all = [], []
    for lab, val, col in groups:
        t = te[te["result"] == val]
        piv = t.pivot_table(index="game_id", columns="move", values="pc1", aggfunc="mean")
        m = piv.mean().to_numpy()
        arr = piv.to_numpy()
        bs = []
        for _ in range(300):
            ix = boot.integers(0, arr.shape[0], arr.shape[0])
            bs.append(np.nanmean(arr[ix], axis=0))
        lo, hi = np.nanpercentile(bs, [2.5, 97.5], axis=0)
        x = piv.columns.to_numpy()
        a.fill_between(x, lo, hi, color=col, alpha=0.2, lw=0)
        a.plot(x, m, color=col, lw=1.8, label=lab)
        lo_all.append(lo.min()); hi_all.append(hi.max())
    ylo, yhi = min(lo_all) - 0.3, max(hi_all) + 0.3
    a.axhspan(ylo, q[0, 0], color=BLUE, alpha=0.10, lw=0)
    a.axhline(q[0, 0], color=MUTED, lw=0.8, ls="--")
    a.text(MAXMOVE, q[0, 0] - 0.08, "Selected region (lower tail of PC1)", ha="right", va="top", fontsize=8, color=INK)
    a.set_ylim(ylo, yhi); a.set_xlim(1, MAXMOVE)
    a.set_xlabel("Move of the player"); a.set_ylabel("PC1")
    a.set_title("A. Position on the first component, move by move", loc="left", fontsize=10, color=INK)
    a.legend(frameon=False, fontsize=8.5, loc="upper left")

    # B. percorso medio sul piano
    for lab, val, col in groups:
        t = te[te["result"] == val].groupby("move")[["pc1", "pc2"]].mean()
        ts = t.rolling(5, center=True, min_periods=1).mean()  # media mobile di 5 mosse, solo per leggibilita
        ts.iloc[:3] = t.iloc[:3]                               # le prime mosse restano grezze (partono dallo start)
        t = ts
        b.plot(t["pc1"], t["pc2"], color=col, lw=1.8, label=lab)
        mk = t.loc[[10, 30, 60]]
        b.scatter(mk["pc1"], mk["pc2"], color=col, s=16, zorder=3)
        if val != 0:
            off = (6, 3) if val == -1 else (7, -4)
            for mv in (10, 30, 60):
                b.annotate(str(mv), (mk.loc[mv, "pc1"], mk.loc[mv, "pc2"]), xytext=off,
                           textcoords="offset points", fontsize=7.5, color=col)
    start = te[te["move"] == 1][["pc1", "pc2"]].mean()
    b.scatter([start["pc1"]], [start["pc2"]], marker="o", s=40, facecolor="white", edgecolor=INK, zorder=4)
    b.annotate("start", (start["pc1"], start["pc2"]), xytext=(6, 4), textcoords="offset points", fontsize=8, color=INK)
    xl, yl = b.get_xlim(), b.get_ylim()
    b.axvspan(xl[0], q[0, 0], color=BLUE, alpha=0.10, lw=0)
    for x in q[:, 0]:
        b.axvline(x, color=MUTED, lw=0.8, ls="--")
    for y in q[:, 1]:
        b.axhline(y, color=MUTED, lw=0.8, ls="--")
    b.set_xlim(xl); b.set_ylim(yl)
    b.set_xlabel("PC1"); b.set_ylabel("PC2")
    b.set_title("B. Average path on the map", loc="left", fontsize=10, color=INK)

    for ax in (a, b):
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(colors=INK, labelsize=8.5)
    fig.tight_layout()
    out = ROOT / "paper/figures/fig0_trajectories.pdf"
    fig.savefig(out); fig.savefig(out.with_suffix(".png"), dpi=200)
    n = te.groupby("result")["game_id"].nunique().to_dict()
    print("partite di test per esito (prospettive contate a parte):", n)
    print("scritta", out)


if __name__ == "__main__":
    main()
