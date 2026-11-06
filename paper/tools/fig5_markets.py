"""Figura (Sez. 3.3): la mappa dei mercati sperimentali (replica di Huber et al., 2024).
Pannello A: loading delle 13 coordinate dello strumento completo su PC1 e PC2 (mappa stimata su tutti
i mercati). Pannello B: gap di occupazione a ricchezza appaiata contro il placebo. Pannello C: rotazione
del piano fra trader ad alta e bassa abilita e fra prima e seconda ripetizione, contro il null.
Nessun nuovo calcolo: legge solo data/outputs/mercati/risultati_mercati.json (scritto da
src/chess_pca/mercati/analisi_mercati.py). Controllo: i numeri devono coincidere con numbers.tex.

Uso (dalla cartella principale della repo):  python paper/tools/fig5_markets.py
"""
import json, re, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
J = json.loads((ROOT / "data/outputs/mercati/risultati_mercati.json").read_text(encoding="utf-8"))
NUM = (ROOT / "paper/numbers.tex").read_text(encoding="utf-8")
BLUE, ORANGE, INK, MUTED = "#2a78d6", "#eb6834", "#1f1f1e", "#8a8a85"
LABELS = {
    "assets_rel": "Asset holdings", "cash_rel": "Cash", "wealth_rel": "Wealth at fundamental value",
    "n_bid": "Bids placed", "n_ask": "Asks placed", "mkt_frac": "Share of market orders",
    "cancel": "Share of orders cancelled", "bid_dev": "Bid price vs value", "ask_dev": "Ask price vs value",
    "buy_q": "Units bought", "sell_q": "Units sold", "buy_dev": "Purchase price vs value",
    "sell_dev": "Sale price vs value",
}


def val(key):
    return float(re.search(r"val@" + re.escape(key) + r"\\endcsname\{([^}]*)\}", NUM).group(1))


def main():
    F, B = J["completo"], J["solo_comportamento"]
    occ = "T2_occupazione_ricchezza_appaiata"
    checks = []
    for k, r in (("full", F), ("beh", B)):
        checks += [(r[occ]["gap_test_pp"], val(f"mkt.{k}.occ.mm"), 1), (r[occ]["placebo_p95"], val(f"mkt.{k}.occ.mm.p95"), 1),
                   (r["T1_abilita_alta_vs_bassa"]["angolo"], val(f"mkt.{k}.rot.skill"), 1),
                   (r["T1_abilita_alta_vs_bassa"]["null_p95"], val(f"mkt.{k}.rot.skill.p95"), 1),
                   (r["T1_rep1_vs_rep2"]["angolo"], val(f"mkt.{k}.rot.rep"), 1),
                   (r["T1_rep1_vs_rep2"]["null_p95"], val(f"mkt.{k}.rot.rep.p95"), 1)]
    for got, ref, nd in checks:
        if abs(got - ref) > 0.5 * 10 ** -nd + 1e-9:   # numbers.tex arrotonda a meta verso l'alto
            sys.exit(f"Numero diverso da numbers.tex: {got} vs {ref}")
    if len(F["loadings_PC1"]) != int(val("mkt.full.nfeat")):
        sys.exit("Numero di coordinate diverso da numbers.tex")
    print("controllo OK")

    fig = plt.figure(figsize=(9.6, 4.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.35, 1], hspace=0.75, wspace=0.42)
    a = fig.add_subplot(gs[:, 0]); b = fig.add_subplot(gs[0, 1]); c = fig.add_subplot(gs[1, 1])

    # A. loading
    l1, l2 = F["loadings_PC1"], F["loadings_PC2"]
    order = sorted(l1, key=lambda k: abs(l1[k]))
    y = np.arange(len(order))
    a.barh(y + 0.19, [l1[k] for k in order], height=0.36, color="#3d3d3a", label="PC1")
    a.barh(y - 0.19, [l2[k] for k in order], height=0.36, color="#b9b8b1", label="PC2")
    a.axvline(0, color=INK, lw=0.8)
    a.set_yticks(y, [LABELS[k] for k in order])
    a.set_xlim(-0.55, 0.55)
    a.set_xlabel("Weight (loading)")
    ev = F["T0_varianza_spiegata_PC1_PC2"] * 100
    a.set_title(f"A. Weights on the market map (PC1 and PC2: {ev:.0f}% of variance)", loc="left", fontsize=10, color=INK)
    a.legend(frameon=False, fontsize=8.5, loc="lower right")
    a.grid(axis="x", color="#e6e5e0", lw=0.6); a.set_axisbelow(True)

    # B. occupazione a ricchezza appaiata
    rows = (("Full\ninstrument", F, BLUE), ("Behavior\nonly", B, ORANGE))
    for i, (lab, r, col) in enumerate(rows):
        g, p95 = r[occ]["gap_test_pp"], r[occ]["placebo_p95"]
        b.bar(i, g, width=0.55, color=col)
        b.plot([i - 0.36, i + 0.36], [p95, p95], color=INK, lw=1.6)
        b.text(i, g + 0.4, f"{g:.1f}", ha="center", fontsize=8.5, color=INK)
    b.text(-0.36, F[occ]["placebo_p95"] + 0.45, "placebo (95th pct.)", ha="left", fontsize=7.5, color=INK)
    b.set_xticks(range(2), [r[0] for r in rows])
    b.set_xlim(-0.5, 1.5); b.set_ylim(0, 12)
    b.set_ylabel("Winners minus losers (pp)")
    b.set_title("B. Time in the EAB, equal wealth", loc="left", fontsize=10, color=INK)

    # C. rotazioni
    keys = (("High vs low\nskill", "T1_abilita_alta_vs_bassa"), ("First vs second\nrepetition", "T1_rep1_vs_rep2"))
    for i, (lab, k) in enumerate(keys):
        for j, (r, col) in enumerate(((F, BLUE), (B, ORANGE))):
            x = i + (j - 0.5) * 0.36
            c.bar(x, r[k]["angolo"], width=0.32, color=col,
                  label=(None if i else ("Full instrument" if j == 0 else "Behavior only")))
            c.plot([x - 0.17, x + 0.17], [r[k]["null_p95"]] * 2, color=INK, lw=1.6)
            c.text(x, r[k]["angolo"] + 0.6, f"{r[k]['angolo']:.1f}°", ha="center", fontsize=7.5, color=INK)
    c.set_xticks(range(2), [k[0] for k in keys])
    c.set_ylim(0, 27)
    c.legend(frameon=False, fontsize=7.5, loc="upper right")
    c.set_ylabel("Rotation (degrees)")
    c.set_title("C. How much the map turns", loc="left", fontsize=10, color=INK)

    for ax in (a, b, c):
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(colors=INK, labelsize=8)
    for ax in (b, c):
        ax.grid(axis="y", color="#e6e5e0", lw=0.6); ax.set_axisbelow(True)
    out = ROOT / "paper/figures/fig5_markets.pdf"
    fig.savefig(out, bbox_inches="tight"); fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    print("scritta", out)


if __name__ == "__main__":
    main()
