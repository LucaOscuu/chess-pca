"""Figura (Sez. 3.2.2): cambiare cio che il gioco premia cambia la mappa.
Pannello A: per ogni coordinata, correlazione con l'esito in depth 6 (gioco standard) contro antichess
e Three-Check (T3_27_09.json). Pannello B: rotazione del piano rispetto a depth 6 e valore critico
del null appaiato (T5_27_09.json). Nessun nuovo calcolo: legge solo i file dei risultati.
Controllo: i numeri devono coincidere con quelli di numbers.tex, altrimenti si ferma.

Uso (dalla cartella principale della repo):  python paper/tools/fig4_chess_objective.py
"""
import json, re, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "data/outputs/analisi_settembre/27_09"
T3 = json.loads((D / "T3_27_09.json").read_text().replace("NaN", "null"))
T5 = json.loads((D / "T5_27_09.json").read_text())
NUM = (ROOT / "paper/numbers.tex").read_text(encoding="utf-8")
BLUE, ORANGE, INK, MUTED = "#2a78d6", "#eb6834", "#1f1f1e", "#8a8a85"
THR = 0.05


def val(key):
    m = re.search(r"val@" + re.escape(key) + r"\\endcsname\{([^}]*)\}", NUM)
    return m.group(1)


def main():
    # controlli contro numbers.tex
    inv = {}
    for v in ("antichess", "3check"):
        det = T3[v]["dettaglio"]
        inv[v] = sum(1 for x in det.values() if x["tgt"] is not None and abs(x["ref"]) > THR
                     and abs(x["tgt"]) > THR and x["ref"] * x["tgt"] < 0)
    checks = [(inv["antichess"], int(val("a2.anti.rawinv"))), (inv["3check"], int(val("a2.3check.rawinv"))),
              (round(T5["antichess"]["angolo"], 1), float(val("rot.anti.angle"))),
              (round(T5["antichess"]["p95_appaiato"], 1), float(val("rot.anti.p95"))),
              (round(T5["3check"]["angolo"], 1), float(val("rot.3check12.angle"))),
              (round(T5["3check"]["p95_appaiato"], 1), float(val("rot.3check12.p95")))]
    for got, ref in checks:
        if abs(got - ref) > 1e-9:
            sys.exit(f"Numero diverso da numbers.tex: {got} vs {ref}")
    print("controllo OK:", inv, [c[0] for c in checks[2:]])

    fig, (a, b) = plt.subplots(1, 2, figsize=(9.2, 3.9), gridspec_kw={"width_ratios": [1.25, 1]})

    # A. correlazione con l'esito, coordinata per coordinata
    allv = [x[k] for v in ("antichess", "3check") for x in T3[v]["dettaglio"].values() for k in ("ref", "tgt")
            if x[k] is not None]
    lim, low = 0.62, min(allv) - 0.05
    a.axhspan(-THR, THR, color="#efeee9", lw=0); a.axvspan(-THR, THR, color="#efeee9", lw=0)
    a.fill_between([THR, lim], low, -THR, color=ORANGE, alpha=0.07, lw=0)
    a.fill_between([-lim, -THR], THR, lim, color=ORANGE, alpha=0.07, lw=0)
    a.plot([low, lim], [low, lim], color=MUTED, lw=0.8, ls="--")
    a.axhline(0, color=INK, lw=0.6); a.axvline(0, color=INK, lw=0.6)
    for v, lab, col, mk in (("3check", "Three-Check (adds a winning condition)", BLUE, "o"),
                            ("antichess", "Antichess (inverts the objective)", ORANGE, "s")):
        det = T3[v]["dettaglio"]
        xs = np.array([x["ref"] for x in det.values() if x["tgt"] is not None])
        ys = np.array([x["tgt"] for x in det.values() if x["tgt"] is not None])
        a.scatter(xs, ys, s=22, color=col, marker=mk, label=lab, zorder=3, edgecolor="white", lw=0.4)
    m = T3["antichess"]["dettaglio"]["material"]
    a.annotate("material", (m["ref"], m["tgt"]), xytext=(0, 8), textcoords="offset points",
               fontsize=8, color=INK, ha="center")
    a.text(0.97 * lim, low + 0.02, "sign inverted", ha="right", va="bottom", fontsize=8, color=ORANGE)
    a.set_xlim(low, lim); a.set_ylim(low, lim)
    a.set_xlabel("Correlation with winning, standard chess (depth 6)")
    a.set_ylabel("Correlation with winning, variant")
    a.set_title("A. Each coordinate's relation with winning", loc="left", fontsize=10, color=INK)
    a.legend(frameon=False, fontsize=8, loc="upper left")

    # B. rotazione del piano contro il null appaiato
    rows = (("Antichess", T5["antichess"], ORANGE), ("Three-Check", T5["3check"], BLUE))
    xs = np.arange(len(rows))
    for i, (lab, r, col) in enumerate(rows):
        b.bar(i, r["angolo"], width=0.55, color=col)
        b.plot([i - 0.36, i + 0.36], [r["p95_appaiato"]] * 2, color=INK, lw=1.6)
        b.text(i, r["angolo"] + 1.5, f"{r['angolo']:.1f}°", ha="center", fontsize=8.5, color=INK)
    b.text(-0.36, T5["antichess"]["p95_appaiato"] + 1.2, "chance (95th pct.)", ha="left", fontsize=8, color=INK)
    b.set_xticks(xs, [r[0] for r in rows])
    b.set_ylim(0, 85)
    b.set_ylabel("Rotation of the plane vs depth 6 (degrees)")
    b.set_title("B. How much the map turns", loc="left", fontsize=10, color=INK)
    b.grid(axis="y", color="#e6e5e0", lw=0.6); b.set_axisbelow(True)

    for ax in (a, b):
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(colors=INK, labelsize=8.5)
    fig.tight_layout()
    out = ROOT / "paper/figures/fig4_chess_objective.pdf"
    fig.savefig(out); fig.savefig(out.with_suffix(".png"), dpi=200)
    print("scritta", out)


if __name__ == "__main__":
    main()
