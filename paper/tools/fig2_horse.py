"""Fig. 2 (Sez. 3): cavallo di Selten simulato, impulse response e reinforcement learning sulla
mappa comune (periodi 150-249), con i profili osservati da NSS (2025) per strangers e partners.

Legge le proiezioni salvate da data/outputs/editori/horse2.py (lanciarlo prima, ~30 s) e
risultati_cavallo_2.json.  Uso:  python paper/tools/fig2_horse.py"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
ED = ROOT / "data" / "outputs" / "editori"
OUT = ROOT / "paper" / "figures"
res = json.loads((ED / "risultati_cavallo_2.json").read_text(encoding="utf-8"))

fig, ax = plt.subplots(figsize=(6.2, 5.0))
for rule, col, lab in (("impulse", "tab:red", "Impulse response"), ("reinforcement", "tab:blue", "Reinforcement learning")):
    P = np.load(ED / f"mappa_comune_{rule}.npy")
    idx = np.random.default_rng(1).choice(len(P), min(4000, len(P)), replace=False)
    ax.scatter(P[idx, 0], P[idx, 1], s=4, alpha=.25, c=col, label=lab, rasterized=True)
for k, m, lab in (("strangers", "s", "Strangers, long run (NSS)"), ("imperfect", "^", "Partners: imperfect equilibrium (NSS)")):
    x, y = res["proiezioni_umane"][k]
    ax.scatter(x, y, marker=m, s=90, facecolor="white", edgecolor="k", zorder=3, label=lab)
ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
leg = ax.legend(fontsize=8, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
for h in leg.legend_handles[:2]: h.set_sizes([25]); h.set_alpha(1)
fig.tight_layout()
OUT.mkdir(exist_ok=True)
fig.savefig(OUT / "fig2_horse.pdf"); fig.savefig(OUT / "fig2_horse.png", dpi=200)
print("scritte paper/figures/fig2_horse.pdf e .png")
