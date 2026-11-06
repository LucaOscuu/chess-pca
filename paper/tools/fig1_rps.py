"""Fig. 1 (Sez. 3): mappe di rock-paper-scissors simulato, gioco stabile S e instabile Ua
(lambda 0.2, scelta di un mix), con le proiezioni di Nash e TASP.

Rilancia data/outputs/editori/rps_sim.py (100 periodi, seme fisso) in una cartella temporanea,
controlla che i risultati coincidano con risultati_rps.json e si ferma se no.
Uso (dalla cartella principale della repo):  python paper/tools/fig1_rps.py   (~1 minuto)"""
import json, subprocess, sys, tempfile, shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
ED = ROOT / "data" / "outputs" / "editori"
OUT = ROOT / "paper" / "figures"

tmp = Path(tempfile.mkdtemp())
shutil.copy(ED / "rps_sim.py", tmp / "rps_sim.py")
subprocess.run([sys.executable, "-I", "rps_sim.py", "100"], cwd=tmp, check=True, stdout=subprocess.DEVNULL)
if json.loads((tmp / "results.json").read_text()) != json.loads((ED / "risultati_rps.json").read_text(encoding="utf-8")):
    sys.exit("La simulazione non riproduce risultati_rps.json: mi fermo.")

fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
for ax, (tag, title) in zip(axes, [("S_0.2_mix", "A. Stable game (S)"), ("Ua_0.2_mix", "B. Unstable game (Ua)")]):
    P = np.load(tmp / f"P_{tag}.npy"); L = np.load(tmp / f"L_{tag}.npy")
    ax.hexbin(P[:, 0], P[:, 1], gridsize=45, cmap="Greys", mincnt=1, bins="log")
    ax.scatter(*L[0], marker="*", s=180, c="tab:red", edgecolor="k", zorder=3, label="Nash")
    if tag.startswith("Ua"):
        ax.scatter(*L[1], marker="D", s=60, c="tab:blue", edgecolor="k", zorder=3, label="TASP")
    ax.set_title(title, fontsize=11); ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.legend(loc="upper right", fontsize=9, frameon=False)
fig.tight_layout()
OUT.mkdir(exist_ok=True)
fig.savefig(OUT / "fig1_rps.pdf"); fig.savefig(OUT / "fig1_rps.png", dpi=200)
shutil.rmtree(tmp)
print("scritte paper/figures/fig1_rps.pdf e .png")
