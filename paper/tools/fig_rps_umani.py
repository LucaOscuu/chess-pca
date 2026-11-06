"""Figura: sasso-carta-forbice giocato da persone (Wang, Xu & Zhou 2014) sulla mappa.
Rilancia data/outputs/editori/rps_umani/rps_umani.py (seme fisso) in una cartella temporanea,
controlla che riproduca risultati_rps_umani.json e si ferma se no.
Uso (dalla cartella principale della repo):  python paper/tools/fig_rps_umani.py"""
import json, subprocess, sys, tempfile, shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
ED = ROOT / "data" / "outputs" / "editori" / "rps_umani"
OUT = ROOT / "paper" / "figures"

tmp = Path(tempfile.mkdtemp())
for f in ["rps_umani.py", "WXZ2014.rda"]:
    shutil.copy(ED / f, tmp / f)
subprocess.run([sys.executable, "rps_umani.py", "WXZ2014.rda", "10"], cwd=tmp, check=True, stdout=subprocess.DEVNULL)
if json.loads((tmp / "risultati_rps_umani.json").read_text()) != json.loads((ED / "risultati_rps_umani.json").read_text(encoding="utf-8")):
    sys.exit("L'analisi non riproduce risultati_rps_umani.json: mi fermo.")

# mappa senza velocità: immagine lineare del simplesso (la versione con velocità ha uno scarto meccanico)
Y = np.load(tmp / "Ypos_umani.npy"); c = np.load(tmp / "nashpos_umani.npy")
fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
ax = axes[0]
P = Y.reshape(-1, 2)
ax.hexbin(P[:, 0], P[:, 1], gridsize=40, cmap="Greys", mincnt=1, bins="log")
ax.scatter(*c, marker="*", s=180, c="tab:red", edgecolor="k", zorder=3, label="Nash")
ax.set_title("A. All groups, all rounds", fontsize=11); ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
ax.legend(loc="upper right", fontsize=9, frameon=False)
ax = axes[1]
# campo medio dei passi su una griglia 8x8 a quantili (come il flusso della Sez. 2)
X0 = Y[:, :-1].reshape(-1, 2); D = np.diff(Y, axis=1).reshape(-1, 2)
qx = np.quantile(X0[:, 0], np.linspace(0, 1, 9)); qy = np.quantile(X0[:, 1], np.linspace(0, 1, 9))
ix = np.clip(np.searchsorted(qx, X0[:, 0]) - 1, 0, 7); iy = np.clip(np.searchsorted(qy, X0[:, 1]) - 1, 0, 7)
ax.hexbin(P[:, 0], P[:, 1], gridsize=40, cmap="Greys", mincnt=1, bins="log", alpha=.35)
for i in range(8):
    for j in range(8):
        m = (ix == i) & (iy == j)
        if m.sum() >= 30:
            x, y = X0[m].mean(0); dx, dy = D[m].mean(0)
            ax.annotate("", xy=(x + 3 * dx, y + 3 * dy), xytext=(x, y),
                        arrowprops=dict(arrowstyle="->", color="tab:blue", lw=1.3))
ax.scatter(*c, marker="*", s=180, c="tab:red", edgecolor="k", zorder=3)
ax.set_title("B. Average step from each cell", fontsize=11); ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
fig.tight_layout()
OUT.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT / "fig_rps_umani.pdf"); fig.savefig(OUT / "fig_rps_umani.png", dpi=200)
shutil.rmtree(tmp)
print("scritte paper/figures/fig_rps_umani.pdf e .png")
