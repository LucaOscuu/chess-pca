"""
run_h3_quick.py - Esegue SOLO H3 (geometric drift) usando dati e PCA già salvati.
"""

import sys
from pathlib import Path

# Aggiungi il percorso principale
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

# ============================================================
# CONFIGURAZIONE - MODIFICA QUESTI PATH SE NECESSARIO
# ============================================================
DATA_DIR = Path("data/outputs/figure_h3_data")
PCA_DIR = Path("data/outputs")
OUT_DIR = Path("data/outputs")

# File già esistenti (generati dalla tua ultima esecuzione corretta)
DRIFT_CSV = DATA_DIR / "drift_data.csv"
PCA_WHITE_PATH = PCA_DIR / "pca_white.pkl"
PCA_BLACK_PATH = PCA_DIR / "pca_black.pkl"

# ============================================================
# CARICA I DATI GIÀ CALCOLATI (veloce!)
# ============================================================
print("=" * 80)
print("📊 H3: REGENERATING BOXPLOT ONLY")
print("=" * 80)

# Carica il drift_data.csv già esistente
if not DRIFT_CSV.exists():
    print(f"❌ Errore: {DRIFT_CSV} non trovato!")
    print("   Esegui prima l'intera pipeline una volta per generare i dati.")
    sys.exit(1)

df_h3 = pd.read_csv(DRIFT_CSV)
print(f"✓ Caricati {len(df_h3):,} drifts")

# ============================================================
# RICALCOLA STATISTICHE (per sicurezza)
# ============================================================
win_vals = df_h3.loc[df_h3["outcome"] == "Win", "drift_z1"].to_numpy()
draw_vals = df_h3.loc[df_h3["outcome"] == "Draw", "drift_z1"].to_numpy()
loss_vals = df_h3.loc[df_h3["outcome"] == "Loss", "drift_z1"].to_numpy()

print(f"\n📊 STATISTICHE:")
print(f"  Win:  n={len(win_vals):,}, median={np.median(win_vals):+.3f}")
print(f"  Draw: n={len(draw_vals):,}, median={np.median(draw_vals):+.3f}")
print(f"  Loss: n={len(loss_vals):,}, median={np.median(loss_vals):+.3f}")

# Mann-Whitney
u_stat, p_val = mannwhitneyu(win_vals, loss_vals, alternative="two-sided")
r_rb = 1.0 - (2.0 * u_stat) / (len(win_vals) * len(loss_vals))
print(f"\n📈 MANN-WHITNEY: U={u_stat:.3g}, p={p_val:.3e}, r={r_rb:+.3f}")

# ============================================================
# RIGENERA SOLO IL BOXPLOT (corretto)
# ============================================================
print("\n📊 Rigenerazione boxplot...")

plt.close('all')
fig, ax = plt.subplots(figsize=(10, 7))

bp = ax.boxplot(
    [win_vals, draw_vals, loss_vals],
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
ax.set_ylabel("Δz₁", fontsize=13, fontweight="bold")
ax.set_xlabel("Outcome", fontsize=13, fontweight="bold")
ax.set_title("H3: Drift Distribution by Outcome", fontsize=14, fontweight="bold")
ax.grid(True, alpha=0.25, axis="y", linestyle="--", linewidth=0.5)

# Aggiungi statistiche sul grafico
y0, y1 = ax.get_ylim()
y_pos = y0 + (y1 - y0) * 0.05
for i, (lab, vals) in enumerate(zip(["Win", "Draw", "Loss"], [win_vals, draw_vals, loss_vals]), 1):
    median_val = np.median(vals)
    n_val = len(vals)
    ax.text(
        i, y_pos,
        f"N={n_val:,}\nMedian={median_val:.2f}",
        ha="center", va="bottom",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8, edgecolor="gray"),
    )

plt.tight_layout()

# Salva SOPRASCRIVENDO il file vecchio
output_path = OUT_DIR / "h3_drift_boxplot.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"✅ Salvato: {output_path.absolute()}")
print(f"   Dimensione file: {output_path.stat().st_size:,} bytes")

# ============================================================
# VERIFICA FINALE
# ============================================================
print("\n" + "=" * 80)
print("📊 VERIFICA FINALE - I valori nel boxplot DOVREBBERO essere:")
print(f"  Win median:  -0.463")
print(f"  Draw median: -0.405")
print(f"  Loss median: -1.352")
print("=" * 80)