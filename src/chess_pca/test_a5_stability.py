"""
==============================================================================
A5 — STABILITA' STATISTICA:  dove si rompe
==============================================================================

A5 dice "le direzioni latenti dominanti sono robuste a perturbazioni
ragionevoli dei dati". Finora l'hai asserita. Questo script la rende
QUANTITATIVA trovando il punto in cui cede.

Procedura: split-half entro lo STESSO corpus a n decrescente. Le due meta'
provengono per costruzione dalla stessa geometria, quindi l'angolo misurato
e' interamente errore di stima. Se A5 vale, l'angolo scala come 1/sqrt(n)
e resta piccolo. Quando esplode, A5 e' violata.

DUE RISULTATI ATTESI, entrambi utili al paper:

  1) la curva angolo(n) da' l'n minimo sotto cui NESSUN confronto fra
     sistemi e' interpretabile;

  2) il confronto depth6 vs random collega A5 ad A3: il gioco casuale ha
     eigengap lambda1-lambda2 piu' grande, quindi il piano principale e'
     PIU' identificato, non meno. Se l'ordinamento e' questo, le tue
     condizioni non sono una lista ma hanno una relazione interna --
     ed e' esattamente cio' che trasforma un elenco in un framework.

USO
---
    python src\\chess_pca\\test_a5_stability.py

Costo: zero run motore, solo cache esistenti. ~3-5 minuti.
==============================================================================
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import subspace_angles
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

RNG = np.random.default_rng(20260727)

FIXED_LEN = 20          # identico a run_t6.py / final_calibration.py
N_REPS = 50             # split-half per ogni n
N_GRID = [100, 200, 400, 800, 1600, 1950]

NON_FEAT = {"game_id", "idx", "ply", "move_num", "result", "outcome",
            "perspective", "opening_fen", "is_resignation", "source_game_id"}

CORPORA = {
    "depth6": ("data/cache_d6/white_data_max2000_min10.parquet",
               "data/cache_d6/black_data_max2000_min10.parquet"),
    "random": ("data/cache_random/white_data_max2000_min10.parquet",
               "data/cache_random/black_data_max2000_min10.parquet"),
    "3check": ("data/cache_3check/white_data_max2000_min10.parquet",
               "data/cache_3check/black_data_max2000_min10.parquet"),
}

# Angoli osservati fra sistemi, da run_t6.py — servono da metro di confronto
OSSERVATI = {"3check vs depth6": 23.4, "random vs depth6": 33.5}


def load(white_path, black_path):
    """Identica a run_t6.load() + troncamento a FIXED_LEN."""
    frames = []
    for path, persp in [(white_path, "White"), (black_path, "Black")]:
        p = Path(path)
        if not p.exists():
            return None
        d = pd.read_parquet(p).reset_index(drop=True)
        d["perspective"] = persp
        d["game_id"] = persp + ":" + d["game_id"].astype(str)
        frames.append(d)
    d = pd.concat(frames, ignore_index=True)
    keep = d.groupby("game_id").size()
    d = d[d.game_id.isin(keep[keep >= FIXED_LEN].index)]
    return d.groupby("game_id", sort=False).head(FIXED_LEN)


def check_constant(d, feats, label):
    """Una feature a varianza nulla manda in NaN np.corrcoef. Va vista prima."""
    v = d[feats].var(ddof=0)
    dead = v[v < 1e-12].index.tolist()
    if dead:
        print(f"  !! [{label}] feature a varianza ~0: {dead}")
        print("     np.corrcoef restituira' NaN su queste. Vanno escluse o")
        print("     dichiarate come violazione di A2 (vedi check_phi_validity.py).")
    return dead


def basis_and_gap(d, feats):
    """Piano principale 2D + eigengap lambda1-lambda2 sulla matrice di correlazione."""
    X = StandardScaler().fit_transform(d[feats].values)
    C = np.corrcoef(X.T)
    lam = np.sort(np.linalg.eigvalsh(C))[::-1]
    V = PCA(n_components=2).fit(X).components_.T
    return V, float(lam[0] - lam[1])


def main():
    print("=" * 74)
    print("A5 — STABILITA' STATISTICA:  curva angolo(n) entro corpus")
    print("=" * 74)

    rows = []
    for name, (w, b) in CORPORA.items():
        d = load(w, b)
        if d is None:
            print(f"\n[{name}] parquet mancante — saltato")
            continue

        feats = [c for c in d.select_dtypes(include=[np.number]).columns
                 if c not in NON_FEAT]
        ids = d.game_id.unique()
        print(f"\n[{name}] {d.game_id.nunique()} traiettorie x {FIXED_LEN} ply, "
              f"{len(feats)} feature")
        check_constant(d, feats, name)

        _, gap_full = basis_and_gap(d, feats)
        print(f"          eigengap lambda1-lambda2 (corpus intero) = {gap_full:.3f}")

        for n in N_GRID:
            if 2 * n > len(ids):
                continue
            ang, gaps = [], []
            for _ in range(N_REPS):
                p = RNG.permutation(ids)
                Va, ga = basis_and_gap(d[d.game_id.isin(p[:n])], feats)
                Vb, gb = basis_and_gap(d[d.game_id.isin(p[n:2 * n])], feats)
                ang.append(np.degrees(subspace_angles(Va, Vb)).max())
                gaps.append(0.5 * (ga + gb))
            ang = np.array(ang)
            rows.append({
                "corpus": name,
                "n_per_lato": n,
                "mediana": round(float(np.median(ang)), 1),
                "p95": round(float(np.percentile(ang, 95)), 1),
                "max": round(float(ang.max()), 1),
                "gap12": round(float(np.mean(gaps)), 3),
            })
            print(f"    n={n:5d}/lato   mediana {rows[-1]['mediana']:5.1f}   "
                  f"p95 {rows[-1]['p95']:5.1f}   max {rows[-1]['max']:5.1f}   "
                  f"gap12 {rows[-1]['gap12']:.3f}")

    if not rows:
        raise SystemExit("Nessun corpus caricato. Controlla i path in CORPORA.")

    tab = pd.DataFrame(rows)

    # ---- scaling: A5 vale se l'angolo va come 1/sqrt(n) --------------------
    print("\n" + "-" * 74)
    print("SCALING  (atteso ~1.41x dimezzando n, se l'errore e' di stima)")
    print("-" * 74)
    for name, g in tab.groupby("corpus"):
        g = g.sort_values("n_per_lato", ascending=False)
        med = g.mediana.values
        ratios = med[1:] / med[:-1]
        print(f"  {name:8s} rapporti mediana: " +
              "  ".join(f"{r:.2f}" for r in ratios))
        if len(ratios) and np.all((ratios > 1.15) & (ratios < 1.75)):
            print("           -> coerente con errore di stima puro: A5 REGGE")
        else:
            print("           -> scaling anomalo: A5 cede, la geometria non e'")
            print("              solo stimata male, e' instabile")

    # ---- n minimo per poter parlare di differenze fra sistemi --------------
    print("\n" + "-" * 74)
    print("N MINIMO PER CONFRONTI FRA SISTEMI")
    print("-" * 74)
    ref = tab[tab.corpus == "depth6"].sort_values("n_per_lato")
    for label, obs in OSSERVATI.items():
        ok = ref[ref.p95 < obs]
        if len(ok):
            n_min = int(ok.n_per_lato.iloc[0])
            print(f"  {label:22s} ({obs:4.1f} deg): serve n >= {n_min:5d}/lato")
        else:
            print(f"  {label:22s} ({obs:4.1f} deg): NON raggiungibile in questa griglia")

    out = Path("data/outputs/a5_stability.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    tab.to_csv(out, index=False)
    print(f"\nSalvato: {out}")


if __name__ == "__main__":
    main()
