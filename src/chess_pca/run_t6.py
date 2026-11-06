"""
==============================================================================
T6 — GRADIENTE DI COMPETENZA
==============================================================================

Confronta la firma geometrica attraverso regimi strategici di competenza
crescente, con DUE appaiamenti che rendono il confronto valido:

  1) n_eff appaiato  -> stesso numero di partite per ogni regime,
                        quindi STESSO bordo di Marchenko-Pastur.
                        Senza questo, il conteggio "autovalori > MP" e'
                        confuso dalla dimensione campionaria: piu' partite
                        = soglia piu' bassa = piu' facile superarla.

  2) lunghezza appaiata -> ogni traiettoria e' troncata campionando una
                        lunghezza dalla distribuzione di riferimento
                        (Stockfish full), cosi' il PR non riflette
                        semplicemente "quanto a lungo si e' giocato".

METRICHE RIPORTATE
------------------
  n_sig      autovalori sopra il bordo MP  -> metrica principale,
             ancorata a soglia teorica
  PR         participation ratio           -> secondaria, sensibile
             alla lunghezza
  eigengap   lambda1 - lambda2             -> struttura dello spettro
  angolo     angolo principale massimo vs Stockfish full -> metrica piu'
             robusta, immune a lunghezza e n_eff

USO
---
    python src\\chess_pca\\run_t6.py
==============================================================================
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import subspace_angles
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

RNG = np.random.default_rng(20260727)

NON_FEATURES = {"game_id", "idx", "ply", "move_num", "result", "outcome",
                "perspective", "opening_fen", "is_resignation",
                "source_game_id"}

# ---------------------------------------------------------------------------
# ADATTA i path ai tuoi file. Commenta le righe dei regimi non ancora generati.
# ---------------------------------------------------------------------------
CORPORA = {
    "random":  ("data/cache_random/white_data_max2000_min10.parquet",
                "data/cache_random/black_data_max2000_min10.parquet"),
    "depth1":  ("data/cache_d1/white_data_max2000_min10.parquet",
                "data/cache_d1/black_data_max2000_min10.parquet"),
    "depth3":  ("data/cache_d3/white_data_max2000_min10.parquet",
                "data/cache_d3/black_data_max2000_min10.parquet"),
    "depth6":  ("data/cache_d6/white_data_max2000_min10.parquet",
                "data/cache_d6/black_data_max2000_min10.parquet"),
    "full":    ("data/cache/white_data_max10000_min10.parquet",
                "data/cache/black_data_max10000_min10.parquet"),
    "3check":  ("data/cache_3check/white_data_max2000_min10.parquet",
                "data/cache_3check/black_data_max2000_min10.parquet"),
}

REFERENCE = "depth6"
N_GAMES = 2000            # partite per prospettiva, uguale per tutti i regimi
FIXED_LEN = 20        # ply, identico per ogni regime

def load(white_path, black_path):
    frames = []
    for path, persp in [(white_path, "White"), (black_path, "Black")]:
        p = Path(path)
        if not p.exists():
            return None
        d = pd.read_parquet(p).reset_index(drop=True)
        d["perspective"] = persp
        d["game_id"] = persp + ":" + d["game_id"].astype(str)
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def subsample_games(df, n_per_persp):
    """Campiona n_per_persp partite per prospettiva -> n_eff identico."""
    keep = []
    for _, g in df.groupby("perspective"):
        ids = g.game_id.unique()
        if len(ids) > n_per_persp:
            ids = RNG.choice(ids, n_per_persp, replace=False)
        keep.append(g[g.game_id.isin(ids)])
    return pd.concat(keep, ignore_index=True)


def truncate_to(df, length_pool):
    """Tronca ogni traiettoria a una lunghezza campionata da length_pool."""
    out = []
    for _, g in df.groupby("game_id", sort=False):
        out.append(g.head(int(RNG.choice(length_pool))))
    return pd.concat(out, ignore_index=True)


def signature(df, features, n_eff):
    X = StandardScaler().fit_transform(df[features].values)
    C = np.corrcoef(X.T)
    lam = np.sort(np.linalg.eigvalsh(C))[::-1]
    edge = (1 + np.sqrt(len(features) / n_eff)) ** 2
    V = PCA(n_components=2).fit(X).components_.T
    return {
        "n_sig": int((lam > edge).sum()),
        "edge": edge,
        "PR": (lam.sum() ** 2) / (lam ** 2).sum(),
        "evr2": (lam[0] + lam[1]) / lam.sum(),
        "gap12": lam[0] - lam[1],
        "lam": lam,
        "V": V,
    }


def main():
    print("=" * 74)
    print("T6 — GRADIENTE DI COMPETENZA")
    print("=" * 74)

    raw = {}
    for name, (w, b) in CORPORA.items():
        d = load(w, b)
        if d is None:
            print(f"[{name}] parquet mancante — saltato")
            continue
        raw[name] = d
        print(f"[{name}] {len(d):,} righe, {d.game_id.nunique()} traiettorie, "
              f"{len(d)/d.game_id.nunique():.0f} ply medi")

    if REFERENCE not in raw:
        raise SystemExit(f"\nRegime di riferimento '{REFERENCE}' assente.")

    features = [c for c in raw[REFERENCE].select_dtypes(include=[np.number]).columns
                if c not in NON_FEATURES]
    print(f"\nFeature usate: {len(features)}")

    # distribuzione di lunghezza di riferimento
    length_pool = raw[REFERENCE].groupby("game_id").size().values

    print(f"\nAppaiamento: {N_GAMES} traiettorie/prospettiva, "
          f"lunghezze campionate da '{REFERENCE}'")
    print("-" * 74)

    sigs, rows = {}, []
    for name, d in raw.items():
        d = subsample_games(d, N_GAMES)
        keep = d.groupby("game_id").size()
        d = d[d.game_id.isin(keep[keep >= FIXED_LEN].index)]
        d = d.groupby("game_id", sort=False).head(FIXED_LEN)
        n_eff = d.game_id.nunique()
        s = signature(d, features, n_eff)
        sigs[name] = s
        rows.append({
            "regime": name,
            "n_eff": n_eff,
            "ply_medi": round(len(d) / n_eff),
            "n_sig": s["n_sig"],
            "PR": round(s["PR"], 2),
            "EVR2": round(s["evr2"], 3),
            "gap12": round(s["gap12"], 3),
        })

    ref_V = sigs[REFERENCE]["V"]
    for r in rows:
        a = np.degrees(subspace_angles(ref_V, sigs[r["regime"]]["V"]))
        r["ang_max"] = round(float(a.max()), 1)
        r["ang_min"] = round(float(a.min()), 1)

    order = ["random", "depth1", "depth3", "depth6", "full","3check"]
    tab = pd.DataFrame(rows)
    tab["_o"] = tab.regime.map({k: i for i, k in enumerate(order)})
    tab = tab.sort_values("_o").drop(columns="_o")

    print()
    print(tab.to_string(index=False))
    print(f"\nBordo MP comune: {sigs[REFERENCE]['edge']:.4f}  "
          f"(identico per tutti: n_eff appaiato)")

    # ---- diagnosi di monotonicita' -------------------------------------
    print("\n" + "-" * 74)
    seq = [r for r in tab.regime if r in order]
    for col, verso in [("n_sig", "crescente"), ("ang_max", "decrescente")]:
        v = tab.set_index("regime").loc[seq, col].values.astype(float)
        d = np.diff(v)
        ok = (d >= 0).all() if verso == "crescente" else (d <= 0).all()
        print(f"{col:8s} {list(v)}  -> {'MONOTONO' if ok else 'non monotono'} "
              f"({verso} atteso)")

    print("\nSpettri (primi 6 autovalori):")
    for name in seq:
        print(f"  {name:8s} " + "  ".join(f"{x:.3f}" for x in sigs[name]["lam"][:6]))

    tab.to_csv("data/outputs/t6_competence_gradient.csv", index=False)
    print("\nSalvato: data/outputs/t6_competence_gradient.csv")


if __name__ == "__main__":
    main()
