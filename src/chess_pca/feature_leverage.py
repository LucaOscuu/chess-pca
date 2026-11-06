"""
==============================================================================
LEVA DELLE COORDINATE  —  quanto ogni feature di phi controlla il piano
==============================================================================

PERCHE' SERVE, E PERCHE' ORA
----------------------------
test_a1_temporal.py ha mostrato che distruggere l'informazione temporale
contenuta in UNA sola feature (delta_mobility, 1 su 18) ruota il piano
principale di ~14.8 gradi, contro un null di 5.0.

14.8 gradi e' piu' del contrasto depth3-vs-depth6 (14.3) e dello stesso
ordine di grandezza del contrasto 3check-vs-depth6 (23.4).

Da qui la domanda che un referee fara' per primo:
    "se togliere una feature muove il piano di 15 gradi, che cosa
     significa davvero un angolo di 23 gradi fra due sistemi?"

Questo script risponde. Per ogni coordinata, ne permuta i valori su tutte
le righe (distruggendone l'informazione ma mantenendo d=18, cosi' i
sottospazi restano confrontabili), rifitta la PCA e misura la rotazione
del piano rispetto al riferimento.

COME LEGGERE IL RISULTATO
-------------------------
  delta_mobility isolato in cima, resto sotto il null
      -> phi e' robusto, la leva temporale e' specifica, e i confronti
         fra sistemi sono interpretabili. Caso migliore.

  molte feature sopra il null, con valori 15-25 gradi
      -> il piano e' controllato dalla COMPOSIZIONE di phi quanto dal
         sistema. Gli angoli fra sistemi non sono attribuibili al sistema
         finche' non si mostra che la differenza eccede questa leva.
         In quel caso il confronto va rifatto a phi ridotto alle
         coordinate a bassa leva, oppure dichiarato condizionale a phi.

In entrambi i casi il numero va nel paper: e' la misura di quanto il
risultato dipende dalla scelta del feature map. Nessuno nella letteratura
scacchi-PCA lo riporta.

USO
---
    python src\\chess_pca\\feature_leverage.py
    python src\\chess_pca\\feature_leverage.py --corpus depth6 --window 20

Costo: zero run motore. ~2-5 minuti.
==============================================================================
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import subspace_angles
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

RNG = np.random.default_rng(20260727)

NON_FEAT = {"game_id", "idx", "ply", "move_num", "result", "outcome",
            "perspective", "opening_fen", "is_resignation", "source_game_id", "base_game"}

CORPORA = {
    "full":   ("data/cache/white_data_max10000_min10.parquet",
               "data/cache/black_data_max10000_min10.parquet"),
    "depth6": ("data/cache_d6/white_data_max2000_min10.parquet",
               "data/cache_d6/black_data_max2000_min10.parquet"),
    "3check": ("data/cache_3check/white_data_max2000_min10.parquet",
               "data/cache_3check/black_data_max2000_min10.parquet"),
    "random": ("data/cache_random/white_data_max2000_min10.parquet",
               "data/cache_random/black_data_max2000_min10.parquet"),
}

# Contrasti fra sistemi gia' misurati in run_t6.py, per il confronto finale
CONTRASTI = {"random vs depth6": 33.5, "full vs depth6": 24.8,
             "3check vs depth6": 23.4, "depth1 vs depth6": 20.7,
             "depth3 vs depth6": 14.3}


def load(white_path, black_path, window):
    frames = []
    for path, persp in [(white_path, "White"), (black_path, "Black")]:
        p = Path(path)
        if not p.exists():
            raise SystemExit(f"Manca {p}")
        d = pd.read_parquet(p).reset_index(drop=True)
        d["base_game"] = d["game_id"].astype(str)
        d["perspective"] = persp
        d["game_id"] = persp + ":" + d["game_id"].astype(str)
        frames.append(d)
    d = pd.concat(frames, ignore_index=True)
    if window:
        keep = d.groupby("game_id").size()
        d = d[d.game_id.isin(keep[keep >= window].index)]
        d = d.groupby("game_id", sort=False).head(window)
    return d


def plane(X):
    return PCA(n_components=2).fit(X).components_.T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="full", choices=list(CORPORA))
    ap.add_argument("--window", type=int, default=0,
                    help="0 = traiettorie intere; 20 = come run_t6.py")
    ap.add_argument("--reps", type=int, default=10,
                    help="permutazioni per feature")
    ap.add_argument("--null-reps", type=int, default=50)
    args = ap.parse_args()

    d = load(*CORPORA[args.corpus], window=args.window)
    feats = [c for c in d.select_dtypes(include=[np.number]).columns
             if c not in NON_FEAT]

    print("=" * 74)
    print(f"LEVA DELLE COORDINATE — corpus '{args.corpus}'"
          f"{f', finestre {args.window} ply' if args.window else ''}")
    print("=" * 74)
    print(f"{len(d):,} righe, {d.game_id.nunique():,} traiettorie, "
          f"{len(feats)} feature")

    Xraw = d[feats].values.astype(float)
    Xz = StandardScaler().fit_transform(Xraw)
    Vref = plane(Xz)

    # ---- null: split-half sullo stesso corpus, stessa n ------------------
    games = d.base_game.unique()
    n_side = len(games) // 2
    null = []
    for _ in range(args.null_reps):
        p = RNG.permutation(games)
        a = plane(StandardScaler().fit_transform(
            d[d.base_game.isin(p[:n_side])][feats].values))
        b = plane(StandardScaler().fit_transform(
            d[d.base_game.isin(p[n_side:2 * n_side])][feats].values))
        null.append(np.degrees(subspace_angles(a, b)).max())
    null_p95 = float(np.percentile(null, 95))
    print(f"\nNull split-half (n={n_side} partite/lato): mediana "
          f"{np.median(null):.2f}  p95 {null_p95:.2f} gradi")

    # ---- leva per feature -------------------------------------------------
    print("\n" + "-" * 74)
    print("Rotazione del piano quando la coordinata viene randomizzata")
    print("-" * 74)
    rows = []
    for j, f in enumerate(feats):
        angs = []
        for _ in range(args.reps):
            Xp = Xraw.copy()
            Xp[:, j] = Xp[RNG.permutation(len(Xp)), j]
            angs.append(np.degrees(subspace_angles(
                Vref, plane(StandardScaler().fit_transform(Xp)))).max())
        rows.append({"feature": f,
                     "leva_deg": round(float(np.mean(angs)), 2),
                     "sd": round(float(np.std(angs)), 2),
                     "sopra_null": bool(np.mean(angs) > null_p95)})

    tab = pd.DataFrame(rows).sort_values("leva_deg", ascending=False)
    print(tab.to_string(index=False))

    sopra = tab[tab.sopra_null]
    print(f"\nFeature sopra il null: {len(sopra)}/{len(feats)}")

    # ---- verdetto ---------------------------------------------------------
    print("\n" + "=" * 74)
    print("LETTURA")
    print("=" * 74)
    leva_max = tab.leva_deg.iloc[0]
    print(f"Leva massima di una singola coordinata: {leva_max:.1f} gradi "
          f"({tab.feature.iloc[0]})")
    print("\nContrasti fra sistemi, confrontati con questa leva:")
    for k, v in sorted(CONTRASTI.items(), key=lambda x: -x[1]):
        if v > leva_max:
            stato = "eccede la leva di phi"
        elif v > null_p95:
            stato = "sopra il null MA dentro la leva di phi"
        else:
            stato = "dentro il null"
        print(f"  {k:20s} {v:5.1f} deg   {stato}")

    print("\nRegola di lettura da scrivere nel paper:")
    print("  un contrasto fra sistemi e' attribuibile al SISTEMA solo se")
    print("  eccede sia il null campionario sia la leva massima di una")
    print("  singola coordinata di phi. Altrimenti e' attribuibile alla")
    print("  composizione del feature map, che e' una scelta del ricercatore.")

    out = Path(f"data/outputs/feature_leverage_{args.corpus}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    tab.to_csv(out, index=False)
    print(f"\nSalvato: {out}")


if __name__ == "__main__":
    main()
