"""
==============================================================================
TEST PERMUTAZIONALE FRA CORPORA
==============================================================================

PERCHE' SERVE
-------------
Finora i contrasti fra sistemi (random 33,5 / full 24,8 / 3check 23,4 /
depth1 20,7 / depth3 14,3 gradi) sono stati letti contro il p95 dello
split-half ENTRO depth6.

Quel confronto dice: "l'angolo supera la variabilita' di stima interna al
corpus di riferimento". Non dice: "i due corpora hanno piani diversi".
Il null split-half non incorpora l'incertezza del secondo corpus, ne' le
differenze di distribuzione fra i due, e non e' costruito sotto l'ipotesi
nulla che i due sistemi condividano lo stesso piano di popolazione.

IL TEST GIUSTO
--------------
    H0: i due corpora provengono dalla stessa popolazione (stesso piano).

    - metti insieme le partite dei due corpora;
    - ridividile a caso in due gruppi delle stesse numerosita' originali;
    - calcola l'angolo principale massimo fra i due piani stimati;
    - ripeti N volte.

Sotto H0 i due pseudo-corpora sono miscele equivalenti, quindi l'angolo e'
piccolo. L'angolo osservato viene confrontato con quella distribuzione.

    p = (b + 1) / (m + 1)     con b = permutazioni >= osservato

Raggruppamento per PARTITA: entrambe le prospettive vanno sempre insieme.
Finestre fisse a FIXED_LEN ply, cosi' la differenza di lunghezza delle
partite non entra nel test.

AVVERTENZA ONESTA
-----------------
Questo e' un test di uguaglianza delle DISTRIBUZIONI, non solo dei piani.
Se i corpora differiscono per qualcosa che non e' la geometria (marginali
delle feature, autocorrelazione), il test lo rileva comunque. Il
troncamento a finestra fissa rimuove la differenza piu' ovvia, la
lunghezza; le altre vanno dichiarate come limite.

USO
---
    python src\\chess_pca\\permutation_contrasts.py
    python src\\chess_pca\\permutation_contrasts.py --perms 500
==============================================================================
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import subspace_angles
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

RNG = np.random.default_rng(20260728)

FIXED_LEN = 20
NON_FEAT = {"game_id", "base_game", "corpus", "idx", "ply", "move_num",
            "result", "outcome", "perspective", "opening_fen",
            "is_resignation", "source_game_id"}

REFERENCE = "depth6"
CORPORA = {
    "depth6":  ("data/cache_d6/white_data_max2000_min10.parquet",
                "data/cache_d6/black_data_max2000_min10.parquet"),
    "random":  ("data/cache_random/white_data_max2000_min10.parquet",
                "data/cache_random/black_data_max2000_min10.parquet"),
    "depth1":  ("data/cache_d1/white_data_max2000_min10.parquet",
                "data/cache_d1/black_data_max2000_min10.parquet"),
    "depth3":  ("data/cache_d3/white_data_max2000_min10.parquet",
                "data/cache_d3/black_data_max2000_min10.parquet"),
    "full":    ("data/cache/white_data_max10000_min10.parquet",
                "data/cache/black_data_max10000_min10.parquet"),
    "3check":  ("data/cache_3check/white_data_max2000_min10.parquet",
                "data/cache_3check/black_data_max2000_min10.parquet"),
}


def load(wp, bp, tag):
    frames = []
    for path, persp in [(wp, "White"), (bp, "Black")]:
        p = Path(path)
        if not p.exists():
            return None
        d = pd.read_parquet(p).reset_index(drop=True)
        d["base_game"] = tag + ":" + d["game_id"].astype(str)
        d["perspective"] = persp
        d["game_id"] = tag + ":" + persp + ":" + d["game_id"].astype(str)
        d["corpus"] = tag
        frames.append(d)
    d = pd.concat(frames, ignore_index=True)
    keep = d.groupby("game_id").size()
    d = d[d.game_id.isin(keep[keep >= FIXED_LEN].index)]
    d = d.groupby("game_id", sort=False).head(FIXED_LEN)
    both = d.groupby("base_game").perspective.nunique()
    return d[d.base_game.isin(both[both == 2].index)]


def plane(d, feats):
    X = StandardScaler().fit_transform(d[feats].values)
    return PCA(n_components=2).fit(X).components_.T


def angle(a, b, feats):
    return float(np.degrees(subspace_angles(plane(a, feats),
                                            plane(b, feats))).max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perms", type=int, default=200)
    ap.add_argument("--reference", default=REFERENCE)
    args = ap.parse_args()

    data = {}
    for name, (w, b) in CORPORA.items():
        d = load(w, b, name)
        if d is None:
            print(f"[{name}] parquet mancante — saltato")
            continue
        data[name] = d

    if args.reference not in data:
        raise SystemExit(f"Riferimento '{args.reference}' assente.")

    ref = data[args.reference]
    feats = [c for c in ref.select_dtypes(include=[np.number]).columns
             if c not in NON_FEAT]

    print("=" * 76)
    print(f"TEST PERMUTAZIONALE FRA CORPORA  (riferimento: {args.reference})")
    print("=" * 76)
    print(f"Feature: {len(feats)}   finestre: {FIXED_LEN} ply   "
          f"permutazioni: {args.perms}\n")

    rows = []
    for name, tgt in data.items():
        if name == args.reference:
            continue

        # appaia le numerosita': campiona entrambi al minimo comune
        g_ref = ref.base_game.unique()
        g_tgt = tgt.base_game.unique()
        n = min(len(g_ref), len(g_tgt))
        a_games = RNG.choice(g_ref, n, replace=False)
        b_games = RNG.choice(g_tgt, n, replace=False)
        A = ref[ref.base_game.isin(a_games)]
        B = tgt[tgt.base_game.isin(b_games)]

        obs = angle(A, B, feats)

        # null: metti insieme e ridividi a caso, sempre per partita
        pool = pd.concat([A, B], ignore_index=True)
        games = pool.base_game.unique()
        null = np.empty(args.perms)
        for i in range(args.perms):
            p = RNG.permutation(games)
            null[i] = angle(pool[pool.base_game.isin(p[:n])],
                            pool[pool.base_game.isin(p[n:2 * n])], feats)

        b_count = int((null >= obs).sum())
        pval = (b_count + 1) / (args.perms + 1)

        rows.append({
            "contrasto": f"{name} vs {args.reference}",
            "n_partite_lato": n,
            "angolo_oss": round(obs, 1),
            "null_mediana": round(float(np.median(null)), 1),
            "null_p95": round(float(np.percentile(null, 95)), 1),
            "null_max": round(float(null.max()), 1),
            "b": b_count,
            "p": round(pval, 4),
        })
        print(f"  {name:8s} vs {args.reference}:  "
              f"osservato {obs:5.1f}   "
              f"null mediana {np.median(null):5.1f} p95 "
              f"{np.percentile(null, 95):5.1f} max {null.max():5.1f}   "
              f"p = {pval:.4f}   (n={n} partite/lato)")

    tab = pd.DataFrame(rows).sort_values("angolo_oss", ascending=False)
    print("\n" + tab.to_string(index=False))

    print("\n" + "-" * 76)
    print("COME RIPORTARLO")
    print("-" * 76)
    print("  Con questo test puoi scrivere: 'i piani dei due corpora")
    print("  differiscono piu' di quanto atteso sotto l'ipotesi di stessa")
    print("  popolazione, p = ...'. Il p minimo ottenibile con m")
    print(f"  permutazioni e' 1/(m+1) = {1/(args.perms+1):.4f}; riportarlo")
    print("  come 'p <= ...', mai come p = 0.")
    print("\n  Limite da dichiarare: il test confronta DISTRIBUZIONI, non")
    print("  solo piani. Il troncamento a finestra fissa rimuove l'effetto")
    print("  della lunghezza delle partite; differenze residue di")
    print("  autocorrelazione o di marginali contribuiscono al risultato.")

    out = Path("data/outputs/permutation_contrasts.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    tab.to_csv(out, index=False)
    print(f"\nSalvato: {out}")


if __name__ == "__main__":
    main()
