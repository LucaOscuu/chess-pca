"""
==============================================================================
RICALCOLO CORRETTO — raggruppamento per partita e eigengap giusto
==============================================================================

Corregge due errori presenti in tutti gli script precedenti.

ERRORE 1 — PSEUDOREPLICAZIONE
    In load() la riga
        d["game_id"] = persp + ":" + d["game_id"].astype(str)
    crea "White:0" e "Black:0" come unita' distinte. Lo split-half permutava
    su quelle, quindi le due prospettive della STESSA partita potevano finire
    in meta' opposte. Non sono indipendenti: stessa sequenza di mosse, stesse
    posizioni, stessa apertura, esito invertito, feature quasi speculari.
    Il null risultava troppo stretto.

    Qui il raggruppamento e' per PARTITA: entrambe le prospettive vanno
    sempre dalla stessa parte dello split.

ERRORE 2 — EIGENGAP SBAGLIATO
    Per la stabilita' del PIANO span(PC1,PC2) conta la separazione fra il
    blocco trattenuto e il resto, cioe' lambda2 - lambda3.
    lambda1 - lambda2 governa la rotazione degli assi DENTRO il piano.
    Tutti gli script precedenti riportavano solo lambda1 - lambda2 e lo
    usavano per parlare di stabilita' del piano.

    Qui vengono riportati entrambi, piu' il bound di Davis-Kahan calcolato
    con quello giusto.

COSA ASPETTARSI
    Il null raggruppato sara' PIU' LARGO di quello precedente. Di quanto,
    non lo so: dipende da quanto le due prospettive si somigliano. Se si
    allarga molto, alcuni contrasti fra sistemi che sembravano sopra il
    null (3check 23,4 / full 24,8 / depth1 20,7) diventano marginali.
    Va saputo prima di scrivere qualsiasi cosa.

USO
---
    python src\\chess_pca\\recompute_grouped.py
==============================================================================
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import subspace_angles
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

RNG = np.random.default_rng(20260728)

FIXED_LEN = 20
N_REPS = 50
N_GRID = [100, 200, 400, 800, 1600, 1950]

NON_FEAT = {"game_id", "base_game", "idx", "ply", "move_num", "result",
            "outcome", "perspective", "opening_fen", "is_resignation",
            "source_game_id"}

CORPORA = {
    "depth6": ("data/cache_d6/white_data_max2000_min10.parquet",
               "data/cache_d6/black_data_max2000_min10.parquet"),
    "random": ("data/cache_random/white_data_max2000_min10.parquet",
               "data/cache_random/black_data_max2000_min10.parquet"),
    "3check": ("data/cache_3check/white_data_max2000_min10.parquet",
               "data/cache_3check/black_data_max2000_min10.parquet"),
}

# valori ottenuti con lo split-half NON raggruppato, per il confronto
PRECEDENTI_P95 = {
    "depth6": {100: 59.0, 200: 40.8, 400: 30.9, 800: 18.8, 1600: 16.2, 1950: 14.4},
    "random": {100: 35.2, 200: 28.8, 400: 19.6, 800: 13.5, 1600: 8.6, 1950: 9.0},
    "3check": {100: 37.3, 200: 30.3, 400: 19.7, 800: 12.8},
}

CONTRASTI = {"random vs depth6": 33.5, "full vs depth6": 24.8,
             "3check vs depth6": 23.4, "depth1 vs depth6": 20.7,
             "depth3 vs depth6": 14.3}


def load(wp, bp):
    """
    base_game  = la partita, unita' di campionamento e di raggruppamento
    game_id    = la traiettoria (partita x prospettiva), unita' di analisi
    """
    frames = []
    for path, persp in [(wp, "White"), (bp, "Black")]:
        p = Path(path)
        if not p.exists():
            return None
        d = pd.read_parquet(p).reset_index(drop=True)
        d["base_game"] = d["game_id"].astype(str)          # <-- la correzione
        d["perspective"] = persp
        d["game_id"] = persp + ":" + d["game_id"].astype(str)
        frames.append(d)
    d = pd.concat(frames, ignore_index=True)
    keep = d.groupby("game_id").size()
    d = d[d.game_id.isin(keep[keep >= FIXED_LEN].index)]
    d = d.groupby("game_id", sort=False).head(FIXED_LEN)
    # tieni solo le partite che sopravvivono da ENTRAMBE le prospettive,
    # altrimenti il raggruppamento e' sbilanciato
    both = d.groupby("base_game").perspective.nunique()
    return d[d.base_game.isin(both[both == 2].index)]


def spectrum(d, feats):
    X = StandardScaler().fit_transform(d[feats].values)
    C = np.corrcoef(X.T)
    lam = np.sort(np.linalg.eigvalsh(C))[::-1]
    V = PCA(n_components=2).fit(X).components_.T
    return lam, V


def davis_kahan(lam, delta=0.1):
    """
    Bound sulla rotazione del sottospazio top-2 sotto perturbazione di norma
    delta. Il denominatore e' la separazione fra il blocco trattenuto e il
    resto: lambda2 - lambda3. Non lambda1 - lambda2.
    Restituisce il seno dell'angolo; > 1 significa bound VACUO.
    """
    gap_plane = lam[1] - lam[2]
    if gap_plane <= 0:
        return np.inf
    return float(delta / gap_plane)


def main():
    print("=" * 78)
    print("RICALCOLO CON RAGGRUPPAMENTO PER PARTITA")
    print("=" * 78)

    data = {}
    for name, (w, b) in CORPORA.items():
        d = load(w, b)
        if d is None:
            print(f"[{name}] parquet mancante — saltato")
            continue
        data[name] = d

    if "depth6" not in data:
        raise SystemExit("Serve almeno depth6.")

    feats = [c for c in data["depth6"].select_dtypes(include=[np.number]).columns
             if c not in NON_FEAT]
    print(f"\nFeature: {len(feats)}   finestre: {FIXED_LEN} ply\n")

    # ------------------------------------------------------------------
    # 1. I DUE EIGENGAP
    # ------------------------------------------------------------------
    print("-" * 78)
    print("1. EIGENGAP: quale governa cosa")
    print("-" * 78)
    print(f"{'corpus':10s} {'partite':>8s} {'traiett.':>9s} "
          f"{'lam1':>7s} {'lam2':>7s} {'lam3':>7s} "
          f"{'g_assi':>8s} {'g_piano':>8s} {'DK(0.1)':>9s}")
    spec = {}
    for name, d in data.items():
        lam, V = spectrum(d, feats)
        spec[name] = (lam, V)
        g_ax = lam[0] - lam[1]
        g_pl = lam[1] - lam[2]
        dk = davis_kahan(lam)
        dk_s = "VACUO" if dk > 1 else f"{dk:.3f}"
        print(f"{name:10s} {d.base_game.nunique():8d} {d.game_id.nunique():9d} "
              f"{lam[0]:7.3f} {lam[1]:7.3f} {lam[2]:7.3f} "
              f"{g_ax:8.3f} {g_pl:8.3f} {dk_s:>9s}")

    print("\n  g_assi  = lambda1-lambda2, governa la rotazione DENTRO il piano")
    print("  g_piano = lambda2-lambda3, governa la stabilita' DEL piano")
    print("  DK      = bound di Davis-Kahan sul piano per perturbazione 0.1;")
    print("            se > 1 il bound non dice nulla e serve calibrazione")
    print("            empirica, che e' l'argomento del paper.")

    # ------------------------------------------------------------------
    # 2. NULL RAGGRUPPATO vs NON RAGGRUPPATO
    # ------------------------------------------------------------------
    print("\n" + "-" * 78)
    print("2. NULL SPLIT-HALF, raggruppato per partita")
    print("-" * 78)

    rows = []
    for name, d in data.items():
        games = d.base_game.unique()
        print(f"\n[{name}] {len(games)} partite disponibili")
        for n in N_GRID:
            if 2 * n > len(games):
                continue
            ang = []
            for _ in range(N_REPS):
                p = RNG.permutation(games)
                a = d[d.base_game.isin(p[:n])]
                b = d[d.base_game.isin(p[n:2 * n])]
                _, Va = spectrum(a, feats)
                _, Vb = spectrum(b, feats)
                ang.append(np.degrees(subspace_angles(Va, Vb)).max())
            ang = np.array(ang)
            p95 = float(np.percentile(ang, 95))
            old = PRECEDENTI_P95.get(name, {}).get(n)
            delta = f"{p95 - old:+5.1f}" if old is not None else "    —"
            rows.append({"corpus": name, "n_partite": n,
                         "mediana": round(float(np.median(ang)), 1),
                         "p95_raggruppato": round(p95, 1),
                         "p95_precedente": old,
                         "differenza": None if old is None else round(p95 - old, 1)})
            print(f"   n={n:5d} partite/lato   mediana {np.median(ang):5.1f}   "
                  f"p95 {p95:5.1f}   (prima {old if old else '—'}, "
                  f"differenza {delta})")

    tab = pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # 3. I CONTRASTI REGGONO ANCORA?
    # ------------------------------------------------------------------
    print("\n" + "-" * 78)
    print("3. I CONTRASTI FRA SISTEMI CONTRO IL NULL CORRETTO")
    print("-" * 78)

    ref = tab[tab.corpus == "depth6"].sort_values("n_partite")
    if len(ref):
        n_max = int(ref.n_partite.iloc[-1])
        p95_max = float(ref.p95_raggruppato.iloc[-1])
        print(f"\n  null di riferimento: p95 = {p95_max:.1f} gradi "
              f"(depth6, n={n_max} partite/lato)")
        print(f"  (prima era {PRECEDENTI_P95['depth6'].get(n_max, '—')})\n")
        for k, v in sorted(CONTRASTI.items(), key=lambda x: -x[1]):
            stato = "SOPRA il null" if v > p95_max else "dentro il null"
            print(f"    {k:20s} {v:5.1f}   {stato}")
        print("\n  Nota: i contrasti in tabella furono calcolati con lo split")
        print("  non raggruppato ma su corpora INTERI, quindi il valore")
        print("  dell'angolo non cambia. Cambia la soglia contro cui va letto.")

    out = Path("data/outputs/null_raggruppato.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    tab.to_csv(out, index=False)
    print(f"\nSalvato: {out}")

    print("\n" + "=" * 78)
    print("DA CORREGGERE NEL REPORT INDIPENDENTEMENTE DA QUESTI NUMERI")
    print("=" * 78)
    print("  - p = 0,000 su 200 permutazioni  ->  p = (0+1)/(200+1) = 0,005")
    print("  - residenza: 4,08 vs 3,49 = +16,9%  (convenzione solo-White).")
    print("    Il 3,27 e' in convenzione White+Black e non ha surrogato")
    print("    appaiato: non confrontarlo con 3,49.")
    print("  - la soglia usata per contare le 3 inversioni A2 e' |r| > 0,05")
    print("    nel corpus di riferimento: va dichiarata nel testo.")


if __name__ == "__main__":
    main()
