"""
CALIBRAZIONE FINALE — due cose, poi stop.

1) Null angolare nelle STESSE condizioni di T6 (finestre da 20 ply,
   n_eff appaiato). Senza questo i 23.4 gradi del 3check non sono
   interpretabili.

2) Verifica della predizione: il loading di checks_avail su PC1 nel
   3check deve essere maggiore che negli scacchi standard.

Uso:  python src\chess_pca\final_calibration.py
"""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.linalg import subspace_angles
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

RNG = np.random.default_rng(20260727)
FIXED_LEN = 20
NON_FEAT = {"game_id", "idx", "ply", "move_num", "result", "outcome",
            "perspective", "opening_fen", "source_game_id"}

PATHS = {
    "depth6": ("data/cache_d6/white_data_max2000_min10.parquet",
               "data/cache_d6/black_data_max2000_min10.parquet"),
    "3check": ("data/cache_3check/white_data_max2000_min10.parquet",
               "data/cache_3check/black_data_max2000_min10.parquet"),
}


def load(wp, bp):
    fr = []
    for p, persp in [(wp, "White"), (bp, "Black")]:
        d = pd.read_parquet(p).reset_index(drop=True)
        d["perspective"] = persp
        d["game_id"] = persp + ":" + d["game_id"].astype(str)
        fr.append(d)
    d = pd.concat(fr, ignore_index=True)
    keep = d.groupby("game_id").size()
    d = d[d.game_id.isin(keep[keep >= FIXED_LEN].index)]
    return d.groupby("game_id", sort=False).head(FIXED_LEN)


def basis(d, feats):
    X = StandardScaler().fit_transform(d[feats].values)
    return PCA(n_components=2).fit(X)


d6 = load(*PATHS["depth6"])
tc = load(*PATHS["3check"])
feats = [c for c in d6.select_dtypes(np.number).columns if c not in NON_FEAT]
print(f"depth6: {d6.game_id.nunique()} traiettorie x {FIXED_LEN} ply")
print(f"3check: {tc.game_id.nunique()} traiettorie x {FIXED_LEN} ply")

# ---------- 1. NULL ANGOLARE, condizioni appaiate ----------
print("\n" + "=" * 62)
print("NULL ANGOLARE  (split-half entro depth6, finestre 20 ply)")
print("=" * 62)
ids = d6.game_id.unique()
for n_side in (1300, 2000):
    if 2 * n_side > len(ids):
        continue
    ang = []
    for _ in range(50):
        p = RNG.permutation(ids)
        a = basis(d6[d6.game_id.isin(p[:n_side])], feats).components_.T
        b = basis(d6[d6.game_id.isin(p[n_side:2 * n_side])], feats).components_.T
        ang.append(np.degrees(subspace_angles(a, b)).max())
    ang = np.array(ang)
    print(f"  n={n_side}/lato : mediana {np.median(ang):5.1f}  "
          f"p95 {np.percentile(ang,95):5.1f}  max {ang.max():5.1f}")

print("\n  Osservato 3check vs depth6 : 23.4 gradi (ang_max)")
print("  Confronta con il null a n=1300 (il 3check ha n_eff 2592,")
print("  cioe' ~1300 per prospettiva).")

# ---------- 2. PREDIZIONE su checks_avail ----------
print("\n" + "=" * 62)
print("PREDIZIONE: checks_avail piu' pesante su PC1 nel 3check")
print("=" * 62)
rows = []
for name, d in [("depth6", d6), ("3check", tc)]:
    p = basis(d, feats)
    L = pd.DataFrame(p.components_.T, index=feats, columns=["PC1", "PC2"])
    # segno arbitrario: orienta PC1 positivamente rispetto a material
    for c in ("PC1", "PC2"):
        if L.loc["material", c] < 0:
            L[c] = -L[c]
    rows.append((name, L))
    top = L.PC1.abs().sort_values(ascending=False).head(5)
    print(f"\n{name} — top 5 loading su PC1:")
    for f in top.index:
        print(f"   {f:22s} {L.loc[f,'PC1']:+.3f}")
    print(f"   -> checks_avail        {L.loc['checks_avail','PC1']:+.3f}   "
          f"(rango {list(L.PC1.abs().sort_values(ascending=False).index).index('checks_avail')+1}/18)")

a = abs(rows[0][1].loc["checks_avail", "PC1"])
b = abs(rows[1][1].loc["checks_avail", "PC1"])
print(f"\n|loading| checks_avail:  depth6 {a:.3f}  ->  3check {b:.3f}")
print(f"  PREDIZIONE {'CONFERMATA' if b > a else 'NON confermata'}"
      f"   (variazione {100*(b-a)/a:+.0f}%)")