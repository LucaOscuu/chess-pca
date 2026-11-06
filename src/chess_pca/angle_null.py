# src\chess_pca\angle_null.py
import numpy as np, pandas as pd
from scipy.linalg import subspace_angles
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

rng = np.random.default_rng(0)
NON_FEAT = {"game_id","idx","ply","move_num","result","outcome",
            "perspective","opening_fen","source_game_id"}

def sub(df, feats):
    X = StandardScaler().fit_transform(df[feats].values)
    return PCA(n_components=2).fit(X).components_.T

df = pd.read_parquet("data/cache/white_data_max10000_min10.parquet")
feats = [c for c in df.select_dtypes(np.number).columns if c not in NON_FEAT]
ids = df.game_id.unique()

out = []
for _ in range(50):
    perm = rng.permutation(ids)
    a = sub(df[df.game_id.isin(perm[:2000])], feats)
    b = sub(df[df.game_id.isin(perm[2000:4000])], feats)
    out.append(np.degrees(subspace_angles(a, b)).max())

out = np.array(out)
print(f"angolo split-half entro Stockfish: mediana {np.median(out):.1f}deg, "
      f"p95 {np.percentile(out,95):.1f}deg, max {out.max():.1f}deg")
print(f"osservato random-vs-full: 37.0deg")