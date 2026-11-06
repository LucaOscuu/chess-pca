import numpy as np, pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

RNG = np.random.default_rng(20260727)
NON_FEAT = {"game_id","idx","ply","move_num","result","outcome",
            "perspective","opening_fen","source_game_id"}
THRESH, BLOCK, N_SURR = -1.0, 10, 200

df = pd.read_parquet("data/cache/white_data_max10000_min10.parquet")
feats = [c for c in df.select_dtypes(np.number).columns if c not in NON_FEAT]
X = StandardScaler().fit_transform(df[feats].values)
df = df.assign(z=PCA(n_components=1).fit_transform(X)[:, 0])

# raggruppa una volta sola in array numpy
groups = [g.z.values for _, g in df.groupby("game_id", sort=False)]
print(f"{len(groups)} traiettorie")

def mean_run(mask):
    if not mask.any():
        return 0.0
    d = np.diff(np.concatenate(([0], mask.view(np.int8), [0])))
    return (np.flatnonzero(d == -1) - np.flatnonzero(d == 1)).mean()

obs = np.mean([mean_run(z < THRESH) for z in groups])

surr = np.empty(N_SURR)
for s in range(N_SURR):
    vals = np.empty(len(groups))
    for i, z in enumerate(groups):
        nb = max(1, len(z) // BLOCK)
        blocks = np.array_split(z, nb)
        RNG.shuffle(blocks)
        vals[i] = mean_run(np.concatenate(blocks) < THRESH)
    surr[s] = vals.mean()
    if (s + 1) % 25 == 0:
        print(f"  {s+1}/{N_SURR}")

p95 = np.percentile(surr, 95)
print(f"\nResidenza osservata : {obs:.2f} ply")
print(f"Surrogati mediana   : {np.median(surr):.2f} ply")
print(f"Surrogati p95       : {p95:.2f} ply")
print(f"C4 {'SODDISFATTA' if obs > p95 else 'VIOLATA'}")