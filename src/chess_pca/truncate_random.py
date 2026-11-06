import pandas as pd, numpy as np
rng = np.random.default_rng(42)

for persp, sf_path, rd_path in [
    ("white", "data/cache/white_data_max10000_min10.parquet",
              "data/cache_random/white_data_max2000_min10.parquet"),
    ("black", "data/cache/black_data_max10000_min10.parquet",
              "data/cache_random/black_data_max2000_min10.parquet")]:
    sf = pd.read_parquet(sf_path)
    rd = pd.read_parquet(rd_path)
    lens = sf.groupby("game_id").size().values
    out = [g.head(int(rng.choice(lens))) for _, g in rd.groupby("game_id")]
    t = pd.concat(out, ignore_index=True)
    t.to_parquet(f"data/cache_random/{persp}_trunc.parquet", index=False)
    print(f"{persp}: {len(t):,} righe, media {len(t)/t.game_id.nunique():.0f} ply")