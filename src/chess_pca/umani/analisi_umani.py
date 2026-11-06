"""Umani vs motore: rotazione del piano, occupazione, obiettivo (vedi PREDIZIONI_UMANI.md)."""
import sys, json, argparse
from pathlib import Path
import numpy as np, pandas as pd
from scipy.linalg import subspace_angles
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

REPO = Path("/mnt/user-data/uploads/chess-pca")
sys.path.insert(0, str(REPO / "src" / "chess_pca"))
import occupancy_grouped as OG
import a2_partial as A2

H = Path("work/umani/data")
CORPORA = {
    "depth6": (REPO / "data/cache_d6/white_data_max2000_min10.parquet", REPO / "data/cache_d6/black_data_max2000_min10.parquet"),
    "full": (REPO / "data/cache/white_data_max10000_min10.parquet", REPO / "data/cache/black_data_max10000_min10.parquet"),
    "classica": (H / "standard_white.parquet", H / "standard_black.parquet"),
    "veloci": (H / "fast_white.parquet", H / "fast_black.parquet"),
}
FEATS = OG.FEATURES
LEN = 20


def load_long(name):
    w, b = CORPORA[name]
    fr = []
    for p, persp in ((w, "W"), (b, "B")):
        d = pd.read_parquet(p)
        d["perspective"] = persp
        d["game_id"] = name + ":" + d["game_id"].astype(str)
        fr.append(d)
    d = pd.concat(fr, ignore_index=True)
    d["result_num"] = d["result"].map({"W": 1, "D": 0, "L": -1})
    return d.sort_values(["game_id", "perspective", "ply"]).reset_index(drop=True)


def window(d, start):
    d = d.copy()
    d["k"] = d.groupby(["game_id", "perspective"]).cumcount()
    d = d[(d.k >= start) & (d.k < start + LEN)]
    n = d.groupby(["game_id", "perspective"]).size().unstack()
    keep = n[(n == LEN).all(axis=1)].index
    return d[d.game_id.isin(keep)]


def plane(d):
    X = StandardScaler().fit_transform(d[FEATS].to_numpy(float))
    return PCA(n_components=2).fit(X).components_.T


def angle(a, b):
    return float(np.degrees(subspace_angles(plane(a), plane(b))).max())


def perm_test(A, B, perms, rng):
    ga, gb = A.game_id.unique(), B.game_id.unique()
    n = min(len(ga), len(gb))
    A = A[A.game_id.isin(rng.choice(ga, n, replace=False))]
    B = B[B.game_id.isin(rng.choice(gb, n, replace=False))]
    obs = angle(A, B)
    pool = pd.concat([A, B]); g = pool.game_id.unique()
    idx = {k: v for k, v in pool.groupby("game_id").groups.items()}
    null = []
    for _ in range(perms):
        p = rng.permutation(g)
        i1 = np.concatenate([idx[x] for x in p[:n]]); i2 = np.concatenate([idx[x] for x in p[n:]])
        null.append(angle(pool.loc[i1], pool.loc[i2]))
    null = np.array(null)
    return dict(n=int(n), angolo=round(obs, 1), null_med=round(float(np.median(null)), 1),
                null_p95=round(float(np.percentile(null, 95)), 1), p=round(float(((null >= obs).sum() + 1) / (perms + 1)), 4))


def occupancy(d, perms, label):
    x = d.rename(columns={})[["game_id", "perspective", "ply", "result_num"] + FEATS].copy()
    x["result"] = x["result_num"]; x["material_cp"] = x["material"]
    a = argparse.Namespace(seed=20260925, perms=perms, min_plies=5)
    return {"label": label, "n_games": int(x.game_id.nunique()),
            "all_plies": OG.run(x, a, matched=False), "material_matched": OG.run(x, a, matched=True)}


def a2(d):
    rows, r_my = A2.partials(d.assign(result=d["result"]), FEATS)
    return pd.DataFrame(rows, index=["grezza", "parziale"]).T, r_my


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--perms", type=int, default=1000)
    a = ap.parse_args()
    out = Path("work/umani/out"); out.mkdir(exist_ok=True)
    D = {k: load_long(k) for k in CORPORA}
    rng = np.random.default_rng(20261003)
    res = {"esiti": {k: D[k].groupby("game_id").result.first().value_counts(normalize=True).round(3).to_dict()
                     for k in D}}
    # 1. rotazione
    rot = []
    for start, lab in ((10, "primaria_post_apertura"), (0, "secondaria_0_19")):
        Wd = {k: window(D[k], start) for k in D}
        for x, y in (("classica", "depth6"), ("veloci", "depth6"), ("classica", "full"), ("veloci", "full"),
                     ("classica", "veloci"), ("full", "depth6")):
            r = perm_test(Wd[x], Wd[y], a.perms, rng); r.update(finestra=lab, contrasto=f"{x} vs {y}")
            rot.append(r); print(r, flush=True)
    pd.DataFrame(rot).to_csv(out / "rotazione.csv", index=False)
    # 2. occupazione (traiettorie intere e finestra primaria)
    occ = {}
    for k in ("classica", "veloci"):
        occ[k + "_intere"] = occupancy(D[k], a.perms, k)
        occ[k + "_post_apertura"] = occupancy(window(D[k], 10), a.perms, k)
        print(k, json.dumps(occ[k + "_intere"]["material_matched"]), flush=True)
    json.dump(occ, open(out / "occupazione.json", "w"), indent=2)
    # 3. obiettivo (A2 parziali vs depth6)
    ref, my_ref = a2(D["depth6"]); tabs = {}
    for k in ("classica", "veloci", "full"):
        t, my = a2(D[k])
        tab = pd.DataFrame({"parz_depth6": ref.parziale, f"parz_{k}": t.parziale, "grezza_depth6": ref.grezza, f"grezza_{k}": t.grezza}).round(3)
        inv = (tab.parz_depth6 * tab[f"parz_{k}"] < 0) & (tab.parz_depth6.abs() > .05) & (tab[f"parz_{k}"].abs() > .05)
        invg = (tab.grezza_depth6 * tab[f"grezza_{k}"] < 0) & (tab.grezza_depth6.abs() > .05) & (tab[f"grezza_{k}"].abs() > .05)
        tabs[k] = dict(r_material_esito=round(my, 3), inv_parziali=tab.index[inv].tolist(), inv_grezze=tab.index[invg].tolist())
        tab.to_csv(out / f"a2_{k}.csv")
    tabs["depth6_r_material_esito"] = round(my_ref, 3)
    res["obiettivo"] = tabs
    json.dump(res, open(out / "esiti_obiettivo.json", "w"), indent=2)
    print(json.dumps(res, indent=2))
