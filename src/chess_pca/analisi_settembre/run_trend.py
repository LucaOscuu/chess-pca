"""Test di trend sulla capacità di calcolo. Criteri in PREREG_TREND.md."""
import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "work")
from occupancy_grouped import FEATURES, fit_space, residence, select_and_test
from run_27_09 import load, window, game_stats, plane_from_cov, ang

SEED = 20260927
CAP = {"random": 0, "depth1": 1, "depth3": 2, "depth6": 3}


def G(reps=200):
    rng = np.random.default_rng(SEED)
    st = {}
    for c in CAP:
        d = window(load(c), 20)
        g, n, S1, S2 = game_stats(d, FEATURES)
        st[c] = (n, S1, S2)
    n6 = len(st["depth6"][0])
    rows = []
    for _ in range(reps):
        p = rng.permutation(n6)
        ref, cmp_ = p[: n6 // 2], p[n6 // 2:]
        n6s, S16, S26 = st["depth6"]
        Vref = plane_from_cov(S16[ref].sum(0), S26[ref].sum(0), n6s[ref].sum())
        m = len(cmp_)
        row = {}
        for c in CAP:
            n, S1, S2 = st[c]
            idx = cmp_ if c == "depth6" else rng.choice(len(n), m, replace=False)
            row[c] = ang(Vref, plane_from_cov(S1[idx].sum(0), S2[idx].sum(0), n[idx].sum()))
        x = np.array([CAP[c] for c in CAP], float)
        y = np.array([row[c] for c in CAP])
        row["beta"] = float(np.polyfit(x, y, 1)[0])
        row["ordine"] = bool(row["depth1"] > row["depth3"] > row["depth6"])
        rows.append(row)
    t = pd.DataFrame(rows)
    out = {c: dict(mediana=float(t[c].median()), p5=float(t[c].quantile(.05)), p95=float(t[c].quantile(.95)))
           for c in CAP}
    out["beta"] = dict(mediana=float(t.beta.median()), int_2_5=float(t.beta.quantile(.025)),
                       int_97_5=float(t.beta.quantile(.975)))
    out["quota_ordine_d1_d3_d6"] = float(t.ordine.mean())
    out["regge"] = bool(out["beta"]["int_97_5"] < 0 and out["quota_ordine_d1_d3_d6"] >= 0.95)
    return out


def occ_gap(d, seeds):
    wr = d[d["perspective"] == "W"].groupby("game_id")["result"].first()
    games = d["game_id"].unique()
    feats = [f for f in FEATURES if d[f].std() > 0]
    gaps = []
    for s in seeds:
        rng = np.random.default_rng(s)
        train = set(rng.choice(games, size=len(games) // 2, replace=False))
        mu, sd, W, q = fit_space(d[d["game_id"].isin(train)], feats)
        R = residence(d, mu, sd, W, q, feats, True, 5)
        o = select_and_test(R, wr, train)
        gaps.append(np.mean([o[p]["signed_gap_test_pp"] for p in ("W", "B")]))
    return float(np.mean(gaps))


def prep(c):
    d = load(c)
    d["material_cp"] = d["material"]
    d["result"] = d["y"]
    return d


def O(perms=200, n_splits=5):
    rng = np.random.default_rng(SEED + 1)
    cs = ["depth1", "depth3", "depth6"]
    x = np.array([1, 2, 3], float)
    data = {c: prep(c) for c in cs}
    seeds = [SEED + 10 + k for k in range(n_splits)]
    obs_g = {c: occ_gap(data[c], seeds) for c in cs}
    obs = float(np.polyfit(x, [obs_g[c] for c in cs], 1)[0])
    # null: si riassegnano le partite fra i tre corpora (nuovi id univoci)
    parts, sizes = [], []
    for k, c in enumerate(cs):
        d = data[c].copy()
        d["game_id"] = d["game_id"].astype(np.int64) + k * 10_000_000
        parts.append(d)
        sizes.append(d["game_id"].nunique())
    pool = pd.concat(parts, ignore_index=True)
    gids = pool["game_id"].unique()
    null = []
    for _ in range(perms):
        p = rng.permutation(gids)
        cuts = np.cumsum([0] + sizes)
        g = [pool[pool["game_id"].isin(set(p[cuts[i]:cuts[i + 1]]))] for i in range(3)]
        null.append(np.polyfit(x, [occ_gap(gi, seeds) for gi in g], 1)[0])
    null = np.array(null)
    p = float((np.sum(null >= obs) + 1) / (perms + 1))
    return dict(gap_per_corpus=obs_g, pendenza=obs, null_p95=float(np.percentile(null, 95)), p=p,
                regge=bool(obs > 0 and p < 0.05))


if __name__ == "__main__":
    what = sys.argv[1]
    out = G() if what == "G" else O(perms=int(sys.argv[2]) if len(sys.argv) > 2 else 200)
    s = json.dumps(out, indent=2, ensure_ascii=False)
    print(s)
    open(f"work/results/trend_{what}.json", "w").write(s)
