"""Analisi del 26/09/2026. Criteri in PREREG_26_09.md, scritti prima del lancio."""
import json
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, "work")
from occupancy_grouped import FEATURES, REGIONS, fit_space, residence, select_and_test

SEED = 20260926
BLOCK = 10


def load(name):
    df = pd.read_parquet(f"work/{name}.parquet")
    df["material_cp"] = df["material"]
    return df.sort_values(["game_id", "perspective", "ply"]).reset_index(drop=True)


def feats_of(df, drop=()):
    return [f for f in FEATURES if f not in drop and df[f].std() > 0]


def traj_bounds(df):
    key = df["game_id"].astype(str) + df["perspective"]
    starts = np.flatnonzero(np.r_[True, key.values[1:] != key.values[:-1]])
    ends = np.r_[starts[1:], len(df)]
    return starts, ends


# ---------------------------------------------------------------- T1
def mean_dwell(inside, starts, ends, drop_censored):
    """inside: (n,4) bool. Durata media delle visite, per regione, poi media."""
    out = []
    for r in range(inside.shape[1]):
        lens = []
        col = inside[:, r]
        for s, e in zip(starts, ends):
            x = col[s:e]
            if not x.any():
                continue
            d = np.diff(np.r_[0, x.astype(np.int8), 0])
            a, b = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
            L = b - a
            if drop_censored:
                L = L[b != len(x)]
            lens.append(L)
        lens = np.concatenate(lens) if lens else np.array([np.nan])
        out.append(lens.mean())
    return float(np.nanmean(out))


def dwell_lists(inside, starts, ends):
    """Per ogni traiettoria: somma durate e numero visite, per regione (per il bootstrap)."""
    n = len(starts)
    S = np.zeros((n, 4))
    N = np.zeros((n, 4))
    for r in range(4):
        col = inside[:, r]
        for i, (s, e) in enumerate(zip(starts, ends)):
            x = col[s:e]
            if not x.any():
                continue
            d = np.diff(np.r_[0, x.astype(np.int8), 0])
            L = np.flatnonzero(d == -1) - np.flatnonzero(d == 1)
            S[i, r], N[i, r] = L.sum(), len(L)
    return S, N


def reorder(n_rows, starts, ends, rng, block):
    idx = np.empty(n_rows, dtype=np.int64)
    for s, e in zip(starts, ends):
        T = e - s
        if block is None:
            idx[s:e] = s + rng.permutation(T)
        else:
            nb = int(np.ceil(T / block))
            order = rng.permutation(nb)
            pieces = [np.arange(b * block, min((b + 1) * block, T)) for b in order]
            idx[s:e] = s + np.concatenate(pieces)
    return idx


def T1(name, n_sur=200, n_boot=200):
    rng = np.random.default_rng(SEED)
    df = load(name)
    feats = feats_of(df)
    mu, sd, W, q = fit_space(df, feats)
    z = ((df[feats].to_numpy(float) - mu) / sd) @ W
    inside_all = np.column_stack([z[:, 0] < q[0, 0], z[:, 0] > q[1, 0],
                                  z[:, 1] < q[0, 1], z[:, 1] > q[1, 1]])
    res = {}
    for p in ("W", "B"):
        m = (df["perspective"] == "W").to_numpy() if p == "W" else (df["perspective"] == "B").to_numpy()
        sub = df[m].reset_index(drop=True)
        ins = inside_all[m]
        st, en = traj_bounds(sub)
        r = {}
        for cens in (False, True):
            obs = mean_dwell(ins, st, en, cens)
            blk = np.array([mean_dwell(ins[reorder(len(sub), st, en, rng, BLOCK)], st, en, cens)
                            for _ in range(n_sur)])
            full = np.array([mean_dwell(ins[reorder(len(sub), st, en, rng, None)], st, en, cens)
                             for _ in range(max(20, n_sur // 10))])
            r["escluse_troncate" if cens else "principale"] = dict(
                osservata=obs, blocchi_media=float(blk.mean()), blocchi_p95=float(np.percentile(blk, 95)),
                p_blocchi=float((np.sum(blk >= obs) + 1) / (n_sur + 1)),
                eccesso_su_blocchi=float(obs / blk.mean() - 1),
                shuffle_completo_media=float(full.mean()))
        # bootstrap sulle partite dell'eccesso (analisi principale)
        S_obs, N_obs = dwell_lists(ins, st, en)
        sur_idx = reorder(len(sub), st, en, rng, BLOCK)
        S_sur, N_sur = dwell_lists(ins[sur_idx], st, en)
        games = sub.groupby(["game_id"], sort=False).ngroup().to_numpy()[st]
        ug = np.unique(games)
        boots = []
        for _ in range(n_boot):
            pick = rng.choice(ug, size=len(ug), replace=True)
            cnt = np.bincount(pick, minlength=games.max() + 1)[games]
            o = np.nanmean((cnt @ S_obs) / (cnt @ N_obs))
            s = np.nanmean((cnt @ S_sur) / (cnt @ N_sur))
            boots.append(o / s - 1)
        r["principale"]["eccesso_IC95"] = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
        r["n_traiettorie"] = len(st)
        res[p] = r
    return res


# ---------------------------------------------------------------- T2
def circ_shift(X, starts, ends, rng):
    lens = ends - starts
    traj = np.repeat(np.arange(len(starts)), lens)
    t = np.arange(len(X)) - np.repeat(starts, lens)
    T = np.repeat(lens, lens)
    off = rng.integers(0, 1 << 30, size=(len(starts), X.shape[1])) % lens[:, None]
    Y = np.empty_like(X)
    for j in range(X.shape[1]):
        Y[:, j] = X[np.repeat(starts, lens) + (t + off[traj, j]) % T, j]
    return Y


def eig_corr(X):
    return np.sort(np.linalg.eigvalsh(np.corrcoef(X.T)))[::-1]


def T2(name, drop=(), n_sur=200):
    rng = np.random.default_rng(SEED)
    df = load(name)
    feats = feats_of(df, drop)
    X = df[feats].to_numpy(float)
    st, en = traj_bounds(df)
    lam = eig_corr(X)
    null = np.array([eig_corr(circ_shift(X, st, en, rng)) for _ in range(n_sur)])
    p95 = np.percentile(null, 95, axis=0)
    above = lam > p95
    k = int(np.argmin(above)) if not above.all() else len(lam)
    return dict(d=len(feats), k_star=k, lam=lam[:5].round(3).tolist(),
                null_p95=p95[:5].round(3).tolist(),
                gap_piano_oss=float(lam[1] - lam[2]))


# ---------------------------------------------------------------- T3
def one_split(df, feats, seed, matched=True):
    rng = np.random.default_rng(seed)
    games = df["game_id"].unique()
    train = set(rng.choice(games, size=len(games) // 2, replace=False))
    mu, sd, W, q = fit_space(df[df["game_id"].isin(train)], feats)
    R = residence(df, mu, sd, W, q, feats, matched, 5)
    wr = df[df["perspective"] == "W"].groupby("game_id")["result"].first()
    return R, wr, train


def T3a(n_splits=100):
    df = load("full")
    feats = feats_of(df)
    rows = []
    for k in range(n_splits):
        R, wr, train = one_split(df, feats, SEED + k)
        o = select_and_test(R, wr, train)
        rows.append({p: o[p]["signed_gap_test_pp"] for p in ("W", "B")})
    t = pd.DataFrame(rows)
    out = {}
    for p in ("W", "B"):
        v = t[p].to_numpy()
        out[p] = dict(mediana=float(np.median(v)), p5=float(np.percentile(v, 5)),
                      p95=float(np.percentile(v, 95)),
                      quota_positivo_e_sopra_1_6=float(np.mean(v > 1.6)))
    return out


def T3b(corpora=("depth1", "depth3", "depth6"), perms=1000):
    res, zs = {}, []
    for c in corpora:
        df = load(c)
        feats = feats_of(df)
        R, wr, train = one_split(df, feats, SEED)
        o = select_and_test(R, wr, train)
        obs = np.mean([o[p]["signed_gap_test_pp"] for p in ("W", "B")])
        rng = np.random.default_rng(SEED + 1)
        b = 0
        for _ in range(perms):
            perm = pd.Series(rng.permutation(wr.to_numpy()), index=wr.index)
            op = select_and_test(R, perm, train)
            b += np.mean([op[p]["signed_gap_test_pp"] for p in ("W", "B")]) >= obs
        p = (b + 1) / (perms + 1)
        res[c] = dict(gap_medio_pp=float(obs), p=float(p))
        zs.append(norm.isf(p))
    z = float(np.sum(zs) / np.sqrt(len(zs)))
    res["stouffer"] = dict(z=z, p=float(norm.sf(z)))
    return res


if __name__ == "__main__":
    what = sys.argv[1]
    if what == "T1":
        out = {c: T1(c) for c in ("full", "depth6")}
    elif what == "T2":
        out = {}
        for c in ("full", "depth6", "random", "3check"):
            out[c] = T2(c)
            out[c + "_senza_material"] = T2(c, drop=("material",))
    elif what == "T3a":
        out = T3a()
    elif what == "T3b":
        out = T3b()
    print(json.dumps(out, indent=2))
    json.dump(out, open(f"work/results/{what}_26_09.json", "w"), indent=2)
