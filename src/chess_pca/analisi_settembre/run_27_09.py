"""Test del 27/09/2026. Criteri in PREREG_27_09.md (scritti prima del lancio).
Deviazioni scoperte dopo la pre-registrazione, dichiarate nel registro:
- il corpus random condivide con depth6 solo ~400 aperture: il contrasto appaiato si fa su quel sottoinsieme;
- engine_d3.pgn contiene solo 204 partite: depth3 non è appaiabile.
"""
import json
import sys

import numpy as np
import pandas as pd
from scipy.linalg import subspace_angles
from sklearn.decomposition import FactorAnalysis

sys.path.insert(0, "work")
from occupancy_grouped import FEATURES, REGIONS, select_and_test

SEED = 20260927
BASE = "/mnt/user-data/uploads/chess-pca/data"
CACHE = {"depth6": "cache_d6", "depth1": "cache_d1", "depth3": "cache_d3", "random": "cache_random",
         "3check": "cache_3check", "antichess": "cache_antichess", "full": "cache"}
MAP = {"W": 1.0, "D": 0.0, "L": -1.0}


def load(name):
    suf = "max10000_min10" if name == "full" else "max2000_min10"
    parts = []
    for p, tag in (("W", "white"), ("B", "black")):
        d = pd.read_parquet(f"{BASE}/{CACHE[name]}/{tag}_data_{suf}.parquet")
        d["perspective"] = p
        d["y"] = d["result"].astype(str).map(MAP)
        parts.append(d)
    d = pd.concat(parts, ignore_index=True)
    return d.sort_values(["game_id", "perspective", "ply"]).reset_index(drop=True)


def window(d, w):
    d = d.copy()
    d["k"] = d.groupby(["game_id", "perspective"]).cumcount()
    d = d[d["k"] < w]
    n = d.groupby(["game_id", "perspective"]).size().unstack()
    keep = n[(n >= w).all(axis=1)].index
    return d[d["game_id"].isin(keep)].reset_index(drop=True)


def ang(A, B):
    return float(np.degrees(subspace_angles(A, B)).max())


def plane_from_cov(S1, S2, n):
    mu = S1 / n
    C = S2 / n - np.outer(mu, mu)
    sd = np.sqrt(np.clip(np.diag(C), 1e-12, None))
    R = C / np.outer(sd, sd)
    lam, V = np.linalg.eigh(R)
    return V[:, np.argsort(lam)[::-1][:2]]


def game_stats(d, feats):
    X = d[feats].to_numpy(float)
    g = d["game_id"].to_numpy()
    ug, inv = np.unique(g, return_inverse=True)
    G = len(ug)
    n = np.bincount(inv, minlength=G).astype(float)
    S1 = np.zeros((G, len(feats)))
    np.add.at(S1, inv, X)
    S2 = np.zeros((G, len(feats), len(feats)))
    for i in range(G):
        Xi = X[inv == i]
        S2[i] = Xi.T @ Xi
    return ug, n, S1, S2


# ---------------------------------------------------------------- T3
def a2_boot(name, feats, B=1000, rng=None):
    d = load(name)
    y = d["y"].to_numpy()
    m = d["material"].to_numpy(float)
    X = d[feats].to_numpy(float)
    ug, inv = np.unique(d["game_id"].to_numpy(), return_inverse=True)
    G = len(ug)

    def agg(v):
        out = np.zeros((G,) + v.shape[1:])
        np.add.at(out, inv, v)
        return out
    per = dict(n=agg(np.ones(len(d))), y=agg(y), yy=agg(y * y), m=agg(m), mm=agg(m * m), my=agg(m * y),
               x=agg(X), xx=agg(X * X), xy=agg(X * y[:, None]), xm=agg(X * m[:, None]))

    def corrs(w):
        s = {k: w @ v for k, v in per.items()}
        n = s["n"]

        def r(sab, sa, sb, saa, sbb):
            cov = sab - sa * sb / n
            va = saa - sa ** 2 / n
            vb = sbb - sb ** 2 / n
            with np.errstate(invalid="ignore", divide="ignore"):
                return cov / np.sqrt(va * vb)
        rfy = r(s["xy"], s["x"], s["y"], s["xx"], s["yy"])
        rfm = r(s["xm"], s["x"], s["m"], s["xx"], s["mm"])
        rmy = r(s["my"], s["m"], s["y"], s["mm"], s["yy"])
        with np.errstate(invalid="ignore", divide="ignore"):
            part = (rfy - rfm * rmy) / np.sqrt(np.clip((1 - rfm ** 2) * (1 - rmy ** 2), 1e-12, None))
        return rfy, part, rmy
    point = corrs(np.ones(G))
    boots = [corrs(np.bincount(rng.integers(0, G, G), minlength=G).astype(float)) for _ in range(B)]
    raw = np.array([b[0] for b in boots])
    par = np.array([b[1] for b in boots])
    my = np.array([b[2] for b in boots])
    ci = lambda a: (np.nanpercentile(a, 2.5, axis=0), np.nanpercentile(a, 97.5, axis=0))
    return dict(raw=point[0], par=point[1], my=point[2], raw_ci=ci(raw), par_ci=ci(par), my_ci=ci(my))


def T3():
    rng = np.random.default_rng(SEED)
    feats = FEATURES
    ref = a2_boot("depth6", feats, rng=rng)
    out = {}
    for tgt in ("antichess", "3check"):
        t = a2_boot(tgt, feats, rng=rng)

        def robust(kind):
            lo_r, hi_r = ref[kind + "_ci"]
            lo_t, hi_t = t[kind + "_ci"]
            rpos, rneg = lo_r > 0, hi_r < 0
            tpos, tneg = lo_t > 0, hi_t < 0
            return (rpos & tneg) | (rneg & tpos)
        rr, rp = robust("raw"), robust("par")
        pt_raw = (np.sign(ref["raw"]) != np.sign(t["raw"])) & (np.abs(ref["raw"]) > .05) & (np.abs(t["raw"]) > .05)
        rp[feats.index("material")] = False
        out[tgt] = dict(
            inversioni_grezze_puntuali=int(np.nansum(pt_raw)),
            inversioni_grezze_robuste=int(rr.sum()),
            inversioni_parziali_robuste=int(rp.sum()),
            coordinate_parziali_robuste=[f for f, b in zip(feats, rp) if b],
            corr_material_esito=dict(ref=float(ref["my"]), ref_ci=[float(ref["my_ci"][0]), float(ref["my_ci"][1])],
                                     tgt=float(t["my"]), tgt_ci=[float(t["my_ci"][0]), float(t["my_ci"][1])]),
            dettaglio={f: dict(ref=round(float(ref["raw"][i]), 3), tgt=round(float(t["raw"][i]), 3),
                               ref_ci=[round(float(ref["raw_ci"][0][i]), 3), round(float(ref["raw_ci"][1][i]), 3)],
                               tgt_ci=[round(float(t["raw_ci"][0][i]), 3), round(float(t["raw_ci"][1][i]), 3)],
                               robusta=bool(rr[i]))
                       for i, f in enumerate(feats)})
    a, c = out["antichess"], out["3check"]
    mat_flip = a["corr_material_esito"]["ref_ci"][0] > 0 and a["corr_material_esito"]["tgt_ci"][1] < 0
    out["A2_regge"] = bool(a["inversioni_grezze_robuste"] >= 10 and c["inversioni_grezze_robuste"] == 0 and mat_flip)
    return out


# ---------------------------------------------------------------- T4 / T5
def starts(name):
    return pd.read_csv(f"work/start_{name}.csv", index_col="game_id")["start"]


def pairs(a, b):
    sa, sb = starts(a), starts(b)
    ga = {}
    for gid, s in sa.items():
        ga.setdefault(s, []).append(gid)
    gb = {}
    for gid, s in sb.items():
        gb.setdefault(s, []).append(gid)
    out = []
    for s, la in ga.items():
        for x, y in zip(la, gb.get(s, [])):
            out.append((x, y))
    return out


def paired_contrast(a, b, w, feats, perms=1000, rng=None):
    da, db = window(load(a), w), window(load(b), w)
    pr = [(x, y) for x, y in pairs(a, b)
          if x in set(da["game_id"]) and y in set(db["game_id"])]
    ids_a = [x for x, _ in pr]
    ids_b = [y for _, y in pr]
    ga, na, S1a, S2a = game_stats(da[da["game_id"].isin(ids_a)], feats)
    gb, nb, S1b, S2b = game_stats(db[db["game_id"].isin(ids_b)], feats)
    ia = {g: i for i, g in enumerate(ga)}
    ib = {g: i for i, g in enumerate(gb)}
    PA = np.array([ia[x] for x in ids_a])
    PB = np.array([ib[y] for y in ids_b])
    nA, s1A, s2A = na[PA], S1a[PA], S2a[PA]
    nB, s1B, s2B = nb[PB], S1b[PB], S2b[PB]

    def stat(flip):
        gA = np.where(flip[:, None], s1B, s1A).sum(0)
        gB = np.where(flip[:, None], s1A, s1B).sum(0)
        hA = np.where(flip[:, None, None], s2B, s2A).sum(0)
        hB = np.where(flip[:, None, None], s2A, s2B).sum(0)
        cA = np.where(flip, nB, nA).sum()
        cB = np.where(flip, nA, nB).sum()
        return ang(plane_from_cov(gA, hA, cA), plane_from_cov(gB, hB, cB))
    P = len(pr)
    obs = stat(np.zeros(P, bool))
    paired = np.array([stat(rng.random(P) < 0.5) for _ in range(perms)])
    # non appaiato sulle stesse partite: riassegna liberamente i 2P giochi
    allS1 = np.vstack([s1A, s1B])
    allS2 = np.concatenate([s2A, s2B])
    alln = np.concatenate([nA, nB])
    unp = []
    for _ in range(perms):
        idx = rng.permutation(2 * P)
        A_, B_ = idx[:P], idx[P:]
        unp.append(ang(plane_from_cov(allS1[A_].sum(0), allS2[A_].sum(0), alln[A_].sum()),
                       plane_from_cov(allS1[B_].sum(0), allS2[B_].sum(0), alln[B_].sum())))
    unp = np.array(unp)
    pv = lambda nul: float((np.sum(nul >= obs) + 1) / (perms + 1))
    return dict(coppie=P, angolo=obs, p_appaiato=pv(paired), p95_appaiato=float(np.percentile(paired, 95)),
                p_non_appaiato=pv(unp), p95_non_appaiato=float(np.percentile(unp, 95)))


def holm(ps):
    items = sorted(ps.items(), key=lambda kv: kv[1])
    k = len(items)
    out, ok = {}, True
    for i, (name, p) in enumerate(items):
        ok = ok and p <= 0.05 / (k - i)
        out[name] = ok
    return out


def T4():
    rng = np.random.default_rng(SEED)
    res = {c: paired_contrast(c, "depth6", 20, FEATURES, rng=rng) for c in ("random", "depth1", "3check")}
    res["holm_appaiato"] = holm({c: res[c]["p_appaiato"] for c in ("random", "depth1", "3check")})
    return res


def T5():
    rng = np.random.default_rng(SEED + 1)
    feats = [f for f in FEATURES if f != "checks_avail"]
    res = {c: paired_contrast(c, "depth6", 12, feats, rng=rng) for c in ("antichess", "3check")}
    res["ipotesi"] = bool(res["antichess"]["angolo"] > res["3check"]["angolo"] and res["antichess"]["p_appaiato"] < 0.05)
    return res


# ---------------------------------------------------------------- T6
def std_plane(X):
    Z = (X - X.mean(0)) / X.std(0)
    lam, V = np.linalg.eigh(np.corrcoef(Z.T))
    return V[:, np.argsort(lam)[::-1][:2]]


def T6(reps=5, nnull=50):
    rng = np.random.default_rng(SEED + 2)
    d = window(load("full"), 20)
    X = d[FEATURES].to_numpy(float)
    games = d["game_id"].unique()
    ref = std_plane(X)
    null = []
    for _ in range(nnull):
        p = rng.permutation(games)
        h = len(p) // 2
        A = X[d["game_id"].isin(p[:h]).to_numpy()]
        B = X[d["game_id"].isin(p[h:2 * h]).to_numpy()]
        null.append(ang(std_plane(A), std_plane(B)))
    lev = {}
    for j, f in enumerate(FEATURES):
        a = []
        for _ in range(reps):
            Y = X.copy()
            Y[:, j] = rng.permutation(Y[:, j])
            a.append(ang(ref, std_plane(Y)))
        lev[f] = float(np.mean(a))
    top = sorted(lev, key=lev.get, reverse=True)
    four = ["pawns_passed_diff", "cnt_pawn", "mobility", "captures_avail"]
    return dict(null_p95=float(np.percentile(null, 95)), null_mediana=float(np.median(null)),
                leva=dict(sorted(lev.items(), key=lambda kv: -kv[1])),
                sopra_null=int(sum(v > np.percentile(null, 95) for v in lev.values())),
                sopra_33_4=[f for f in top if lev[f] > 33.4],
                criterio=bool(all(lev[f] > 33.4 for f in four) or set(top[:4]) == set(four)))


# ---------------------------------------------------------------- T2
def paf(R, k=2, iters=200, tol=1e-6):
    """Analisi fattoriale per assi principali sulla matrice di correlazione."""
    Rinv = np.linalg.pinv(R)
    h2 = np.clip(1 - 1 / np.diag(Rinv), 0.01, 0.999)
    for _ in range(iters):
        Rr = R.copy()
        np.fill_diagonal(Rr, h2)
        lam, V = np.linalg.eigh(Rr)
        o = np.argsort(lam)[::-1][:k]
        L = V[:, o] * np.sqrt(np.clip(lam[o], 1e-9, None))
        new = np.clip((L ** 2).sum(1), 0.005, 0.999)
        if np.max(np.abs(new - h2)) < tol:
            h2 = new
            break
        h2 = new
    return L


def fa_plane(X, seed=0):
    Z = (X - X.mean(0)) / X.std(0)
    return paf(np.corrcoef(Z.T))


def fa_plane_from_cov(S1, S2, n):
    mu = S1 / n
    C = S2 / n - np.outer(mu, mu)
    sd = np.sqrt(np.clip(np.diag(C), 1e-12, None))
    return paf(C / np.outer(sd, sd))


def T2a(perms=500):
    rng = np.random.default_rng(SEED + 3)
    da, db = window(load("random"), 20), window(load("depth6"), 20)
    ga, gb = da["game_id"].unique(), db["game_id"].unique()
    n = min(len(ga), len(gb))
    A = da[da["game_id"].isin(rng.choice(ga, n, replace=False))]
    B = db[db["game_id"].isin(rng.choice(gb, n, replace=False))]
    _, na, S1a, S2a = game_stats(A, FEATURES)
    _, nb, S1b, S2b = game_stats(B, FEATURES)
    obs = ang(fa_plane_from_cov(S1a.sum(0), S2a.sum(0), na.sum()), fa_plane_from_cov(S1b.sum(0), S2b.sum(0), nb.sum()))
    obs_pca = ang(plane_from_cov(S1a.sum(0), S2a.sum(0), na.sum()), plane_from_cov(S1b.sum(0), S2b.sum(0), nb.sum()))
    S1, S2, nn = np.vstack([S1a, S1b]), np.concatenate([S2a, S2b]), np.concatenate([na, nb])
    null = []
    for _ in range(perms):
        p = rng.permutation(len(nn))
        a_, b_ = p[:n], p[n:]
        null.append(ang(fa_plane_from_cov(S1[a_].sum(0), S2[a_].sum(0), nn[a_].sum()),
                        fa_plane_from_cov(S1[b_].sum(0), S2[b_].sum(0), nn[b_].sum())))
    null = np.array(null)
    return dict(angolo_FA=obs, angolo_PCA_stesso_campione=obs_pca, null_p95=float(np.percentile(null, 95)),
                p=float((np.sum(null >= obs) + 1) / (perms + 1)))


def T2b(reps=3):
    rng = np.random.default_rng(SEED + 4)
    d = window(load("depth6"), 20)
    X = d[FEATURES].to_numpy(float)
    ref = fa_plane(X)
    lev = {}
    for j, f in enumerate(FEATURES):
        a = []
        for _ in range(reps):
            Y = X.copy()
            Y[:, j] = rng.permutation(Y[:, j])
            a.append(ang(ref, fa_plane(Y)))
        lev[f] = float(np.mean(a))
    top6 = sorted(lev, key=lev.get, reverse=True)[:6]
    four = ["pawns_passed_diff", "cnt_pawn", "mobility", "captures_avail"]
    return dict(leva_FA=dict(sorted(lev.items(), key=lambda kv: -kv[1])), top6=top6,
                criterio=bool(all(f in top6 for f in four)))


def T2c(n_splits=20):
    d = load("full")
    d["material_cp"] = d["material"]
    d = d.rename(columns={"y": "y_num"})
    d["result"] = d["y_num"]
    wr = d[d["perspective"] == "W"].groupby("game_id")["result"].first()
    games = d["game_id"].unique()
    gaps = {"W": [], "B": []}
    for k in range(n_splits):
        rng = np.random.default_rng(SEED + 100 + k)
        train = set(rng.choice(games, size=len(games) // 2, replace=False))
        tr = d[d["game_id"].isin(train)]
        mu, sd = tr[FEATURES].to_numpy(float).mean(0), tr[FEATURES].to_numpy(float).std(0)
        Ztr = (tr[FEATURES].to_numpy(float) - mu) / sd
        Rtr = np.corrcoef(Ztr.T)
        L = paf(Rtr)
        Wsc = np.linalg.pinv(Rtr) @ L          # punteggi fattoriali per regressione (Thurstone)
        z_tr = Ztr @ Wsc
        q = np.percentile(z_tr, [25, 75], axis=0)
        rows = d[d["material_cp"].abs() <= 50]
        z = ((rows[FEATURES].to_numpy(float) - mu) / sd) @ Wsc
        ind = np.column_stack([z[:, 0] < q[0, 0], z[:, 0] > q[1, 0], z[:, 1] < q[0, 1], z[:, 1] > q[1, 1]]).astype(float)
        tmp = pd.DataFrame(ind, columns=REGIONS)
        tmp["game_id"] = rows["game_id"].to_numpy()
        tmp["perspective"] = rows["perspective"].to_numpy()
        g = tmp.groupby(["game_id", "perspective"])
        R = g[REGIONS].mean()[g.size() >= 5].reset_index()
        o = select_and_test(R, wr, train)
        for p in ("W", "B"):
            gaps[p].append(o[p]["signed_gap_test_pp"])
    out = {p: dict(mediana=float(np.median(v)), min=float(np.min(v)), quota_sopra_1_6=float(np.mean(np.array(v) > 1.6)))
           for p, v in gaps.items()}
    out["regge"] = bool(all(out[p]["quota_sopra_1_6"] >= 0.9 for p in ("W", "B")))
    return out


if __name__ == "__main__":
    what = sys.argv[1]
    fn = dict(T3=T3, T4=T4, T5=T5, T6=T6, T2a=T2a, T2b=T2b, T2c=T2c)[what]
    out = fn()
    s = json.dumps(out, indent=2, ensure_ascii=False, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print(s)
    open(f"work/results/{what}_27_09.json", "w").write(s)
