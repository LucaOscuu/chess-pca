"""Morphological Framework sul poker (vedi PREDIZIONI_POKER.md)."""
import json
import numpy as np, pandas as pd
from scipy.linalg import subspace_angles
from scipy.stats import mannwhitneyu

RNG = np.random.default_rng(20261004)
B = pd.read_parquet("work/poker/panel_poker_1000.parquet")
F = ["vpip", "pfr", "fold_pre", "saw_flop", "showdown", "pre_aggr", "post_aggr", "af"]
B["af"] = B.af.clip(upper=B.af.quantile(.99))
REG = ["PC1_low", "PC1_high", "PC2_low", "PC2_high"]

def fit(d):
    X = d[F].to_numpy(float); mu, sd = X.mean(0), X.std(0); sd[sd == 0] = 1
    Z = (X - mu) / sd; zm = Z.mean(0); _, s, vt = np.linalg.svd(Z - zm, full_matrices=False)
    W = vt[:2].T
    return (lambda e: ((e[F].to_numpy(float) - mu) / sd - zm) @ W), W, s ** 2 / (len(Z) - 1)
def angle(a, b):
    return float(np.degrees(subspace_angles(fit(a)[1], fit(b)[1])).max())

out = {}
# T0
_, W, ev = fit(B)
X = B[F].to_numpy(float); nul = []
for _ in range(200):
    Xs = np.column_stack([RNG.permutation(X[:, j]) for j in range(X.shape[1])]); Z = (Xs - Xs.mean(0)) / Xs.std(0)
    nul.append(np.linalg.svd(Z, compute_uv=False)[:2] ** 2 / (len(Z) - 1))
out["T0_autovalori"] = [round(float(x), 3) for x in ev[:4]]
out["T0_null_p95"] = [round(float(x), 3) for x in np.percentile(nul, 95, axis=0)]
out["loadings_PC1"] = dict(zip(F, np.round(W[:, 0], 3).tolist())); out["loadings_PC2"] = dict(zip(F, np.round(W[:, 1], 3).tolist()))
pl = B.traj.unique(); sh = []
for _ in range(200):
    p = RNG.permutation(pl); h = len(p) // 2
    sh.append(angle(B[B.traj.isin(p[:h])], B[B.traj.isin(p[h:])]))
out["T0_split_half_mediana_p95"] = [round(float(np.median(sh)), 1), round(float(np.percentile(sh, 95)), 1)]

# T1 bassa vs alta posta
lo, hi = B[B.stake == "low"], B[B.stake == "high"]
obs = angle(lo, hi); tl = B.groupby("traj").stake.first(); nul = []
for _ in range(500):
    perm = pd.Series(RNG.permutation(tl.to_numpy()), index=tl.index); l = B.traj.map(perm).to_numpy() == "low"
    nul.append(angle(B[l], B[~l]))
out["T1_bassa_vs_alta"] = dict(angolo=round(obs, 1), null_p95=round(float(np.percentile(nul, 95)), 1),
                               p=round(float((np.sum(np.array(nul) >= obs) + 1) / 501), 4))

# T2 bacino per posta, 20 divisioni
def basin_test(d, perms):
    players = d.traj.unique(); tr = set(RNG.choice(players, len(players) // 2, replace=False))
    proj = fit(d[d.traj.isin(tr)])[0]; S = proj(d); itr = d.traj.isin(tr).to_numpy()
    q = np.percentile(S[itr], [25, 75], axis=0)
    ind = np.column_stack([S[:, 0] < q[0, 0], S[:, 0] > q[1, 0], S[:, 1] < q[0, 1], S[:, 1] > q[1, 1]]).astype(float)
    T = pd.DataFrame(ind, columns=REG); T["traj"] = d.traj.to_numpy()
    R = T.groupby("traj").mean().join(d.groupby("traj").result_num.first().rename("res"))
    R = R[R.res != 0]; R["train"] = R.index.isin(tr)
    def sel(res):
        trm = R.train.to_numpy(); V = R[REG].to_numpy()
        g = V[trm & (res == 1)].mean(0) - V[trm & (res == -1)].mean(0); j = int(np.argmax(g))
        return j, 100 * (V[~trm & (res == 1), j].mean() - V[~trm & (res == -1), j].mean())
    res = R.res.to_numpy(); j, gap = sel(res)
    nul = np.array([sel(RNG.permutation(res))[1] for _ in range(perms)])
    te = R[~R.train]; w = te[te.res == 1][REG[j]]; l = te[te.res == -1][REG[j]]
    auc = mannwhitneyu(w, l).statistic / (len(w) * len(l))
    # profilo del bacino: medie delle feature dentro vs fuori (osservazioni di test)
    inb = ind[:, j].astype(bool) & ~itr
    prof = {f: [round(float(d[f].to_numpy()[inb].mean()), 3), round(float(d[f].to_numpy()[~inb & ~itr].mean()), 3)] for f in F}
    return dict(regione=REG[j], gap=gap, p=(np.sum(nul >= gap) + 1) / (perms + 1), p95=float(np.percentile(nul, 95)),
                auc=auc, vinc=w.mean(), perd=l.mean(), n_test=len(te), profilo=prof)
for st in ("low", "high"):
    d = B[B.stake == st]; rs = [basin_test(d, 1000 if i == 0 else 200) for i in range(20)]
    R = pd.DataFrame(rs)
    out[f"T2_bacino_{st}"] = dict(gap_mediano_pp=round(float(R.gap.median()), 2), p_mediano=round(float(R.p.median()), 4),
                                 quota_p05=round(float((R.p <= 0.05).mean()), 2), placebo_p95=round(float(R.p95.median()), 2),
                                 auc_mediana=round(float(R.auc.median()), 3), tempo_vinc_perd=[round(float(R.vinc.median()), 3), round(float(R.perd.median()), 3)],
                                 regioni=R.regione.value_counts().to_dict(), n_test=int(R.n_test.median()),
                                 profilo_bacino_dentro_fuori_split1=rs[0]["profilo"])

# T3 flusso bassa vs alta
proj = fit(B)[0]; S = proj(B)
E = B[["traj", "stake", "block"]].copy(); E["p1"], E["p2"] = S[:, 0], S[:, 1]
E = E.sort_values(["traj", "block"]); g = E.groupby("traj")
E["v1"] = g.p1.shift(-1) - E.p1; E["v2"] = g.p2.shift(-1) - E.p2; E = E.dropna()
qx = np.quantile(E.p1, [.25, .5, .75]); qy = np.quantile(E.p2, [.25, .5, .75]); E["cell"] = np.digitize(E.p1, qx) * 4 + np.digitize(E.p2, qy)
def fd(a, b):
    fa = a.groupby("cell")[["v1", "v2"]].mean(); fb = b.groupby("cell")[["v1", "v2"]].mean()
    w = np.minimum(a.groupby("cell").size(), b.groupby("cell").size()).reindex(fa.index).fillna(0); ok = w >= 30
    return float((w[ok] * ((fa - fb.reindex(fa.index)) ** 2).sum(axis=1)[ok]).sum() / w[ok].sum())
obs = fd(E[E.stake == "low"], E[E.stake == "high"]); tl = E.groupby("traj").stake.first(); nul = []
for _ in range(500):
    perm = pd.Series(RNG.permutation(tl.to_numpy()), index=tl.index); l = E.traj.map(perm).to_numpy() == "low"
    nul.append(fd(E[l], E[~l]))
out["T3_flusso_bassa_vs_alta"] = dict(dist=round(obs, 5), rapporto=round(obs / float(np.median(nul)), 2),
                                     p=round(float((np.sum(np.array(nul) >= obs) + 1) / 501), 4))
out["n"] = B.groupby("stake").traj.nunique().to_dict()
print(json.dumps(out, indent=1, ensure_ascii=False))
json.dump(out, open("work/poker/risultati_poker_1000.json", "w"), indent=1, ensure_ascii=False)
