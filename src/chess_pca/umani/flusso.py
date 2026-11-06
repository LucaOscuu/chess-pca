"""Stessa mappa, flusso diverso? (vedi PREDIZIONI_FLUSSO.md)"""
import json, sys
import numpy as np, pandas as pd
sys.path.insert(0, "work/umani")
from analisi_umani import load_long, FEATS

RNG = np.random.default_rng(20261004)
NAMES = ["classica", "veloci", "full", "depth6"]
PERMS = 500

D = {k: load_long(k) for k in NAMES}
for k, d in D.items():
    d["corpus"] = k
    d["k"] = d.groupby(["game_id", "perspective"]).cumcount()
    D[k] = d[d.k >= 10]

# mappa comune: pool bilanciato per partite
n = min(d.game_id.nunique() for d in D.values())
pool = pd.concat([d[d.game_id.isin(RNG.choice(d.game_id.unique(), n, replace=False))] for d in D.values()])
X = pool[FEATS].to_numpy(float); mu, sd = X.mean(0), X.std(0); sd[sd == 0] = 1
Z = (X - mu) / sd
_, _, vt = np.linalg.svd(Z - Z.mean(0), full_matrices=False)
W = vt[:2].T; zmean = Z.mean(0)

A = pd.concat(D.values(), ignore_index=True)
S = ((A[FEATS].to_numpy(float) - mu) / sd - zmean) @ W
A["p1"], A["p2"] = S[:, 0], S[:, 1]
A = A.sort_values(["corpus", "game_id", "perspective", "k"])
g = A.groupby(["game_id", "perspective"])
A["v1"] = g.p1.shift(-1) - A.p1
A["v2"] = g.p2.shift(-1) - A.p2
A["dmat"] = A.material - g.material.shift(1)
A["step"] = np.hypot(A.v1, A.v2)
A["stato"] = pd.cut(A.material, [-1e9, -50.5, 50.5, 1e9], labels=["svantaggio", "pari", "vantaggio"])
A = A.dropna(subset=["v1"])
out = {"n_partite_pool_per_corpus": int(n)}

# M1 velocità: media per partita, poi media fra partite; permutazione per partita
def m1(a, b, stato):
    x = A[(A.corpus.isin([a, b])) & (A.stato == stato)].groupby(["corpus", "game_id"]).step.mean().reset_index()
    obs = x[x.corpus == b].step.mean() - x[x.corpus == a].step.mean()
    lab = (x.corpus == b).to_numpy(); vals = x.step.to_numpy(); null = []
    for _ in range(PERMS):
        p = RNG.permutation(lab); null.append(vals[p].mean() - vals[~p].mean())
    null = np.abs(null)
    return dict(media_a=round(float(x[x.corpus == a].step.mean()), 4), media_b=round(float(x[x.corpus == b].step.mean()), 4),
                diff_b_meno_a=round(float(obs), 4), p_bilaterale=round(float(((null >= abs(obs)).sum() + 1) / (PERMS + 1)), 4))
out["M1_velocita"] = {f"{a} vs {b} | {s}": m1(a, b, s) for a, b in [("classica", "veloci"), ("full", "classica"), ("full", "veloci")]
                      for s in ["svantaggio", "pari", "vantaggio"]}
out["M1_medie"] = A.groupby(["corpus", "stato"], observed=True).step.median().round(4).unstack().to_dict()

# M2 campo di flusso su griglia 6x6 (quantili del pool)
qx = np.quantile(A.p1, np.linspace(0, 1, 7)[1:-1]); qy = np.quantile(A.p2, np.linspace(0, 1, 7)[1:-1])
A["cell"] = np.digitize(A.p1, qx) * 6 + np.digitize(A.p2, qy)
def per_game(c):
    x = A[A.corpus == c]
    gid = x.game_id.astype("category")
    G = gid.cat.codes.to_numpy(); C = x.cell.to_numpy()
    ng = gid.cat.categories.size
    cnt = np.zeros((ng, 36)); s1 = np.zeros((ng, 36)); s2 = np.zeros((ng, 36))
    np.add.at(cnt, (G, C), 1); np.add.at(s1, (G, C), x.v1); np.add.at(s2, (G, C), x.v2)
    return cnt, s1, s2
PG = {c: per_game(c) for c in NAMES}
def field_dist(ca, cb):
    (na, a1, a2), (nb, b1, b2) = ca, cb
    Na, Nb = na.sum(0), nb.sum(0); w = np.minimum(Na, Nb); ok = w >= 30
    fa = np.stack([a1.sum(0), a2.sum(0)]) / np.maximum(Na, 1); fb = np.stack([b1.sum(0), b2.sum(0)]) / np.maximum(Nb, 1)
    d2 = ((fa - fb) ** 2).sum(0)
    return float((w[ok] * d2[ok]).sum() / w[ok].sum())
def m2(a, b):
    ca, cb = PG[a], PG[b]
    m = min(ca[0].shape[0], cb[0].shape[0])
    ia = RNG.choice(ca[0].shape[0], m, replace=False); ib = RNG.choice(cb[0].shape[0], m, replace=False)
    ca = tuple(t[ia] for t in ca); cb = tuple(t[ib] for t in cb)
    obs = field_dist(ca, cb)
    allg = tuple(np.vstack([x, y]) for x, y in zip(ca, cb)); null = []
    for _ in range(PERMS):
        p = RNG.permutation(2 * m)
        null.append(field_dist(tuple(t[p[:m]] for t in allg), tuple(t[p[m:]] for t in allg)))
    null = np.array(null)
    return dict(n=int(m), dist_oss=round(obs, 5), null_med=round(float(np.median(null)), 5),
                null_p95=round(float(np.percentile(null, 95)), 5), rapporto_oss_null=round(obs / float(np.median(null)), 2),
                p=round(float(((null >= obs).sum() + 1) / (PERMS + 1)), 4))
out["M2_campo"] = {f"{a} vs {b}": m2(a, b) for a, b in [("classica", "veloci"), ("classica", "full"), ("veloci", "full"),
                                                         ("full", "depth6"), ("classica", "depth6")]}

# M3 risposta all'impulso
games = A.game_id.unique(); train = set(RNG.choice(games, len(games) // 2, replace=False))
A["train"] = A.game_id.isin(train)
tr = A[A.train & (A.material.abs() <= 50) & (A.result_num != 0)]
pos = tr.groupby(["game_id", "perspective"]).agg(p1=("p1", "mean"), p2=("p2", "mean"), r=("result_num", "first"))
u = pos[pos.r == 1][["p1", "p2"]].mean() - pos[pos.r == -1][["p1", "p2"]].mean()
u = (u / np.hypot(*u)).to_numpy()
out["M3_direzione_u"] = [round(float(x), 3) for x in u]
te = A[~A.train].copy()
te["proj"] = te.v1 * u[0] + te.v2 * u[1]
te["impulso"] = te.dmat <= -100
m3 = {}
for c in NAMES:
    x = te[te.corpus == c]
    gi = x[x.impulso].groupby("game_id").proj.mean(); gn = x[~x.impulso].groupby("game_id").proj.mean()
    bs = [RNG.choice(gi.to_numpy(), len(gi)).mean() for _ in range(2000)]
    m3[c] = dict(n_partite_con_impulso=int(len(gi)), proj_dopo_impulso=round(float(gi.mean()), 4),
                 ic95=[round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)],
                 proj_senza_impulso=round(float(gn.mean()), 4))
out["M3_impulso"] = m3
# confronto classica vs veloci sulla risposta all'impulso (permutazione per partita)
x = te[te.impulso & te.corpus.isin(["classica", "veloci"])].groupby(["corpus", "game_id"]).proj.mean().reset_index()
lab = (x.corpus == "veloci").to_numpy(); vals = x.proj.to_numpy()
obs = vals[lab].mean() - vals[~lab].mean(); null = []
for _ in range(2000):
    p = RNG.permutation(lab); null.append(vals[p].mean() - vals[~p].mean())
out["M3_veloci_meno_classica"] = dict(diff=round(float(obs), 4), p_bilaterale=round(float((np.sum(np.abs(null) >= abs(obs)) + 1) / 2001), 4))
json.dump(out, open("work/umani/out/flusso.json", "w"), indent=2, ensure_ascii=False)
print(json.dumps(out, indent=2, ensure_ascii=False))
