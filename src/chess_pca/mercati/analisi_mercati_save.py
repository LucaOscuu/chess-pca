"""Morphological Framework sui mercati sperimentali (vedi PREDIZIONI_MERCATI.md)."""
import json, sys
import numpy as np, pandas as pd
from scipy.linalg import subspace_angles

RNG = np.random.default_rng(20261004)
PERMS = 1000
D = pd.read_parquet("work/mercati/work/panel.parquet")
FULL = ['assets_rel', 'cash_rel', 'wealth_rel', 'n_bid', 'n_ask', 'mkt_frac', 'cancel', 'bid_dev', 'ask_dev',
        'buy_q', 'sell_q', 'buy_dev', 'sell_dev']
BEHAV = ['assets_rel', 'n_bid', 'n_ask', 'mkt_frac', 'cancel', 'bid_dev', 'ask_dev', 'buy_q', 'sell_q']
for c in FULL:  # winsorizzazione 1-99
    lo, hi = D[c].quantile([.01, .99]); D[c] = D[c].clip(lo, hi)
D["market"] = D.slab + ":" + D.mlab
D["skill_hi"] = D.skills > D.groupby("market").skills.transform("median")
markets = D.market.unique()

def std_fit(X):
    mu, sd = X.mean(0), X.std(0); sd[sd == 0] = 1; return mu, sd
def plane(d, F):
    X = d[F].to_numpy(float); mu, sd = std_fit(X); Z = (X - mu) / sd
    _, s, vt = np.linalg.svd(Z - Z.mean(0), full_matrices=False)
    return vt[:2].T, s ** 2 / (len(Z) - 1)
def angle(a, b, F):
    return float(np.degrees(subspace_angles(plane(a, F)[0], plane(b, F)[0])).max())

out = {}
SAVE = {}
import pickle
for lab, F in (("completo", FULL), ("solo_comportamento", BEHAV)):
    r = {}
    # T0 spettro vs null colonne mescolate
    W, ev = plane(D, F)
    X = D[F].to_numpy(float); nulls = []
    for _ in range(200):
        Xs = np.column_stack([RNG.permutation(X[:, j]) for j in range(X.shape[1])])
        mu, sd = std_fit(Xs); Z = (Xs - mu) / sd
        nulls.append(np.linalg.svd(Z - Z.mean(0), compute_uv=False)[:3] ** 2 / (len(Z) - 1))
    nulls = np.array(nulls)
    r["T0_autovalori"] = [round(float(x), 3) for x in ev[:4]]
    r["T0_null_p95"] = [round(float(x), 3) for x in np.percentile(nulls, 95, axis=0)]
    r["T0_varianza_spiegata_PC1_PC2"] = round(float(ev[:2].sum() / ev.sum()), 3)
    sh = []
    for _ in range(200):
        m = RNG.permutation(markets); h = len(m) // 2
        sh.append(angle(D[D.market.isin(m[:h])], D[D.market.isin(m[h:])], F))
    r["T0_split_half_mediana_p95"] = [round(float(np.median(sh)), 1), round(float(np.percentile(sh, 95)), 1)]
    # loadings
    r["loadings_PC1"] = dict(zip(F, np.round(W[:, 0], 3).tolist()))
    r["loadings_PC2"] = dict(zip(F, np.round(W[:, 1], 3).tolist()))

    # T1 mappa: rep1 vs rep2 (permuto la ripetizione dentro il trader), alta vs bassa abilità (permuto dentro mercato)
    def perm_angle(labcol, groupcol):
        d = D.dropna(subset=[labcol]) if D[labcol].dtype != bool else D
        obs = angle(d[d[labcol] == True], d[d[labcol] == False], F)
        null = []
        units = d.groupby(groupcol)
        keys = d[groupcol].to_numpy()
        lab_by_unit = d.groupby("traj")[labcol].first()
        for _ in range(300):
            # permuta le etichette fra traiettorie dentro ciascun gruppo
            perm = lab_by_unit.groupby(d.groupby("traj")[groupcol].first()).transform(lambda s: RNG.permutation(s.to_numpy()))
            l = d.traj.map(perm).to_numpy().astype(bool)
            null.append(angle(d[l], d[~l], F))
        null = np.array(null)
        return dict(angolo=round(obs, 1), null_med=round(float(np.median(null)), 1), null_p95=round(float(np.percentile(null, 95)), 1),
                    p=round(float(((null >= obs).sum() + 1) / (len(null) + 1)), 4))
    D["rep2"] = D.rep == 2
    D["pl"] = D.plab
    r["T1_rep1_vs_rep2"] = perm_angle("rep2", "pl")
    r["T1_abilita_alta_vs_bassa"] = perm_angle("skill_hi", "market")

    # T2 occupazione
    def occupancy(matched):
        m = RNG.permutation(markets); tr = set(m[: len(m) // 2])
        Dtr = D[D.market.isin(tr)]
        X = Dtr[F].to_numpy(float); mu, sd = std_fit(X); Z = (X - mu) / sd; zm = Z.mean(0)
        _, _, vt = np.linalg.svd(Z - zm, full_matrices=False); Wt = vt[:2].T
        S = ((D[F].to_numpy(float) - mu) / sd - zm) @ Wt
        q = np.percentile(S[D.market.isin(tr).to_numpy()], [25, 75], axis=0)
        ind = np.column_stack([S[:, 0] < q[0, 0], S[:, 0] > q[1, 0], S[:, 1] < q[0, 1], S[:, 1] > q[1, 1]]).astype(float)
        keep = (D.wealth_rel.between(0.9, 1.1)).to_numpy() if matched else np.ones(len(D), bool)
        T = pd.DataFrame(ind[keep], columns=["PC1_low", "PC1_high", "PC2_low", "PC2_high"])
        T["traj"] = D.traj.to_numpy()[keep]
        R = T.groupby("traj").mean()
        R = R[T.groupby("traj").size() >= 3]
        info = D.groupby("traj").agg(market=("market", "first"), res=("result_num", "first"))
        R = R.join(info); R = R[R.res != 0]
        R["train"] = R.market.isin(tr)
        def sel_test(res):
            trm = R.train.to_numpy(); V = R[["PC1_low", "PC1_high", "PC2_low", "PC2_high"]].to_numpy()
            g = V[trm & (res == 1)].mean(0) - V[trm & (res == -1)].mean(0); j = int(np.argmax(np.abs(g))); s = np.sign(g[j])
            return j, s, 100 * s * (V[~trm & (res == 1), j].mean() - V[~trm & (res == -1), j].mean())
        res = R.res.to_numpy(); j, s, obs = sel_test(res)
        SAVE[(lab, matched)] = dict(Wt=Wt, mu=mu, sd=sd, zm=zm, q=q, tr=sorted(tr), j=j, s=s, obs=obs, F=F)
        null = []
        mk = R.market.to_numpy()
        for _ in range(PERMS):
            rp = res.copy()
            for mm in np.unique(mk):
                ix = np.where(mk == mm)[0]; rp[ix] = RNG.permutation(rp[ix])
            null.append(sel_test(rp)[2])
        null = np.array(null)
        return dict(regione=["PC1_low", "PC1_high", "PC2_low", "PC2_high"][j], gap_test_pp=round(float(obs), 2),
                    placebo_p95=round(float(np.percentile(null, 95)), 2), p=round(float(((null >= obs).sum() + 1) / (PERMS + 1)), 4),
                    n_traiettorie_test=int((~R.train).sum()))
    r["T2_occupazione_tutte"] = occupancy(False)
    r["T2_occupazione_ricchezza_appaiata"] = occupancy(True)

    # T3 flusso: campo di velocità su griglia 4x4
    S = D[F].to_numpy(float); mu, sd = std_fit(S); Z = (S - mu) / sd; zm = Z.mean(0)
    P = (Z - zm) @ W
    E = D[["traj", "market", "period", "skill_hi", "rep"]].copy(); E["p1"], E["p2"] = P[:, 0], P[:, 1]
    E = E.sort_values(["traj", "period"])
    E["v1"] = E.groupby("traj").p1.shift(-1) - E.p1; E["v2"] = E.groupby("traj").p2.shift(-1) - E.p2
    E = E.dropna(subset=["v1"])
    qx = np.quantile(E.p1, [.25, .5, .75]); qy = np.quantile(E.p2, [.25, .5, .75])
    E["cell"] = np.digitize(E.p1, qx) * 4 + np.digitize(E.p2, qy)
    def field_dist(sub_a, sub_b):
        fa = sub_a.groupby("cell")[["v1", "v2"]].mean(); fb = sub_b.groupby("cell")[["v1", "v2"]].mean()
        na = sub_a.groupby("cell").size(); nb = sub_b.groupby("cell").size()
        w = np.minimum(na, nb).reindex(fa.index).fillna(0); ok = w >= 30
        d2 = ((fa - fb.reindex(fa.index)) ** 2).sum(1)
        return float((w[ok] * d2[ok]).sum() / w[ok].sum())
    def perm_field(labcol, groupcol):
        obs = field_dist(E[E[labcol]], E[~E[labcol]])
        lab_u = E.groupby("traj")[labcol].first(); grp_u = E.groupby("traj")[groupcol].first()
        null = []
        for _ in range(300):
            perm = lab_u.groupby(grp_u).transform(lambda s: RNG.permutation(s.to_numpy()))
            l = E.traj.map(perm).to_numpy().astype(bool)
            null.append(field_dist(E[l], E[~l]))
        null = np.array(null)
        return dict(dist=round(obs, 5), null_med=round(float(np.median(null)), 5), rapporto=round(obs / float(np.median(null)), 2),
                    p=round(float(((null >= obs).sum() + 1) / (len(null) + 1)), 4))
    E["rep2"] = E.rep == 2
    E["pl"] = E.traj.str.split(":").str[0]
    r["T3_flusso_abilita"] = perm_field("skill_hi", "market")
    r["T3_flusso_rep1_vs_rep2"] = perm_field("rep2", "pl")
    out[lab] = r
    print(lab, json.dumps(r, indent=1, ensure_ascii=False), flush=True)
json.dump(out, open("work/mercati/work/risultati_mercati.json", "w"), indent=1, ensure_ascii=False)
pickle.dump(SAVE, open("work/mercati/work/occ_maps.pkl","wb"))
