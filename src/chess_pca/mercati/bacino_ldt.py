"""Test B (bacino e abilità) e C (learning direction). Vedi PREDIZIONI_BACINO_LDT.md."""
import json
import numpy as np, pandas as pd
from scipy.stats import spearmanr

RNG = np.random.default_rng(20261004)
D = pd.read_parquet("work/mercati/work/panel.parquet")
FULL = ['assets_rel', 'cash_rel', 'wealth_rel', 'n_bid', 'n_ask', 'mkt_frac', 'cancel', 'bid_dev', 'ask_dev',
        'buy_q', 'sell_q', 'buy_dev', 'sell_dev']
BEHAV = ['assets_rel', 'n_bid', 'n_ask', 'mkt_frac', 'cancel', 'bid_dev', 'ask_dev', 'buy_q', 'sell_q']
RAW_BUY_DEV, RAW_SELL_DEV, RAW_BIDDEV = D.buy_dev.copy(), D.sell_dev.copy(), D.bid_dev.copy()
for c in FULL:
    lo, hi = D[c].quantile([.01, .99]); D[c] = D[c].clip(lo, hi)
D["market"] = D.slab + ":" + D.mlab
D["skill_hi"] = D.skills > D.groupby("market").skills.transform("median")
D["earn_rel"] = D.earnings / D.groupby(["market", "rep"]).earnings.transform("mean")
markets = D.market.unique()
REG = ["PC1_low", "PC1_high", "PC2_low", "PC2_high"]

def fit(train, F):
    X = train[F].to_numpy(float); mu, sd = X.mean(0), X.std(0); sd[sd == 0] = 1
    Z = (X - mu) / sd; zm = Z.mean(0); _, _, vt = np.linalg.svd(Z - zm, full_matrices=False)
    W = vt[:2].T
    proj = lambda d: ((d[F].to_numpy(float) - mu) / sd - zm) @ W
    return proj

def residences(F, tr):
    proj = fit(D[D.market.isin(tr)], F)
    S = proj(D); itr = D.market.isin(tr).to_numpy()
    q = np.percentile(S[itr], [25, 75], axis=0)
    ind = np.column_stack([S[:, 0] < q[0, 0], S[:, 0] > q[1, 0], S[:, 1] < q[0, 1], S[:, 1] > q[1, 1]]).astype(float)
    T = pd.DataFrame(ind, columns=REG); T["traj"] = D.traj.to_numpy()
    R = T.groupby("traj").mean()
    info = D.groupby("traj").agg(market=("market", "first"), res=("result_num", "first"), skills=("skills", "first"),
                                 earn_rel=("earn_rel", "first"), w0=("wealth_rel", "first"))
    R = R.join(info); R["train"] = R.market.isin(tr)
    trm = R.train & (R.res != 0)
    gaps = R[trm & (R.res == 1)][REG].mean() - R[trm & (R.res == -1)][REG].mean()
    basin = gaps.idxmax(); sign = 1.0
    R["basin"] = R[basin]
    return R, basin, sign, S

def test_B(F):
    rows = []
    for s in range(20):
        m = RNG.permutation(markets); tr = set(m[: len(m) // 2])
        R, basin, sign, _ = residences(F, tr)
        te = R[~R.train].dropna(subset=["skills"])
        rho = spearmanr(te.skills, te.basin).statistic
        null = []
        mk = te.market.to_numpy(); sk = te.skills.to_numpy(); b = te.basin.to_numpy()
        for _ in range(1000 if s == 0 else 200):
            sp = sk.copy()
            for mm in np.unique(mk):
                ix = np.where(mk == mm)[0]; sp[ix] = RNG.permutation(sp[ix])
            null.append(spearmanr(sp, b).statistic)
        null = np.array(null)
        # B2: regressione guadagno ~ abilità (+ bacino), standardizzate, solo test
        y = (te.earn_rel - te.earn_rel.mean()) / te.earn_rel.std()
        xs = (te.skills - te.skills.mean()) / te.skills.std(); xb = (te.basin - te.basin.mean()) / te.basin.std()
        b1 = np.linalg.lstsq(np.column_stack([np.ones(len(y)), xs]), y, rcond=None)[0][1]
        b2 = np.linalg.lstsq(np.column_stack([np.ones(len(y)), xs, xb]), y, rcond=None)[0]
        rows.append(dict(bacino=f"{basin}{'+' if sign > 0 else ' (complemento)'}", rho=rho,
                         p=((null >= rho).sum() + 1) / (len(null) + 1), beta_abilita_solo=b1,
                         beta_abilita_con_bacino=b2[1], beta_bacino=b2[2], n=len(te)))
    R = pd.DataFrame(rows)
    return dict(rho_mediana=round(float(R.rho.median()), 3), rho_min_max=[round(float(R.rho.min()), 3), round(float(R.rho.max()), 3)],
                p_mediana=round(float(R.p.median()), 4), quota_split_p05=round(float((R.p <= 0.05).mean()), 2),
                beta_abilita_solo=round(float(R.beta_abilita_solo.median()), 3),
                beta_abilita_con_bacino=round(float(R.beta_abilita_con_bacino.median()), 3),
                beta_bacino=round(float(R.beta_bacino.median()), 3), bacini_scelti=R.bacino.value_counts().to_dict(),
                n_traiettorie_test=int(R.n.median()))

def test_C(F):
    m = RNG.permutation(markets); tr = set(m[: len(m) // 2])
    R, basin, sign, S = residences(F, tr)
    E = D[["traj", "market", "period", "skill_hi", "result_num", "buy_q", "sell_q"]].copy()
    E["buy_dev_raw"], E["sell_dev_raw"], E["bid_raw"] = RAW_BUY_DEV.values, RAW_SELL_DEV.values, RAW_BIDDEV.values
    E["p1"], E["p2"] = S[:, 0], S[:, 1]
    E["n_bidraw"] = D.n_bid.values
    itr = E.market.isin(tr)
    pos = E[itr & (E.result_num != 0)].groupby("traj").agg(p1=("p1", "mean"), p2=("p2", "mean"), r=("result_num", "first"))
    u = (pos[pos.r == 1][["p1", "p2"]].mean() - pos[pos.r == -1][["p1", "p2"]].mean()).to_numpy(); u = u / np.hypot(*u)
    E = E.sort_values(["traj", "period"]); g = E.groupby("traj")
    E["v1"] = g.p1.shift(-1) - E.p1; E["v2"] = g.p2.shift(-1) - E.p2
    E["dbid"] = g.bid_raw.shift(-1) - E.bid_raw
    E["has_bid_next"] = g.n_bidraw.shift(-1) > 0
    E["has_bid_now"] = E.n_bidraw > 0
    E = E.dropna(subset=["v1"])
    E["errore"] = ((E.buy_q > 0) & (E.buy_dev_raw > 0.1)) | ((E.sell_q > 0) & (E.sell_dev_raw < -0.1))
    E["overpay"] = (E.buy_q > 0) & (E.buy_dev_raw > 0.1)
    E["proj"] = E.v1 * u[0] + E.v2 * u[1]
    te = E[~E.market.isin(tr)]
    # C1: variazione del prezzo dei bid dopo aver pagato troppo (solo periodi con bid in t e t+1)
    c1 = te[te.has_bid_now & te.has_bid_next]
    a = c1[c1.overpay].groupby("traj").dbid.mean(); b = c1[~c1.overpay].groupby("traj").dbid.mean()
    # C2: differenza per trader (stesso trader con e senza errore)
    pe = te[te.errore].groupby("traj").proj.mean(); pn = te[~te.errore].groupby("traj").proj.mean()
    both = pd.concat([pe, pn], axis=1, keys=["e", "n"]).dropna(); diff = both.e - both.n
    sk = te.groupby("traj").skill_hi.first().reindex(diff.index)
    def signflip(x, k=5000):
        x = x.to_numpy(); obs = x.mean(); nul = [(x * RNG.choice([-1, 1], len(x))).mean() for _ in range(k)]
        return round(float(obs), 4), round(float((np.sum(np.abs(nul) >= abs(obs)) + 1) / (k + 1)), 4)
    c1both = pd.concat([a, b], axis=1, keys=["e", "n"]).dropna(); c1d = c1both.e - c1both.n
    hi, lo = diff[sk == True], diff[sk == False]
    obs3 = hi.mean() - lo.mean(); lab = (sk == True).to_numpy(); vals = diff.to_numpy(); nul = []
    for _ in range(5000):
        p = RNG.permutation(lab); nul.append(vals[p].mean() - vals[~p].mean())
    return dict(direzione_u=[round(float(x), 3) for x in u],
                C1_delta_bid_errore_meno_senza=signflip(c1d), C1_n_trader=int(len(c1d)),
                C2_proiezione_errore_meno_senza=signflip(diff), C2_n_trader=int(len(diff)),
                C2_media_dopo_errore=round(float(both.e.mean()), 4), C2_media_senza=round(float(both.n.mean()), 4),
                C3_alta_meno_bassa=round(float(obs3), 4), C3_p=round(float((np.sum(np.abs(nul) >= abs(obs3)) + 1) / 5001), 4),
                C3_alta=round(float(hi.mean()), 4), C3_bassa=round(float(lo.mean()), 4))

out = {}
for lab, F in (("completo", FULL), ("solo_comportamento", BEHAV)):
    out[lab] = {"B_bacino_abilita": test_B(F), "C_learning_direction": test_C(F)}
    print(lab, json.dumps(out[lab], indent=1, ensure_ascii=False), flush=True)
json.dump(out, open("work/mercati/work/risultati_bacino_ldt.json", "w"), indent=1, ensure_ascii=False)
