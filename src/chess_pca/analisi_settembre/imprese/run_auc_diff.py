"""Differenza fra AUC con bootstrap appaiato. Criteri in PREREG_AUC.md."""
import json

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA
from sklearn.metrics import roc_auc_score

exec(open("run_a6.py").read().split("firms = d.groupby('company_name').y.first()")[0])  # d, Z1, Z2
d["company_name"] = pd.factorize(d.company_name)[0]
name = d.company_name.values
firms = d.groupby("company_name").y.first()
allids = firms.index.values


def shared(Z1s, Z2s, fitmask, evalmask, n_cc=2):
    p1 = PCA(2, random_state=0).fit(Z1s[fitmask]); p2 = PCA(2, random_state=0).fit(Z2s[fitmask])
    a_fit, b_fit = p1.transform(Z1s[fitmask]), p2.transform(Z2s[fitmask])
    a_ev, b_ev = p1.transform(Z1s[evalmask]), p2.transform(Z2s[evalmask])
    cca = CCA(n_cc, max_iter=2000).fit(a_fit, b_fit)
    u, v = cca.transform(a_ev, b_ev)
    return u, v, a_ev, b_ev


def scores_B(seed):
    """Replica esatta di run_consolidation.py: stessa sequenza del generatore fino allo split."""
    rng = np.random.default_rng(seed)
    ids = allids[rng.permutation(len(allids))]
    inA = np.isin(name, list(set(ids[: len(ids) // 2])))
    u, v, a_ev, b_ev = shared(Z1.values, Z2.values, inA, ~inA)
    sgn = lambda x, y: np.sign(np.corrcoef(x, y)[0, 1])
    S_sh = u[:, 0] / u[:, 0].std() + sgn(u[:, 0], v[:, 0]) * v[:, 0] / v[:, 0].std()
    S_2 = u[:, 1] / u[:, 1].std() + sgn(u[:, 1], v[:, 1]) * v[:, 1] / v[:, 1].std()
    dB = pd.DataFrame({"firm": name[~inA], "y": d.y.values[~inA], "shared": S_sh, "second": S_2,
                       "map1_PC1": a_ev[:, 0]})
    g = dB.groupby("firm").mean()
    y = g.y.astype(int).values
    out = {}
    for k in ("shared", "second", "map1_PC1"):
        s = g[k].values
        out[k] = -s if roc_auc_score(y, s) < 0.5 else s
    return y, out


if __name__ == "__main__":
    orig = json.load(open("orig/results_consolidation.json"))["T3"]
    rng_boot = np.random.default_rng(20260927)

    # Split identico all'originale: si riesegue run_consolidation.py e se ne prendono i punteggi del lato B
    import runpy, io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ns = runpy.run_path("run_consolidation.py")
    g, yB = ns["g"], ns["yB"]
    sc = {}
    for k in ("shared", "second", "map1_PC1"):
        s = g[k].values if k != "shared" else ns["dB"].groupby("firm").mean()["shared"].values
        sc[k] = -s if roc_auc_score(yB, s) < 0.5 else s
    point = {k: roc_auc_score(yB, sc[k]) for k in sc}
    check = {k: (round(point[k], 3), orig[k]["auc"]) for k in sc}

    n = len(yB)
    d1, d2 = [], []
    for _ in range(2000):
        b = rng_boot.integers(0, n, n)
        if yB[b].min() == yB[b].max():
            continue
        a_sh = roc_auc_score(yB[b], sc["shared"][b])
        d1.append(roc_auc_score(yB[b], sc["second"][b]) - a_sh)
        d2.append(roc_auc_score(yB[b], sc["map1_PC1"][b]) - a_sh)
    ci = lambda x: [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]
    res = {"verifica_AUC_puntuali (nuovo, originale)": check,
           "delta1_seconda_meno_condivisa": {"stima": point["second"] - point["shared"], "IC95": ci(d1)},
           "delta2_PC1map1_meno_condivisa": {"stima": point["map1_PC1"] - point["shared"], "IC95": ci(d2)}}
    res["criterio"] = bool(res["delta1_seconda_meno_condivisa"]["IC95"][0] > 0 and
                           res["delta2_PC1map1_meno_condivisa"]["IC95"][0] > 0)

    # stabilità su 20 split diversi
    D1, D2 = [], []
    for k in range(20):
        y, s = scores_B(1000 + k)
        a = roc_auc_score(y, s["shared"])
        D1.append(roc_auc_score(y, s["second"]) - a)
        D2.append(roc_auc_score(y, s["map1_PC1"]) - a)
    res["stabilita_20_split"] = {
        "delta1": {"mediana": float(np.median(D1)), "min": float(np.min(D1)), "quota_positivi": float(np.mean(np.array(D1) > 0))},
        "delta2": {"mediana": float(np.median(D2)), "min": float(np.min(D2)), "quota_positivi": float(np.mean(np.array(D2) > 0))}}
    s = json.dumps(res, indent=2, ensure_ascii=False)
    print(s)
    open("results_auc_diff.json", "w").write(s)
