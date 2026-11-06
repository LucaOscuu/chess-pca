"""Test di trasferimento alla curva dei tassi. Criteri in PREREG_TASSI.md."""
import json

import numpy as np
import pandas as pd
import pyreadr
from scipy.linalg import subspace_angles

RNG = np.random.default_rng(20260926)
MAT = ["R_3M", "R_6M", "R_1Y", "R_2Y", "R_3Y", "R_5Y", "R_7Y", "R_10Y"]

Y = pyreadr.read_r("FedYieldCurve.rda")["FedYieldCurve"]
Y.index = pd.period_range("1982-01", periods=len(Y), freq="M")
Y = Y[MAT]
P1 = (Y.index >= pd.Period("1982-01")) & (Y.index <= pd.Period("1994-12"))
P2 = (Y.index >= pd.Period("1995-01")) & (Y.index <= pd.Period("2008-11"))
ZLB = Y.index >= pd.Period("2008-12")
PRE = P1 | P2
REC = [pd.Period("1990-07"), pd.Period("2001-03"), pd.Period("2007-12")]


def instrument(df, s):
    X = df.to_numpy(float)
    if s == "S3":
        X = np.diff(X, axis=0)
    if s == "S4":
        X = np.column_stack([X[:, 0], X[:, 1:] - X[:, [0]]])
    return X


def pca(X, corr=True):
    Xc = X - X.mean(0)
    if corr:
        Xc = Xc / Xc.std(0)
    C = Xc.T @ Xc / len(Xc)
    lam, V = np.linalg.eigh(C)
    o = np.argsort(lam)[::-1]
    return lam[o], V[:, o], Xc


def plane(X, corr=True):
    return pca(X, corr)[1][:, :2]


def angle(A, B):
    return float(np.degrees(subspace_angles(A, B)).max())


def blocks(mask, size=12):
    idx = np.flatnonzero(mask)
    return [idx[i:i + size] for i in range(0, len(idx), size)]


def split_null(mask, n=500, corr=True):
    bl = blocks(mask)
    out = []
    for _ in range(n):
        p = RNG.permutation(len(bl))
        a = np.concatenate([bl[i] for i in p[: len(bl) // 2]])
        b = np.concatenate([bl[i] for i in p[len(bl) // 2:]])
        out.append(angle(plane(Y.to_numpy(float)[np.sort(a)], corr),
                         plane(Y.to_numpy(float)[np.sort(b)], corr)))
    return np.array(out)


def contrast(maskA, maskB, n=1000):
    X = Y.to_numpy(float)
    obs = angle(plane(X[maskA]), plane(X[maskB]))
    bA, bB = blocks(maskA), blocks(maskB)
    allb = bA + bB
    null = []
    for _ in range(n):
        p = RNG.permutation(len(allb))
        a = np.concatenate([allb[i] for i in p[: len(bA)]])
        b = np.concatenate([allb[i] for i in p[len(bA):]])
        null.append(angle(plane(X[np.sort(a)]), plane(X[np.sort(b)])))
    null = np.array(null)
    return dict(angolo=obs, null_p95=float(np.percentile(null, 95)),
                p=float((np.sum(null >= obs) + 1) / (n + 1)))


out = {}

# ---------------- Y1
lam, V, _ = pca(Y[PRE].to_numpy(float))
nul = split_null(PRE)
out["Y1"] = dict(autovalori=lam.round(4).tolist(),
                 quota_varianza=(lam / lam.sum()).round(4).tolist(),
                 gap_piano=float(lam[1] - lam[2]),
                 loading_PC1=V[:, 0].round(3).tolist(), loading_PC2=V[:, 1].round(3).tolist(),
                 split_mediana=float(np.median(nul)), split_p95=float(np.percentile(nul, 95)))

# ---------------- Y2
base = Y[PRE]
ref = plane(instrument(base, "S1"))
strum = {"S1-S2 (corr vs cov)": angle(ref, plane(instrument(base, "S2"), corr=False)),
         "S1-S3 (livelli vs variazioni)": angle(ref, plane(instrument(base, "S3"))),
         "S1-S4 (livelli vs spread)": angle(ref, plane(instrument(base, "S4")))}
out["Y2"] = dict(strumento=strum, sistema_P1_vs_P2=contrast(P1, P2))
out["Y2"]["ipotesi_strumento_supera_sistema"] = bool(max(strum.values()) > out["Y2"]["sistema_P1_vs_P2"]["angolo"])

# ---------------- Y3
pre, z = Y[PRE], Y[ZLB]
var_drop = {m: float(1 - z[m].var() / pre[m].var()) for m in ["R_3M", "R_6M"]}
c_pre, c_z = float(pre["R_3M"].corr(pre["R_10Y"])), float(z["R_3M"].corr(z["R_10Y"]))
zc = contrast(PRE, ZLB)
out["Y3"] = dict(calo_varianza=var_drop, corr_3M_10Y_pre=c_pre, corr_3M_10Y_ZLB=c_z,
                 angolo_pre_vs_ZLB=zc, correlazioni_pre=pre.corr()["R_10Y"].round(3).to_dict(),
                 correlazioni_ZLB=z.corr()["R_10Y"].round(3).to_dict())
out["Y3"]["previsione_1"] = all(v > 0.9 for v in var_drop.values())
out["Y3"]["previsione_2"] = bool(c_pre > 0.8 and c_z < 0.5)
out["Y3"]["previsione_3"] = bool(zc["p"] < 0.05 and zc["angolo"] > out["Y2"]["sistema_P1_vs_P2"]["angolo"])

# ---------------- Y4
spread = (Y["R_10Y"] - Y["R_3M"]).to_numpy()
res4 = {}
for s in ["S1", "S2", "S4", "S3"]:
    X = instrument(Y, s)
    lam_s, V_s, Xc = pca(X, corr=(s != "S2"))
    sc = Xc @ V_s[:, :2]
    sp = spread[1:] if s == "S3" else spread
    if s == "S3":
        sp = np.diff(spread)
    # componente di pendenza = quella più correlata (in valore assoluto) con lo spread
    k = int(np.argmax([abs(np.corrcoef(sc[:, j], sp)[0, 1]) for j in range(2)]))
    slope = sc[:, k] * np.sign(np.corrcoef(sc[:, k], sp)[0, 1])
    inreg = slope < np.quantile(slope, 0.25)
    idx = Y.index[1:] if s == "S3" else Y.index
    pre_mask = np.zeros(len(idx), bool)
    per_rec = []
    for r in REC:
        w = (idx >= r - 12) & (idx < r)
        pre_mask |= w
        per_rec.append(float(inreg[w].mean()))
    other = float(inreg[~pre_mask].mean())
    res4[s] = dict(componente_pendenza=f"PC{k+1}", quota_prima_di_ogni_recessione=per_rec,
                   quota_altri_mesi=other, robusto=bool(all(q > other for q in per_rec)))
out["Y4"] = res4
out["Y4"]["criterio_S1_S2_S4"] = bool(all(res4[s]["robusto"] for s in ["S1", "S2", "S4"]))

print(json.dumps(out, indent=2, ensure_ascii=False))
json.dump(out, open("risultati_tassi.json", "w"), indent=2, ensure_ascii=False)
