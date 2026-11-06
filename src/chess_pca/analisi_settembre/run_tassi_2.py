"""Test rafforzati sui tassi. Criteri in PREREG_TASSI_2.md."""
import json

import numpy as np
import pandas as pd
import pyreadr
from scipy.linalg import subspace_angles

RNG = np.random.default_rng(20260927)

# ------------------------------------------------------------------ dati
CMT = pyreadr.read_r("FedYieldCurve.rda")["FedYieldCurve"]
CMT.index = pd.period_range("1982-01", periods=len(CMT), freq="M")
MK = pyreadr.read_r("Irates.rda")["Irates"]
MK.index = pd.period_range("1946-12", periods=len(MK), freq="M")
ECB = pyreadr.read_r("ECBYieldCurve.rda")["ECBYieldCurve"]
ECB.index = pd.bdate_range("2006-12-29", periods=len(ECB))
REC_MK = [pd.Period(x) for x in ["1948-11", "1953-07", "1957-08", "1960-04", "1969-12",
                                 "1973-11", "1980-01", "1981-07"]]
REC_CMT = [pd.Period(x) for x in ["1990-07", "2001-03", "2007-12"]]


def pca(X, corr=True):
    Xc = X - X.mean(0)
    if corr:
        Xc = Xc / Xc.std(0)
    lam, V = np.linalg.eigh(Xc.T @ Xc / len(Xc))
    o = np.argsort(lam)[::-1]
    return lam[o], V[:, o], Xc


def plane(X, corr=True):
    return pca(X, corr)[1][:, :2]


def ang(A, B):
    return float(np.degrees(subspace_angles(A, B)).max())


def blocks(n, size):
    return [np.arange(i, min(i + size, n)) for i in range(0, n, size)]


def split_null(X, size, n=500, corr=True):
    bl = blocks(len(X), size)
    out = []
    for _ in range(n):
        p = RNG.permutation(len(bl))
        a = np.sort(np.concatenate([bl[i] for i in p[: len(bl) // 2]]))
        b = np.sort(np.concatenate([bl[i] for i in p[len(bl) // 2:]]))
        out.append(ang(plane(X[a], corr), plane(X[b], corr)))
    return np.array(out)


def block_boot_idx(n, size):
    bl = blocks(n, size)
    pick = RNG.integers(0, len(bl), len(bl))
    return np.concatenate([bl[i] for i in pick])


def slope_score(df, short, long, corr=True):
    X = df.to_numpy(float)
    lam, V, Xc = pca(X, corr)
    sc = Xc @ V[:, :2]
    sp = (df[long] - df[short]).to_numpy()
    k = int(np.argmax([abs(np.corrcoef(sc[:, j], sp)[0, 1]) for j in range(2)]))
    return sc[:, k] * np.sign(np.corrcoef(sc[:, k], sp)[0, 1])


# ------------------------------------------------------------------ R1
def pre_windows(index, recs):
    m = np.zeros(len(index), bool)
    per = []
    for r in recs:
        w = (index >= r - 12) & (index < r)
        per.append(w)
        m |= w
    return m, per


def occ_stat(regions):
    """regions: lista di (inreg bool array, index, recs)."""
    diffs, per_event = [], []
    for inreg, idx, recs in regions:
        allpre, per = pre_windows(idx, recs)
        other = inreg[~allpre].mean()
        for w in per:
            per_event.append(float(inreg[w].mean() - other))
    return float(np.mean(per_event)), per_event


def R1(n=5000):
    def region(df, short, long, corr=True, thr_mask=None):
        s = slope_score(df, short, long, corr)
        base = s if thr_mask is None else s[thr_mask]
        return s < np.quantile(base, 0.25)

    variants = {
        "principale (livelli, correlazione)": dict(mk=MK, cmt=CMT, corr=True),
        "livelli, covarianza": dict(mk=MK, cmt=CMT, corr=False),
        "MK con scadenze come CMT": dict(mk=MK[["r3", "r6", "r12", "r36", "r60", "r120"]], cmt=CMT, corr=True),
    }
    out = {}
    for name, v in variants.items():
        mk_short = "r1" if "r1" in v["mk"].columns else "r3"
        reg = [(region(v["mk"], mk_short, "r120", v["corr"]), v["mk"].index, REC_MK),
               (region(v["cmt"], "R_3M", "R_10Y", v["corr"]), v["cmt"].index, REC_CMT)]
        obs, per = occ_stat(reg)
        null = []
        for _ in range(n):
            shifted = []
            for inreg, idx, recs in reg:
                k = RNG.integers(len(idx))
                shifted.append((np.roll(inreg, k), idx, recs))
            null.append(occ_stat(shifted)[0])
        null = np.array(null)
        out[name] = dict(differenza_media_pp=100 * obs,
                         per_evento_pp=[round(100 * x, 1) for x in per],
                         eventi_positivi=int(sum(x > 0 for x in per)),
                         p=float((np.sum(null >= obs) + 1) / (n + 1)))
    out["criterio"] = bool(out["principale (livelli, correlazione)"]["p"] < 0.05
                           and out["principale (livelli, correlazione)"]["eventi_positivi"] >= 9)
    # fuori campione nel tempo: soglia stimata su 1946-1969
    s = slope_score(MK, "r1", "r120")
    early = MK.index <= pd.Period("1969-06")
    inreg = s < np.quantile(s[early], 0.25)
    late_recs = [r for r in REC_MK if r >= pd.Period("1969-12")]
    late = MK.index >= pd.Period("1968-12")
    _, per = occ_stat([(inreg[late], MK.index[late], late_recs)])
    out["fuori_campione_1969_1981"] = dict(per_evento_pp=[round(100 * x, 1) for x in per])
    return out


# ------------------------------------------------------------------ R2
def cancorr(X, Y):
    Xc = X - X.mean(0)
    Yc = Y - Y.mean(0)
    Qx, _ = np.linalg.qr(Xc)
    Qy, _ = np.linalg.qr(Yc)
    return np.linalg.svd(Qx.T @ Qy, compute_uv=False)


def R2():
    common = MK.index.intersection(CMT.index)
    A = MK.loc[common]
    B = CMT.loc[common]
    Xa, Xb = A.to_numpy(float), B.to_numpy(float)
    _, Va, Za = pca(Xa)
    _, Vb, Zb = pca(Xb)
    Sa, Sb = Za @ Va[:, :2], Zb @ Vb[:, :2]
    obs_angle = ang(Sa, Sb)
    cc = cancorr(Za, Zb)
    ra = slope_score(A, "r1", "r120") < np.quantile(slope_score(A, "r1", "r120"), 0.25)
    rb = slope_score(B, "R_3M", "R_10Y") < np.quantile(slope_score(B, "R_3M", "R_10Y"), 0.25)
    po = np.mean(ra == rb)
    pe = ra.mean() * rb.mean() + (1 - ra.mean()) * (1 - rb.mean())
    kappa = (po - pe) / (1 - pe)
    corr_pc = [float(abs(np.corrcoef(Sa[:, j], Sb[:, j])[0, 1])) for j in range(2)]
    return dict(mesi=len(common), angolo_percorsi_PCA=obs_angle,
                corr_PC1=corr_pc[0], corr_PC2=corr_pc[1],
                correlazioni_canoniche=[round(float(x), 4) for x in cc[:4]],
                kappa_regione=float(kappa), accordo_grezzo=float(po),
                criterio=bool(cc[0] > 0.9 and cc[1] > 0.9 and kappa > 0.6))


# ------------------------------------------------------------------ R3
def instr_vs_noise(X, size, n=500):
    lev = X
    chg = np.diff(X, axis=0)
    obs = ang(plane(lev), plane(chg))
    cov_obs = ang(plane(lev), plane(lev, corr=False))
    boot = []
    for _ in range(n):
        idx = np.sort(block_boot_idx(len(X), size))
        idx = idx[idx < len(X) - 1]
        boot.append(ang(plane(lev[idx]), plane(chg[idx])))
    null = split_null(X, size, n)
    return dict(livelli_vs_variazioni=obs, IC95=[float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                correlazione_vs_covarianza=cov_obs,
                null_split_p95=float(np.percentile(null, 95)), null_split_mediana=float(np.median(null)))


def R3():
    pre = CMT.index <= pd.Period("2008-11")
    out = {"MK_1946_1991": instr_vs_noise(MK.to_numpy(float), 12),
           "CMT_1982_2008": instr_vs_noise(CMT[pre].to_numpy(float), 12)}
    out["criterio"] = bool(all(out[k]["IC95"][0] > out[k]["null_split_p95"] for k in out))
    return out


# ------------------------------------------------------------------ R4
def R4(n_half=5):
    eras = {"1947-1965": ("1947-01", "1965-12"), "1966-1979": ("1966-01", "1979-09"),
            "1979-1991": ("1979-10", "1991-02")}
    P, sl, il = [], [], []
    for e, (a, b) in eras.items():
        df = MK[(MK.index >= pd.Period(a)) & (MK.index <= pd.Period(b))].to_numpy(float)
        bl = blocks(len(df), 12)
        for _ in range(n_half):
            p = RNG.permutation(len(bl))
            idx = np.sort(np.concatenate([bl[i] for i in p[: len(bl) // 2]]))
            for ins in ("S1", "S2", "S3"):
                if ins == "S3":
                    X = np.diff(df, axis=0)[idx[idx < len(df) - 1]]
                    V = plane(X)
                elif ins == "S2":
                    V = plane(df[idx], corr=False)
                else:
                    V = plane(df[idx])
                P.append((V @ V.T).ravel())
                sl.append(e)
                il.append(ins)
    P = np.array(P)
    g = P.mean(0)
    tot = ((P - g) ** 2).sum()
    d = pd.DataFrame({"s": sl, "i": il})
    ss = lambda col: sum(((P[d[col] == v].mean(0) - g) ** 2).sum() * (d[col] == v).sum() for v in d[col].unique())
    ss_s, ss_i = ss("s"), ss("i")
    ss_cell, ss_res = 0.0, 0.0
    for _, idx in d.groupby(["s", "i"]).groups.items():
        m = P[idx].mean(0)
        ss_cell += ((m - g) ** 2).sum() * len(idx)
        ss_res += ((P[idx] - m) ** 2).sum()
    out = {k: float(v / tot) for k, v in dict(sistema=ss_s, strumento=ss_i, interazione=ss_cell - ss_s - ss_i,
                                               residuo_campionamento=ss_res).items()}
    out["ipotesi_strumento_supera_sistema"] = out["strumento"] > out["sistema"]
    return out


# ------------------------------------------------------------------ R5
def contrast_blocks(XA, XB, size, n=1000):
    obs = ang(plane(XA), plane(XB))
    bA, bB = blocks(len(XA), size), blocks(len(XB), size)
    allb = [XA[b] for b in bA] + [XB[b] for b in bB]
    null = []
    for _ in range(n):
        p = RNG.permutation(len(allb))
        A = np.vstack([allb[i] for i in p[: len(bA)]])
        B = np.vstack([allb[i] for i in p[len(bA):]])
        null.append(ang(plane(A), plane(B)))
    return dict(angolo=obs, null_p95=float(np.percentile(null, 95)),
                p=float((np.sum(np.array(null) >= obs) + 1) / (n + 1)))


def R5():
    X = ECB.to_numpy(float)
    lam, _, _ = pca(X)
    r = instr_vs_noise(X, 20)
    lehman = ECB.index < pd.Timestamp("2008-09-15")
    r["pre_vs_post_Lehman"] = contrast_blocks(X[lehman], X[~lehman], 20)
    r["quota_varianza_PC1_PC2"] = [float(lam[0] / lam.sum()), float(lam[1] / lam.sum())]
    r["criterio"] = bool(r["IC95"][0] > r["null_split_p95"])
    return r


if __name__ == "__main__":
    out = dict(R1=R1(), R2=R2(), R3=R3(), R4=R4(), R5=R5())
    s = json.dumps(out, indent=2, ensure_ascii=False)
    print(s)
    open("risultati_tassi_2.json", "w").write(s)
