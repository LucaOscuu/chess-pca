"""Analisi B del 26/09/2026. Criteri in PREREG_26_09_B.md, scritti prima del lancio."""
import json
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

sys.path.insert(0, "work")
from occupancy_grouped import FEATURES, REGIONS, fit_space, residence, select_and_test

SEED = 20260926
HIGH_LEV = ["pawns_passed_diff", "cnt_pawn", "mobility", "captures_avail"]
MAT_FAM = ["material", "cnt_pawn", "cnt_knight", "cnt_bishop", "cnt_rook", "cnt_queen",
           "pawns_passed_diff", "hanging_diff"]
INSTR = {
    "I0_tutte": FEATURES,
    "I1_senza_leva_alta": [f for f in FEATURES if f not in HIGH_LEV],
    "I2_solo_materiale": MAT_FAM,
    "I3_solo_posizionali": [f for f in FEATURES if f not in MAT_FAM],
}


def load(name):
    df = pd.read_parquet(f"work/{name}.parquet")
    df["material_cp"] = df["material"]
    return df.sort_values(["game_id", "perspective", "ply"]).reset_index(drop=True)


def window(df, w=20):
    df = df.copy()
    df["k"] = df.groupby(["game_id", "perspective"]).cumcount()
    df = df[df["k"] < w]
    full = df.groupby(["game_id", "perspective"]).size().unstack()
    keep = full[(full >= w).all(axis=1)].index
    return df[df["game_id"].isin(keep)].reset_index(drop=True)


# ------------------------------------------------------------------ B1
def split(df, seed):
    rng = np.random.default_rng(seed)
    games = df["game_id"].unique()
    return set(rng.choice(games, size=len(games) // 2, replace=False))


def run_instr(df, feats, train):
    feats = [f for f in feats if df[f].std() > 0]
    mu, sd, W, q = fit_space(df[df["game_id"].isin(train)], feats)
    R = residence(df, mu, sd, W, q, feats, True, 5)
    wr = df[df["perspective"] == "W"].groupby("game_id")["result"].first()
    return R, wr


def B1(n_splits=20, perms=1000):
    df = load("full")
    out = {}
    oriented = {}
    for name, feats in INSTR.items():
        gaps = {"W": [], "B": []}
        for k in range(n_splits):
            train = split(df, SEED + k)
            R, wr = run_instr(df, feats, train)
            o = select_and_test(R, wr, train)
            for p in ("W", "B"):
                gaps[p].append(o[p]["signed_gap_test_pp"])
            if k == 0:
                # placebo per partita sulla prima divisione
                rng = np.random.default_rng(SEED + 99)
                b = {"W": 0, "B": 0}
                for _ in range(perms):
                    perm = pd.Series(rng.permutation(wr.to_numpy()), index=wr.index)
                    op = select_and_test(R, perm, train)
                    for p in ("W", "B"):
                        b[p] += op[p]["signed_gap_test_pp"] >= o[p]["signed_gap_test_pp"]
                pvals = {p: (b[p] + 1) / (perms + 1) for p in ("W", "B")}
                # residence share orientata (alto = vittoria) sul test, per l'accordo
                is_test = ~R["game_id"].isin(train)
                s = pd.Series(0.0, index=R.index)
                for p in ("W", "B"):
                    m = (R["perspective"] == p) & is_test
                    reg = o[p]["region"]
                    s[m] = R.loc[m, reg] * o[p]["sign_train"]
                oriented[name] = s[is_test].set_axis(
                    R.loc[is_test, "game_id"].astype(str) + R.loc[is_test, "perspective"])
                regions = {p: o[p]["region"] for p in ("W", "B")}
        res = {}
        for p in ("W", "B"):
            v = np.array(gaps[p])
            res[p] = dict(mediana=float(np.median(v)), min=float(v.min()), max=float(v.max()),
                          quota_sopra_1_6=float(np.mean(v > 1.6)), placebo_p=float(pvals[p]),
                          regione_split0=regions[p])
        res["regge"] = bool(all(res[p]["quota_sopra_1_6"] >= 0.9 and res[p]["placebo_p"] < 0.05
                                for p in ("W", "B")))
        res["d"] = len([f for f in feats if df[f].std() > 0])
        out[name] = res
    ref = oriented["I0_tutte"]
    for name in INSTR:
        if name == "I0_tutte":
            continue
        a, b = ref.align(oriented[name], join="inner")
        out[name]["spearman_con_I0"] = float(spearmanr(a, b).correlation)
    n_ok = sum(out[n]["regge"] for n in list(INSTR)[1:])
    out["affermazione_3"] = "SOSTENUTA" if n_ok == 3 else ("PARZIALE" if n_ok == 2 else "NON SOSTENUTA")
    return out


# ------------------------------------------------------------------ B2
def proj(X):
    Z = (X - X.mean(0)) / np.where(X.std(0) > 0, X.std(0), 1)
    lam, V = np.linalg.eigh(np.corrcoef(Z.T))
    V = V[:, np.argsort(lam)[::-1][:2]]
    return (V @ V.T).ravel()


def transform(X, kind, j, rng):
    if kind == "orig":
        return X
    if kind == "rank":
        return np.column_stack([rankdata(X[:, c]) for c in range(X.shape[1])])
    if kind == "slog":
        return np.sign(X) * np.log1p(np.abs(X))
    if kind == "rand":
        Y = X.copy()
        Y[:, j] = rng.permutation(Y[:, j])
        return Y


def decompose(P, sys_lab, ins_lab):
    P = np.asarray(P)
    g = P.mean(0)
    tot = ((P - g) ** 2).sum()
    df = pd.DataFrame({"s": sys_lab, "i": ins_lab})
    ss_s = sum(((P[df.s == s].mean(0) - g) ** 2).sum() * (df.s == s).sum() for s in df.s.unique())
    ss_i = sum(((P[df.i == i].mean(0) - g) ** 2).sum() * (df.i == i).sum() for i in df.i.unique())
    ss_cell = 0.0
    ss_res = 0.0
    for (s, i), idx in df.groupby(["s", "i"]).groups.items():
        m = P[idx].mean(0)
        ss_cell += ((m - g) ** 2).sum() * len(idx)
        ss_res += ((P[idx] - m) ** 2).sum()
    ss_int = ss_cell - ss_s - ss_i
    return {k: float(v / tot) for k, v in
            dict(sistema=ss_s, strumento=ss_i, interazione=ss_int, residuo_campionamento=ss_res).items()}


def B2(n_half=5):
    rng = np.random.default_rng(SEED)
    systems = ["full", "depth1", "depth3", "depth6", "random", "3check"]
    data = {}
    for s in systems:
        d = window(load(s))
        data[s] = d
    feats = [f for f in FEATURES if all(data[s][f].std() > 0 for s in systems)]
    out = {"d": len(feats)}
    for design, instruments in {
        "A_randomizzazione": [("orig", None)] + [("rand", j) for j in range(len(feats))],
        "B_trasformazioni_monotone": [("orig", None), ("rank", None), ("slog", None)],
    }.items():
        P, sl, il = [], [], []
        for s in systems:
            d = data[s]
            games = d["game_id"].unique()
            for h in range(n_half):
                half = rng.choice(games, size=len(games) // 2, replace=False)
                X = d[d["game_id"].isin(half)][feats].to_numpy(float)
                for kind, j in instruments:
                    P.append(proj(transform(X, kind, j, rng)))
                    sl.append(s)
                    il.append(f"{kind}{'' if j is None else '_' + feats[j]}")
        out[design] = decompose(P, sl, il)
    out["A_ipotesi_strumento_supera_sistema"] = out["A_randomizzazione"]["strumento"] > out["A_randomizzazione"]["sistema"]
    out["B_ipotesi_strumento_sotto_sistema"] = out["B_trasformazioni_monotone"]["strumento"] < out["B_trasformazioni_monotone"]["sistema"]
    return out


if __name__ == "__main__":
    what = sys.argv[1]
    out = B1() if what == "B1" else B2()
    print(json.dumps(out, indent=2, ensure_ascii=False))
    json.dump(out, open(f"work/results/{what}_26_09.json", "w"), indent=2, ensure_ascii=False)
