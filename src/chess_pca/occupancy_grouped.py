"""
occupancy_grouped.py — controllo dell'occupazione con placebo raggruppato per partita.

Scopo: verificare se il risultato di occupazione (REPORT_FINALE_v4, III.5) regge
quando il placebo permuta le PARTITE e non le traiettorie.

CRITERI FISSATI PRIMA DEL LANCIO (non modificarli dopo aver visto i numeri):
  Primario  — corpus full, materiale appaiato (|material| <= 50 cp):
              REGGE se p_placebo_raggruppato <= 0,05 e il gap sul test ha lo
              stesso segno selezionato sul training.
  Secondario — stesso test su depth6 e random, a finestre di 20 ply:
              se il gap a materiale appaiato resta in ENTRAMBI, l'occupazione
              riflette il gioco (obiettivo) e non la qualità del giocatore.

Procedura (nessun esito usato per costruire lo spazio o le regioni):
  1. Le partite si dividono a caso in metà training e metà test (per partita).
  2. Standardizzazione e PCA stimate sul training, senza esiti.
  3. Regioni: per PC1 e PC2, coda bassa (< quartile 25 del training) e coda
     alta (> quartile 75 del training). Quattro regioni in tutto.
  4. Residence share per traiettoria (partita x prospettiva) = frazione di ply
     nella regione.
  5. Sul training, per ogni prospettiva, si sceglie la regione con il gap
     vittorie-sconfitte più grande in valore assoluto. Unico uso degli esiti.
  6. Sul test si misura quel gap (punti percentuali) e il rango biseriale.
  7. Placebo: si permutano gli esiti fra PARTITE (il risultato del Bianco;
     quello del Nero è speculare) e si ripetono 5 e 6. p = (b+1)/(m+1).
     Le regioni del punto 3 non dipendono dagli esiti e restano fisse.

Formato dei dati: una riga per ply e prospettiva, con colonne
  game_id, perspective ('W'/'B'), ply, result (dal punto di vista della
  prospettiva: 1 vittoria, 0 patta, -1 sconfitta), material (centipawn, dal
  punto di vista della prospettiva) e le 18 coordinate del feature map.
Se i nomi delle colonne sono diversi, usare le opzioni --col-*.

Uso:
  python occupancy_grouped.py corpus.parquet --label full
  python occupancy_grouped.py depth6.parquet --label depth6 --window 20
  python occupancy_grouped.py random.parquet --label random --window 20
"""
import argparse
import json

import numpy as np
import pandas as pd

FEATURES = [
    "material", "mobility", "captures_avail", "checks_avail", "attack_overlap",
    "hanging_diff", "center_core", "center_ext", "king_safety_diff",
    "pawns_doubled_diff", "pawns_isolated_diff", "pawns_passed_diff",
    "cnt_pawn", "cnt_knight", "cnt_bishop", "cnt_rook", "cnt_queen",
    "delta_mobility",
]
REGIONS = ["PC1_low", "PC1_high", "PC2_low", "PC2_high"]


def load(path, a):
    df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
    if a.col_material not in df.columns:
        raise SystemExit(f"Colonna del materiale mancante: {a.col_material}")
    df["material_cp"] = df[a.col_material]
    df = df.rename(columns={a.col_game: "game_id", a.col_persp: "perspective",
                            a.col_ply: "ply", a.col_result: "result"})
    missing = [c for c in FEATURES + ["game_id", "perspective", "ply", "result"]
               if c not in df.columns]
    if missing:
        raise SystemExit(f"Colonne mancanti: {missing}")
    df = df.sort_values(["game_id", "perspective", "ply"])
    if a.window:
        df["k"] = df.groupby(["game_id", "perspective"]).cumcount()
        df = df[df["k"] < a.window]
        # tieni solo le partite che arrivano alla finestra da entrambe le prospettive
        full = df.groupby(["game_id", "perspective"]).size().unstack()
        keep = full[(full >= a.window).all(axis=1)].index
        df = df[df["game_id"].isin(keep)]
    return df.reset_index(drop=True)


def fit_space(train_rows, feats):
    X = train_rows[feats].to_numpy(float)
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1.0
    Z = (X - mu) / sd
    _, _, vt = np.linalg.svd(Z - Z.mean(0), full_matrices=False)
    W = vt[:2].T
    scores = Z @ W
    q = np.percentile(scores, [25, 75], axis=0)  # righe: q25, q75; colonne: PC1, PC2
    return mu, sd, W, q


def residence(df, mu, sd, W, q, feats, matched, min_plies):
    rows = df if not matched else df[df["material_cp"].abs() <= 50]
    Z = (rows[feats].to_numpy(float) - mu) / sd
    s = Z @ W
    ind = np.column_stack([s[:, 0] < q[0, 0], s[:, 0] > q[1, 0],
                           s[:, 1] < q[0, 1], s[:, 1] > q[1, 1]]).astype(float)
    tmp = pd.DataFrame(ind, columns=REGIONS)
    tmp["game_id"] = rows["game_id"].to_numpy()
    tmp["perspective"] = rows["perspective"].to_numpy()
    g = tmp.groupby(["game_id", "perspective"])
    R = g[REGIONS].mean()
    R = R[g.size() >= min_plies]
    return R.reset_index()


def rank_biserial(x_win, x_loss):
    from scipy.stats import mannwhitneyu
    u = mannwhitneyu(x_win, x_loss, alternative="two-sided").statistic
    return 2 * u / (len(x_win) * len(x_loss)) - 1


def select_and_test(R, white_res, train_games):
    """white_res: Series game_id -> risultato del Bianco (1/0/-1)."""
    res = R["game_id"].map(white_res).to_numpy()
    res = np.where(R["perspective"].to_numpy() == "W", res, -res)
    is_train = R["game_id"].isin(train_games).to_numpy()
    out = {}
    for p in ("W", "B"):
        m = R["perspective"].to_numpy() == p
        tr_w, tr_l = m & is_train & (res == 1), m & is_train & (res == -1)
        te_w, te_l = m & ~is_train & (res == 1), m & ~is_train & (res == -1)
        if min(tr_w.sum(), tr_l.sum(), te_w.sum(), te_l.sum()) < 10:
            out[p] = None
            continue
        V = R[REGIONS].to_numpy()
        gaps_tr = V[tr_w].mean(0) - V[tr_l].mean(0)
        j = int(np.argmax(np.abs(gaps_tr)))
        sign = np.sign(gaps_tr[j])
        gap_te = V[te_w, j].mean() - V[te_l, j].mean()
        out[p] = dict(region=REGIONS[j], sign_train=float(sign),
                      gap_test_pp=100 * float(gap_te),
                      signed_gap_test_pp=100 * float(sign * gap_te),
                      rb_test=float(rank_biserial(V[te_w, j], V[te_l, j])),
                      n_test_win=int(te_w.sum()), n_test_loss=int(te_l.sum()))
    return out


def run(df, a, matched):
    rng = np.random.default_rng(a.seed)
    games = df["game_id"].unique()
    train_games = set(rng.choice(games, size=len(games) // 2, replace=False))
    feats = [f for f in FEATURES if df[f].std() > 0]
    mu, sd, W, q = fit_space(df[df["game_id"].isin(train_games)], feats)
    R = residence(df, mu, sd, W, q, feats, matched, a.min_plies)

    wr = (df[df["perspective"] == "W"].groupby("game_id")["result"].first())
    obs = select_and_test(R, wr, train_games)

    # placebo raggruppato: permuta il risultato del Bianco fra partite
    stat_obs = {p: (v["signed_gap_test_pp"] if v else None) for p, v in obs.items()}
    exceed = {p: 0 for p in stat_obs}
    null = {p: [] for p in stat_obs}
    vals, idx = wr.to_numpy(), wr.index
    for _ in range(a.perms):
        perm = pd.Series(rng.permutation(vals), index=idx)
        o = select_and_test(R, perm, train_games)
        for p in stat_obs:
            if stat_obs[p] is None or o[p] is None:
                continue
            s = o[p]["signed_gap_test_pp"]
            null[p].append(s)
            exceed[p] += s >= stat_obs[p]
    for p in obs:
        if obs[p]:
            obs[p]["placebo_p"] = (exceed[p] + 1) / (a.perms + 1)
            obs[p]["placebo_p95_pp"] = float(np.percentile(null[p], 95))
    return obs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--label", default="corpus")
    ap.add_argument("--window", type=int, default=0, help="0 = traiettorie intere")
    ap.add_argument("--perms", type=int, default=1000)
    ap.add_argument("--min-plies", type=int, default=5)
    ap.add_argument("--seed", type=int, default=20260925)
    ap.add_argument("--col-game", default="game_id")
    ap.add_argument("--col-persp", default="perspective")
    ap.add_argument("--col-ply", default="ply")
    ap.add_argument("--col-result", default="result")
    ap.add_argument("--col-material", default="material")
    a = ap.parse_args()

    df = load(a.path, a)
    out = {"label": a.label, "window": a.window, "perms": a.perms,
           "n_games": int(df["game_id"].nunique()),
           "all_plies": run(df, a, matched=False),
           "material_matched": run(df, a, matched=True)}
    print(json.dumps(out, indent=2))
    with open(f"occupancy_grouped_{a.label}.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
