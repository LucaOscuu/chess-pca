"""Cosa contiene l'EAB dei mercati (Sez. 3.3.3, paragrafo 'What does this region contain?').
Strumento completo, regione a ricchezza appaiata (0.9-1.1) scelta da analisi_mercati_save.py (stesso seed di
analisi_mercati.py; salva la mappa della meta di training in occ_maps.pkl). Confronto dentro/fuori sulla meta di test.
Uso: dalla cartella che contiene work/mercati/work/ (panel.parquet, occ_maps.pkl)."""
import pickle, json, numpy as np, pandas as pd
W = "work/mercati/work/"
S = pickle.load(open(W + "occ_maps.pkl", "rb"))[("completo", True)]
D = pd.read_parquet(W + "panel.parquet"); RAW = D.copy(); F = S["F"]
for c in F:
    lo, hi = D[c].quantile([.01, .99]); D[c] = D[c].clip(lo, hi)
D["market"] = D.slab + ":" + D.mlab
P = ((D[F].to_numpy(float) - S["mu"]) / S["sd"] - S["zm"]) @ S["Wt"]; q = S["q"]; j = S["j"]
inreg = [P[:, 0] < q[0, 0], P[:, 0] > q[1, 0], P[:, 1] < q[0, 1], P[:, 1] > q[1, 1]][j]
eab = inreg if S["s"] > 0 else ~inreg
sel = (~D.market.isin(set(S["tr"])).to_numpy()) & D.wealth_rel.between(0.9, 1.1).to_numpy()
R = RAW[sel]; e = eab[sel]
buy = (R.buy_q > 0).to_numpy()
anyo = ((np.expm1(R.n_bid) + np.expm1(R.n_ask)) > 0).to_numpy()
out = dict(
    regione=["PC1_low", "PC1_high", "PC2_low", "PC2_high"][j], segno=float(S["s"]), gap_test_pp=round(float(S["obs"]), 2),
    n_dentro=int(e.sum()), n_fuori=int((~e).sum()),
    quota_periodi_con_acquisti=dict(dentro=round(float(buy[e].mean()), 4), fuori=round(float(buy[~e].mean()), 4)),
    log_prezzo_acquisto_vs_valore_se_acquista=dict(dentro=round(float(R.buy_dev[e & buy].mean()), 4), fuori=round(float(R.buy_dev[~e & buy].mean()), 4),
                                                   n_dentro=int((e & buy).sum()), n_fuori=int((~e & buy).sum())),
    quota_ordini_al_mercato_se_ordina=dict(dentro=round(float(R.mkt_frac[e & anyo].mean()), 4), fuori=round(float(R.mkt_frac[~e & anyo].mean()), 4)),
)
json.dump(out, open(W + "eab_profilo_mercati.json", "w"), indent=1); print(json.dumps(out, indent=1))
