"""Cavallo di Selten, secondo lancio (vedi PREREG_CAVALLO_2.md).
Stessa simulazione di horse.py (stesso modulo, stesso seme, stesso ordine delle regole);
cambiano solo le misure: PCA comune alle due regole, zona degli strangers di NSS (2025),
densita' sulla mappa nei profili umani.  Uso: python horse2.py  (dalla cartella editori/)"""
import json, sys
from pathlib import Path
import numpy as np
from scipy.stats import gaussian_kde

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import horse  # crea rng = default_rng(7) all'import, come nel primo lancio

ref = {r["rule"]: r for r in json.loads((HERE / "risultati_cavallo.json").read_text(encoding="utf-8"))}
RULES = ["impulse", "reinforcement"]

late = {}
for rule in RULES:                       # stesso ordine di horse.py: impulse, poi reinforcement
    R = horse.simulate(rule)
    L = R[R[:, 1] >= 150]
    p = L[:, 2:5]
    near_tpe = (p[:, 0] > .8) & (p[:, 1] > .8) & (p[:, 2] < .35)
    got, want = round(float(near_tpe.mean()), 4), round(ref[rule]["share_near_TPE"], 4)
    if got != want:
        sys.exit(f"RIPRODUZIONE FALLITA per {rule}: {got} contro {want}. Mi fermo.")
    late[rule] = L
print("riproduzione OK")

def expected_payoffs(p1, p2, p3):
    """payoff attesi dei tre ruoli per un profilo di probabilita' di R (albero di horse.py)."""
    a = (1 - p1) * ((1 - p3) * horse.PA[('A', 0)] + p3 * horse.PA[('A', 1)])
    b = p1 * p2 * horse.END
    c = p1 * (1 - p2) * ((1 - p3) * horse.PA[('B', 0)] + p3 * horse.PA[('B', 1)])
    return a + b + c

X_all = np.vstack([late[r][:, 2:8] for r in RULES])
mu, sd = X_all.mean(0), X_all.std(0) + 1e-9
Z = (X_all - mu) / sd
w, V = np.linalg.eigh(np.cov(Z.T)); w, V = w[::-1], V[:, ::-1]
W = V[:, :2]
def proj(X): return ((X - mu) / sd) @ W

HUMAN = {"strangers": (0.5, 0.35, 0.7), "imperfect": (0.0, 0.5, 1.0)}
hum_pts = {k: proj(np.r_[np.array(v), expected_payoffs(*v)][None, :])[0] for k, v in HUMAN.items()}

out = {"evr_comune_PC1_PC2": float(w[:2].sum() / w.sum()), "evr_PC1": float(w[0] / w.sum()), "evr_PC2": float(w[1] / w.sum()),
       "loadings_PC1": dict(zip(['p1', 'p2', 'p3', 'u1', 'u2', 'u3'], W[:, 0].round(2).tolist())),
       "loadings_PC2": dict(zip(['p1', 'p2', 'p3', 'u1', 'u2', 'u3'], W[:, 1].round(2).tolist())),
       "proiezioni_umane": {k: v.round(3).tolist() for k, v in hum_pts.items()}, "regole": {}}
rng_kde = np.random.default_rng(20261010)
for rule in RULES:
    L = late[rule]; p = L[:, 2:5]
    strangers = (np.abs(p[:, 0] - .5) <= .15) & (np.abs(p[:, 2] - .7) <= .15) & (p[:, 1] >= .2) & (p[:, 1] <= .5)
    imperfect = (p[:, 0] < .2) & (p[:, 1] > .3) & (p[:, 2] > .8)
    P = proj(L[:, 2:8])
    sub = P[rng_kde.choice(len(P), min(20000, len(P)), replace=False)]
    kde = gaussian_kde(sub.T); dens = kde(sub.T)
    out["regole"][rule] = {
        "share_near_TPE": float(((p[:, 0] > .8) & (p[:, 1] > .8) & (p[:, 2] < .35)).mean()),
        "share_zona_strangers": float(strangers.mean()),
        "share_zona_imperfetto": float(imperfect.mean()),
        "dens_pct_strangers": float((dens < kde(hum_pts["strangers"][:, None])[0]).mean()),
        "dens_pct_imperfetto": float((dens < kde(hum_pts["imperfect"][:, None])[0]).mean()),
        "media_mappa": P.mean(0).round(3).tolist(),
    }
    np.save(HERE / f"mappa_comune_{rule}.npy", P.astype(np.float32))
(HERE / "risultati_cavallo_2.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
print(json.dumps(out, indent=1))
