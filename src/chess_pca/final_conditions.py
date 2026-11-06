"""
==============================================================================
ULTIME TRE ANALISI
==============================================================================

Tutto su cache. Nessun run motore. ~5 minuti.

TEST 1 — A4 sa accorgersi della propria assenza?
    Nel corpus casuale l'87% sono patte: la partizione vinta/persa esiste
    nominalmente ma non porta informazione. Se l'occupazione differenziata
    per esito sopravvive anche li', A4 non discrimina nulla e va tolta.

TEST 2 — A3 e A4 vanno sciolte
    Il mover casuale viola A3 (nessuna organizzazione strategica) E A4
    (esiti quasi degeneri) contemporaneamente. Il controllo negativo
    confonde due condizioni. Restringendosi alle sole partite decisive del
    corpus casuale, A4 torna soddisfatta mentre A3 resta violata. Se la
    firma spettrale di A3 sopravvive, non era guidata dalle patte.

TEST 3 — A6, identificabilita' degli assi dentro il piano
    A5 dice che il PIANO e' recuperabile da sottocampioni indipendenti.
    Non dice nulla sulle direzioni al suo interno. Con lambda1-lambda2 =
    0,182 le due cose si separano: il piano puo' essere stabile mentre PC1
    ruota liberamente dentro di esso. E' il problema della rotazione di
    Thurstone, misurato.

    PREVISIONE CHE LA METTE A RISCHIO: se A6 dipende dall'eigengap, il
    corpus casuale (gap 0,52) deve avere assi PIU' stabili di depth6
    (gap 0,22). Se non li ha, A6 non e' quello che pensiamo e va ripensata.

USO
---
    python src\\chess_pca\\final_conditions.py
==============================================================================
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import subspace_angles
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

RNG = np.random.default_rng(20260728)

FIXED_LEN = 20
N_REPS = 50
N_PERM = 200

NON_FEAT = {"game_id", "idx", "ply", "move_num", "result", "outcome",
            "perspective", "opening_fen", "is_resignation", "source_game_id"}

CORPORA = {
    "depth6": ("data/cache_d6/white_data_max2000_min10.parquet",
               "data/cache_d6/black_data_max2000_min10.parquet"),
    "random": ("data/cache_random/white_data_max2000_min10.parquet",
               "data/cache_random/black_data_max2000_min10.parquet"),
}


def load(wp, bp):
    frames = []
    for path, persp in [(wp, "White"), (bp, "Black")]:
        p = Path(path)
        if not p.exists():
            raise SystemExit(f"Manca {p}")
        d = pd.read_parquet(p).reset_index(drop=True)
        d["perspective"] = persp
        d["game_id"] = persp + ":" + d["game_id"].astype(str)
        frames.append(d)
    d = pd.concat(frames, ignore_index=True)
    keep = d.groupby("game_id").size()
    d = d[d.game_id.isin(keep[keep >= FIXED_LEN].index)]
    return d.groupby("game_id", sort=False).head(FIXED_LEN)


def fit(d, feats):
    """Ritorna scaler, pca(2), autovalori della matrice di correlazione."""
    X = StandardScaler().fit_transform(d[feats].values)
    C = np.corrcoef(X.T)
    lam = np.sort(np.linalg.eigvalsh(C))[::-1]
    p = PCA(n_components=2).fit(X)
    return p, X, lam


def axis_angle(v, w):
    """Angolo fra due direzioni, insensibile al segno (sono rette, non frecce)."""
    c = np.clip(abs(float(np.dot(v, w))) / (np.linalg.norm(v) * np.linalg.norm(w)),
                0.0, 1.0)
    return float(np.degrees(np.arccos(c)))


# =========================================================================
# TEST 1 + 2 — occupazione per esito, e scioglimento A3/A4
# =========================================================================

def occupancy_gap(d, feats, q=0.25):
    """
    Frazione media di ply che una traiettoria passa sotto la soglia,
    confrontata fra traiettorie vinte e perse. Ritorna il gap in punti
    percentuali, piu' il conteggio di traiettorie decisive.
    """
    lab = d[d.result.isin(["W", "L"])]
    if lab.game_id.nunique() < 40:
        return None
    p, X, _ = fit(d, feats)                       # geometria dal corpus intero
    z = p.transform(StandardScaler().fit_transform(d[feats].values))[:, 0]
    thr = np.quantile(z, q)

    tmp = pd.DataFrame({"g": d.game_id.values, "r": d.result.values,
                        "below": (z < thr).astype(float)})
    per_traj = tmp.groupby(["g", "r"], sort=False)["below"].mean().reset_index()
    per_traj = per_traj[per_traj.r.isin(["W", "L"])]
    if per_traj.r.nunique() < 2:
        return None

    w = per_traj[per_traj.r == "W"].below.values
    l = per_traj[per_traj.r == "L"].below.values
    obs = 100.0 * (w.mean() - l.mean())

    # placebo: permuta le etichette di esito fra traiettorie
    pool = np.concatenate([w, l])
    nw = len(w)
    null = np.empty(N_PERM)
    for i in range(N_PERM):
        s = RNG.permutation(pool)
        null[i] = 100.0 * (s[:nw].mean() - s[nw:].mean())
    p_val = float((np.abs(null) >= abs(obs)).mean())

    return {"gap_pp": obs, "p_perm": p_val,
            "n_W": len(w), "n_L": len(l),
            "null_p95": float(np.percentile(np.abs(null), 95))}


def run_occupancy(data, feats):
    print("=" * 74)
    print("TEST 1 — A4 sa accorgersi della propria assenza?")
    print("=" * 74)

    for name, d in data.items():
        dec = d[d.result.isin(["W", "L"])].game_id.nunique()
        tot = d.game_id.nunique()
        print(f"\n[{name}] {tot} traiettorie, {dec} decisive "
              f"({100*dec/tot:.1f}%)")
        r = occupancy_gap(d, feats)
        if r is None:
            print("   troppo poche traiettorie decisive: A4 VIOLATA per")
            print("   degenerazione della partizione (non misurabile)")
            continue
        print(f"   gap di occupazione W-L : {r['gap_pp']:+.2f} pp")
        print(f"   placebo |gap| p95      : {r['null_p95']:.2f} pp")
        print(f"   p permutazionale       : {r['p_perm']:.3f}  "
              f"({r['n_W']} W / {r['n_L']} L)")
        verdict = "A4 SODDISFATTA" if r["p_perm"] < 0.05 else "A4 VIOLATA"
        print(f"   -> {verdict}")

    print("\n  Lettura: se il gap sopravvive nel corpus casuale, A4 non")
    print("  discrimina e va tolta dalla lista. Se svanisce, A4 sa")
    print("  riconoscere una partizione degenere ed e' una condizione vera.")


def run_disentangle(data, feats):
    print("\n" + "=" * 74)
    print("TEST 2 — A3 e A4 sciolte (solo partite decisive del casuale)")
    print("=" * 74)

    rnd = data["random"]
    dec_ids = rnd[rnd.result.isin(["W", "L"])].game_id.unique()
    rnd_dec = rnd[rnd.game_id.isin(dec_ids)]
    n_dec = rnd_dec.game_id.nunique()
    print(f"\n  sottoinsieme decisivo del casuale: {n_dec} traiettorie")
    if n_dec < 100:
        print("  troppo piccolo per un confronto spettrale utile.")
        return

    # depth6 sottocampionato allo stesso n, cosi' il bordo MP e' identico
    d6 = data["depth6"]
    ids6 = RNG.choice(d6.game_id.unique(), n_dec, replace=False)
    d6_m = d6[d6.game_id.isin(ids6)]

    print(f"  depth6 sottocampionato allo stesso n: {d6_m.game_id.nunique()}")
    print()
    for name, d in [("random-decisive", rnd_dec), ("depth6-matched", d6_m)]:
        _, X, lam = fit(d, feats)
        n_eff = d.game_id.nunique()
        edge = (1 + np.sqrt(len(feats) / n_eff)) ** 2
        n_sig = int((lam > edge).sum())
        pr = (lam.sum() ** 2) / (lam ** 2).sum()
        print(f"  {name:18s} n_sig {n_sig}/{len(feats)}   "
              f"PR {pr:5.2f}   gap12 {lam[0]-lam[1]:.3f}   "
              f"(bordo MP {edge:.3f})")

    print("\n  Lettura: se il casuale-decisivo conserva la firma spettrale")
    print("  del casuale completo (gap12 alto, n_sig basso), allora il")
    print("  risultato su A3 non era guidato dall'87% di patte, e le due")
    print("  condizioni sono separabili. Se collassa, il tuo controllo")
    print("  negativo stava misurando A4 e non A3.")


# =========================================================================
# TEST 3 — A6, identificabilita' degli assi
# =========================================================================

def run_identifiability(data, feats):
    print("\n" + "=" * 74)
    print("TEST 3 — A6: il piano e' stabile, gli assi lo sono?")
    print("=" * 74)
    print("\n  Split-half. Per ogni ripetizione misura:")
    print("    - angolo massimo fra i due PIANI      (questo e' A5)")
    print("    - angolo fra i due PC1 presi da soli  (questo sarebbe A6)")
    print()

    rows = []
    for name, d in data.items():
        ids = d.game_id.unique()
        n_side = len(ids) // 2
        _, _, lam_full = fit(d, feats)
        gap = lam_full[0] - lam_full[1]

        plane, ax1, ax2 = [], [], []
        for _ in range(N_REPS):
            p = RNG.permutation(ids)
            pa, _, _ = fit(d[d.game_id.isin(p[:n_side])], feats)
            pb, _, _ = fit(d[d.game_id.isin(p[n_side:2 * n_side])], feats)
            Va, Vb = pa.components_.T, pb.components_.T
            plane.append(np.degrees(subspace_angles(Va, Vb)).max())
            ax1.append(axis_angle(Va[:, 0], Vb[:, 0]))
            ax2.append(axis_angle(Va[:, 1], Vb[:, 1]))

        rows.append({
            "corpus": name, "n_lato": n_side, "gap12": round(gap, 3),
            "piano_med": round(float(np.median(plane)), 1),
            "PC1_med": round(float(np.median(ax1)), 1),
            "PC2_med": round(float(np.median(ax2)), 1),
            "PC1_p95": round(float(np.percentile(ax1, 95)), 1),
        })

    tab = pd.DataFrame(rows)
    print(tab.to_string(index=False))

    print("\n  " + "-" * 70)
    for r in rows:
        ratio = r["PC1_med"] / max(r["piano_med"], 1e-9)
        print(f"  {r['corpus']:8s} gap12 {r['gap12']:.3f}   "
              f"PC1 ruota {ratio:.1f}x piu' del piano")

    if len(rows) == 2:
        a, b = sorted(rows, key=lambda x: x["gap12"])
        print(f"\n  Previsione: eigengap piu' grande -> assi piu' stabili.")
        print(f"    {a['corpus']} (gap {a['gap12']:.3f}): PC1 {a['PC1_med']:.1f} gradi")
        print(f"    {b['corpus']} (gap {b['gap12']:.3f}): PC1 {b['PC1_med']:.1f} gradi")
        ok = b["PC1_med"] < a["PC1_med"]
        print(f"    -> previsione {'CONFERMATA' if ok else 'SMENTITA'}")
        if ok:
            print("       A6 e' una condizione distinta da A5 e dipende")
            print("       dall'eigengap. E' il problema della rotazione di")
            print("       Thurstone, misurato sui tuoi dati.")
        else:
            print("       A6 non dipende dall'eigengap come previsto.")
            print("       Non aggiungerla finche' non sai da cosa dipende.")

    out = Path("data/outputs/a6_identifiability.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    tab.to_csv(out, index=False)
    print(f"\n  Salvato: {out}")


def main():
    data = {k: load(*v) for k, v in CORPORA.items()}
    feats = [c for c in data["depth6"].select_dtypes(include=[np.number]).columns
             if c not in NON_FEAT]
    print(f"Feature: {len(feats)}   finestre: {FIXED_LEN} ply\n")

    run_occupancy(data, feats)
    run_disentangle(data, feats)
    run_identifiability(data, feats)


if __name__ == "__main__":
    main()
