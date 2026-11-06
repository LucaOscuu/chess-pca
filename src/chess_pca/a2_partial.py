"""
==============================================================================
A2 — CONTROLLO SULLE INVERSIONI:  indipendenti o propagate?
==============================================================================

check_phi_validity.py ha trovato 14 coordinate su 18 che invertono la
relazione con l'esito passando da depth6 ad antichess.

L'obiezione: quasi tutte quelle coordinate sono funzioni del materiale o
fortemente correlate con esso (conteggi pezzi, mobilita', controllo del
centro, pezzi appesi, sicurezza del re). Se il materiale inverte, tutto
cio' che gli sta a valle inverte con lui. Non sarebbero 14 inversioni
indipendenti ma UNA che si propaga.

Questo script calcola la correlazione PARZIALE fra ogni feature e l'esito
tenendo fermo il materiale:

    r(f,y | m) = [ r(f,y) - r(f,m) r(m,y) ] / sqrt[ (1-r(f,m)^2)(1-r(m,y)^2) ]

DUE ESITI POSSIBILI, entrambi utilizzabili:

  Alcune inversioni sopravvivono
      -> sono semanticamente indipendenti. A2 fallisce in piu' punti
         separati del feature map.

  Tutte collassano
      -> l'inversione e' una sola e si propaga. Il risultato diventa:
         le coordinate di phi non hanno semantica modulare, perche' sono
         tutte a valle di una quantita' sola. Per A2 e' una violazione
         piu' profonda, non piu' debole: "ogni coordinata mantiene
         un'interpretazione stabile" e' una richiesta che questo phi
         strutturalmente non puo' soddisfare.

USO
---
    python src\\chess_pca\\a2_partial.py
    python src\\chess_pca\\a2_partial.py --target 3check
==============================================================================
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

NON_FEAT = {"game_id", "idx", "ply", "move_num", "result", "outcome",
            "perspective", "opening_fen", "is_resignation", "source_game_id"}

CACHE = {
    "depth6": "data/cache_d6",
    "antichess": "data/cache_antichess",
    "3check": "data/cache_3check",
    "random": "data/cache_random",
}
STEM = "{p}_data_max2000_min10.parquet"
CONTROL = "material"


def load(cache_dir):
    frames = []
    for persp in ("white", "black"):
        p = Path(cache_dir) / STEM.format(p=persp)
        if not p.exists():
            raise SystemExit(f"Manca {p}")
        d = pd.read_parquet(p).reset_index(drop=True)
        d["perspective"] = persp
        d["game_id"] = persp + ":" + d["game_id"].astype(str)
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def corr_safe(a, b):
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def partials(d, feats):
    """Correlazione con l'esito, grezza e parziale (controllando material)."""
    y = d["result"].map({"W": 1.0, "D": 0.0, "L": -1.0})
    keep = y.notna().values
    y = y[keep].values
    m = d.loc[keep, CONTROL].values.astype(float)
    r_my = corr_safe(m, y)

    rows = {}
    for f in feats:
        x = d.loc[keep, f].values.astype(float)
        r_fy = corr_safe(x, y)
        if f == CONTROL:
            rows[f] = (r_fy, np.nan)      # non si controlla per se stesso
            continue
        r_fm = corr_safe(x, m)
        if np.isnan(r_fy) or np.isnan(r_fm) or np.isnan(r_my):
            rows[f] = (r_fy, np.nan)
            continue
        den = np.sqrt(max((1 - r_fm ** 2) * (1 - r_my ** 2), 1e-12))
        rows[f] = (r_fy, (r_fy - r_fm * r_my) / den)
    return rows, r_my


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="antichess")
    ap.add_argument("--reference", default="depth6")
    ap.add_argument("--min-r", type=float, default=0.05,
                    help="soglia sotto cui una correlazione e' troppo debole "
                         "perche' l'inversione significhi qualcosa")
    args = ap.parse_args()

    ref = load(CACHE[args.reference])
    tgt = load(CACHE[args.target])
    feats = [c for c in ref.select_dtypes(include=[np.number]).columns
             if c not in NON_FEAT and c in tgt.columns]

    r_ref, my_ref = partials(ref, feats)
    r_tgt, my_tgt = partials(tgt, feats)

    print("=" * 78)
    print(f"A2 — INVERSIONI INDIPENDENTI O PROPAGATE?   "
          f"{args.target} vs {args.reference}")
    print("=" * 78)
    print(f"\ncorr(material, esito):  {args.reference} {my_ref:+.3f}   "
          f"{args.target} {my_tgt:+.3f}")
    if my_ref * my_tgt < 0:
        print("  -> il materiale stesso inverte. E' la propagazione da testare.")

    tab = pd.DataFrame({
        "grezza_ref": {f: r_ref[f][0] for f in feats},
        "grezza_tgt": {f: r_tgt[f][0] for f in feats},
        "parziale_ref": {f: r_ref[f][1] for f in feats},
        "parziale_tgt": {f: r_tgt[f][1] for f in feats},
    }).round(3)

    def inverted(a, b):
        return (a * b < 0) & (a.abs() > args.min_r) & (b.abs() > args.min_r)

    tab["inv_grezza"] = inverted(tab.grezza_ref, tab.grezza_tgt)
    tab["inv_parziale"] = inverted(tab.parziale_ref, tab.parziale_tgt)
    tab = tab.sort_values("inv_parziale", ascending=False)

    print("\n" + tab.to_string())

    n_g = int(tab.inv_grezza.sum())
    n_p = int(tab.inv_parziale.fillna(False).sum())
    survivors = tab[tab.inv_parziale.fillna(False)].index.tolist()

    print("\n" + "-" * 78)
    print(f"Inversioni sulla correlazione GREZZA    : {n_g}/{len(feats)}")
    print(f"Inversioni sulla correlazione PARZIALE  : {n_p}/{len(feats)}")
    print("-" * 78)

    if n_p == 0:
        print("\nTUTTE le inversioni spariscono controllando per il materiale.")
        print("Affermazione da fare nel paper:")
        print("  Non 14 violazioni semantiche indipendenti, ma UNA che si")
        print("  propaga. Le coordinate di phi non hanno semantica modulare:")
        print("  sono tutte a valle di una quantita' sola, quindi un singolo")
        print("  cambio di regola ne capovolge il significato in blocco.")
        print("  Per A2 e' una violazione piu' profonda, non piu' debole.")
    elif n_p < n_g:
        print(f"\nNe sopravvivono {n_p}: {', '.join(survivors)}")
        print("Affermazione da fare nel paper:")
        print("  Parte delle inversioni e' propagazione dal materiale, parte")
        print("  no. Riporta il numero PARZIALE come conteggio di violazioni")
        print("  semantiche indipendenti, e la propagazione separatamente.")
    else:
        print("\nLe inversioni reggono al controllo: sono indipendenti.")
        print("Verifica pero' che il materiale non sia esso stesso a valle")
        print("di altro prima di dichiararlo.")

    out = Path(f"data/outputs/a2_partial_{args.target}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    tab.to_csv(out)
    print(f"\nSalvato: {out}")


if __name__ == "__main__":
    main()
