"""
==============================================================================
A2 — VALIDITA' DEL FEATURE MAP phi SU UN ALTRO SISTEMA
==============================================================================

A2 chiede che phi fornisca "una descrizione consistente e comparabile, con
ogni coordinata che mantiene un'interpretazione semantica stabile".
Finora e' asserita. Questo script la misura.

Una coordinata puo' fallire in due modi, entrambi diagnosticabili:

  1) COLLASSO DI VARIANZA. La feature diventa costante o quasi. Esempio
     previsto: in antichess non esiste lo scacco, quindi checks_avail e'
     identicamente 0. Una coordinata a varianza nulla non descrive piu'
     nulla, e in piu' manda in NaN np.corrcoef dentro run_t6.signature().

  2) INVERSIONE DI SEGNO. La feature conserva varianza ma cambia relazione
     con l'esito. Esempio previsto: in antichess il MATERIALE e' un
     handicap, non un vantaggio.

Questi due numeri sono il test di A2. Non sono controlli tecnici: sono il
risultato. E danno anche il meccanismo dietro la predizione fallita sul
3check — se una feature e' schiacciata contro il suo massimo, la sua
varianza si comprime e il loading crolla. Qui lo vedi in forma estrema.

USO
---
    python src\\chess_pca\\check_phi_validity.py --target antichess
    python src\\chess_pca\\check_phi_validity.py --target 3check

Costo: zero. Lancialo PRIMA di run_t6.py su qualsiasi corpus nuovo.
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
    "3check": "data/cache_3check",
    "antichess": "data/cache_antichess",
    "atomic": "data/cache_atomic",
    "kingofthehill": "data/cache_koth",
    "random": "data/cache_random",
}
STEM = "{p}_data_max2000_min10.parquet"


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


def outcome_corr(d, feats):
    """Correlazione di ogni feature con l'esito (Win=+1, Draw=0, Loss=-1)."""
    y = d["result"].map({"W": 1.0, "D": 0.0, "L": -1.0})
    keep = y.notna()
    if keep.sum() < 100:
        return pd.Series(np.nan, index=feats)
    return pd.Series(
        {f: np.corrcoef(d.loc[keep, f].values, y[keep].values)[0, 1]
         for f in feats})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, help="es. antichess")
    ap.add_argument("--reference", default="depth6")
    ap.add_argument("--dead-threshold", type=float, default=0.01,
                    help="rapporto di varianza sotto cui la feature e' morta")
    args = ap.parse_args()

    ref = load(CACHE[args.reference])
    tgt = load(CACHE[args.target])
    feats = [c for c in ref.select_dtypes(include=[np.number]).columns
             if c not in NON_FEAT]

    print("=" * 78)
    print(f"A2 — VALIDITA' DI phi:  {args.target}  vs  {args.reference}")
    print("=" * 78)
    print(f"{args.reference:12s} {ref.game_id.nunique():6d} traiettorie, "
          f"{len(ref):,} righe")
    print(f"{args.target:12s} {tgt.game_id.nunique():6d} traiettorie, "
          f"{len(tgt):,} righe")

    missing = [f for f in feats if f not in tgt.columns]
    if missing:
        print(f"\n!! Feature assenti nel target: {missing}")
        feats = [f for f in feats if f in tgt.columns]

    v_ref = ref[feats].var(ddof=0)
    v_tgt = tgt[feats].var(ddof=0)
    ratio = (v_tgt / v_ref.replace(0, np.nan))

    c_ref = outcome_corr(ref, feats)
    c_tgt = outcome_corr(tgt, feats)

    tab = pd.DataFrame({
        "var_ref": v_ref.round(3),
        "var_tgt": v_tgt.round(3),
        "rapporto": ratio.round(3),
        "corr_esito_ref": c_ref.round(3),
        "corr_esito_tgt": c_tgt.round(3),
    })
    tab["segno_invertito"] = (
        (tab.corr_esito_ref * tab.corr_esito_tgt < 0)
        & (tab.corr_esito_ref.abs() > 0.05)
        & (tab.corr_esito_tgt.abs() > 0.05)
    )
    tab = tab.sort_values("rapporto")

    print("\n" + tab.to_string())

    dead = tab[(tab.var_tgt < 1e-12) | (tab.rapporto < args.dead_threshold)]
    flipped = tab[tab.segno_invertito]

    print("\n" + "-" * 78)
    print("ESITO DEL TEST DI A2")
    print("-" * 78)

    if len(dead):
        print(f"\n  COLLASSO DI VARIANZA — {len(dead)}/{len(feats)} coordinate:")
        for f, r in dead.iterrows():
            print(f"    {f:24s} varianza {r.var_ref:8.3f} -> {r.var_tgt:8.3f}")
        print("\n  Queste coordinate non descrivono piu' nulla in questo sistema.")
        print("  np.corrcoef restituira' NaN: escludile prima di run_t6.py con")
        print("     feats = [f for f in feats if d[f].var() > 1e-12]")
        print("  e RIPORTA l'esclusione nel paper. E' una violazione misurata")
        print("  di A2, non un dettaglio implementativo.")
    else:
        print("\n  Nessun collasso di varianza: tutte le coordinate restano vive.")

    if len(flipped):
        print(f"\n  INVERSIONE DI SEGNO — {len(flipped)}/{len(feats)} coordinate:")
        for f, r in flipped.iterrows():
            print(f"    {f:24s} corr esito {r.corr_esito_ref:+.3f} -> "
                  f"{r.corr_esito_tgt:+.3f}")
        print("\n  La coordinata sopravvive ma cambia significato strategico.")
        print("  Questa e' la forma piu' interessante di violazione di A2:")
        print("  phi resta calcolabile e sembra funzionare, ma misura")
        print("  l'opposto. Un confronto geometrico fra i due sistemi e'")
        print("  ancora possibile; la lettura dei loading no.")
    else:
        print("\n  Nessuna inversione di segno rilevata.")

    n_ok = len(feats) - len(dead)
    print(f"\n  Coordinate utilizzabili per il confronto: {n_ok}/{len(feats)}")
    if n_ok < len(feats):
        print("  ATTENZIONE: run_t6.py confronta piani 2D stimati su INSIEMI")
        print("  DI FEATURE DIVERSI se escludi coordinate solo nel target.")
        print("  Escludile in ENTRAMBI i corpora, altrimenti l'angolo non")
        print("  e' definito su uno spazio comune e non significa nulla.")

    out = Path(f"data/outputs/a2_phi_{args.target}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    tab.to_csv(out)
    print(f"\nSalvato: {out}")


if __name__ == "__main__":
    main()
