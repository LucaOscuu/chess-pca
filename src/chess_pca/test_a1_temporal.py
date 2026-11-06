"""
==============================================================================
A1 — ORDINE TEMPORALE:  cosa fa davvero, e cosa non fa
==============================================================================

PREMESSA CHE CAMBIA IL TEST
---------------------------
La PCA e' fittata su tutte le righe insieme. Una permutazione di righe NON
cambia la matrice di covarianza. Quindi mescolare i ply e rifittare la PCA
darebbe 0.0 gradi per identita' algebrica, non per un risultato.

L'unico canale attraverso cui l'ordine entra nella geometria e'
delta_mobility, 1 feature su 18. Quindi il test corretto e':

  A) mescola i ply DENTRO ogni traiettoria e RICALCOLA delta_mobility
     dalla sequenza mescolata -> misura quanto ruota la geometria;

  B) proietta i dati mescolati sullo STESSO asse di riferimento e misura
     la residenza (la statistica di T9) -> misura quanto crolla la
     persistenza.

RISULTATO ATTESO, e va scritto cosi' nel paper:
  A1 non serve a RECUPERARE la geometria (che e' invariante all'ordine);
  serve a rendere interpretabile la PERSISTENZA. Sono due ruoli diversi
  e finora nel paper erano confusi in uno.

Questo e' anche il punto in cui l'affermazione dell'Introduzione — "la
letteratura tratta le posizioni come istantanee scollegate, noi no" — va
riscritta onestamente: vale per le misure di residenza, non per la
geometria.

USO
---
    python src\\chess_pca\\test_a1_temporal.py

Costo: zero run motore. ~2-4 minuti sul corpus full.
==============================================================================
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import subspace_angles
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

RNG = np.random.default_rng(20260727)

WHITE = "data/cache/white_data_max10000_min10.parquet"
BLACK = "data/cache/black_data_max10000_min10.parquet"

THRESH = -1.0        # identico a t9_fast.py
N_NULL = 50          # split-half per il null angolare
N_SHUF = 20          # ripetizioni del mescolamento

NON_FEAT = {"game_id", "idx", "ply", "move_num", "result", "outcome",
            "perspective", "opening_fen", "is_resignation", "source_game_id", "base_game"}


def load():
    frames = []
    for path, persp in [(WHITE, "White"), (BLACK, "Black")]:
        p = Path(path)
        if not p.exists():
            raise SystemExit(f"Manca {p}")
        d = pd.read_parquet(p).reset_index(drop=True)
        d["base_game"] = d["game_id"].astype(str)
        d["perspective"] = persp
        d["game_id"] = persp + ":" + d["game_id"].astype(str)
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def shuffle_within(df, feats):
    """Permuta i ply dentro ogni traiettoria e RICALCOLA delta_mobility."""
    codes = pd.factorize(df.game_id)[0]
    key = RNG.random(len(df))
    idx = np.lexsort((key, codes))          # ordina per gruppo, casuale dentro
    out = df.iloc[idx].reset_index(drop=True)
    if "delta_mobility" in feats:
        out["delta_mobility"] = (
            out.groupby("game_id", sort=False)["mobility"].diff().fillna(0.0)
        )
    return out


def mean_run(mask):
    """Lunghezza media dei tratti consecutivi sotto soglia. Da t9_fast.py."""
    if not mask.any():
        return 0.0
    d = np.diff(np.concatenate(([0], mask.view(np.int8), [0])))
    return float((np.flatnonzero(d == -1) - np.flatnonzero(d == 1)).mean())


def residence(df, z):
    """Residenza media per traiettoria, su un asse z gia' proiettato."""
    tmp = pd.DataFrame({"g": df.game_id.values, "z": z})
    return float(np.mean([mean_run(v.values < THRESH)
                          for _, v in tmp.groupby("g", sort=False)["z"]]))


def main():
    df = load()
    feats = [c for c in df.select_dtypes(include=[np.number]).columns
             if c not in NON_FEAT]
    print("=" * 74)
    print("A1 — ORDINE TEMPORALE")
    print("=" * 74)
    print(f"{len(df):,} righe, {df.game_id.nunique():,} traiettorie, "
          f"{len(feats)} feature")

    # riferimento: scaler + PCA fittati UNA volta sui dati originali
    scaler = StandardScaler().fit(df[feats].values)
    Xref = scaler.transform(df[feats].values)
    pca = PCA(n_components=2).fit(Xref)
    Vref = pca.components_.T
    zref = pca.transform(Xref)[:, 0]

    # ---------------------------------------------------------------- null
    print("\n" + "-" * 74)
    print("NULL ANGOLARE  (split-half entro il corpus, traiettorie intere)")
    print("-" * 74)
    games = df.base_game.unique()
    n_side = len(games) // 2
    null = []
    for _ in range(N_NULL):
        p = RNG.permutation(games)
        Va = PCA(2).fit(StandardScaler().fit_transform(
            df[df.base_game.isin(p[:n_side])][feats].values)).components_.T
        Vb = PCA(2).fit(StandardScaler().fit_transform(
            df[df.base_game.isin(p[n_side:2 * n_side])][feats].values)).components_.T
        null.append(np.degrees(subspace_angles(Va, Vb)).max())
    null = np.array(null)
    print(f"  n={n_side}/lato : mediana {np.median(null):.2f}  "
          f"p95 {np.percentile(null, 95):.2f}  max {null.max():.2f} gradi")

    # ------------------------------------------------- A) geometria
    print("\n" + "-" * 74)
    print("A) GEOMETRIA sotto mescolamento dei ply")
    print("-" * 74)

    # controllo di identita': permutare senza ricalcolare delta_mobility
    perm = df.iloc[RNG.permutation(len(df))].reset_index(drop=True)
    Vperm = PCA(2).fit(StandardScaler().fit_transform(
        perm[feats].values)).components_.T
    print(f"  permutazione pura (delta_mobility NON ricalcolata): "
          f"{np.degrees(subspace_angles(Vref, Vperm)).max():.3f} gradi")
    print("     ^ e' zero per identita' algebrica. Non e' un esperimento:")
    print("       serve solo a rendere esplicito perche' il test ingenuo e' vuoto.")

    angs = []
    for _ in range(N_SHUF):
        sh = shuffle_within(df, feats)
        Vsh = PCA(2).fit(StandardScaler().fit_transform(
            sh[feats].values)).components_.T
        angs.append(np.degrees(subspace_angles(Vref, Vsh)).max())
    angs = np.array(angs)
    print(f"\n  mescolamento entro traiettoria (delta_mobility RICALCOLATA):")
    print(f"     mediana {np.median(angs):.2f}  max {angs.max():.2f} gradi")
    dentro = angs.max() < np.percentile(null, 95)
    print(f"     -> {'DENTRO' if dentro else 'FUORI'} il null "
          f"(p95 = {np.percentile(null, 95):.2f})")
    print("     " + ("La geometria NON dipende dall'ordine temporale."
                     if dentro else
                     "La geometria dipende dall'ordine: delta_mobility pesa."))

    # ------------------------------------------------- B) persistenza
    print("\n" + "-" * 74)
    print("B) PERSISTENZA sotto mescolamento dei ply")
    print("-" * 74)
    obs = residence(df, zref)
    print(f"  residenza osservata            : {obs:.2f} ply")

    res_sh = []
    for _ in range(N_SHUF):
        sh = shuffle_within(df, feats)
        z = pca.transform(scaler.transform(sh[feats].values))[:, 0]
        res_sh.append(residence(sh, z))
    res_sh = np.array(res_sh)
    print(f"  residenza con ply mescolati    : {res_sh.mean():.2f} ply "
          f"(p95 {np.percentile(res_sh, 95):.2f})")
    print(f"  (riferimento T9, block-bootstrap: 3.49 mediana / 3.51 p95)")

    ecc = 100 * (obs - res_sh.mean()) / res_sh.mean()
    print(f"\n  eccesso di persistenza attribuibile all'ordine: {ecc:+.0f}%")

    print("\n" + "=" * 74)
    print("LETTURA PER IL PAPER")
    print("=" * 74)
    print("  A1 non e' una condizione per RECUPERARE la geometria.")
    print("  E' una condizione per poter MISURARE la persistenza.")
    print("  Nel paper vanno separate: la geometria richiede A2/A3/A5,")
    print("  la residenza richiede A1. L'Introduzione va corretta di conseguenza.")

    out = Path("data/outputs/a1_temporal.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "metrica": ["null_p95_deg", "angolo_shuffle_max_deg",
                    "residenza_osservata_ply", "residenza_shuffle_ply"],
        "valore": [np.percentile(null, 95), angs.max(), obs, res_sh.mean()],
    }).to_csv(out, index=False)
    print(f"\nSalvato: {out}")


if __name__ == "__main__":
    main()
