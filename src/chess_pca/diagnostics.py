"""
==============================================================================
SUITE DIAGNOSTICA — Morphological Framework
==============================================================================

Ordine di esecuzione: T0 -> T1 -> T2 -> T3 -> T4 -> T5, poi il resto.
I test T1 e T2 sono BLOCCANTI: se falliscono, non scrivere il paper
nella forma attuale.

INPUT ATTESO
------------
Un DataFrame `df` con almeno queste colonne:
    game_id      int/str   identificativo partita
    ply          int       indice temporale entro la partita
    outcome      str       'Win' | 'Draw' | 'Loss'  (dalla prospettiva scelta)
    perspective  str       'White' | 'Black'
    f1 ... f18   float     le feature del feature map phi
    material     float     differenziale di materiale (deve essere fra le f)

Se il tuo codice attuale non produce questo formato, adattalo qui una volta
sola: tutti i test sotto assumono questo schema.

Esecuzione:  python diagnostics.py
==============================================================================
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.linalg import subspace_angles
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

RNG = np.random.default_rng(20260727)

# Feature prodotte da chess_pca.features.numeric_features().
# Vengono risolte dinamicamente dal DataFrame, così lo script resta coerente
# con eventuali aggiunte future alla feature map.
NON_FEATURE_COLUMNS = {
    "game_id", "idx", "ply", "move_num", "result", "outcome",
    "perspective", "opening_fen", "is_resignation", "source_game_id",
}

# Nomi reali delle feature materiali nella pipeline corrente.
MATERIAL_FAMILY = [
    "material",
    "cnt_pawn",
    "cnt_knight",
    "cnt_bishop",
    "cnt_rook",
    "cnt_queen",
    "pawns_passed_diff",
    "hanging_diff",
]


def infer_features(df: pd.DataFrame) -> list[str]:
    """Replica la logica di numeric_features senza importare il package."""
    features = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c not in NON_FEATURE_COLUMNS
    ]
    if not features:
        raise ValueError("Nessuna feature numerica trovata nel dataset.")
    return features


def adapt_pipeline_dataframe(
    df: pd.DataFrame,
    perspective: str,
) -> pd.DataFrame:
    """Adatta un parquet White/Black della pipeline allo schema diagnostico."""
    out = df.copy().reset_index(drop=True)

    required = {"game_id", "ply", "result"}
    missing = required - set(out.columns)
    if missing:
        raise KeyError(f"Colonne mancanti nel dataset {perspective}: {sorted(missing)}")

    mapping = {"W": "Win", "D": "Draw", "L": "Loss"}
    normalized = out["result"].astype(str).str.strip().str.upper().map(mapping)
    if normalized.isna().any():
        bad = sorted(out.loc[normalized.isna(), "result"].astype(str).unique())
        raise ValueError(f"Valori result non riconosciuti in {perspective}: {bad}")

    out["outcome"] = normalized
    out["perspective"] = perspective
    # Evita collisioni tra la traiettoria White e Black della stessa partita.
    out["source_game_id"] = out["game_id"]
    out["game_id"] = perspective + ":" + out["game_id"].astype(str)
    return out


def load_pipeline_data(
    white_path: str | Path,
    black_path: str | Path,
    perspective: str = "both",
) -> pd.DataFrame:
    """Carica i parquet della pipeline e restituisce il DataFrame diagnostico."""
    white_path = Path(white_path)
    black_path = Path(black_path)

    frames = []
    if perspective in {"white", "both"}:
        if not white_path.exists():
            raise FileNotFoundError(f"Parquet White non trovato: {white_path}")
        frames.append(adapt_pipeline_dataframe(pd.read_parquet(white_path), "White"))

    if perspective in {"black", "both"}:
        if not black_path.exists():
            raise FileNotFoundError(f"Parquet Black non trovato: {black_path}")
        frames.append(adapt_pipeline_dataframe(pd.read_parquet(black_path), "Black"))

    if not frames:
        raise ValueError("Perspective deve essere white, black o both.")

    df = pd.concat(frames, ignore_index=True)
    features = infer_features(df)
    df[features] = df[features].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df


# =============================================================================
# T0 — SANITY CHECK  (10 minuti)
# =============================================================================
# Regola: nessuna regola di stop. Serve a chiudere le incoerenze che un
# referee troverebbe. Ma se d != 18 o i conteggi non tornano, fermati e
# sistema PRIMA di eseguire il resto: tutti i test successivi ereditano
# l'errore.

def T0_sanity(df, features=None):
    print("\n" + "=" * 70)
    print("T0 — SANITY CHECK")
    print("=" * 70)

    present = infer_features(df) if features is None else list(features)
    print(f"Feature numeriche rilevate: {len(present)}")
    print("  " + ", ".join(present))
    if len(present) != 18:
        print(f"  !! La pipeline corrente produce d={len(present)}, non d=18.")
        print("     Il paper deve riportare il numero effettivo e l'elenco completo.")

    print(f"\nPartite uniche: {df.game_id.nunique()}")
    print(f"Osservazioni (ply-stato): {len(df)}")
    print(f"Ply medi per partita: {len(df) / df.game_id.nunique():.1f}")

    print("\nConteggi per prospettiva x esito:")
    print(df.groupby(["perspective", "outcome"]).game_id.nunique())
    print("\n  -> Verifica che i N nelle figure residence corrispondano.")
    print("     Nel PDF, White e Black riportano ENTRAMBI N_win=1846 /")
    print("     N_loss=1069. Per Black devono essere invertiti.")

    print("\nRange delle feature (scala):")
    desc = df[present].agg(["min", "max", "std"]).T
    desc["range"] = desc["max"] - desc["min"]
    print(desc.sort_values("range", ascending=False).head(8))
    print("\n  -> Se i range differiscono di ordini di grandezza e la PCA e'")
    print("     su COVARIANZA, PC1 e' determinata dalla scala, non dalla")
    print("     struttura. Dichiara esplicitamente cov vs corr nel paper.")
    return present


# =============================================================================
# T1 — LEAKAGE MATERIALE  *** BLOCCANTE ***  (20 minuti)
# =============================================================================
# DOMANDA: PC1/PC2 sono qualcosa di piu' del differenziale di materiale?
#
# REGOLA DI DECISIONE:
#   |r| > 0.85  -> il nucleo empirico e' tautologico. Il paper non e'
#                  pubblicabile con queste feature. Ricostruisci phi.
#   0.60-0.85   -> forte confondimento. Obbligatorio T2 e la versione
#                  senza feature materiali.
#   |r| < 0.60  -> procedi, ma riporta comunque il numero nel paper.
#
# Riporta SEMPRE questo valore nel paper. Un referee lo chiedera'.

def T1_material_leakage(df, features):
    print("\n" + "=" * 70)
    print("T1 — LEAKAGE MATERIALE  *** BLOCCANTE ***")
    print("=" * 70)

    X = StandardScaler().fit_transform(df[features].values)
    pca = PCA(n_components=4).fit(X)
    Z = pca.transform(X)

    # Convenzione deterministica: quando disponibile, orienta PC1 in modo
    # che sia positivamente correlata con il vantaggio materiale. Il segno
    # della PCA e' arbitrario; fissarlo evita inversioni tra esecuzioni.
    if "material" in df.columns:
        r_pc1_material = stats.pearsonr(Z[:, 0], df["material"].to_numpy(dtype=float))[0]
        if np.isfinite(r_pc1_material) and r_pc1_material < 0:
            Z[:, 0] *= -1
            pca.components_[0] *= -1
            print("\nPC1 orientata: segno invertito per ottenere corr(PC1, material) >= 0.")

    for name in ["material", "eval_cp"]:  # eval_cp = valutazione engine
        if name not in df.columns:
            print(f"\n[{name}] non presente — salta "
                  f"(se hai la eval di Stockfish, aggiungila: e' il test piu' forte)")
            continue
        v = df[name].values
        print(f"\nCorrelazione con {name}:")
        for k in range(3):
            r, p = stats.pearsonr(Z[:, k], v)
            flag = ("*** TAUTOLOGICO ***" if abs(r) > 0.85
                    else "!! confondimento forte" if abs(r) > 0.60
                    else "ok")
            print(f"  PC{k+1}: r = {r:+.3f}   {flag}")

    # Varianza di PC1 spiegata dall'intera famiglia materiale
    material_family = [c for c in MATERIAL_FAMILY if c in df.columns]
    if material_family:
        print(f"\nFamiglia materiale usata: {material_family}")
        M = StandardScaler().fit_transform(df[material_family].values)
        M1 = np.column_stack([np.ones(len(M)), M])
        beta, *_ = np.linalg.lstsq(M1, Z[:, 0], rcond=None)
        r2 = 1 - np.sum((Z[:, 0] - M1 @ beta) ** 2) / np.sum(
            (Z[:, 0] - Z[:, 0].mean()) ** 2)
        print(f"\nR^2 di PC1 sulla famiglia materiale: {r2:.3f}")
        if r2 > 0.75:
            print("  *** PC1 E' essenzialmente il materiale. ***")

    return Z, pca


# =============================================================================
# T2 — RESIDENCE A MATERIALE APPAIATO  *** IL TEST DECISIVO ***  (1 ora)
# =============================================================================
# DOMANDA: l'asimmetria di occupazione sopravvive quando le posizioni
# hanno materiale EQUIVALENTE?
#
# Questo e' il test che trasforma "chi ha piu' pezzi vince" in un
# risultato. E' il singolo esperimento con il piu' alto valore per il
# paper.
#
# REGOLA DI DECISIONE:
#   asimmetria sopravvive (>= 8 punti percentuali, p<0.01 dopo correzione)
#       -> HAI UN RISULTATO. E' la figura principale del paper.
#   asimmetria svanisce
#       -> la geometria e' il materiale. Riposiziona come risultato
#          negativo, oppure ricostruisci phi con feature posizionali pure.

def T2_matched_material(df, Z, tol=0.5, thresholds=(-1.0, 0.75)):
    print("\n" + "=" * 70)
    print("T2 — RESIDENCE A MATERIALE APPAIATO  *** DECISIVO ***")
    print("=" * 70)

    d = df.copy()
    d["pc1"], d["pc2"] = Z[:, 0], Z[:, 1]

    mask = d["material"].abs() <= tol
    print(f"\nPly con |material| <= {tol}: {mask.sum():,} "
          f"({100*mask.mean():.1f}% del totale)")
    if mask.sum() < 20000:
        print("  !! pochi ply appaiati — allarga tol o usa piu' partite")

    sub = d[mask]
    lo, hi = thresholds

    # Con PC1 orientata positivamente rispetto al materiale, valori alti
    # indicano vantaggio per la prospettiva: regione win a destra, loss a sinistra.
    for region, cond in [("Omega_loss (PC1 < %.2f)" % lo, sub.pc1 < lo),
                         ("Omega_win (PC1 > %.2f)" % hi, sub.pc1 > hi)]:
        tmp = sub.assign(inside=cond)
        res = (tmp.groupby(["game_id", "outcome"])
                  .inside.mean().reset_index())
        w = res[res.outcome == "Win"].inside
        l = res[res.outcome == "Loss"].inside
        if len(w) < 30 or len(l) < 30:
            print(f"\n{region}: campione insufficiente")
            continue
        u, p = stats.mannwhitneyu(w, l, alternative="two-sided")
        rb = 2 * u / (len(w) * len(l)) - 1  # rank-biserial
        gap = 100 * (w.median() - l.median())
        print(f"\n{region}")
        print(f"  mediana Win  : {100*w.median():.1f}%  (n={len(w)})")
        print(f"  mediana Loss : {100*l.median():.1f}%  (n={len(l)})")
        print(f"  gap          : {gap:+.1f} punti percentuali")
        print(f"  rank-biserial: {rb:+.3f}   p = {p:.2e}")
        print("  -> RIPORTA IL GAP E L'EFFECT SIZE, NON IL p-VALUE.")


# =============================================================================
# T3 — SOGLIA BBP / NULLO MARCHENKO-PASTUR  (15 minuti)
# =============================================================================
# Sostituisce il nullo shuffled del paper, che e' non informativo:
# distrugge TUTTE le correlazioni, quindi qualunque dato reale lo batte.
#
# NOTA CRITICA sulla dimensione campionaria efficace:
# gli stati entro una partita sono fortemente autocorrelati. n NON e' il
# numero di ply. Usa n_eff = numero di PARTITE (conservativo). Questo
# alza la soglia e rende il test onesto.
#
# REGOLA: C3 e' soddisfatta se lambda_1 > (1 + sqrt(gamma))^2 con
# gamma = d / n_eff, su matrice di CORRELAZIONE.

def T3_bbp_threshold(df, features, pca):
    print("\n" + "=" * 70)
    print("T3 — SOGLIA BBP / MARCHENKO-PASTUR")
    print("=" * 70)

    d = len(features)
    n_ply = len(df)
    n_eff = df.game_id.nunique()

    C = np.corrcoef(StandardScaler().fit_transform(df[features].values).T)
    lam = np.sort(np.linalg.eigvalsh(C))[::-1]

    for label, n in [("ply (ottimistico)", n_ply), ("partite (onesto)", n_eff)]:
        g = d / n
        edge = (1 + np.sqrt(g)) ** 2
        n_above = int((lam > edge).sum())
        print(f"\nn = {n:,}  [{label}]   gamma = {g:.2e}")
        print(f"  bordo MP: {edge:.4f}")
        print(f"  lambda_1 = {lam[0]:.4f}  -> "
              f"{'C3 SODDISFATTA' if lam[0] > edge else 'C3 VIOLATA'}")
        print(f"  autovalori sopra il bordo: {n_above} / {d}")

    pr = (lam.sum() ** 2) / (lam ** 2).sum()
    print(f"\nParticipation ratio: {pr:.2f} / {d}")
    print(f"Dimensionalita' effettiva relativa: {pr/d:.1%}")
    print("  -> NON e' un problema: e' una QUANTITA' COMPARATIVA.")
    print("     Confrontala con quella del random mover (T5).")

    print(f"\nSpettro: " + "  ".join(f"{v:.3f}" for v in lam[:6]))
    print(f"Eigengap lambda1-lambda2 = {lam[0]-lam[1]:.4f}")
    print(f"          lambda2-lambda3 = {lam[1]-lam[2]:.4f}")
    if (lam[0] - lam[1]) < 2 * (lam[1] - lam[2]):
        print("  !! PC1 e PC2 quasi degeneri: i SINGOLI assi non sono")
        print("     identificati. Parla di SOTTOSPAZIO 2D, non di 'PC1 vs PC2'.")
        print("     Questo invalida l'asimmetria White-PC2 / Black-PC1.")
    return lam


# =============================================================================
# T4 — APPAIAMENTO FISHTEST  *** BLOCCANTE ***  (30 minuti)
# =============================================================================
# Fishtest gioca partite a COPPIE dalla stessa posizione di libro con
# colori invertiti. Se e' il tuo caso, una frazione degli esiti e'
# determinata dallo squilibrio del book, non dalla dinamica.
#
# REGOLA:
#   se le partite sono appaiate -> DEVI rifare la residence asymmetry
#   ENTRO le coppie. Se sparisce, il risultato era il libro di aperture.

def T4_opening_pairing(df):
    print("\n" + "=" * 70)
    print("T4 — APPAIAMENTO / LIBRO DI APERTURE")
    print("=" * 70)

    if "opening_fen" not in df.columns:
        print("\nCampo 'opening_fen' assente.")
        return

    if "perspective" not in df.columns:
        raise KeyError("T4 richiede la colonna 'perspective'.")

    perspective_counts = (
        df.groupby("opening_fen")["perspective"]
        .nunique()
        .value_counts()
        .sort_index()
    )

    print("\nNumero di prospettive per opening_fen:")
    print(perspective_counts)

    # Una sola prospettiva equivale a una riga per partita reale.
    real_df = df[df["perspective"] == "White"].copy()

    counts = real_df.groupby("opening_fen")["game_id"].nunique()

    print(f"\nPosizioni di apertura distinte: {counts.size}")
    print("Partite reali per posizione:")
    print(counts.value_counts().sort_index())

    n_games = int(counts.sum())
    n_unique_games = int((counts == 1).sum())
    n_shared_games = int(counts[counts > 1].sum())

    print(f"\nPartite reali totali: {n_games}")
    print(
        f"Partite con posizione unica: {n_unique_games} "
        f"({100 * n_unique_games / n_games:.1f}%)"
    )
    print(
        f"Partite in posizioni condivise: {n_shared_games} "
        f"({100 * n_shared_games / n_games:.1f}%)"
    )

    if counts.mode().iloc[0] == 1:
        print("\n✓ Nessun appaiamento sistematico rilevato.")
        print("  La moda corretta, a livello di partita reale, è 1.")
    else:
        print("\n!! Possibile struttura sistematica di appaiamento.")

    multi_openings = counts[counts > 1].index
    print(
        f"\nAperture disponibili per il controllo within-opening: "
        f"{len(multi_openings)}"
    )

    return multi_openings


# =============================================================================
# T5 — CONTROLLO NEGATIVO: RANDOM MOVER  *** BLOCCANTE ***  (1 notte)
# =============================================================================
# Genera ~2000 partite di gioco casuale legale con lo STESSO phi e la
# STESSA pipeline. Poi confronta le tre firme geometriche.
#
# GENERAZIONE (script separato, python-chess):
#
#   import chess, chess.pgn, random
#   for i in range(2000):
#       b = chess.Board()
#       while not b.is_game_over() and b.fullmove_number < 200:
#           b.push(random.choice(list(b.legal_moves)))
#       # salva PGN, estrai le stesse 18 feature
#
# REGOLA DI DECISIONE:
#   PR_random > PR_stockfish (di almeno ~2 unita')  E
#   angoli principali fra i sottospazi > ~20 gradi
#       -> LA GEOMETRIA E' UNA PROPRIETA' DEL GIOCO COMPETENTE.
#          E' la tesi centrale del paper, dimostrata.
#
#   geometrie indistinguibili
#       -> la geometria viene dalle REGOLE degli scacchi, non dal gioco
#          strategico. Tesi falsificata. Riposiziona come risultato
#          negativo (pubblicabile, ma diverso).

def T5_negative_control(df_stockfish, df_random, features):
    print("\n" + "=" * 70)
    print("T5 — CONTROLLO NEGATIVO: RANDOM MOVER  *** BLOCCANTE ***")
    print("=" * 70)

    if not features:
        raise ValueError("T5 richiede almeno una feature comune.")

    out = {}
    for name, d in [("Stockfish", df_stockfish), ("Random", df_random)]:
        missing = [f for f in features if f not in d.columns]
        if missing:
            raise ValueError(f"T5: nel dataset {name} mancano le feature: {missing}")
        if d.empty:
            raise ValueError(f"T5: il dataset {name} e' vuoto.")

        X = StandardScaler().fit_transform(d[features].to_numpy(dtype=float))
        p = PCA().fit(X)
        lam = p.explained_variance_
        pr = (lam.sum() ** 2) / (lam ** 2).sum()

        # Marchenko-Pastur sulla matrice di correlazione, con numero di
        # partite come numerosita' effettiva conservativa.
        C = np.corrcoef(X.T)
        lam_c = np.sort(np.linalg.eigvalsh(C))[::-1]
        n_eff = int(d["game_id"].nunique())
        if n_eff <= 0:
            raise ValueError(f"T5: n_eff non valido per {name}.")
        mp_edge = (1 + np.sqrt(len(features) / n_eff)) ** 2
        n_sig_mp = int((lam_c > mp_edge).sum())

        out[name] = {
            "PR": float(pr),
            "EVR2": float(p.explained_variance_ratio_[:2].sum()),
            "V": p.components_[:2].T,
            "lam": lam,
            "mp_edge": float(mp_edge),
            "n_sig_mp": n_sig_mp,
            "n_eff": n_eff,
        }

        print(f"\n{name}:")
        print(f"  osservazioni     = {len(d):,}")
        print(f"  n_eff (partite)  = {n_eff:,}")
        print(f"  PR               = {pr:.2f} / {len(features)}")
        print(f"  EVR(PC1+PC2)     = {out[name]['EVR2']:.3f}")
        print(f"  eigengap 1-2     = {lam[0]-lam[1]:.4f}")
        print(
            f"  autovalori > MP  = {n_sig_mp} / {len(features)} "
            f"(bordo={mp_edge:.4f})"
        )

    ang = np.degrees(subspace_angles(out["Stockfish"]["V"], out["Random"]["V"]))
    dpr = out["Random"]["PR"] - out["Stockfish"]["PR"]
    delta_mp = out["Stockfish"]["n_sig_mp"] - out["Random"]["n_sig_mp"]

    print(f"\nAngoli principali fra i sottospazi 2D: "
          f"{ang[0]:.1f}deg, {ang[1]:.1f}deg")
    print(f"Delta PR (random - stockfish): {dpr:+.2f}")
    print(f"Delta autovalori MP (stockfish - random): {delta_mp:+d}")

    print("\n" + "-" * 70)
    if dpr > 2 and ang.max() > 20:
        print("  *** TESI SUPPORTATA ***")
        print("  Il gioco competente produce una geometria piu' compressa")
        print("  e orientata diversamente.")
    elif dpr > 1 or ang.max() > 15 or delta_mp != 0:
        print("  Segnale comparativo presente, ma non conclusivo su tutte le metriche.")
        print("  Interpreta insieme PR, angoli principali e conteggio sopra MP.")
    else:
        print("  *** TESI NON SUPPORTATA da questo confronto ***")
        print("  La geometria non distingue chiaramente gioco casuale da competente.")

    return out


# =============================================================================
# T6 — GRADIENTE DI COMPETENZA  (1 notte)
# =============================================================================
# Genera corpora con Stockfish a depth 1, 3, 6 e confronta PR, EVR2 e
# angoli principali rispetto al full-strength.
#
# REGOLA: se PR decresce monotonicamente con la profondita', hai una
# RELAZIONE, non solo un contrasto. E' il risultato piu' forte ottenibile
# nei tempi disponibili.

def T6_competence_gradient(corpora, features):
    """corpora: dict {'random':df, 'depth1':df, 'depth3':df, 'sf':df}"""
    print("\n" + "=" * 70)
    print("T6 — GRADIENTE DI COMPETENZA")
    print("=" * 70)
    ref = None
    rows = []
    for name, d in corpora.items():
        X = StandardScaler().fit_transform(d[features].values)
        p = PCA().fit(X)
        lam = p.explained_variance_
        V = p.components_[:2].T
        if ref is None:
            ref = V
        ang = np.degrees(subspace_angles(ref, V)).max()
        rows.append({
            "regime": name,
            "PR": (lam.sum() ** 2) / (lam ** 2).sum(),
            "EVR2": p.explained_variance_ratio_[:2].sum(),
            "angolo_vs_ref": ang,
        })
    tab = pd.DataFrame(rows)
    print("\n", tab.to_string(index=False))
    print("\n  -> Monotonicita' di PR = risultato principale del paper.")
    return tab


# =============================================================================
# T7 — TRAIN/TEST SPLIT  (30 minuti)
# =============================================================================
# Rompe la circolarita': PCA e soglie stimate SOLO su train, asimmetria
# misurata SOLO su test.
#
# REGOLA: se il gap out-of-sample e' < 50% del gap in-sample, avevi
# overfitting sulle soglie.

def T7_train_test(df, features, frac=0.7):
    print("\n" + "=" * 70)
    print("T7 — TRAIN/TEST SPLIT")
    print("=" * 70)

    games = df.game_id.unique()
    RNG.shuffle(games)
    cut = int(frac * len(games))
    tr = df[df.game_id.isin(games[:cut])]
    te = df[df.game_id.isin(games[cut:])]
    print(f"\nTrain: {cut} partite | Test: {len(games)-cut} partite")

    sc = StandardScaler().fit(tr[features].values)
    p = PCA(n_components=2).fit(sc.transform(tr[features].values))

    z_tr = p.transform(sc.transform(tr[features].values))[:, 0]
    # soglie derivate SOLO dal train (quantili, non "a occhio")
    lo, hi = np.quantile(z_tr, [0.25, 0.75])
    print(f"Soglie da train (quartili): {lo:.3f}, {hi:.3f}")

    z_te = p.transform(sc.transform(te[features].values))[:, 0]
    t = te.assign(inside=(z_te < lo))
    res = t.groupby(["game_id", "outcome"]).inside.mean().reset_index()
    w = res[res.outcome == "Win"].inside
    l = res[res.outcome == "Loss"].inside
    if len(w) > 20 and len(l) > 20:
        u, pv = stats.mannwhitneyu(w, l, alternative="two-sided")
        print(f"\nOUT-OF-SAMPLE  gap = "
              f"{100*(w.median()-l.median()):+.1f} pp   p = {pv:.2e}")
        print(f"  rank-biserial = {2*u/(len(w)*len(l))-1:+.3f}")


# =============================================================================
# T8 — PLACEBO  (30 minuti)
# =============================================================================
# Ripeti l'intera pipeline con etichette finte. Se produce "bacini" anche
# li', la pipeline e' il risultato.
#
# REGOLA: il gap placebo deve essere < 1/3 del gap reale.

def T8_placebo(df, Z, n_perm=200, threshold=-1.0):
    print("\n" + "=" * 70)
    print("T8 — PLACEBO")
    print("=" * 70)

    d = df.assign(inside=(Z[:, 0] < threshold))
    per_game = d.groupby("game_id").agg(
        inside=("inside", "mean"), outcome=("outcome", "first"))
    dec = per_game[per_game.outcome.isin(["Win", "Loss"])]
    real = (dec[dec.outcome == "Win"].inside.median()
            - dec[dec.outcome == "Loss"].inside.median())

    # Converti esplicitamente in un ndarray NumPy standard. Con pandas
    # ArrowStringArray, RNG.shuffle puo' produrre warning e comportamento
    # non garantito a causa delle view semantics.
    labels = dec["outcome"].astype("string").to_numpy(dtype=str, copy=True)
    inside = dec["inside"].to_numpy(dtype=float, copy=True)

    null = np.empty(n_perm, dtype=float)
    for i in range(n_perm):
        permuted = RNG.permutation(labels)
        win_vals = inside[permuted == "Win"]
        loss_vals = inside[permuted == "Loss"]
        null[i] = np.median(win_vals) - np.median(loss_vals)

    # Correzione +1: evita di stampare p=0 con un numero finito di permutazioni.
    exceedances = int(np.count_nonzero(np.abs(null) >= abs(real)))
    p = (exceedances + 1) / (n_perm + 1)

    print(f"\nGap reale                : {100*real:+.1f} pp")
    print(f"Gap permutato (|mediana|): {100*np.median(np.abs(null)):.1f} pp")
    print(f"p empirico corretto (+1): {p:.4f}")
    if p > 0.01:
        print("  *** IL PLACEBO RIPRODUCE IL RISULTATO. ***")

    # Placebo semantico: outcome finto scorrelato dalla strategia
    fake = (df.groupby("game_id").ply.max() % 2 == 0)
    print("\n(Ripeti con outcome finti: parita' della lunghezza partita,")
    print(" prima mossa 1.e4 vs 1.d4. Devono dare gap ~ 0.)")


# =============================================================================
# T9 — PERSISTENZA vs SURROGATI  (condizione C4)  (1 ora)
# =============================================================================
# Il nullo shuffled del paper distrugge tutto. Il nullo corretto preserva
# l'autocorrelazione e distrugge solo la struttura regionale.
#
# REGOLA: C4 e' soddisfatta se il tempo medio di residenza osservato
# eccede il 95-esimo percentile dei surrogati a blocchi.

def T9_persistence(df, Z, threshold=-1.0, block=10, n_surr=200):
    print("\n" + "=" * 70)
    print("T9 — PERSISTENZA vs SURROGATI A BLOCCHI  (C4)")
    print("=" * 70)

    def mean_run(mask):
        runs, cur = [], 0
        for v in mask:
            if v:
                cur += 1
            elif cur:
                runs.append(cur); cur = 0
        if cur:
            runs.append(cur)
        return np.mean(runs) if runs else 0.0

    d = df.assign(z=Z[:, 0])
    obs = [mean_run((g.z.values < threshold))
           for _, g in d.groupby("game_id")]
    obs = np.mean(obs)

    surr = []
    for _ in range(n_surr):
        vals = []
        for _, g in d.groupby("game_id"):
            z = g.z.values
            nb = max(1, len(z) // block)
            blocks = np.array_split(z, nb)
            RNG.shuffle(blocks)
            vals.append(mean_run(np.concatenate(blocks) < threshold))
        surr.append(np.mean(vals))
    surr = np.array(surr)

    print(f"\nResidenza media osservata : {obs:.2f} ply")
    print(f"Surrogati a blocchi (p95) : {np.percentile(surr, 95):.2f} ply")
    print(f"  -> C4 {'SODDISFATTA' if obs > np.percentile(surr,95) else 'VIOLATA'}")


# =============================================================================
def run_core_suite(df: pd.DataFrame, tol: float = 50.0) -> None:
    """Esegue i test disponibili sul dataset della pipeline corrente."""
    features = infer_features(df)
    features = T0_sanity(df, features)
    Z, pca = T1_material_leakage(df, features)
    T2_matched_material(df, Z, tol=tol)
    T3_bbp_threshold(df, features, pca)
    T4_opening_pairing(df)
    T7_train_test(df, features)
    T8_placebo(df, Z)
    T9_persistence(df, Z)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Suite diagnostica adattata ai parquet prodotti da chess_pca.pipeline"
    )
    parser.add_argument(
        "--white",
        default="data/cache/white_data_max10000_min10.parquet",
        help="Percorso del parquet White",
    )
    parser.add_argument(
        "--black",
        default="data/cache/black_data_max10000_min10.parquet",
        help="Percorso del parquet Black",
    )
    parser.add_argument(
        "--perspective",
        choices=["white", "black", "both"],
        default="both",
        help="Dataset da analizzare",
    )
    parser.add_argument(
        "--material-tol",
        type=float,
        default=50.0,
        help=(
            "Tolleranza T2 nelle unità della feature material. "
            "La pipeline usa centipawn: 50 equivale a mezzo pedone."
        ),
    )
    parser.add_argument(
        "--core-only",
        action="store_true",
        help="Esegue solo T0-T4, evitando i test più lenti T7-T9",
    )

    parser.add_argument(
        "--random-white",
        default=None,
        help="Parquet White del corpus random mover",
    )
    parser.add_argument(
        "--random-black",
        default=None,
        help="Parquet Black del corpus random mover",
    )
    args = parser.parse_args()

    df = load_pipeline_data(args.white, args.black, args.perspective)
    features = infer_features(df)

    print("\nDataset diagnostico caricato")
    print(f"  prospettiva: {args.perspective}")
    print(f"  osservazioni: {len(df):,}")
    print(f"  partite uniche: {df['game_id'].nunique():,}")
    print(f"  feature: {len(features)}")

    features = T0_sanity(df, features)
    Z, pca = T1_material_leakage(df, features)
    T2_matched_material(df, Z, tol=args.material_tol)
    T3_bbp_threshold(df, features, pca)
    T4_opening_pairing(df)

    if args.random_white or args.random_black:
        if not (args.random_white and args.random_black):
            raise ValueError(
                "Per T5 devi passare entrambi: --random-white e --random-black."
            )

        df_rand = load_pipeline_data(
            args.random_white,
            args.random_black,
            args.perspective,
        )
        random_features = infer_features(df_rand)
        missing_random = sorted(set(features) - set(random_features))
        extra_random = sorted(set(random_features) - set(features))

        if missing_random:
            raise ValueError(
                "T5: nel dataset random mancano queste feature: "
                f"{missing_random}"
            )
        if extra_random:
            print(f"\n[T5] Feature extra nel random ignorate: {extra_random}")

        T5_negative_control(df, df_rand, features)
    else:
        print("\n[T5 saltato: passa --random-white e --random-black]")

    if not args.core_only:
        T7_train_test(df, features)
        T8_placebo(df, Z)
        T9_persistence(df, Z)

