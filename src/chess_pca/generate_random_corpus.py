"""
==============================================================================
GENERATORE CORPUS RANDOM MOVER  —  input per T5
==============================================================================

Produce PGN di gioco casuale legale nello stesso formato dei PGN Fishtest,
così che la TUA pipeline esistente estragga esattamente le stesse 18 feature.
Questo e' essenziale: se il feature map differisce, il confronto T5 non ha
alcun valore.

WORKFLOW
--------
1)  python generate_random_corpus.py --games 2000 --out data/random_games.pgn
2)  fai girare la tua pipeline su quel PGN per produrre:
        data/cache/random_white.parquet
        data/cache/random_black.parquet
3)  aggiungi la chiamata a T5 nel main di diagnostics_updated.py (patch sotto)

DIPENDENZE:  pip install python-chess

NOTA SUGLI ESITI
----------------
Il gioco casuale finisce quasi sempre in patta (materiale insufficiente,
50 mosse, ripetizione). Attenditi il 90-97% di Draw.
NON e' un problema per T5: le metriche geometriche centrali — PR, autovalori
sopra il bordo di Marchenko-Pastur, angoli principali — NON usano l'esito.
L'esito serve solo al confronto opzionale di residence asymmetry; se i
decisivi sono meno di ~200 per lato, salta quel confronto e dichiaralo.

MODALITA' --match-openings (consigliata)
----------------------------------------
Se passi il file delle posizioni iniziali reali, le partite casuali partono
dalle STESSE posizioni di libro delle partite Stockfish. Questo isola
l'effetto del GIOCO da quello della POSIZIONE INIZIALE ed e' il controllo
metodologicamente piu' forte.
==============================================================================
"""

import argparse
import random
from pathlib import Path

import chess
import chess.pgn


def play_random_game(start_fen: str | None, max_plies: int, rng: random.Random):
    """Gioca una partita interamente casuale fra mosse legali."""
    board = chess.Board(start_fen) if start_fen else chess.Board()
    moves = []
    while not board.is_game_over(claim_draw=True) and len(moves) < max_plies:
        legal = list(board.legal_moves)
        if not legal:
            break
        mv = rng.choice(legal)
        board.push(mv)
        moves.append(mv)
    return board, moves


def board_result(board: chess.Board, truncated: bool) -> str:
    if truncated:
        return "1/2-1/2"          # troncata = patta per convenzione
    r = board.result(claim_draw=True)
    return r if r != "*" else "1/2-1/2"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=2000)
    ap.add_argument("--out", default="data/random_games.pgn")
    ap.add_argument("--max-plies", type=int, default=300,
                    help="Taglio di sicurezza. 300 ply ~ 150 mosse.")
    ap.add_argument("--seed", type=int, default=20260727)
    ap.add_argument("--match-openings", default=None,
                    help="File di testo con un FEN per riga: le partite "
                         "casuali partiranno da queste posizioni.")
    args = ap.parse_args()

    rng = random.Random(args.seed)

    openings = None
    if args.match_openings:
        openings = [l.strip() for l in Path(args.match_openings).read_text().splitlines()
                    if l.strip()]
        print(f"Posizioni di apertura caricate: {len(openings)}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    counts = {"1-0": 0, "0-1": 0, "1/2-1/2": 0}

    with out.open("w", encoding="utf-8") as fh:
        for i in range(args.games):
            fen = rng.choice(openings) if openings else None
            board, moves = play_random_game(fen, args.max_plies, rng)
            truncated = len(moves) >= args.max_plies
            res = board_result(board, truncated)
            counts[res] += 1

            game = chess.pgn.Game()
            game.headers["Event"] = "RandomMoverControl"
            game.headers["Site"] = "synthetic"
            game.headers["White"] = "RandomMover"
            game.headers["Black"] = "RandomMover"
            game.headers["Round"] = str(i + 1)
            game.headers["Result"] = res
            if fen:
                game.headers["FEN"] = fen
                game.headers["SetUp"] = "1"

            node = game
            for mv in moves:
                node = node.add_variation(mv)

            print(game, file=fh, end="\n\n")

            if (i + 1) % 250 == 0:
                print(f"  {i+1}/{args.games} partite generate")

    tot = sum(counts.values())
    print(f"\nScritto: {out}")
    print("Distribuzione esiti:")
    for k, v in counts.items():
        print(f"  {k:8s} {v:6d}  ({100*v/tot:.1f}%)")
    dec = counts["1-0"] + counts["0-1"]
    print(f"\nPartite decisive: {dec} ({100*dec/tot:.1f}%)")
    if dec < 400:
        print("  !! Pochi decisivi: usa T5 solo per le metriche geometriche")
        print("     (PR, autovalori sopra MP, angoli principali), che non")
        print("     dipendono dall'esito. Dichiaralo nel paper.")


if __name__ == "__main__":
    main()


# =============================================================================
# ESTRARRE LE POSIZIONI DI APERTURA REALI  (per --match-openings)
# =============================================================================
#
#   import pandas as pd
#   df = pd.read_parquet("data/cache/white_data_max10000_min10.parquet")
#   fens = df.groupby("game_id").opening_fen.first().dropna().unique()
#   Path("data/openings.txt").write_text("\n".join(fens))
#
# poi:
#   python generate_random_corpus.py --games 2000 \
#          --match-openings data/openings.txt
#
#
# =============================================================================
# PATCH PER diagnostics_updated.py
# =============================================================================
#
# 1) Aggiungi gli argomenti CLI dopo --core-only:
#
#     parser.add_argument("--random-white", default=None,
#                         help="Parquet White del corpus random mover")
#     parser.add_argument("--random-black", default=None,
#                         help="Parquet Black del corpus random mover")
#
# 2) Aggiungi in fondo al main, DOPO T4:
#
#     if args.random_white and args.random_black:
#         df_rand = load_pipeline_data(
#             args.random_white, args.random_black, args.perspective)
#         common = [f for f in features if f in df_rand.columns]
#         if len(common) != len(features):
#             print(f"!! Feature non allineate: {set(features) - set(common)}")
#             print("   Il confronto T5 richiede lo STESSO feature map.")
#         T5_negative_control(df, df_rand, common)
#     else:
#         print("\n[T5 saltato: passa --random-white e --random-black]")
#
# 3) Esegui:
#
#     python diagnostics_updated.py \
#         --random-white data/cache/random_white.parquet \
#         --random-black data/cache/random_black.parquet
#
#
# =============================================================================
# AGGIUNTA A T5: il conteggio autovalori sopra Marchenko-Pastur
# =============================================================================
# E' la metrica migliore che hai — ancorata a una soglia teorica, non a una
# tua scelta. Aggiungila dentro il loop di T5_negative_control:
#
#     import numpy as np
#     C = np.corrcoef(StandardScaler().fit_transform(d[features].values).T)
#     lam_c = np.sort(np.linalg.eigvalsh(C))[::-1]
#     n_eff = d.game_id.nunique()
#     edge = (1 + np.sqrt(len(features) / n_eff)) ** 2
#     n_sig = int((lam_c > edge).sum())
#     print(f"  autovalori sopra MP = {n_sig} / {len(features)}  "
#           f"(bordo {edge:.4f}, n_eff={n_eff})")
#
# Riferimento Stockfish: 7 su 18.
# =============================================================================
