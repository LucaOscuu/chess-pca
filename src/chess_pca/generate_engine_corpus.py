"""
==============================================================================
GENERATORE CORPORA A COMPETENZA VARIABILE  —  input per T6
==============================================================================

Genera partite Stockfish self-play a profondita' di ricerca limitata.
La profondita' e' il proxy di competenza: depth 1 = quasi casuale con
un filtro tattico minimo, depth crescente = gioco progressivamente forte.

PREREQUISITO: binario di Stockfish
    https://stockfishchess.org/download/
    Scarica, estrai, e passa il path con --engine

USO
---
    python src\\chess_pca\\generate_engine_corpus.py --depth 1 --games 2000 \\
        --engine C:\\stockfish\\stockfish.exe \\
        --match-openings data\\openings.txt \\
        --out data\\engine_d1.pgn

Ripeti per --depth 3 e --depth 6 (o 5, se il tempo stringe).

TEMPI INDICATIVI (2000 partite, CPU desktop)
--------------------------------------------
    depth 1   ~5-10 min
    depth 3   ~20-40 min
    depth 5   ~1.5-3 h
    depth 6   ~3-6 h

Se il tempo e' poco: fai depth 1 e 3 (bastano per stabilire monotonicita'
su quattro punti insieme a random e full).

NOTA SULLA DIVERSITA'
---------------------
A profondita' fissa Stockfish e' deterministico: dalla stessa posizione
produce sempre la stessa partita. La varieta' viene dalle 2000 posizioni
di apertura distinte campionate da openings.txt. Usa SEMPRE
--match-openings, altrimenti otterresti 2000 copie della stessa partita.
==============================================================================
"""

import argparse
import random
from pathlib import Path

import chess
import chess.engine
import chess.pgn


def play_engine_game(engine, start_fen, depth, max_plies):
    board = chess.Board(start_fen) if start_fen else chess.Board()
    moves = []
    limit = chess.engine.Limit(depth=depth)
    while not board.is_game_over(claim_draw=True) and len(moves) < max_plies:
        try:
            res = engine.play(board, limit)
        except chess.engine.EngineError:
            break
        if res.move is None:
            break
        board.push(res.move)
        moves.append(res.move)
    return board, moves


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True, help="Path al binario Stockfish")
    ap.add_argument("--depth", type=int, required=True)
    ap.add_argument("--games", type=int, default=2000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-plies", type=int, default=300)
    ap.add_argument("--seed", type=int, default=20260727)
    ap.add_argument("--match-openings", default="data/openings.txt")
    ap.add_argument("--threads", type=int, default=1,
                    help="Tienilo a 1: piu' thread rendono la ricerca "
                         "non deterministica e sporcano il confronto.")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    openings = [l.strip() for l in Path(args.match_openings).read_text().splitlines()
                if l.strip()]
    print(f"Aperture caricate: {len(openings)}")

    # Campiona SENZA rimpiazzo: 2000 aperture distinte -> 2000 partite distinte
    chosen = rng.sample(openings, min(args.games, len(openings)))

    engine = chess.engine.SimpleEngine.popen_uci(args.engine)
    engine.configure({"Threads": args.threads, "Hash": 64})

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    counts = {"1-0": 0, "0-1": 0, "1/2-1/2": 0}

    try:
        with out.open("w", encoding="utf-8") as fh:
            for i, fen in enumerate(chosen):
                board, moves = play_engine_game(
                    engine, fen, args.depth, args.max_plies)
                trunc = len(moves) >= args.max_plies
                r = "1/2-1/2" if trunc else board.result(claim_draw=True)
                if r == "*":
                    r = "1/2-1/2"
                counts[r] += 1

                g = chess.pgn.Game()
                g.headers["Event"] = f"EngineDepth{args.depth}"
                g.headers["White"] = f"SF_d{args.depth}"
                g.headers["Black"] = f"SF_d{args.depth}"
                g.headers["Round"] = str(i + 1)
                g.headers["Result"] = r
                g.headers["FEN"] = fen
                g.headers["SetUp"] = "1"
                node = g
                for mv in moves:
                    node = node.add_variation(mv)
                print(g, file=fh, end="\n\n")

                if (i + 1) % 100 == 0:
                    print(f"  {i+1}/{len(chosen)}")
    finally:
        engine.quit()

    tot = sum(counts.values())
    print(f"\nScritto: {out}")
    for k, v in counts.items():
        print(f"  {k:8s} {v:5d}  ({100*v/tot:.1f}%)")
    print(f"Decisive: {counts['1-0']+counts['0-1']} "
          f"({100*(counts['1-0']+counts['0-1'])/tot:.1f}%)")


if __name__ == "__main__":
    main()


# =============================================================================
# DOPO LA GENERAZIONE: costruisci i parquet
# =============================================================================
# Crea build_engine.py nella stessa cartella:
#
#   import sys
#   from chess_pca.pipeline import build_or_load_dataset
#   d = sys.argv[1]
#   w, b = build_or_load_dataset(
#       input_path=f"data/engine_d{d}.pgn",
#       max_games=2000, min_moves=10,
#       cache_dir=f"data/cache_d{d}", force_recompute=True)
#   print(f"d{d}: {len(w):,} righe, {w.game_id.nunique()} partite, "
#         f"{len(w)/w.game_id.nunique():.0f} ply/partita")
#
# Poi:
#   python src\chess_pca\build_engine.py 1
#   python src\chess_pca\build_engine.py 3
#   python src\chess_pca\build_engine.py 6
# =============================================================================
