"""
==============================================================================
GENERATORE CORPUS VARIANTE  —  v2
==============================================================================

CHE COSA CAMBIA RISPETTO ALLA v1
--------------------------------

1) VARIANTI CHE RICHIEDONO L'HEADER [Variant]
   La v1 ometteva l'header di default, con questa motivazione: "tutte le
   mosse sono mosse legali standard, cambia solo QUANDO la partita
   finisce". E' vero per 3check, kingofthehill, racingkings, horde.
   E' FALSO per atomic, antichess, crazyhouse:

     - atomic     : una cattura fa esplodere i pezzi adiacenti. Su una
                    chess.Board normale la cattura e' standard, quindi
                    dalla PRIMA cattura in poi la scacchiera ricostruita
                    non e' piu' quella della partita vera. Le feature
                    verrebbero calcolate su posizioni mai esistite,
                    senza alcun errore visibile.
     - antichess  : catture obbligatorie, re catturabile, si puo' muovere
                    "sotto scacco". Mosse non replicabili su board standard.
                    Il re puo' sparire: features.king_safety gestisce gia'
                    king() = None, ma solo perche' e' difensiva.
                    In antichess non esiste lo scacco, quindi checks_avail
                    e' identicamente 0 -> varianza nulla -> np.corrcoef
                    restituisce NaN in run_t6.signature(). Vedi
                    check_phi_validity.py PRIMA di lanciare T6.
     - crazyhouse : i drop non sono mosse standard, l'estrazione fallisce.

   Ora l'header e' FORZATO per queste varianti. features.py chiama
   game.board(), e chess.pgn legge [Variant] e costruisce da solo la
   board giusta: nessuna modifica a features.py o pipeline.py.

2) VALIDAZIONE DELLE APERTURE PRIMA DI GENERARE
   racingkings e horde hanno una posizione iniziale diversa: i FEN di
   data/openings.txt contengono pedoni e non sono legali. La v1 li
   catturava come ValueError e li contava in "skipped": avresti visto
   "scartate: 2000" senza capire perche'. Ora il controllo e' in testa e
   fallisce subito con un messaggio esplicito.

3) NUOVE VARIANTI:  antichess, crazyhouse, horde

USO — Antichess (il test di A2)
-------------------------------
    python src\\chess_pca\\generate_variant_corpus.py \\
        --variant antichess --depth 6 --games 2000 \\
        --engine "$fsf" \\
        --match-openings data\\openings.txt \\
        --out data\\variant_antichess.pgn

Perche' antichess: l'obiettivo e' invertito, quindi il MATERIALE cambia
segno come indicatore strategico. E' la perturbazione semantica piu'
forte di phi che si possa ottenere mantenendo la stessa posizione
iniziale e le stesse aperture. E' il test di A2.

AVVERTENZA ONESTA sulle aperture: i FEN provengono da partite standard,
quindi sono posizioni che in antichess non nascerebbero mai. Il controllo
"stesse aperture" resta valido come controllo, ma va dichiarato nel paper
che le posizioni iniziali sono fuori distribuzione per la variante.
Vale identicamente per il 3check che hai gia' fatto: e' coerente.
==============================================================================
"""

import argparse
import random
from pathlib import Path

import chess
import chess.engine
import chess.pgn
import chess.variant

VARIANTS = {
    "kingofthehill": chess.variant.KingOfTheHillBoard,
    "3check": chess.variant.ThreeCheckBoard,
    "atomic": chess.variant.AtomicBoard,
    "racingkings": chess.variant.RacingKingsBoard,
    "antichess": chess.variant.AntichessBoard,
    "crazyhouse": chess.variant.CrazyhouseBoard,
    "horde": chess.variant.HordeBoard,
}

# Varianti le cui mosse NON sono replicabili su una chess.Board standard.
# Per queste l'header [Variant] e' obbligatorio, altrimenti la pipeline
# estrae feature da posizioni sbagliate senza segnalare nulla.
REQUIRES_HEADER = {"atomic", "antichess", "crazyhouse"}

# Varianti con posizione iniziale incompatibile con data/openings.txt
INCOMPATIBLE_OPENINGS = {"racingkings", "horde"}


def validate_openings(board_cls, openings, variant):
    """Fallisce subito e chiaramente invece di scartare in silenzio."""
    ok = 0
    for fen in openings[:50]:
        try:
            board_cls(fen)
            ok += 1
        except ValueError:
            pass
    if ok == 0:
        raise SystemExit(
            f"\nERRORE: nessuna delle prime 50 aperture e' legale in "
            f"'{variant}'.\n"
            f"La posizione iniziale di questa variante e' diversa da quella\n"
            f"degli scacchi standard, quindi data/openings.txt non e'\n"
            f"utilizzabile. Rilancia senza --match-openings (ma perdi il\n"
            f"controllo sulle aperture, e va dichiarato nel paper), oppure\n"
            f"scegli una variante che parte dalla posizione standard\n"
            f"(3check, kingofthehill, atomic, antichess, crazyhouse).")
    if ok < 50:
        print(f"  Attenzione: {50-ok}/50 aperture di prova non legali in "
              f"'{variant}'. Verranno scartate.")


def play(engine, board_cls, fen, depth, max_plies):
    board = board_cls(fen) if fen else board_cls()
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
    ap.add_argument("--engine", required=True, help="Path a Fairy-Stockfish")
    ap.add_argument("--variant", required=True, choices=list(VARIANTS))
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--games", type=int, default=2000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-plies", type=int, default=300)
    ap.add_argument("--seed", type=int, default=20260727)
    ap.add_argument("--match-openings", default="data/openings.txt")
    ap.add_argument("--no-openings", action="store_true",
                    help="Parti dalla posizione iniziale della variante.")
    ap.add_argument("--keep-variant-header", action="store_true",
                    help="Forzato comunque per atomic/antichess/crazyhouse.")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    board_cls = VARIANTS[args.variant]

    keep_header = args.keep_variant_header or args.variant in REQUIRES_HEADER
    if args.variant in REQUIRES_HEADER and not args.keep_variant_header:
        print(f"[!] '{args.variant}' richiede l'header [Variant]: forzato.")
        print("    Senza, la pipeline ricostruirebbe le partite su una board")
        print("    standard e le feature sarebbero calcolate su posizioni")
        print("    mai esistite, in silenzio.")

    if args.no_openings or args.variant in INCOMPATIBLE_OPENINGS:
        if args.variant in INCOMPATIBLE_OPENINGS and not args.no_openings:
            print(f"[!] '{args.variant}' non e' compatibile con "
                  f"data/openings.txt: parto dalla posizione iniziale.")
        chosen = [None] * args.games
        print("Aperture: posizione iniziale della variante (nessun controllo)")
    else:
        openings = [l.strip() for l in
                    Path(args.match_openings).read_text().splitlines() if l.strip()]
        validate_openings(board_cls, openings, args.variant)
        chosen = rng.sample(openings, min(args.games, len(openings)))
        print(f"Aperture: {len(chosen)} campionate da {len(openings)}")

    print(f"Variante: {args.variant}   depth {args.depth}   "
          f"header [Variant]: {'SI' if keep_header else 'no'}")

    engine = chess.engine.SimpleEngine.popen_uci(args.engine)
    # NON impostare UCI_Variant: python-chess la deduce dalla classe del board.
    engine.configure({"Threads": 1, "Hash": 64})

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    counts, skipped, lens = {"1-0": 0, "0-1": 0, "1/2-1/2": 0}, 0, []

    try:
        with out.open("w", encoding="utf-8") as fh:
            for i, fen in enumerate(chosen):
                try:
                    board, moves = play(engine, board_cls, fen,
                                        args.depth, args.max_plies)
                except ValueError:
                    skipped += 1
                    continue
                if not moves:
                    skipped += 1
                    continue

                trunc = len(moves) >= args.max_plies
                r = "1/2-1/2" if trunc else board.result(claim_draw=True)
                if r == "*":
                    r = "1/2-1/2"
                counts[r] += 1
                lens.append(len(moves))

                g = chess.pgn.Game()
                # L'header [Variant] va impostato PRIMA di add_variation,
                # altrimenti il SAN viene generato su una board standard.
                if keep_header:
                    g.headers["Variant"] = args.variant
                g.headers["Event"] = f"{args.variant}_d{args.depth}"
                g.headers["White"] = f"FSF_d{args.depth}"
                g.headers["Black"] = f"FSF_d{args.depth}"
                g.headers["Round"] = str(i + 1)
                g.headers["Result"] = r
                if fen:
                    g.headers["FEN"] = fen
                    g.headers["SetUp"] = "1"

                node = g
                for mv in moves:
                    node = node.add_variation(mv)
                print(g, file=fh, end="\n\n")

                if (i + 1) % 200 == 0:
                    print(f"  {i+1}/{len(chosen)}")
    finally:
        engine.quit()

    tot = max(sum(counts.values()), 1)
    print(f"\nScritto: {out}   (scartate: {skipped})")
    for k, v in counts.items():
        print(f"  {k:8s} {v:5d}  ({100*v/tot:.1f}%)")
    dec = counts["1-0"] + counts["0-1"]
    print(f"Decisive: {dec} ({100*dec/tot:.1f}%)")
    if lens:
        import statistics
        med = statistics.median(lens)
        short = sum(1 for L in lens if L < 20)
        print(f"Lunghezza mediana: {med:.0f} ply   "
              f"partite sotto 20 ply: {short} ({100*short/len(lens):.1f}%)")
        if short > 0.3 * len(lens):
            print("  !! Molte partite corte: con min_moves=10 (che si applica")
            print("     per prospettiva, cioe' ~20 ply) ne perderai parecchie.")
            print("     Controlla n_eff dopo build_or_load_dataset PRIMA di T6.")
    print("\nRiferimento standard depth 6: 29.3 / 26.8 / 44.0,  decisive 56.0%")


if __name__ == "__main__":
    main()


# =============================================================================
# DOPO LA GENERAZIONE  (antichess)
# =============================================================================
#
# 1) Costruisci i parquet:
#
#    python -c "from chess_pca.pipeline import build_or_load_dataset as B; \
#    w,b=B(input_path='data/variant_antichess.pgn',max_games=2000,min_moves=10, \
#    cache_dir='data/cache_antichess',force_recompute=True); \
#    print(f'{w.game_id.nunique()} partite, {len(w)/w.game_id.nunique():.0f} ply'); \
#    print(list(w.columns))"
#
#    VERIFICA 1: quante partite sono sopravvissute al filtro min_moves.
#    VERIFICA 2: le colonne devono essere identiche ai corpora standard.
#
# 2) PRIMA di T6, lancia il controllo su phi:
#
#    python src\chess_pca\check_phi_validity.py --target antichess
#
#    Se checks_avail ha varianza ~0 (previsto), run_t6.signature() produce
#    NaN via np.corrcoef. Escludi la feature dal confronto e riportalo come
#    violazione misurata di A2 — non come dettaglio tecnico: e' il risultato.
#
# 3) In run_t6.py aggiungi dentro CORPORA:
#
#    "antichess": ("data/cache_antichess/white_data_max2000_min10.parquet",
#                  "data/cache_antichess/black_data_max2000_min10.parquet"),
#
#    e nella lista `order`.
# =============================================================================
