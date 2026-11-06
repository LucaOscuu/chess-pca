# Data

- `raw/fishtest_2018/`: Stockfish self-play games from Fishtest (main corpus, `.pgn.gz`).
- `outputs/`: numerical results and figures written by the scripts.
- `cache/` (not tracked): parquet tables of features, rebuilt by `scripts/run_pipeline.py`.

The corpora below were generated with the scripts in `src/chess_pca/`. The random and engine corpora start from the same opening book (`openings.txt`), so comparisons between them are paired by opening.

| File | Content | Script |
|---|---|---|
| `openings.txt` | Opening book extracted from the Fishtest corpus | `extract_openings.py` |
| `random_games.pgn`, `random_games_100.pgn` | Uniform random mover | `generate_random_corpus.py` |
| `engine_d1.pgn`, `engine_d3.pgn`, `engine_d6.pgn` | Stockfish self-play at search depth 1, 3, 6 | `generate_engine_corpus.py` |
| `variant_3check.pgn` | Three-Check (win by giving three checks) | `generate_variant_corpus.py` |
| `variant_antichess.pgn` | Antichess (objective inverted: lose all pieces to win) | `generate_variant_corpus.py` |
