from chess_pca.pipeline import build_or_load_dataset

w, b = build_or_load_dataset(
    input_path="data/random_games.pgn",
    max_games=2000,
    min_moves=10,
    cache_dir="data/cache_random",   # cartella separata: nessuna collisione
    force_recompute=True,
)
print(f"White: {len(w):,} righe, {w.game_id.nunique()} partite")
print(f"Black: {len(b):,} righe, {b.game_id.nunique()} partite")
print(f"Colonne: {list(w.columns)}")