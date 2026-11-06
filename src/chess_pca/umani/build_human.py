"""Converte le partite umane (TSV dal browser) in parquet nel formato di pipeline.py."""
import io, sys
from pathlib import Path
import numpy as np, pandas as pd, chess.pgn
sys.path.insert(0, "/mnt/user-data/uploads/chess-pca/src")
from chess_pca.features import extract_features_both_colors, numeric_features

src, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
meta, W, B = [], {}, {}
for line in src.read_text(encoding="utf-8").splitlines():
    p = line.split("\t")
    if len(p) != 10:
        continue
    tour, wn, bn, we, be, cls, tc, date, res, mv = p
    pgn = f'[White "{wn}"]\n[Black "{bn}"]\n[Result "{res}"]\n\n{mv}\n'
    g = chess.pgn.read_game(io.StringIO(pgn))
    if g is None or g.errors:
        continue
    dw, db = extract_features_both_colors(g, min_moves=10)
    if dw is None:
        continue
    gid = len(meta)
    meta.append(dict(game_id=gid, tour=tour, white=wn, black=bn, welo=int(we), belo=int(be),
                     cls=cls, tc=tc, date=date, result=res, plies=len(list(g.mainline_moves()))))
    for d, store in ((dw, W), (db, B)):
        d.insert(0, "game_id", gid)
        store.setdefault(cls, []).append(d)
meta = pd.DataFrame(meta)
meta.to_csv(out / "meta.csv", index=False)
for cls in W:
    w, b = pd.concat(W[cls], ignore_index=True), pd.concat(B[cls], ignore_index=True)
    f = numeric_features(w)
    for d in (w, b):
        d[f] = d[f].replace([np.inf, -np.inf], np.nan).fillna(0)
    w.to_parquet(out / f"{cls}_white.parquet", index=False)
    b.to_parquet(out / f"{cls}_black.parquet", index=False)
    print(cls, w.game_id.nunique())
print(meta.groupby("cls").result.value_counts().unstack())
