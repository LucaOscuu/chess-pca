"""
io_pgn.py

PGN ingestion utilities.
- Supports single .pgn / .pgn.gz files or directories (recursive).
- Yields chess.pgn.Game objects, optionally capped by `limit`.

Designed to be memory-efficient: streams games instead of loading everything at once.
"""

import gzip
from pathlib import Path
import chess.pgn


def is_pgn_file(path: Path) -> bool:
    if not path.is_file():
        return False
    name = path.name.lower()
    return name.endswith(".pgn") or name.endswith(".pgn.gz")


def open_pgn(path: Path):
    # .pgn.gz
    if path.suffix.lower() == ".gz" or path.name.lower().endswith(".pgn.gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="ignore")
    # .pgn
    return open(path, "r", encoding="utf-8", errors="ignore")


def iter_games_from_path(path: Path, limit: int | None = None):
    """
    Yield chess.pgn.Game objects from:
    - a single PGN/PGN.GZ file
    - a directory containing many .pgn / .pgn.gz
    """
    path = Path(path)
    count = 0

    if path.is_file():
        if not is_pgn_file(path):
            raise ValueError(f"File is not a PGN: {path}")
        with open_pgn(path) as f:
            while True:
                if limit is not None and count >= limit:
                    return
                game = chess.pgn.read_game(f)
                if game is None:
                    break
                yield game
                count += 1
        return

    if not path.exists():
        raise FileNotFoundError(f"Path not found: {path}")

    files = sorted(list(path.rglob("*.pgn")) + list(path.rglob("*.pgn.gz")))
    if not files:
        raise FileNotFoundError(f"No PGN files found under: {path}")

    for fp in files:
        with open_pgn(fp) as f:
            while True:
                if limit is not None and count >= limit:
                    return
                game = chess.pgn.read_game(f)
                if game is None:
                    break
                yield game
                count += 1
