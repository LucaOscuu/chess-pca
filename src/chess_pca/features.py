"""
features.py

Feature extraction from chess positions (python-chess).

Main entrypoint:
- extract_features_both_colors(game, min_moves): iterates through mainline moves and
  builds two datasets: one from White's perspective and one from Black's perspective.

Feature families:
- material & piece-count differences (perspective-based)
- mobility / captures / checks availability
- center control (core + extended)
- pawn structure (doubled / isolated / passed)
- king safety proxy and hanging pieces
- attack overlap (shared attacked squares)

Outputs:
- df_white, df_black with one row per move for each player's perspective.
- `result` encoded as W/D/L from each perspective.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import chess

PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 20000,
}

CENTER_SQ = [chess.D4, chess.E4, chess.D5, chess.E5]
EXT_CENTER = [
    chess.C3, chess.C4, chess.C5, chess.C6,
    chess.D3, chess.E3, chess.D6, chess.E6,
    chess.F3, chess.F4, chess.F5, chess.F6,
]


def material_eval(b: chess.Board, perspective_color: bool) -> int:
    """Material balance from perspective_color viewpoint."""
    mat = sum(
        PIECE_VALUES[pt] * (len(b.pieces(pt, chess.WHITE)) - len(b.pieces(pt, chess.BLACK)))
        for pt in PIECE_VALUES
    )
    return mat if perspective_color == chess.WHITE else -mat


def piece_counts(b: chess.Board, perspective_color: bool) -> dict[str, int]:
    """Piece count difference from perspective."""
    diff = 1 if perspective_color == chess.WHITE else -1
    return {
        f"cnt_{chess.piece_name(pt)}": diff
        * (len(b.pieces(pt, chess.WHITE)) - len(b.pieces(pt, chess.BLACK)))
        for pt in [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]
    }


def mobility_stats(b: chess.Board, color: bool) -> tuple[int, int, int]:
    """Mobility, captures, checks for given color."""
    tmp = b.copy(stack=False)
    tmp.turn = color
    moves = list(tmp.legal_moves)
    return (
        len(moves),
        sum(1 for m in moves if tmp.is_capture(m)),
        sum(1 for m in moves if tmp.gives_check(m)),
    )


def attack_overlap(b: chess.Board) -> int:
    """Squares attacked by both colors."""
    return sum(1 for sq in chess.SQUARES if b.attackers(chess.WHITE, sq) and b.attackers(chess.BLACK, sq))


def hanging_pieces(b: chess.Board, perspective_color: bool) -> int:
    """Undefended pieces (positive = advantage for perspective_color)."""

    def count(color: bool) -> int:
        return sum(
            1
            for sq in chess.SquareSet(b.occupied_co[color])
            if b.attackers(not color, sq) and not b.attackers(color, sq)
        )

    diff = count(not perspective_color) - count(perspective_color)
    return diff


def center_control(b: chess.Board, perspective_color: bool) -> tuple[int, int]:
    """Center control from perspective."""
    opp = not perspective_color

    def ctrl(c: bool, sqs) -> int:
        return sum(1 for sq in sqs if b.attackers(c, sq))

    core = ctrl(perspective_color, CENTER_SQ) - ctrl(opp, CENTER_SQ)
    ext = ctrl(perspective_color, EXT_CENTER) - ctrl(opp, EXT_CENTER)
    return core, ext


def king_safety(b: chess.Board, perspective_color: bool) -> int:
    """King safety difference (positive = advantage)."""

    def safety(c: bool) -> int:
        ksq = b.king(c)
        if ksq is None:
            return 0
        kf, kr = chess.square_file(ksq), chess.square_rank(ksq)
        ring = [
            chess.square(kf + df, kr + dr)
            for df in (-1, 0, 1)
            for dr in (-1, 0, 1)
            if (df or dr) and 0 <= kf + df <= 7 and 0 <= kr + dr <= 7
        ]
        attacks = sum(1 for sq in ring if b.attackers(not c, sq))
        shield_rank = kr + (1 if c == chess.WHITE else -1)
        shield = 0
        if 0 <= shield_rank <= 7:
            for fo in (-1, 0, 1):
                ff = kf + fo
                if 0 <= ff <= 7:
                    sq = chess.square(ff, shield_rank)
                    if (
                        b.piece_type_at(sq) == chess.PAWN
                        and b.color_at(sq) == c
                    ):
                        shield += 1
        return shield - attacks

    return safety(perspective_color) - safety(not perspective_color)


def pawn_structure(b: chess.Board, perspective_color: bool) -> tuple[int, int, int]:
    """Pawn structure from perspective: (doubled_diff, isolated_diff, passed_diff)."""

    def feat(c: bool) -> tuple[int, int, int]:
        pawns = b.pieces(chess.PAWN, c)
        files = [chess.square_file(sq) for sq in pawns]

        doubled = sum(max(0, files.count(f) - 1) for f in range(8))

        isolated = 0
        for f in range(8):
            if files.count(f) > 0:
                left = files.count(f - 1) if f - 1 >= 0 else 0
                right = files.count(f + 1) if f + 1 <= 7 else 0
                if left == 0 and right == 0:
                    isolated += 1

        passed = 0
        for sq in pawns:
            sq_file = chess.square_file(sq)
            if c == chess.WHITE:
                ahead = range(sq + 8, 64, 8)
            else:
                ahead = range(sq - 8, -1, -8)

            blocked = False
            for a in ahead:
                a_file = chess.square_file(a)
                if a_file in (sq_file - 1, sq_file, sq_file + 1):
                    if b.piece_type_at(a) == chess.PAWN and b.color_at(a) != c:
                        blocked = True
                        break
            if not blocked:
                passed += 1

        return doubled, isolated, passed

    opp = not perspective_color
    dP, iP, pP = feat(perspective_color)
    dO, iO, pO = feat(opp)
    return (dP - dO, iP - iO, pP - pO)


def extract_features_both_colors(game, min_moves: int = 10):
    """
    Extract features for BOTH White and Black moves.
    Returns (df_white, df_black) or (None, None) if invalid.
    """
    board = game.board()
    white_rows = []
    black_rows = []
    ply = 0

    try:
        for move in game.mainline_moves():
            board.push(move)
            ply += 1

            # who just moved?
            player_color = chess.WHITE if board.turn == chess.BLACK else chess.BLACK

            mob, cap, chk = mobility_stats(board, player_color)
            mat = material_eval(board, player_color)
            counts = piece_counts(board, player_color)
            core, ext = center_control(board, player_color)
            pawn_d, pawn_i, pawn_p = pawn_structure(board, player_color)

            row = {
                "ply": ply,
                "material": mat,
                "mobility": mob,
                "captures_avail": cap,
                "checks_avail": chk,
                "attack_overlap": attack_overlap(board),
                "hanging_diff": hanging_pieces(board, player_color),
                "center_core": core,
                "center_ext": ext,
                "king_safety_diff": king_safety(board, player_color),
                "pawns_doubled_diff": pawn_d,
                "pawns_isolated_diff": pawn_i,
                "pawns_passed_diff": pawn_p,
                **counts,
            }

            if player_color == chess.WHITE:
                row["move_num"] = len(white_rows) + 1
                white_rows.append(row)
            else:
                row["move_num"] = len(black_rows) + 1
                black_rows.append(row)

    except Exception:
        return None, None

    if len(white_rows) < min_moves or len(black_rows) < min_moves:
        return None, None

    result = game.headers.get("Result", "*")
    if result not in ["1-0", "0-1", "1/2-1/2"]:
        return None, None

    df_white = pd.DataFrame(white_rows)
    df_black = pd.DataFrame(black_rows)

    if result == "1-0":
        df_white["result"] = "W"
        df_black["result"] = "L"
    elif result == "0-1":
        df_white["result"] = "L"
        df_black["result"] = "W"
    else:
        df_white["result"] = "D"
        df_black["result"] = "D"

    df_white["delta_mobility"] = df_white["mobility"].diff().fillna(0)
    df_black["delta_mobility"] = df_black["mobility"].diff().fillna(0)

    return df_white, df_black


def numeric_features(df: pd.DataFrame) -> list[str]:
    """
    Return numeric feature columns to use for PCA (exclude ids / indices).
    """
    cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in ["ply", "move_num", "game_id"]]
    return cols
