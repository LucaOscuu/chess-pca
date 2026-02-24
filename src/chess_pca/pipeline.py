"""
pipeline.py

End-to-end pipeline for chess PCA experiments.

Stages:
1) build_or_load_dataset:
   - streams PGN games, extracts per-move features (white & black perspectives),
   - caches results to parquet (speed-up on reruns).

2) run_unified_analysis:
   - computes unified PCA for white and black datasets,
   - generates loading profiles and trajectory plots,
   - optionally runs validation/robustness figures (H1-H4).

Outputs:
- cached datasets: data/cache/*.parquet
- figures/tables: data/outputs/**

Notes:
- This module is intended to be the main CLI entrypoint in the repo.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

from chess_pca.io_pgn import iter_games_from_path
from chess_pca.features import extract_features_both_colors, numeric_features
from chess_pca.analysis import compute_unified_pca, compute_all_trajectories
from chess_pca.plotting import plot_unified_trajectories
from chess_pca.figure_h1 import run_figure_h1
from chess_pca.figure_h2 import run_figure_h2
from chess_pca.figure_h3 import run_figure_h3
from chess_pca.figure_h4_residence import run_residence_analysis



def _ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _cache_paths(cache_dir: str | Path, max_games: int, min_moves: int):
    cache_dir = _ensure_dir(cache_dir)
    w = cache_dir / f"white_data_max{max_games}_min{min_moves}.parquet"
    b = cache_dir / f"black_data_max{max_games}_min{min_moves}.parquet"
    return w, b


def build_or_load_dataset(
    input_path: str | Path,
    max_games: int = 10000,
    min_moves: int = 10,
    cache_dir: str | Path = "data/cache",
    force_recompute: bool = False,
):
    """
    Returns (white_data, black_data).
    Uses parquet cache if available.
    """
    w_path, b_path = _cache_paths(cache_dir, max_games=max_games, min_moves=min_moves)

    if (not force_recompute) and w_path.exists() and b_path.exists():
        white_data = pd.read_parquet(w_path)
        black_data = pd.read_parquet(b_path)
        return white_data, black_data

    input_path = Path(input_path)
    games = iter_games_from_path(input_path, limit=max_games)

    white_games = []
    black_games = []
    skipped = 0

    for game in tqdm(games, desc="Processing games", total=max_games):
        df_w, df_b = extract_features_both_colors(game, min_moves=min_moves)
        if df_w is None or df_b is None:
            skipped += 1
            continue
        white_games.append(df_w)
        black_games.append(df_b)

    if not white_games:
        raise ValueError("No valid games found (all skipped). Check input_path / filters.")

    white_data = (
        pd.concat(white_games, keys=range(len(white_games)), names=["game_id", "idx"])
        .reset_index(level=0)
    )
    black_data = (
        pd.concat(black_games, keys=range(len(black_games)), names=["game_id", "idx"])
        .reset_index(level=0)
    )

    # clean numeric
    features = numeric_features(white_data)
    for df in (white_data, black_data):
        df[features] = df[features].replace([np.inf, -np.inf], np.nan).fillna(0)

    white_data.to_parquet(w_path, index=False)
    black_data.to_parquet(b_path, index=False)

    print(f"\n✅ Cached datasets saved:")
    print(f"   • {w_path}")
    print(f"   • {b_path}")
    print(f"   • Games kept: {len(white_games)} (skipped {skipped})\n")

    return white_data, black_data


def run_unified_analysis(
    input_path: str | Path,
    max_games: int = 10000,
    min_moves: int = 10,
    cache_dir: str | Path = "data/cache",
    out_dir: str | Path = "data/outputs",
    force_recompute: bool = False,
    random_state: int = 42,
    max_moves_traj: int = 50,
    min_obs_traj: int = 50,
    run_h1: bool = False,
    h1_m_null: int = 500,
    run_h2: bool = False,
    h2_n_bootstrap: int = 80,
    h2_sample_fraction: float = 0.8,
    run_h3: bool = False,
    h3_start_move: int = 5,
    h3_exclude_resignation: bool = False,
    h3_bins_main: int = 50,
    h3_save_prefix: str = "h3",
    run_h4: bool = False,

):
    out_dir = _ensure_dir(out_dir)

    white_data, black_data = build_or_load_dataset(
        input_path=input_path,
        max_games=max_games,
        min_moves=min_moves,
        cache_dir=cache_dir,
        force_recompute=force_recompute,
    )

    FEATURES = numeric_features(white_data)

    white_res = compute_unified_pca(white_data, FEATURES, random_state=random_state)
    black_res = compute_unified_pca(black_data, FEATURES, random_state=random_state)

    from chess_pca.figure_loadings import run_loading_profiles

    run_loading_profiles(
        white_loadings=white_res["loadings"],
        black_loadings=black_res["loadings"],
        out_dir=Path(out_dir) / "loading_profiles",
)
    
    if run_h1:
        run_figure_h1(
            white_data=white_data,
            black_data=black_data,
            features=FEATURES,
            out_dir=out_dir,
            M_null=h1_m_null,
            random_state=random_state,
        )

    if run_h2:
        run_figure_h2(
            white_data=white_data,
            black_data=black_data,
            features=FEATURES,
            out_dir=out_dir,
            n_bootstrap=h2_n_bootstrap,
            sample_fraction=h2_sample_fraction,
            random_state=random_state,
        )
    if run_h3:
        run_figure_h3(
            white_data=white_data,
            black_data=black_data,
            features=FEATURES,
            pca_white=white_res["pca"],
            pca_black=black_res["pca"],
            out_dir=out_dir,
            start_move=h3_start_move,
            exclude_resignation=h3_exclude_resignation,
            bins_main=h3_bins_main,
            save_prefix=h3_save_prefix,
    )

    if run_h4:
        run_residence_analysis(
            white_data=white_data,
            black_data=black_data,
            features=FEATURES,
            out_dir=out_dir,
            omega_win_thresh=-1.5,
            omega_loss_thresh=+1.0,
            random_state=random_state,
        )


    var1_w, var2_w = white_res["explained_variance_ratio"]
    var1_b, var2_b = black_res["explained_variance_ratio"]

    print("📈 VARIANCE EXPLAINED")
    print(f"White (unified): PCA1={var1_w:.2%}, PCA2={var2_w:.2%}, Total={(var1_w+var2_w):.2%}")
    print(f"Black (unified): PCA1={var1_b:.2%}, PCA2={var2_b:.2%}, Total={(var1_b+var2_b):.2%}\n")

    traj = compute_all_trajectories(
        white_res,
        black_res,
        max_moves=max_moves_traj,
        min_obs=min_obs_traj,
    )

    out_png = Path(out_dir) / "chess_unified_pca.png"
    plot_unified_trajectories(traj, out_png)
    print(f"💾 Output image saved to: {out_png}\n")

    def _print_loadings(loadings, title):
        print(title)
        print("─" * 60)
        top1 = loadings.sort_values("PCA1", key=abs, ascending=False).head(5)
        print("PCA1 (Top 5):")
        for feat, val in top1["PCA1"].items():
            print(f"  {'+' if val > 0 else '-'} {feat:25s}: {val:+.3f}")
        top2 = loadings.sort_values("PCA2", key=abs, ascending=False).head(5)
        print("PCA2 (Top 5):")
        for feat, val in top2["PCA2"].items():
            print(f"  {'+' if val > 0 else '-'} {feat:25s}: {val:+.3f}")
        print()

    _print_loadings(white_res["loadings"], "WHITE (Unified PCA)")
    _print_loadings(black_res["loadings"], "BLACK (Unified PCA)")

    return white_res, black_res, traj
