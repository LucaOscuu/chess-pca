"""
analysis.py

Core PCA utilities:
- compute_unified_pca: fit a 2D PCA on all observations of a perspective (white/black),
  then attach PCA1/PCA2 coordinates and split by outcome (W/D/L).
- compute_trajectory: build mean PCA trajectory by move number (with standard error).
- compute_all_trajectories: convenience wrapper for both colors and all outcomes.

Assumptions:
- `data` contains a `result` column with values in {"W","D","L"}.
- `move_num` exists for trajectory computations.
- `features` are numeric columns.

Returns:
- sklearn PCA + scaler, loadings table, subsets, explained variance ratios.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def compute_unified_pca(data: pd.DataFrame, features: list[str], random_state: int = 42):
    """
    Fit ONE PCA on ALL observations of a given color dataset.
    Then project and split by result (W/D/L).
    """
    X = data[features].to_numpy(dtype=float, copy=True)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2, random_state=random_state)
    components = pca.fit_transform(X_scaled)

    out = data.copy()
    out["PCA1"] = components[:, 0]
    out["PCA2"] = components[:, 1]

    loadings = pd.DataFrame(
        pca.components_.T,
        index=features,
        columns=["PCA1", "PCA2"],
    )

    subsets = {
        "all": out,
        "win": out[out["result"] == "W"].copy(),
        "draw": out[out["result"] == "D"].copy(),
        "loss": out[out["result"] == "L"].copy(),
    }

    return {
        "pca": pca,
        "scaler": scaler,
        "loadings": loadings,
        "subsets": subsets,
        "explained_variance_ratio": pca.explained_variance_ratio_,
    }


def compute_trajectory(subset: pd.DataFrame, max_moves: int = 50, min_obs: int = 50):
    """
    Average trajectory and standard error by move_num for a subset.
    Returns (avg_df, se_df) or (None, None).
    """
    if subset is None or len(subset) < min_obs:
        return None, None

    sub = subset[subset["move_num"] <= max_moves].copy()
    if sub.empty:
        return None, None

    avg = sub.groupby("move_num")[["PCA1", "PCA2"]].mean()
    std = sub.groupby("move_num")[["PCA1", "PCA2"]].std()
    n = sub.groupby("move_num").size().clip(lower=1)
    se = std.div(np.sqrt(n), axis=0)
    return avg, se


def compute_all_trajectories(white_res: dict, black_res: dict, max_moves: int = 50, min_obs: int = 50):
    """
    Convenience wrapper: compute trajectories for W/D/L for both colors.
    """
    w = white_res["subsets"]
    b = black_res["subsets"]

    traj = {}

    for key in ("win", "draw", "loss"):
        traj[f"white_{key}"] = compute_trajectory(w.get(key), max_moves=max_moves, min_obs=min_obs)
        traj[f"black_{key}"] = compute_trajectory(b.get(key), max_moves=max_moves, min_obs=min_obs)

    return traj