"""
plotting.py

Shared plotting utilities.

Currently:
- plot_unified_trajectories: generates a 2x2 panel (White/Black × PCA1/PCA2)
  showing mean trajectories for Win/Draw/Loss with optional standard error bands.

Input contract:
- `traj` is a dict: keys like "white_win", "black_draw", each value is (avg_df, se_df).
- avg_df is indexed by move number and includes columns ["PCA1","PCA2"].
"""

from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt


def plot_unified_trajectories(traj: dict, out_path: str | Path):
    """
    traj contains entries like:
      "white_win": (avg_df, se_df), etc.
    Saves a 2x2 figure to out_path.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Colors (kept simple)
    style = {
        "win":  {"label_w": "White Win",  "label_b": "Black Win",  "color_w": "green",    "color_b": "darkgreen"},
        "draw": {"label_w": "White Draw", "label_b": "Black Draw", "color_w": "gray",     "color_b": "dimgray"},
        "loss": {"label_w": "White Loss", "label_b": "Black Loss", "color_w": "red",      "color_b": "darkred"},
    }

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Unified PCA Trajectories: White vs Black, Win/Draw/Loss", fontsize=16, fontweight="bold", y=0.98)

    def plot_line(ax, avg, se, ycol, label, color):
        x = avg.index
        ax.plot(x, avg[ycol], color=color, linewidth=2.5, label=label, alpha=0.9)
        if se is not None:
            ax.fill_between(x, avg[ycol] - se[ycol], avg[ycol] + se[ycol], color=color, alpha=0.15)

    # --- WHITE PCA1 ---
    ax = axes[0, 0]
    for k in ("win", "draw", "loss"):
        avg, se = traj.get(f"white_{k}", (None, None))
        if avg is not None:
            plot_line(ax, avg, se, "PCA1", style[k]["label_w"], style[k]["color_w"])
    ax.set_title("WHITE - PCA1 Trajectory (Unified PCA)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Move Number", fontweight="bold")
    ax.set_ylabel("PCA1", fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=10)

    # --- WHITE PCA2 ---
    ax = axes[0, 1]
    for k in ("win", "draw", "loss"):
        avg, se = traj.get(f"white_{k}", (None, None))
        if avg is not None:
            plot_line(ax, avg, se, "PCA2", style[k]["label_w"], style[k]["color_w"])
    ax.set_title("WHITE - PCA2 Trajectory (Unified PCA)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Move Number", fontweight="bold")
    ax.set_ylabel("PCA2", fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=10)

    # --- BLACK PCA1 ---
    ax = axes[1, 0]
    for k in ("win", "draw", "loss"):
        avg, se = traj.get(f"black_{k}", (None, None))
        if avg is not None:
            plot_line(ax, avg, se, "PCA1", style[k]["label_b"], style[k]["color_b"])
    ax.set_title("BLACK - PCA1 Trajectory (Unified PCA)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Move Number", fontweight="bold")
    ax.set_ylabel("PCA1", fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=10)

    # --- BLACK PCA2 ---
    ax = axes[1, 1]
    for k in ("win", "draw", "loss"):
        avg, se = traj.get(f"black_{k}", (None, None))
        if avg is not None:
            plot_line(ax, avg, se, "PCA2", style[k]["label_b"], style[k]["color_b"])
    ax.set_title("BLACK - PCA2 Trajectory (Unified PCA)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Move Number", fontweight="bold")
    ax.set_ylabel("PCA2", fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=10)

    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
